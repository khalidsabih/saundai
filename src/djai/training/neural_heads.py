from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any,Literal
import numpy as np
from ..intelligence.model_registry import ensure_default_registry,checkpoint_fingerprint
from ..intelligence.neural_policy import HEAD_SCHEMA_VERSION

_TASK={"transition_family":("transition_family_policy","multiclass"),"transition_family_policy":("transition_family_policy","multiclass"),"stem_choreography":("stem_choreography_policy","multiclass"),"stem_choreography_policy":("stem_choreography_policy","multiclass"),"duration":("duration_policy","scalar"),"duration_policy":("duration_policy","scalar"),"rendered_judge":("rendered_dj_judge","scalar"),"rendered_dj_judge":("rendered_dj_judge","scalar"),"next_track":("next_track_policy","scalar"),"next_track_policy":("next_track_policy","scalar")}
def _bucket(v):return int(hashlib.sha256(v.encode()).hexdigest()[:8],16)%10
def _rows(path,rights=True):
    rows=[]
    for n,line in enumerate(Path(path).expanduser().read_text().splitlines(),1):
        if not line.strip():continue
        try:r=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f"Invalid JSONL at line {n}: {exc}") from exc
        if not isinstance(r,dict) or not isinstance(r.get("features"),dict):continue
        if rights and not bool(r.get("rights_cleared",False)):continue
        rows.append(r)
    return rows
def _matrix(rows):
    names=sorted({str(k) for r in rows for k in r["features"]}); x=np.asarray([[float(r["features"].get(n,0)) for n in names] for r in rows],float); mean=x.mean(0); scale=x.std(0); scale=np.where(scale<1e-6,1,scale); return names,(x-mean)/scale,{n:float(v) for n,v in zip(names,mean,strict=True)},{n:float(v) for n,v in zip(names,scale,strict=True)}
def _split(rows):
    group_values=np.asarray([str(r.get("group") or r.get("context_id") or r.get("id") or i) for i,r in enumerate(rows)],dtype=object)
    b=np.asarray([_bucket(str(value)) for value in group_values]); va=np.flatnonzero(b>=8); tr=np.flatnonzero(b<8)
    unique=sorted(set(str(value) for value in group_values))
    # Preserve group isolation even when the deterministic hash happens to put
    # every group on the same side. Row-level fallback would leak the same DJ
    # transition pair into both train and validation.
    if len(unique)>=2 and (len(va)==0 or len(tr)==0):
        heldout=unique[-1]
        va=np.flatnonzero(group_values==heldout)
        tr=np.flatnonzero(group_values!=heldout)
    elif len(va)==0 and len(rows)>=5:
        va=np.asarray([len(rows)-1]);tr=np.arange(len(rows)-1)
    if len(tr)==0:
        tr=np.arange(len(rows));va=np.asarray([],int)
    return tr,va
def _sw(rows):return np.asarray([max(.05,min(10,float(r.get("weight",1)))) for r in rows],float)
def train_neural_head(task,dataset,output,*,require_rights_cleared=True,iterations=800,learning_rate=.035,l2=.002,initial_model_path=None):
    if task not in _TASK:raise ValueError(f"Unsupported neural task: {task}")
    model_id,kind=_TASK[task]; rows=_rows(dataset,require_rights_cleared); minimum=8 if kind=="multiclass" else 6
    if len(rows)<minimum:raise ValueError(f"Need at least {minimum} rights-cleared examples for {model_id}; found {len(rows)}.")
    names,x,means,scales=_matrix(rows); tr,va=_split(rows); sw=_sw(rows); payload={"schema_version":HEAD_SCHEMA_VERSION,"model_id":model_id,"task":model_id,"kind":kind,"feature_names":names,"feature_mean":means,"feature_scale":scales,"rights_cleared_only":require_rights_cleared}
    initial={}
    if initial_model_path:
        try:initial=json.loads(Path(initial_model_path).expanduser().resolve().read_text())
        except Exception:initial={}
    if initial and list(initial.get("feature_names",[]))!=names:initial={}
    if kind=="multiclass":
        labels=sorted({str(r.get("label","")) for r in rows if str(r.get("label",""))});
        if len(labels)<2:raise ValueError("Multiclass training requires at least two labels.")
        idx={l:i for i,l in enumerate(labels)}; y=np.asarray([idx[str(r["label"])] for r in rows]); w=np.zeros((len(labels),x.shape[1])); b=np.zeros(len(labels))
        if initial and list(initial.get("labels",[]))==labels:
            iw=np.asarray(initial.get("weights",[]),float); ib=np.asarray(initial.get("bias",[]),float)
            if iw.shape==w.shape:w=iw.copy()
            if ib.shape==b.shape:b=ib.copy()
        for _ in range(max(50,int(iterations))):
            xt=x[tr]; logits=xt@w.T+b; logits-=logits.max(1,keepdims=True); p=np.exp(np.clip(logits,-60,60));p/=p.sum(1,keepdims=True);t=np.zeros_like(p);t[np.arange(len(tr)),y[tr]]=1; s=sw[tr,None]; err=(p-t)*s; den=max(float(s.sum()),1e-9); w-=learning_rate*(err.T@xt/den+l2*w); b-=learning_rate*err.sum(0)/den
        acc=lambda ind:0. if len(ind)==0 else float(np.mean(np.argmax(x[ind]@w.T+b,1)==y[ind])); metrics={"train_accuracy":round(acc(tr),6),"validation_accuracy":round(acc(va),6),"training_examples":int(len(tr)),"validation_examples":int(len(va)),"examples":len(rows)}; payload.update(labels=labels,weights=w.tolist(),bias=b.tolist(),metrics=metrics)
    else:
        y=[]
        for r in rows:
            if "target" in r:v=float(r["target"])
            elif "label" in r:v=1. if str(r["label"]).lower() in {"1","true","yes","accept","preferred","good"} else 0.
            else:raise ValueError("Scalar rows require target or label")
            y.append(float(np.clip(v,0,1)))
        y=np.asarray(y);w=np.zeros(x.shape[1]);b=0.
        if initial:
            iw=np.asarray(initial.get("weights",[]),float)
            if iw.ndim==2 and iw.shape[0]==1:iw=iw[0]
            if iw.shape==w.shape:w=iw.copy()
            ib=initial.get("bias",0.)
            b=float(ib[0] if isinstance(ib,list) and ib else ib)
        for _ in range(max(50,int(iterations))):
            xt=x[tr]; p=1/(1+np.exp(-np.clip(xt@w+b,-40,40)));err=(p-y[tr])*sw[tr];den=max(float(sw[tr].sum()),1e-9);w-=learning_rate*(xt.T@err/den+l2*w);b-=learning_rate*float(err.sum()/den)
        def met(ind):
            if len(ind)==0:return 0.,0.
            p=1/(1+np.exp(-np.clip(x[ind]@w+b,-40,40)));return float(np.mean((p>=.5)==(y[ind]>=.5))),float(np.mean(abs(p-y[ind])))
        ta,tm=met(tr);va_a,va_m=met(va);metrics={"train_accuracy":round(ta,6),"train_mae":round(tm,6),"validation_accuracy":round(va_a,6),"validation_mae":round(va_m,6),"training_examples":int(len(tr)),"validation_examples":int(len(va)),"examples":len(rows)};payload.update(weights=w.tolist(),bias=float(b),metrics=metrics)
    if initial:
        payload["fine_tuned_from_model_id"]=str(initial.get("model_id") or Path(initial_model_path).stem)
    out=Path(output).expanduser();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n");return out

def promote_model(
    model_id,
    *,
    authority: Literal["shadow", "advisory", "production"],
    registry_path=None,
    rights_reviewed=False,
    governance_approved=False,
    arena_report_payload=None,
    human_release_gate_payload=None,
):
    path = ensure_default_registry(registry_path)
    payload = json.loads(path.read_text())
    target = next(
        (row for row in payload.get("models", []) if isinstance(row, dict) and row.get("id") == model_id),
        None,
    )
    if target is None:
        raise ValueError(f"Unknown model id: {model_id}")
    cp = target.get("checkpoint")
    if not cp or not Path(cp).expanduser().exists():
        raise ValueError(f"Checkpoint is not installed for {model_id}.")
    hp = json.loads(Path(cp).expanduser().read_text())
    metrics = dict(hp.get("metrics", {}))
    accuracy = float(metrics.get("validation_accuracy", 0))
    validation_examples = int(metrics.get("validation_examples", 0))
    if authority == "advisory" and (validation_examples < 10 or accuracy < 0.55):
        raise ValueError("Advisory promotion requires >=10 held-out examples and >=55% validation accuracy.")
    release_gate = dict(human_release_gate_payload or {})
    if authority == "production":
        if not rights_reviewed:
            raise ValueError("Production promotion requires an explicit rights review.")
        if validation_examples < 50 or accuracy < 0.62:
            raise ValueError("Production promotion requires >=50 held-out examples and >=62% validation accuracy.")
        if not governance_approved:
            raise ValueError("Production promotion requires explicit model-governance approval.")
        if str(release_gate.get("release_gate_version") or "") != "1.0":
            raise ValueError("Production promotion requires a Phase 11 locked human release-gate report.")
        if not bool(release_gate.get("benchmark_locked")):
            raise ValueError("Production promotion requires a locked human release benchmark.")
        if not release_gate.get("benchmark_id") or not release_gate.get("locked_sha256"):
            raise ValueError("Production promotion requires benchmark identity and locked SHA-256.")
        if not bool(release_gate.get("deployment_ready")):
            raise ValueError("Production promotion requires a deployment-ready Phase 11 human release gate.")

    target["authority"] = authority
    target["production_approved"] = authority == "production" and rights_reviewed and governance_approved
    target["checkpoint_sha256"] = checkpoint_fingerprint(cp)
    metadata = target.setdefault("metadata", {})
    metadata["validation_metrics"] = metrics
    metadata["rights_reviewed"] = bool(rights_reviewed)
    metadata["governance_approved"] = bool(governance_approved)
    metadata["arena_report"] = dict(arena_report_payload or {})
    metadata["human_release_gate"] = release_gate
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def evaluate_neural_head(task,dataset,model_path,*,require_rights_cleared=False):
    if task not in _TASK:raise ValueError(f"Unsupported neural task: {task}")
    model_id,kind=_TASK[task];rows=_rows(dataset,require_rights_cleared)
    if not rows:return {"examples":0,"accuracy":None,"mae":None}
    payload=json.loads(Path(model_path).expanduser().resolve().read_text())
    names=list(payload.get("feature_names",[]))
    if not names:return {"examples":0,"accuracy":None,"mae":None}
    x=np.asarray([[float(r["features"].get(n,0)) for n in names] for r in rows],float)
    means=payload.get("feature_mean",{}) if isinstance(payload.get("feature_mean"),dict) else {}
    scales=payload.get("feature_scale",{}) if isinstance(payload.get("feature_scale"),dict) else {}
    x=np.asarray([[(row[i]-float(means.get(n,0)))/max(1e-6,abs(float(scales.get(n,1)))) for i,n in enumerate(names)] for row in x],float)
    if kind=="multiclass":
        labels=[str(x) for x in payload.get("labels",[])]
        label_index={label:i for i,label in enumerate(labels)}
        usable=[i for i,r in enumerate(rows) if str(r.get("label","")) in label_index]
        if not usable:return {"examples":0,"accuracy":None,"mae":None}
        idx=np.asarray(usable,int);w=np.asarray(payload.get("weights",[]),float);b=np.asarray(payload.get("bias",[]),float)
        y=np.asarray([label_index[str(rows[i]["label"])] for i in usable],int)
        pred=np.argmax(x[idx]@w.T+b,axis=1)
        return {"examples":len(idx),"accuracy":round(float(np.mean(pred==y)),6),"mae":None}
    y=[]
    for r in rows:
        if "target" in r:y.append(float(np.clip(float(r["target"]),0,1)))
        elif "label" in r:y.append(1. if str(r["label"]).lower() in {"1","true","yes","accept","preferred","good"} else 0.)
        else:y.append(0.5)
    y=np.asarray(y,float);w=np.asarray(payload.get("weights",[]),float)
    if w.ndim==2 and w.shape[0]==1:w=w[0]
    bp=payload.get("bias",0.);b=float(bp[0] if isinstance(bp,list) and bp else bp)
    probs=1/(1+np.exp(-np.clip(x@w+b,-40,40)))
    return {
        "examples":len(rows),
        "accuracy":round(float(np.mean((probs>=.5)==(y>=.5))),6),
        "mae":round(float(np.mean(np.abs(probs-y))),6),
    }
