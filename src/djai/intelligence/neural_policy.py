from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json, math, os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from ..environment import config_path
from .model_registry import checkpoint_fingerprint, load_model_registry

if TYPE_CHECKING:
    from ..brain.set_state import SetState
    from ..brain.transition_router import TransitionRoute
    from ..models import TrackAnalysis, TransitionPlan

HEAD_SCHEMA_VERSION="1.0"
TRANSITION_FAMILIES=("direct_blend","layer_handoff","breakdown_drop","rhythmic_switch","hard_reset","bridge_preferred")
STEM_PROFILES=("progressive_handoff","outgoing_vocal_bridge","instrumental_reset","drum_first_build","bass_ownership_swap")
RENDER_JUDGE_EMBEDDING_DIMS=24
_STEM_PROFILE_SET=set(STEM_PROFILES)

@dataclass(frozen=True,slots=True)
class HeadPrediction:
    model_id:str; available:bool; labels:tuple[str,...]=(); probabilities:tuple[float,...]=(); confidence:float=0.; authority:float=0.; top_label:str|None=None; checkpoint_fingerprint:str=""; reason:str=""
    def probability(self,label,default=0.):
        try:return float(self.probabilities[self.labels.index(str(label))])
        except (ValueError,IndexError):return float(default)
    def ordered_labels(self,allowed:Iterable[str]|None=None):
        allow=set(allowed) if allowed is not None else None
        return tuple(label for label,_ in sorted(zip(self.labels,self.probabilities,strict=False),key=lambda x:(x[1],x[0]),reverse=True) if allow is None or label in allow)
    def to_dict(self):return {"model_id":self.model_id,"available":self.available,"labels":list(self.labels),"probabilities":[round(float(x),6) for x in self.probabilities],"confidence":round(self.confidence,6),"authority":round(self.authority,6),"top_label":self.top_label,"checkpoint_fingerprint":self.checkpoint_fingerprint,"reason":self.reason}
@dataclass(frozen=True,slots=True)
class ScalarPrediction:
    model_id:str; available:bool; score:float=.5; confidence:float=0.; authority:float=0.; checkpoint_fingerprint:str=""; reason:str=""
    def to_dict(self):return {"model_id":self.model_id,"available":self.available,"score":round(self.score,6),"confidence":round(self.confidence,6),"authority":round(self.authority,6),"checkpoint_fingerprint":self.checkpoint_fingerprint,"reason":self.reason}

def _sigmoid(v): v=max(-40.,min(40.,float(v))); return 1/(1+math.exp(-v))
def _softmax(v):
    if v.size==0:return v
    x=v-np.max(v); e=np.exp(np.clip(x,-60,60)); d=float(np.sum(e)); return e/d if d>0 and np.isfinite(d) else np.full_like(v,1/max(1,v.size),dtype=float)
def _cosine(a,b):
    if not a or not b:return 0.
    n=min(len(a),len(b)); x=np.asarray(a[:n],float); y=np.asarray(b[:n],float); d=float(np.linalg.norm(x)*np.linalg.norm(y)); return float(np.clip(np.dot(x,y)/d,-1,1)) if d>1e-12 else 0.

def _pair_features(a:"TrackAnalysis",b:"TrackAnalysis",state:"SetState|None"=None):
    from ..compatibility import genre_compatibility,harmonic_compatibility,match_bpm
    ba=max(float(a.bpm),1e-6); bb=max(float(b.bpm),1e-6); _,pd,_=match_bpm(ba,bb)
    return {"bpm_direct_delta":min(1,abs(bb-ba)/ba),"bpm_perceived_delta":min(1,float(pd)),"harmonic_similarity":float(harmonic_compatibility(a.key,b.key)),"genre_similarity":float(genre_compatibility(a,b)),"embedding_similarity":float(np.clip(.5+.5*_cosine(a.music_embedding,b.music_embedding),0,1)),"energy_a":float(a.energy),"energy_b":float(b.energy),"energy_delta":float(b.energy-a.energy),"percussive_a":float(a.percussive_ratio),"percussive_b":float(b.percussive_ratio),"bass_a":float(a.bass_activity),"bass_b":float(b.bass_activity),"vocal_a":float(a.vocal_probability),"vocal_b":float(b.vocal_probability),"brightness_a":float(a.brightness),"brightness_b":float(b.brightness),"beat_confidence_a":float(a.beat_grid_confidence),"beat_confidence_b":float(b.beat_grid_confidence),"structure_confidence_a":float(a.structure_confidence),"structure_confidence_b":float(b.structure_confidence),"state_energy":float(getattr(state,"energy",.5)) if state else .5,"state_tension":float(getattr(state,"tension",.5)) if state else .5,"state_vocal_density":float(getattr(state,"vocal_density",.5)) if state else .5}


def _safe_feature_float(value: Any, default: float = 0.0) -> float:
    try:
        result=float(value)
    except (TypeError,ValueError):
        return float(default)
    return float(result) if math.isfinite(result) else float(default)


def _render_embedding_features(values: Any) -> dict[str,float]:
    if not isinstance(values,(list,tuple)) or not values:
        return {"render_embedding_present":0.0,**{f"render_embedding_{index:02d}":0.0 for index in range(RENDER_JUDGE_EMBEDDING_DIMS)}}
    vector=np.asarray(values,dtype=float).reshape(-1)
    vector=np.nan_to_num(vector,nan=0.0,posinf=0.0,neginf=0.0)
    folded=np.zeros(RENDER_JUDGE_EMBEDDING_DIMS,dtype=float)
    for index,value in enumerate(vector):
        slot=index%RENDER_JUDGE_EMBEDDING_DIMS
        folded[slot]+=float(value) if (index//RENDER_JUDGE_EMBEDDING_DIMS)%2==0 else -float(value)
    norm=float(np.linalg.norm(folded))
    if norm>1e-12:folded/=norm
    return {"render_embedding_present":1.0,**{f"render_embedding_{index:02d}":float(value) for index,value in enumerate(folded)}}


def rendered_judge_payload_features(payload: Mapping[str,Any]) -> dict[str,float]:
    """Feature view shared by Training Lab exports and the live rendered judge."""
    compatibility=payload.get("compatibility") if isinstance(payload.get("compatibility"),Mapping) else {}
    measurement=payload.get("measurement") if isinstance(payload.get("measurement"),Mapping) else {}
    stem_evidence=bool(measurement.get("stem_evidence",False))
    render_bass=(measurement.get("stem_bass_cleanliness") if stem_evidence else measurement.get("bass_cleanliness"))
    features={
        "plan_score":_safe_feature_float(payload.get("analytical_score",payload.get("score",0.5)),0.5),
        "cue_confidence":_safe_feature_float(payload.get("cue_confidence"),0.5),
        "intent_score":_safe_feature_float(payload.get("intent_score"),0.5),
        "technique_score":_safe_feature_float(payload.get("technique_score"),0.5),
        "compat_tempo":_safe_feature_float(compatibility.get("tempo"),0.5),
        "compat_harmonic":_safe_feature_float(compatibility.get("harmonic"),0.5),
        "compat_energy":_safe_feature_float(compatibility.get("energy"),0.5),
        "compat_texture":_safe_feature_float(compatibility.get("texture"),0.5),
        "compat_phrase":_safe_feature_float(compatibility.get("phrase"),0.5),
        "compat_sections":_safe_feature_float(compatibility.get("sections"),0.5),
        "compat_beat_grid":_safe_feature_float(compatibility.get("beat_grid"),0.5),
        "compat_vocal":_safe_feature_float(compatibility.get("vocal_overlap"),0.5),
        "compat_bass":_safe_feature_float(compatibility.get("bass_conflict"),0.5),
        "bars":_safe_feature_float(payload.get("transition_bars"),8.0),
        "render_overall":_safe_feature_float(measurement.get("overall",payload.get("measured_score",0.5)),0.5),
        "render_musical":_safe_feature_float(measurement.get("musical_overall"),0.5),
        "render_rhythm":_safe_feature_float(measurement.get("rhythm_coherence"),0.5),
        "render_transient":_safe_feature_float(measurement.get("transient_clarity"),0.5),
        "render_groove":_safe_feature_float(measurement.get("groove_continuity"),0.5),
        "render_harmonic":_safe_feature_float(measurement.get("harmonic_continuity"),0.5),
        "render_vocal":_safe_feature_float(measurement.get("vocal_cleanliness"),0.5),
        "render_bass":_safe_feature_float(render_bass,0.5),
        "render_energy_fit":_safe_feature_float(measurement.get("energy_intent_fit"),0.5),
        "render_drop_payoff":_safe_feature_float(measurement.get("drop_payoff"),0.5),
    }
    features.update(_render_embedding_features(payload.get("rendered_transition_embedding")))
    return features


def transition_feature_dict(a:"TrackAnalysis",b:"TrackAnalysis",*,route:"TransitionRoute|None"=None,plan:"TransitionPlan|None"=None,set_state:"SetState|None"=None,duration_bars:int|None=None):
    f=_pair_features(a,b,set_state)
    if route:
        f.update(route_confidence=float(route.confidence),route_direct_confidence=float(route.direct_transition_confidence),route_bridge_preference=float(route.bridge_preference),route_groove_similarity=float(route.groove_similarity),route_layer_capability=float(route.layer_capability)); f.update({f"route_is_{x}":1. if route.family==x else 0. for x in TRANSITION_FAMILIES})
    if plan:
        c=plan.compatibility
        payload={
            "analytical_score":float(plan.score),
            "cue_confidence":float(plan.cue_confidence),
            "intent_score":float(plan.intent_score),
            "technique_score":float(plan.technique_score),
            "transition_bars":float(plan.transition_bars),
            "compatibility":{
                "tempo":float(c.tempo),"harmonic":float(c.harmonic),"energy":float(c.energy),"texture":float(c.texture),
                "phrase":float(c.phrase),"sections":float(c.sections),"beat_grid":float(c.beat_grid),
                "vocal_overlap":float(c.vocal_overlap),"bass_conflict":float(c.bass_conflict),
            },
            "measurement":plan.measurement.to_dict() if plan.measurement else None,
            "rendered_transition_embedding":list((plan.metadata or {}).get("rendered_transition_embedding",[])),
        }
        f.update(rendered_judge_payload_features(payload))
    if duration_bars is not None:f["duration_bars"]=float(duration_bars); f["duration_log2"]=math.log2(max(1,float(duration_bars)))
    return f

def _file_state(path: Path) -> tuple[str, int, int]:
    try:
        resolved=path.expanduser().resolve()
    except OSError:
        resolved=path.expanduser().absolute()
    try:
        stat=resolved.stat()
        return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)
    except OSError:
        return str(resolved), -1, -1

@lru_cache(maxsize=16)
def _registry_cached(path_str: str, mtime_ns: int, size: int):
    del mtime_ns, size
    return load_model_registry(path_str)

def clear_neural_policy_cache() -> None:
    """Refresh registry/checkpoint state after local shadow training."""
    _registry_cached.cache_clear()
    _checkpoint_cached.cache_clear()


def _registry():
    path=config_path(os.environ.get("DJAI_MODEL_REGISTRY","model-registry.json"))
    key=_file_state(path)
    registry=_registry_cached(*key)
    # A missing registry can be created by load_model_registry. Refresh the cache key once
    # so subsequent graph-search calls reuse the created file rather than a synthetic -1 key.
    if key[1] < 0:
        refreshed=_file_state(path)
        if refreshed[1] >= 0:
            registry=_registry_cached(*refreshed)
    return registry

@lru_cache(maxsize=64)
def _checkpoint_cached(model_id: str, path_str: str, mtime_ns: int, size: int, expected_sha: str):
    del mtime_ns, size
    p=Path(path_str)
    try:
        payload=json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return None,f"checkpoint unreadable: {exc}"
    if str(payload.get("schema_version"))!=HEAD_SCHEMA_VERSION:
        return None,"unsupported checkpoint schema"
    checkpoint_model_id=str(payload.get("model_id",model_id))
    checkpoint_task=str(payload.get("task",""))
    if checkpoint_model_id!=model_id:
        # Pairwise next-track training emits a research checkpoint id while the
        # runtime registry exposes the stable policy id. Allow that one
        # task-compatible alias; all other model ids remain exact-match.
        if not (model_id=="next_track_policy" and checkpoint_task=="next_track_policy"):
            return None,"checkpoint model id mismatch"
    fp=checkpoint_fingerprint(p)
    if expected_sha and fp.lower()!=expected_sha.lower():
        return None,"checkpoint hash does not match registry"
    payload["_fingerprint"]=fp
    return payload,""

def _load(model_id):
    registry=_registry(); manifest=registry.get(model_id)
    if not manifest or not manifest.checkpoint:
        return None,registry,"model not registered or checkpoint unset"
    checkpoint_path=config_path(manifest.checkpoint)
    checkpoint_state=_file_state(checkpoint_path)
    if checkpoint_state[1] < 0:
        return None,registry,"checkpoint not installed"
    payload,reason=_checkpoint_cached(model_id,*checkpoint_state,str(manifest.checkpoint_sha256 or ""))
    return payload,registry,reason
def _vector(payload,features):
    names=[str(x) for x in payload.get("feature_names",[])]; means=payload.get("feature_mean",{}) if isinstance(payload.get("feature_mean"),dict) else {}; scales=payload.get("feature_scale",{}) if isinstance(payload.get("feature_scale"),dict) else {}
    return np.asarray([(float(features.get(n,0))-float(means.get(n,0)))/max(1e-6,abs(float(scales.get(n,1)))) for n in names],float)
def predict_head(model_id,features):
    p,r,reason=_load(model_id)
    if p is None:return HeadPrediction(model_id,False,reason=reason)
    labels=tuple(str(x) for x in p.get("labels",[])); w=np.asarray(p.get("weights",[]),float); b=np.asarray(p.get("bias",[0]*len(labels)),float); x=_vector(p,features)
    if not labels or w.shape!=(len(labels),len(x)) or b.shape!=(len(labels),):return HeadPrediction(model_id,False,reason="checkpoint dimensions invalid")
    probs=_softmax(w@x+b); conf=float(np.max(probs)); auth=float(r.authority(model_id,conf)); top=labels[int(np.argmax(probs))]
    return HeadPrediction(model_id,True,labels,tuple(float(v) for v in probs),conf,auth,top,str(p.get("_fingerprint",""))[:16],"authoritative" if auth>0 else "shadow prediction only")
def predict_scalar_head(model_id,features):
    p,r,reason=_load(model_id)
    if p is None:return ScalarPrediction(model_id,False,reason=reason)
    x=_vector(p,features)
    kind=str(p.get("kind",""))
    if kind=="mlp_pairwise_ranker":
        params=p.get("parameters",{}) if isinstance(p.get("parameters"),dict) else {}
        try:
            w1=np.asarray(params.get("w1",[]),float); b1=np.asarray(params.get("b1",[]),float)
            w2=np.asarray(params.get("w2",[]),float); b2=np.asarray(params.get("b2",[]),float)
            w3=np.asarray(params.get("w3",[]),float); b3=np.asarray(params.get("b3",[]),float)
            if w1.ndim!=2 or w1.shape[0]!=x.shape[0] or b1.shape!=(w1.shape[1],):
                return ScalarPrediction(model_id,False,reason="checkpoint MLP input dimensions invalid")
            if w2.ndim!=2 or w2.shape[0]!=w1.shape[1] or b2.shape!=(w2.shape[1],):
                return ScalarPrediction(model_id,False,reason="checkpoint MLP hidden dimensions invalid")
            if w3.ndim!=2 or w3.shape!=(w2.shape[1],1) or b3.size!=1:
                return ScalarPrediction(model_id,False,reason="checkpoint MLP output dimensions invalid")
            a1=np.maximum(x@w1+b1,0.0)
            a2=np.maximum(a1@w2+b2,0.0)
            raw=float((a2@w3+b3.reshape(-1))[0])
            score=_sigmoid(raw)
        except (TypeError,ValueError,IndexError):
            return ScalarPrediction(model_id,False,reason="checkpoint MLP parameters invalid")
    else:
        w=np.asarray(p.get("weights",[]),float); w=w[0] if w.ndim==2 and w.shape[0]==1 else w
        if w.shape!=x.shape:return ScalarPrediction(model_id,False,reason="checkpoint dimensions invalid")
        bp=p.get("bias",0.); bias=float(bp[0] if isinstance(bp,list) and bp else bp); score=_sigmoid(float(np.dot(w,x))+bias)
    conf=float(np.clip(abs(score-.5)*2,0,1))
    auth=float(r.authority(model_id,conf))
    return ScalarPrediction(model_id,True,score,conf,auth,str(p.get("_fingerprint",""))[:16],"authoritative" if auth>0 else "shadow prediction only")
def transition_family_prediction(a,b,*,route=None,set_state=None):return predict_head("transition_family_policy",transition_feature_dict(a,b,route=route,set_state=set_state))
def rank_duration_values(a,b,route,durations,*,set_state=None):
    scored=[]; preds=[]; ma=0.
    for i,bars in enumerate(durations):
        p=predict_scalar_head("duration_policy",transition_feature_dict(a,b,route=route,set_state=set_state,duration_bars=int(bars))); preds.append({"bars":int(bars),**p.to_dict()}); ma=max(ma,p.authority); h=1-i/max(1,len(durations)); influence=min(.75,2*p.authority); scored.append(((1-influence)*h+influence*p.score,int(bars)))
    if ma<=0:return tuple(int(x) for x in durations),{"model_id":"duration_policy","authority":0.,"predictions":preds,"changed_order":False}
    ordered=tuple(v for _,v in sorted(scored,key=lambda x:(x[0],x[1]),reverse=True)); return ordered,{"model_id":"duration_policy","authority":round(ma,6),"predictions":preds,"changed_order":ordered!=tuple(durations)}
def rank_stem_profiles(a,b,route,plan,profiles,*,set_state=None):
    p=predict_head("stem_choreography_policy",transition_feature_dict(a,b,route=route,plan=plan,set_state=set_state)); original=tuple(str(x) for x in profiles)
    if not p.available or p.authority<=0:return original,p.to_dict()
    learned=[x for x in p.ordered_labels(STEM_PROFILES) if x in _STEM_PROFILE_SET]; merged=tuple(dict.fromkeys([*learned,*original])); return merged,{**p.to_dict(),"changed_order":merged!=original}
def scalar_head_uses_feature_prefix(model_id: str, prefix: str, *, require_authority: bool = False) -> bool:
    payload, registry, _ = _load(model_id)
    if payload is None:
        return False
    if require_authority:
        manifest = registry.get(model_id)
        if manifest is None or manifest.authority in {"disabled", "shadow"} or manifest.effective_weight <= 0.0:
            return False
    return any(str(name).startswith(prefix) for name in payload.get("feature_names", []))

def rendered_judge_prediction(a,b,plan,*,set_state=None):return predict_scalar_head("rendered_dj_judge",transition_feature_dict(a,b,plan=plan,set_state=set_state))
def next_track_prediction(a,b,*,set_state=None):return predict_scalar_head("next_track_policy",transition_feature_dict(a,b,set_state=set_state))

def experimental_next_track_prediction(a,b,*,set_state=None):
    """Use the best available shadow next-track MLP for retrieval experiments.

    Prefer the mixed synthetic + Phase16 real-rehearsal checkpoint when present;
    otherwise fall back to the research-synthetic checkpoint. In both cases the
    model has zero final authority and may only narrow/rank candidates before
    deterministic rehearsal and safety.
    """
    candidates=(
        (
            config_path("real-retrieval-models"),
            "NEXT_TRACK_REAL_RETRIEVAL_LATEST",
            "next_track_real_retrieval_filter",
        ),
        (
            config_path("research-synthetic-models"),
            "NEXT_TRACK_RESEARCH_LATEST",
            "next_track_research_filter",
        ),
    )
    payload=None
    path=None
    model_label="next_track_research_filter"
    for root,pointer_name,label in candidates:
        pointer=root/pointer_name
        try:
            filename=pointer.read_text(encoding="utf-8").strip()
            if not filename:
                continue
            candidate_path=(root/filename).resolve()
            candidate_payload=json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):
            continue
        payload=candidate_payload
        path=candidate_path
        model_label=label
        break
    if payload is None or path is None:
        return ScalarPrediction("next_track_research_filter",False,reason="no experimental retrieval checkpoint installed")
    if str(payload.get("task",""))!="next_track_policy" and str(payload.get("learning_head",""))!="next_track_policy":
        return ScalarPrediction(model_label,False,reason="experimental checkpoint task mismatch")
    features=transition_feature_dict(a,b,set_state=set_state)
    x=_vector(payload,features)
    kind=str(payload.get("kind",""))
    if kind!="mlp_pairwise_ranker":
        return ScalarPrediction(model_label,False,reason="experimental checkpoint is not the MLP pairwise ranker")
    params=payload.get("parameters",{}) if isinstance(payload.get("parameters"),dict) else {}
    try:
        w1=np.asarray(params.get("w1",[]),float); b1=np.asarray(params.get("b1",[]),float)
        w2=np.asarray(params.get("w2",[]),float); b2=np.asarray(params.get("b2",[]),float)
        w3=np.asarray(params.get("w3",[]),float); b3=np.asarray(params.get("b3",[]),float)
        if w1.ndim!=2 or w1.shape[0]!=x.shape[0] or b1.shape!=(w1.shape[1],):
            return ScalarPrediction(model_label,False,reason="experimental MLP input dimensions invalid")
        if w2.ndim!=2 or w2.shape[0]!=w1.shape[1] or b2.shape!=(w2.shape[1],):
            return ScalarPrediction(model_label,False,reason="experimental MLP hidden dimensions invalid")
        if w3.ndim!=2 or w3.shape!=(w2.shape[1],1) or b3.size!=1:
            return ScalarPrediction(model_label,False,reason="experimental MLP output dimensions invalid")
        a1=np.maximum(x@w1+b1,0.0)
        a2=np.maximum(a1@w2+b2,0.0)
        raw=float((a2@w3+b3.reshape(-1))[0])
        score=_sigmoid(raw)
    except (TypeError,ValueError,IndexError):
        return ScalarPrediction(model_label,False,reason="experimental MLP parameters invalid")
    confidence=float(np.clip(abs(score-.5)*2,0,1))
    return ScalarPrediction(
        model_label,
        True,
        score,
        confidence,
        0.0,
        checkpoint_fingerprint(path)[:16],
        "shadow-only retrieval filter; final authority remains deterministic",
    )
