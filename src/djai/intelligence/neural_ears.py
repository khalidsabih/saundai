from __future__ import annotations
from dataclasses import asdict,dataclass
import os
from pathlib import Path
from typing import Sequence
import numpy as np
from ..environment import config_path
from .model_registry import load_model_registry

@dataclass(frozen=True,slots=True)
class NeuralBeatResult:
    backend:str; model:str; available:bool; beats_seconds:tuple[float,...]; downbeats_seconds:tuple[float,...]; bpm:float; confidence:float; agreement:float; authority:float; used:bool; reason:str
    def to_dict(self): return asdict(self)

def _truthy(v): return str(v or "").strip().lower() in {"1","true","yes","on"}
def _grid_confidence(beats):
    b=np.asarray(beats,dtype=float)
    if len(b)<4:return 0.0
    d=np.diff(b); m=float(np.median(d));
    if m<=1e-8:return 0.0
    return float(np.clip(1.0-float(np.std(d)/m)/0.12,0,1))
def _agreement(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if len(a)<2 or len(b)<2:return 0.5
    tol=0.07; matches=sum(float(np.min(np.abs(a-x)))<=tol for x in b)
    return float(matches/max(1,len(b)))
def _estimate_bpm(beats):
    b=np.asarray(beats,dtype=float)
    if len(b)<2:return 0.0
    m=float(np.median(np.diff(b))); return 60.0/m if m>1e-8 else 0.0


def downbeat_spacing_integrity(beats:Sequence[float],downbeats:Sequence[float])->tuple[bool,str]:
    """Reject impossible one-downbeat-per-beat grids in opt-in safety trials.

    This is a structural check only, not a claim of correct musical meter or
    downbeat phase. Genuine 3-, 4- and 6-pulse groupings remain possible.
    """
    b=np.asarray(beats,dtype=float); d=np.asarray(downbeats,dtype=float)
    if len(b)<4 or len(d)<2 or not np.isfinite(b).all() or not np.isfinite(d).all():
        return False,"insufficient or nonfinite beat/downbeat evidence"
    if np.any(np.diff(b)<=0) or np.any(np.diff(d)<=0):
        return False,"unordered beat or downbeat timestamps"
    beat_period=float(np.median(np.diff(b)))
    if beat_period<=0:
        return False,"invalid beat period"
    down_periods=np.diff(d)/beat_period
    if float(np.median(down_periods))<2.5:
        return False,"neural downbeats are too dense relative to beats"
    return True,"structural downbeat spacing passed"



def predict_beats(path,*,fallback_beats:Sequence[float]=(),fallback_downbeats:Sequence[float]=()):
    enabled=os.environ.get("DJAI_NEURAL_BEATS","auto").strip().lower()
    if enabled in {"0","off","false","no","disabled"}: return NeuralBeatResult("none","none",False,(),(),0,0,0,0,False,"neural beats disabled")
    checkpoint=os.environ.get("DJAI_BEAT_THIS_MODEL","small0"); explicit=os.environ.get("DJAI_BEAT_THIS_CHECKPOINT","").strip(); allow=_truthy(os.environ.get("DJAI_MODEL_ALLOW_DOWNLOAD")) or _truthy(os.environ.get("DJAI_BEAT_THIS_ALLOW_DOWNLOAD")); value=explicit or checkpoint
    if explicit and not Path(explicit).expanduser().exists(): return NeuralBeatResult("beat_this",value,False,(),(),0,0,0,0,False,"explicit Beat This! checkpoint is missing")
    marker=config_path(os.environ.get("DJAI_BEAT_THIS_READY_MARKER","neural/beat-this-ready.json"))
    if not explicit and not allow and not marker.exists(): return NeuralBeatResult("beat_this",value,False,(),(),0,0,0,0,False,"checkpoint not bootstrapped; run `djai models-bootstrap` or provide DJAI_BEAT_THIS_CHECKPOINT")
    try: from beat_this.inference import File2Beats
    except ImportError: return NeuralBeatResult("beat_this",value,False,(),(),0,0,0,0,False,"beat-this package is not installed")
    device=os.environ.get("DJAI_BEAT_THIS_DEVICE","auto").strip().lower()
    if device=="auto":
        # Upstream Beat This! documents CUDA/CPU inference. Keep Apple Silicon
        # on the reliable CPU path by default; advanced users may opt into MPS
        # explicitly with DJAI_BEAT_THIS_DEVICE=mps.
        try:
            import torch
            device="cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device="cpu"
    try: beats,downbeats=File2Beats(checkpoint_path=value,device=device,dbn=False)(str(Path(path).expanduser().resolve()))
    except Exception as exc: return NeuralBeatResult("beat_this",value,False,(),(),0,0,0,0,False,f"Beat This! inference unavailable: {type(exc).__name__}")
    bv=tuple(float(x) for x in np.asarray(beats).reshape(-1) if float(x)>=0); dv=tuple(float(x) for x in np.asarray(downbeats).reshape(-1) if float(x)>=0)
    grid=_grid_confidence(bv); agreement=_agreement(fallback_beats,bv); support=min(1.0,len(dv)/max(1.0,len(bv)/4.5)) if bv else 0; confidence=float(np.clip(.50*grid+.32*agreement+.18*support,0,1)); authority=load_model_registry().authority("beat_this",confidence); threshold=float(os.environ.get("DJAI_NEURAL_BEAT_USE_THRESHOLD","0.70")); used=authority>0 and confidence>=max(.5,min(.95,threshold)) and len(dv)>=2
    reason="neural beat grid passed authority/confidence gate" if used else "neural beat grid recorded as evidence but deterministic grid retained"
    # Explicit opt-in for trials only: protect the complete grid + BPM from
    # implausibly dense downbeats, never promote a scalar tempo alone.
    if used and _truthy(os.environ.get("DJAI_NEURAL_DOWNBEAT_INTEGRITY")):
        valid,integrity_reason=downbeat_spacing_integrity(bv,dv)
        if not valid:
            used=False
            reason="opt-in neural downbeat integrity rejected grid: "+integrity_reason
    return NeuralBeatResult("beat_this",value,True,bv,dv,round(_estimate_bpm(bv),4),round(confidence,4),round(agreement,4),round(authority,4),used,reason)

