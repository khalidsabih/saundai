from __future__ import annotations
from dataclasses import asdict,dataclass
import os
import hashlib
from pathlib import Path
from typing import Sequence
import numpy as np
from ..environment import config_path
from .model_registry import load_model_registry

@dataclass(frozen=True,slots=True)
class NeuralBeatResult:
    backend:str; model:str; available:bool; beats_seconds:tuple[float,...]; downbeats_seconds:tuple[float,...]; bpm:float; confidence:float; agreement:float; authority:float; used:bool; reason:str
    tempo_confidence:float=0.0
    beat_alignment_confidence:float=0.0
    downbeat_confidence:float=0.0
    tempo_used:bool=False
    tempo_reason:str="not evaluated"
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
    """Use multi-beat slopes to reduce Beat This!'s 20ms frame quantization.

    Single adjacent frame intervals can round a 122 BPM pulse to 120 BPM.
    A robust 8-beat span preserves long-term tempo without trusting nominal
    synthetic BPM or a reference label.
    """
    b = np.asarray(beats, dtype=float)
    if b.size < 2 or not np.isfinite(b).all() or np.any(np.diff(b) <= 0):
        return 0.0
    stride = min(8, max(1, b.size // 4))
    intervals = (b[stride:] - b[:-stride]) / stride
    median = float(np.median(intervals))
    return 60.0 / median if median > 1e-8 else 0.0


def _tempo_confidence(beats: Sequence[float]) -> float:
    """Internal tempo stability, not independently verified BPM accuracy."""
    values = np.asarray(beats, dtype=float)
    if values.size < 16 or not np.all(np.isfinite(values)):
        return 0.0
    intervals = np.diff(values)
    if np.any(intervals <= 0.0):
        return 0.0
    median = float(np.median(intervals))
    if median <= 1e-8:
        return 0.0
    relative_deviation = float(np.median(np.abs(intervals - median)) / median)
    coverage = min(1.0, intervals.size / 64.0)
    return float(np.clip((1.0 - relative_deviation / 0.08) * coverage, 0.0, 1.0))


def _downbeat_confidence(beats: Sequence[float], downbeats: Sequence[float]) -> float:
    """Measure geometric support only; musical bar-one accuracy is unverified."""
    if len(beats) < 8 or len(downbeats) < 2:
        return 0.0
    b = np.asarray(beats, dtype=float)
    d = np.asarray(downbeats, dtype=float)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(d)):
        return 0.0
    nearest = np.asarray([np.min(np.abs(b - value)) for value in d])
    aligned = float(np.mean(nearest <= 0.07))
    support = min(1.0, len(d) / max(1.0, len(b) / 4.5))
    return float(np.clip(aligned * support, 0.0, 1.0))


def tempo_only_adoption(
    result: NeuralBeatResult, fallback_bpm: float
) -> tuple[bool, str]:
    """Never use tempo-only evidence without independent tempo agreement.

    Default is observation-only until real-audio evaluation supports promotion.
    This gate never authorizes replacing beats or downbeats.
    """
    if not _truthy(os.environ.get("DJAI_NEURAL_TEMPO_ONLY", "0")):
        return False, "tempo-only adoption disabled (observation only)"
    if not result.available or not np.isfinite(result.bpm) or not np.isfinite(fallback_bpm):
        return False, "tempo evidence unavailable"
    if not (40.0 <= result.bpm <= 240.0 and 40.0 <= fallback_bpm <= 240.0):
        return False, "tempo outside supported range"
    if result.tempo_confidence < 0.80:
        return False, "insufficient neural tempo stability"
    if abs(result.bpm - fallback_bpm) / fallback_bpm > 0.02:
        return False, "independent tempo estimates disagree"
    authority = load_model_registry().authority("beat_this", result.tempo_confidence)
    if authority <= 0:
        return False, "model lacks tempo advisory authority"
    return True, "independent tempo agreement passed; beat and downbeat grids unchanged"


def downbeat_bar_integrity(beats: Sequence[float], downbeats: Sequence[float]) -> tuple[bool, str]:
    """Conservative meter/phase-coherence veto before neural bars enter planning.

    Structural evidence only: regular bar spacing cannot prove the correct
    downbeat phase. Irregular/variable-meter music abstains by design.
    """
    b = np.asarray(beats, dtype=float)
    d = np.asarray(downbeats, dtype=float)
    if b.size < 12 or d.size < 3 or not np.isfinite(b).all() or not np.isfinite(d).all():
        return False, "insufficient or nonfinite neural beat/downbeat evidence"
    if np.any(np.diff(b) <= 0) or np.any(np.diff(d) <= 0):
        return False, "unordered neural beat/downbeat timestamps"
    period = float(np.median(np.diff(b)))
    if period <= 0:
        return False, "invalid neural beat period"
    positions = np.searchsorted(b, d)
    left = np.clip(positions - 1, 0, b.size - 1)
    right = np.clip(positions, 0, b.size - 1)
    nearest = np.where(np.abs(b[left] - d) <= np.abs(b[right] - d), left, right)
    tolerance = min(0.07, 0.20 * period)
    if np.any(np.abs(b[nearest] - d) > tolerance):
        return False, "neural downbeats are not aligned to predicted beats"
    jumps = np.diff(nearest)
    if np.any(jumps <= 0):
        return False, "duplicate or unordered neural bar-start beat indices"
    # A constant 3/4/6 pulse bar length is the only proven-safe structure
    # for this trial; we do not invent/correct bars from unstable predictions.
    if int(jumps[0]) not in (3, 4, 6) or np.any(jumps != jumps[0]):
        return False, "irregular neural bar spacing or surplus downbeat labels"
    return True, "regular neural bar spacing and beat alignment passed"


def _checkpoint_integrity(path: str, expected_sha256: str) -> tuple[bool, str]:
    """Optional release pin; reject path/name-based models when a hash is required."""
    if not expected_sha256:
        return True, "checkpoint hash not requested"
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        return False, "invalid DJAI_BEAT_THIS_SHA256 (must be 64 hex characters)"
    source = Path(path).expanduser()
    if not source.is_file():
        return False, "SHA256 pin requires an explicit local checkpoint file"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False, "unable to read pinned Beat This! checkpoint"
    if digest.hexdigest() != expected:
        return False, "Beat This! checkpoint SHA256 mismatch"
    return True, "Beat This! checkpoint SHA256 verified"


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
    integrity_ok, integrity_reason = _checkpoint_integrity(value, os.environ.get("DJAI_BEAT_THIS_SHA256", ""))
    if not integrity_ok: return NeuralBeatResult("beat_this",value,False,(),(),0,0,0,0,False,integrity_reason)
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
    # Always fail closed: neural beats and bars are an indivisible input to
    # cue/phrase planning. An inconsistent bar-start sequence is not permitted
    # to replace the deterministic grid, regardless of environment flags.
    # Candidate timestamps remain visible for offline comparison in evidence.
    if used:
        coherent, bar_reason = downbeat_bar_integrity(bv, dv)
        if not coherent:
            used = False
            reason = "neural bar-grid safety veto: " + bar_reason
    return NeuralBeatResult("beat_this",value,True,bv,dv,round(_estimate_bpm(bv),4),round(confidence,4),round(agreement,4),round(authority,4),used,reason,round(_tempo_confidence(bv),4),round(agreement,4),round(_downbeat_confidence(bv,dv),4))

