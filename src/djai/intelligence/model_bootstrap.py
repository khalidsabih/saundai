from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any

def _resolve_device(value: str) -> str:
    device=str(value or "auto").strip().lower()
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

def bootstrap_models(*, beat_this: bool=True, mert: bool=False) -> dict[str,Any]:
    results: dict[str,Any]={}
    if beat_this:
        try:
            from beat_this.inference import File2Beats
            model=os.environ.get("DJAI_BEAT_THIS_MODEL","small0"); device=_resolve_device(os.environ.get("DJAI_BEAT_THIS_DEVICE","auto"))
            File2Beats(checkpoint_path=model,device=device,dbn=False)
            marker=Path(os.environ.get("DJAI_BEAT_THIS_READY_MARKER","config/neural/beat-this-ready.json")).expanduser(); marker.parent.mkdir(parents=True,exist_ok=True); marker.write_text(json.dumps({"model":model,"device":device,"ready":True},indent=2)+"\n")
            results["beat_this"]={"ready":True,"model":model,"device":device,"marker":str(marker)}
        except Exception as exc: results["beat_this"]={"ready":False,"error":str(exc)}
    if mert:
        try:
            from transformers import AutoFeatureExtractor,AutoModel
            model_id=os.environ.get("DJAI_MERT_MODEL","m-a-p/MERT-v1-95M")
            AutoFeatureExtractor.from_pretrained(model_id,trust_remote_code=True); AutoModel.from_pretrained(model_id,trust_remote_code=True)
            results["mert"]={"ready":True,"model":model_id}
        except Exception as exc: results["mert"]={"ready":False,"error":str(exc)}
    results["model_cache"]={"TORCH_HOME":os.environ.get("TORCH_HOME"),"HF_HOME":os.environ.get("HF_HOME")}
    return results
