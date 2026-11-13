from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..environment import config_path

MODEL_REGISTRY_VERSION = "5.0"
ModelAuthority = Literal["disabled", "shadow", "advisory", "production"]
_AUTHORITY_WEIGHTS = {"disabled": 0.0, "shadow": 0.0, "advisory": 0.35, "production": 0.75}

@dataclass(frozen=True, slots=True)
class ModelManifest:
    id: str
    task: str
    provider: str
    model: str
    version: str
    authority: ModelAuthority = "shadow"
    weight: float = 0.0
    minimum_confidence: float = 0.55
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    source: str = ""
    license: str = "review"
    training_data_provenance: str = "unknown"
    production_approved: bool = False
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_weight(self) -> float:
        # An explicit zero is a hard zero. Authority defines only the ceiling;
        # it must never silently substitute a non-zero weight.
        ceiling = _AUTHORITY_WEIGHTS.get(self.authority, 0.0)
        return max(0.0, min(ceiling, float(self.weight)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class ModelRegistry:
    manifests: dict[str, ModelManifest]
    path: Path | None = None

    def get(self, model_id: str) -> ModelManifest | None:
        return self.manifests.get(model_id)

    def authority(self, model_id: str, confidence: float = 1.0) -> float:
        manifest = self.get(model_id)
        if manifest is None or manifest.authority in {"disabled", "shadow"}:
            return 0.0
        confidence = max(0.0, min(1.0, float(confidence)))
        if confidence < manifest.minimum_confidence:
            return 0.0
        if manifest.authority == "production" and not manifest.production_approved:
            return 0.0
        return manifest.effective_weight * confidence

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": MODEL_REGISTRY_VERSION, "models": [m.to_dict() for m in self.manifests.values()]}

def _default_registry_path() -> Path:
    return config_path(os.environ.get("DJAI_MODEL_REGISTRY", "model-registry.json"))

def default_manifests() -> list[ModelManifest]:
    return [
        ModelManifest(id="beat_this", task="beat_downbeat", provider="CPJKU", model="Beat This! small0", version="1.1.0", authority="advisory", weight=0.34, minimum_confidence=0.66, source="https://github.com/CPJKU/beat_this", license="MIT code and published weights; product training-data review required", training_data_provenance="published Beat This! training corpus", notes="Neural rhythm evidence is fused with DJAI diagnostics; timing validation remains authoritative."),
        ModelManifest(id="music_encoder", task="music_representation", provider="DJAI adapter", model="MusicFM or MERT", version="5.0-adapter", authority="advisory", weight=0.35, minimum_confidence=0.50, source="https://github.com/minzwon/musicfm ; https://github.com/yizhilll/MERT", license="backend-specific", training_data_provenance="backend-specific", notes="Foundation embeddings are sensory evidence, not DJ taste."),
        ModelManifest(id="transition_family_policy", task="transition_family", provider="DJAI", model="DJAI Transition Family Head", version="1.0", authority="shadow", checkpoint="config/neural/transition-family.json", license="DJAI-owned weights only", training_data_provenance="rights-cleared demonstrations/preferences", notes="Promote only after held-out/Arena validation."),
        ModelManifest(id="duration_policy", task="transition_duration", provider="DJAI", model="DJAI Duration Head", version="1.0", authority="shadow", checkpoint="config/neural/duration.json", license="DJAI-owned weights only", training_data_provenance="rights-cleared demonstrations/preferences"),
        ModelManifest(id="stem_choreography_policy", task="stem_choreography", provider="DJAI", model="DJAI Stem Choreography Head", version="1.0", authority="shadow", checkpoint="config/neural/stem-choreography.json", license="DJAI-owned weights only", training_data_provenance="rights-cleared stems + preference data"),
        ModelManifest(id="rendered_dj_judge", task="rendered_transition_preference", provider="DJAI", model="DJAI Rendered Transition Judge", version="1.0", authority="shadow", checkpoint="config/neural/rendered-judge.json", license="DJAI-owned weights only", training_data_provenance="A/B/Tie/Neither + expert demonstrations"),
        ModelManifest(id="next_track_policy", task="next_track", provider="DJAI", model="DJAI Next Track Head", version="1.0", authority="shadow", checkpoint="config/neural/next-track.json", license="DJAI-owned weights only", training_data_provenance="rights-cleared full-set demonstrations/preferences"),
    ]

def ensure_default_registry(path: str | Path | None = None) -> Path:
    destination = Path(path).expanduser() if path is not None else _default_registry_path()
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": MODEL_REGISTRY_VERSION, "policy": {"shadow": "Predictions are logged but cannot change a final plan.", "advisory": "Predictions may reorder/narrow search with bounded authority.", "production": "Requires production_approved=true, rights review, held-out validation, blind Arena readiness, governance approval and checkpoint pinning.", "default_deny": True}, "models": [m.to_dict() for m in default_manifests()]}
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination

def _manifest_from_dict(p: dict[str, Any]) -> ModelManifest:
    authority = str(p.get("authority", "shadow"))
    if authority not in _AUTHORITY_WEIGHTS:
        authority = "shadow"
    return ModelManifest(id=str(p.get("id","unknown")), task=str(p.get("task","unknown")), provider=str(p.get("provider","unknown")), model=str(p.get("model","unknown")), version=str(p.get("version","unknown")), authority=authority, weight=float(p.get("weight",0.0) or 0.0), minimum_confidence=float(p.get("minimum_confidence",0.55) or 0.55), checkpoint=str(p["checkpoint"]) if p.get("checkpoint") else None, checkpoint_sha256=str(p["checkpoint_sha256"]) if p.get("checkpoint_sha256") else None, source=str(p.get("source","")), license=str(p.get("license","review")), training_data_provenance=str(p.get("training_data_provenance","unknown")), production_approved=bool(p.get("production_approved",False)), notes=str(p.get("notes","")), metadata=dict(p.get("metadata",{})) if isinstance(p.get("metadata"),dict) else {})

def load_model_registry(path: str | Path | None = None) -> ModelRegistry:
    source = Path(path).expanduser() if path is not None else _default_registry_path()
    if not source.exists():
        try:
            source = ensure_default_registry(source)
        except OSError:
            # Optional model configuration must not prevent deterministic
            # planning on read-only hosts (e.g. native macOS /config).
            return ModelRegistry({m.id: m for m in default_manifests()}, path=None)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ModelRegistry({m.id:m for m in default_manifests()}, path=None)
    manifests = {}
    for item in payload.get("models",[]):
        if isinstance(item,dict):
            m=_manifest_from_dict(item); manifests[m.id]=m
    for default in default_manifests():
        manifests.setdefault(default.id,default)
    return ModelRegistry(manifests, source)

def checkpoint_fingerprint(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def registry_status(path: str | Path | None = None) -> dict[str, Any]:
    registry=load_model_registry(path); rows=[]
    for manifest in registry.manifests.values():
        checkpoint_path = config_path(manifest.checkpoint) if manifest.checkpoint else None
        exists=bool(checkpoint_path and checkpoint_path.exists())
        verified=None
        if exists and manifest.checkpoint_sha256:
            try: verified=checkpoint_fingerprint(checkpoint_path or "")==manifest.checkpoint_sha256
            except OSError: verified=False
        rows.append({**manifest.to_dict(),"effective_weight":round(manifest.effective_weight,4),"checkpoint_exists":exists,"checkpoint_verified":verified})
    return {"schema_version":MODEL_REGISTRY_VERSION,"registry_path":str(registry.path) if registry.path else None,"models":rows}
