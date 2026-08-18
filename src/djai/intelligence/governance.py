from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_registry import checkpoint_fingerprint, load_model_registry

MODEL_GOVERNANCE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class GovernanceReport:
    version: str
    ready: bool
    production_models: int
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        payload["warnings"] = list(self.warnings)
        return payload


def _resolve_checkpoint(registry_path: Path | None, checkpoint: str) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_absolute():
        return candidate
    if registry_path is not None:
        root = registry_path.parent
        # Registry normally lives in config/ while checkpoint strings are
        # project-relative (config/neural/...). Try both contracts.
        direct = (root / candidate).resolve()
        project = (root.parent / candidate).resolve()
        if direct.exists() or not project.exists():
            return direct
        return project
    return candidate.resolve()


def audit_model_governance(registry_path: str | Path | None = None) -> GovernanceReport:
    registry = load_model_registry(registry_path)
    blockers: list[str] = []
    warnings: list[str] = []
    production_models = 0
    for model in registry.manifests.values():
        if model.authority != "production":
            if model.authority == "advisory" and model.license.strip().casefold() in {"", "review", "unknown"}:
                warnings.append(f"{model.id}: advisory model license still requires review")
            continue
        production_models += 1
        prefix = f"{model.id}:"
        if not model.production_approved:
            blockers.append(f"{prefix} production_approved is false")
        if not model.checkpoint:
            blockers.append(f"{prefix} production model has no checkpoint")
            continue
        checkpoint = _resolve_checkpoint(registry.path, model.checkpoint)
        if not checkpoint.exists():
            blockers.append(f"{prefix} checkpoint is missing")
            continue
        if not model.checkpoint_sha256:
            blockers.append(f"{prefix} checkpoint has no pinned SHA-256")
        else:
            try:
                if checkpoint_fingerprint(checkpoint) != model.checkpoint_sha256:
                    blockers.append(f"{prefix} checkpoint SHA-256 mismatch")
            except OSError:
                blockers.append(f"{prefix} checkpoint could not be fingerprinted")
        metadata = dict(model.metadata)
        if not bool(metadata.get("rights_reviewed", False)):
            blockers.append(f"{prefix} rights review is missing")
        metrics = dict(metadata.get("validation_metrics", {})) if isinstance(metadata.get("validation_metrics"), dict) else {}
        if int(metrics.get("validation_examples", 0) or 0) < 50:
            blockers.append(f"{prefix} fewer than 50 held-out validation examples")
        if float(metrics.get("validation_accuracy", 0.0) or 0.0) < 0.62:
            blockers.append(f"{prefix} held-out validation accuracy is below 62%")
        arena = dict(metadata.get("arena_report", {})) if isinstance(metadata.get("arena_report"), dict) else {}
        if not bool(arena.get("deployment_ready", False)):
            blockers.append(f"{prefix} blind Arena evidence is not deployment-ready")
        if not bool(metadata.get("governance_approved", False)):
            blockers.append(f"{prefix} governance approval is missing")

    return GovernanceReport(
        version=MODEL_GOVERNANCE_VERSION,
        ready=not blockers,
        production_models=production_models,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def load_arena_evidence(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Arena report must contain a JSON object.")
    return payload
