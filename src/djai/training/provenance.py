from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

DataLane = Literal["research", "production"]


@dataclass(frozen=True, slots=True)
class DataProvenance:
    dataset_id: str
    name: str
    lane: DataLane
    source: str
    license: str = "unknown"
    commercial_training_allowed: bool | None = None
    derivative_models_allowed: bool | None = None
    redistribution_allowed: bool | None = None
    attribution: str = ""
    rights_reviewed: bool = False
    audio_bundled: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce(item: dict[str, Any]) -> DataProvenance:
    return DataProvenance(
        dataset_id=str(item.get("id", item.get("dataset_id", "unknown"))),
        name=str(item.get("name", item.get("id", "unknown"))),
        lane=str(item.get("lane", "research")),  # type: ignore[arg-type]
        source=str(item.get("source", item.get("project_page", item.get("paper", "")))),
        license=str(item.get("license", "unknown")),
        commercial_training_allowed=item.get("commercial_training_allowed"),
        derivative_models_allowed=item.get("derivative_models_allowed"),
        redistribution_allowed=item.get("redistribution_allowed"),
        attribution=str(item.get("attribution", "")),
        rights_reviewed=bool(item.get("rights_reviewed", False)),
        audio_bundled=bool(item.get("audio_bundled", False)),
        notes=str(item.get("notes", "")),
    )


def load_dataset_registry(path: str | Path = "datasets/registry.json") -> list[DataProvenance]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload.get("datasets", []) if isinstance(payload, dict) else []
    return [_coerce(item) for item in items if isinstance(item, dict)]


def audit_dataset_registry(
    path: str | Path = "datasets/registry.json",
    *,
    lane: DataLane = "production",
) -> dict[str, Any]:
    entries = load_dataset_registry(path)
    selected = [item for item in entries if item.lane == lane]
    blockers: list[dict[str, str]] = []
    for item in selected:
        if lane == "production":
            if not item.rights_reviewed:
                blockers.append({"dataset_id": item.dataset_id, "reason": "rights_not_reviewed"})
            if item.commercial_training_allowed is not True:
                blockers.append({"dataset_id": item.dataset_id, "reason": "commercial_training_not_explicitly_allowed"})
            if item.derivative_models_allowed is not True:
                blockers.append({"dataset_id": item.dataset_id, "reason": "derivative_model_rights_not_explicitly_allowed"})
    return {
        "registry": str(Path(path).expanduser().resolve()),
        "lane": lane,
        "datasets": [item.to_dict() for item in selected],
        "dataset_count": len(selected),
        "blockers": blockers,
        "ready": not blockers,
    }
