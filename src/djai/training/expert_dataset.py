from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

PreferenceChoice = Literal["a", "b", "tie", "neither"]


@dataclass(slots=True)
class ExpertTransitionDecision:
    expert_id: str
    track_a_id: str
    track_b_id: str
    context: dict[str, Any]
    candidate_a: dict[str, Any]
    candidate_b: dict[str, Any]
    choice: PreferenceChoice
    reasons: list[str] = field(default_factory=list)
    confidence: float | None = None
    source_rights_confirmed: bool = False
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = "expert-transition-decision/1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpertSetDemonstration:
    expert_id: str
    set_id: str
    track_ids: list[str]
    transitions: list[dict[str, Any]]
    journey: str = "unspecified"
    context: dict[str, Any] = field(default_factory=dict)
    source_rights_confirmed: bool = False
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = "expert-set-demonstration/1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_expert_record(
    record: ExpertTransitionDecision | ExpertSetDemonstration,
    path: str | Path,
    *,
    production: bool = False,
) -> Path:
    if production and not record.source_rights_confirmed:
        raise ValueError("Production expert data requires source_rights_confirmed=true.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return destination


def expert_dataset_stats(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    counts = {"transition_decisions": 0, "set_demonstrations": 0, "rights_confirmed": 0}
    experts: set[str] = set()
    if not source.exists():
        return {"path": str(source), **counts, "experts": 0}
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        schema = str(item.get("schema", ""))
        if schema.startswith("expert-transition"):
            counts["transition_decisions"] += 1
        elif schema.startswith("expert-set"):
            counts["set_demonstrations"] += 1
        if item.get("source_rights_confirmed") is True:
            counts["rights_confirmed"] += 1
        if item.get("expert_id"):
            experts.add(str(item["expert_id"]))
    return {"path": str(source), **counts, "experts": len(experts)}
