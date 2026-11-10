from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

CONSENSUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class TeacherConsensus:
    scenario_id: str
    task: str
    decisive: bool
    consensus_choice: str | None
    agreement: float
    mean_confidence: float
    teacher_count: int
    reason_codes: tuple[str, ...]
    labels: tuple[dict[str, Any], ...]
    label_source: str = "llm_teacher_consensus_research"
    human_preference: bool = False
    production_authority: bool = False
    schema_version: str = CONSENSUS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        payload["labels"] = [dict(item) for item in self.labels]
        return payload


def consensus_for_scenario(
    judgments: Iterable[Mapping[str, Any]],
    *,
    minimum_teachers: int = 2,
    minimum_agreement: float = 0.67,
    minimum_mean_confidence: float = 0.55,
) -> TeacherConsensus:
    rows = [dict(item) for item in judgments]
    if not rows:
        raise ValueError("Consensus requires at least one teacher judgment.")
    scenario_ids = {str(item.get("scenario_id") or "") for item in rows}
    scenario_ids.discard("")
    if len(scenario_ids) != 1:
        raise ValueError("Consensus rows must belong to exactly one scenario.")
    tasks = {str(item.get("task") or "") for item in rows}
    tasks.discard("")
    if len(tasks) != 1:
        raise ValueError("Consensus rows must use exactly one task.")

    votes = Counter(str(item.get("choice") or "") for item in rows if str(item.get("choice") or ""))
    winner: str | None = None
    agreement = 0.0
    if votes:
        winner, count = votes.most_common(1)[0]
        agreement = count / max(1, sum(votes.values()))

    confidences = [max(0.0, min(1.0, float(item.get("confidence", 0.0) or 0.0))) for item in rows]
    mean_confidence = sum(confidences) / max(1, len(confidences))
    reason_counter: Counter[str] = Counter()
    for item in rows:
        for code in item.get("reason_codes", []) or []:
            reason_counter[str(code)] += 1
    reasons = tuple(code for code, _ in reason_counter.most_common())

    decisive = bool(
        len(rows) >= minimum_teachers
        and winner is not None
        and agreement >= minimum_agreement
        and mean_confidence >= minimum_mean_confidence
    )
    return TeacherConsensus(
        scenario_id=next(iter(scenario_ids)),
        task=next(iter(tasks)),
        decisive=decisive,
        consensus_choice=winner if decisive else None,
        agreement=round(float(agreement), 6),
        mean_confidence=round(float(mean_confidence), 6),
        teacher_count=len(rows),
        reason_codes=reasons,
        labels=tuple(rows),
    )


def load_judgments(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("scenario_id"):
                rows.append(payload)
    return rows


def build_consensus_file(
    judgments_path: str | Path,
    output_path: str | Path,
    *,
    minimum_teachers: int = 2,
    minimum_agreement: float = 0.67,
    minimum_mean_confidence: float = 0.55,
) -> Path:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_judgments(judgments_path):
        grouped[str(row["scenario_id"])].append(row)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for scenario_id in sorted(grouped):
            consensus = consensus_for_scenario(
                grouped[scenario_id],
                minimum_teachers=minimum_teachers,
                minimum_agreement=minimum_agreement,
                minimum_mean_confidence=minimum_mean_confidence,
            )
            handle.write(json.dumps(consensus.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination
