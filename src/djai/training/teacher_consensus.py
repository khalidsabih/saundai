from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .teacher_quality import (
    calibrated_confidence,
    choice_entropy,
    deduplicate_judgments,
    diversity_metrics,
)

CONSENSUS_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class TeacherConsensus:
    scenario_id: str
    task: str
    decisive: bool
    consensus_choice: str | None
    agreement: float
    mean_confidence: float
    calibrated_mean_confidence: float
    teacher_count: int
    raw_teacher_count: int
    duplicate_count: int
    distinct_providers: int
    distinct_models: int
    distinct_families: int
    disagreement_entropy: float
    quality_flags: tuple[str, ...]
    reason_codes: tuple[str, ...]
    labels: tuple[dict[str, Any], ...]
    label_source: str = "llm_teacher_consensus_research"
    human_preference: bool = False
    production_authority: bool = False
    schema_version: str = CONSENSUS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["quality_flags"] = list(self.quality_flags)
        payload["reason_codes"] = list(self.reason_codes)
        payload["labels"] = [dict(item) for item in self.labels]
        return payload


def consensus_for_scenario(
    judgments: Iterable[Mapping[str, Any]],
    *,
    minimum_teachers: int = 2,
    minimum_agreement: float = 0.67,
    minimum_mean_confidence: float = 0.55,
    minimum_distinct_providers: int = 2,
    minimum_distinct_models: int = 2,
    minimum_distinct_families: int = 2,
    calibration: Mapping[str, Any] | None = None,
) -> TeacherConsensus:
    raw_rows = [dict(item) for item in judgments]
    rows, duplicate_count = deduplicate_judgments(raw_rows)
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
    calibrated = [calibrated_confidence(item, calibration) for item in rows]
    mean_confidence = sum(confidences) / max(1, len(confidences))
    calibrated_mean_confidence = sum(calibrated) / max(1, len(calibrated))
    diversity = diversity_metrics(rows)
    entropy = choice_entropy(rows)
    provenance_available = any(
        str(item.get("teacher_provider") or "unknown") != "unknown"
        or str(item.get("teacher_model") or "unknown") != "unknown"
        for item in rows
    )
    quality_flags: list[str] = []
    if duplicate_count:
        quality_flags.append("duplicate_judgments_removed")
    if not provenance_available:
        quality_flags.append("missing_teacher_provenance")
    if diversity["distinct_providers"] < minimum_distinct_providers:
        quality_flags.append("insufficient_provider_diversity")
    if diversity["distinct_models"] < minimum_distinct_models:
        quality_flags.append("insufficient_model_diversity")
    if diversity["distinct_families"] < minimum_distinct_families:
        quality_flags.append("correlated_teacher_family_risk")
    if entropy >= 0.55:
        quality_flags.append("high_teacher_disagreement")
    reason_counter: Counter[str] = Counter()
    for item in rows:
        for code in item.get("reason_codes", []) or []:
            reason_counter[str(code)] += 1
    reasons = tuple(code for code, _ in reason_counter.most_common())

    diversity_ok = (
        not provenance_available
        or (
            diversity["distinct_providers"] >= minimum_distinct_providers
            and diversity["distinct_models"] >= minimum_distinct_models
            and diversity["distinct_families"] >= minimum_distinct_families
        )
    )
    decisive = bool(
        len(rows) >= minimum_teachers
        and winner is not None
        and agreement >= minimum_agreement
        and calibrated_mean_confidence >= minimum_mean_confidence
        and diversity_ok
    )
    return TeacherConsensus(
        scenario_id=next(iter(scenario_ids)),
        task=next(iter(tasks)),
        decisive=decisive,
        consensus_choice=winner if decisive else None,
        agreement=round(float(agreement), 6),
        mean_confidence=round(float(mean_confidence), 6),
        calibrated_mean_confidence=round(float(calibrated_mean_confidence), 6),
        teacher_count=len(rows),
        raw_teacher_count=len(raw_rows),
        duplicate_count=duplicate_count,
        distinct_providers=int(diversity["distinct_providers"]),
        distinct_models=int(diversity["distinct_models"]),
        distinct_families=int(diversity["distinct_families"]),
        disagreement_entropy=round(float(entropy), 6),
        quality_flags=tuple(quality_flags),
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
    minimum_distinct_providers: int = 2,
    minimum_distinct_models: int = 2,
    minimum_distinct_families: int = 2,
    calibration: Mapping[str, Any] | None = None,
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
                minimum_distinct_providers=minimum_distinct_providers,
                minimum_distinct_models=minimum_distinct_models,
                minimum_distinct_families=minimum_distinct_families,
                calibration=calibration,
            )
            handle.write(json.dumps(consensus.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination
