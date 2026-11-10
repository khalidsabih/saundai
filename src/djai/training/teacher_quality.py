from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TEACHER_QUALITY_SCHEMA_VERSION = "1.0"


def teacher_identity(row: Mapping[str, Any]) -> str:
    return ":".join(
        (
            str(row.get("teacher_provider") or "unknown"),
            str(row.get("teacher_model") or "unknown"),
            str(row.get("teacher_model_version") or ""),
        )
    )


def teacher_family(row: Mapping[str, Any]) -> str:
    provider = str(row.get("teacher_provider") or "unknown").strip().lower()
    model = str(row.get("teacher_model") or "unknown").strip().lower()
    root = model.split("-", 1)[0].split("/", 1)[0]
    return f"{provider}:{root}"


def judgment_fingerprint(row: Mapping[str, Any]) -> str:
    normalized = {
        "scenario_id": str(row.get("scenario_id") or ""),
        "teacher": teacher_identity(row),
        "choice": str(row.get("choice") or ""),
        "ranking": list(row.get("ranking") or []),
        "confidence": round(float(row.get("confidence", 0.0) or 0.0), 6),
        "reason_codes": sorted(str(item) for item in (row.get("reason_codes") or [])),
        "seed": str(row.get("seed") or ""),
        "prompt_version": str(row.get("prompt_version") or ""),
    }
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def deduplicate_judgments(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        payload = dict(row)
        fingerprint = judgment_fingerprint(payload)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        output.append(payload)
    return output, duplicates


def diversity_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    providers = {str(row.get("teacher_provider") or "unknown") for row in rows}
    models = {teacher_identity(row) for row in rows}
    families = {teacher_family(row) for row in rows}
    return {
        "distinct_providers": len(providers),
        "distinct_models": len(models),
        "distinct_families": len(families),
        "providers": sorted(providers),
        "models": sorted(models),
        "families": sorted(families),
    }


def choice_entropy(rows: Sequence[Mapping[str, Any]]) -> float:
    votes = Counter(str(row.get("choice") or "") for row in rows if str(row.get("choice") or ""))
    total = sum(votes.values())
    if total <= 1 or len(votes) <= 1:
        return 0.0
    entropy = 0.0
    for count in votes.values():
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    maximum = math.log(max(2, len(votes)), 2)
    return float(entropy / maximum) if maximum > 0.0 else 0.0


def calibrated_confidence(
    row: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
) -> float:
    raw = max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0)))
    if not calibration:
        # Synthetic teacher confidence is not probability truth. Keep a small
        # shrinkage toward uncertainty until human/reference calibration exists.
        return round(0.5 + 0.85 * (raw - 0.5), 6)
    identity = teacher_identity(row)
    profile = calibration.get(identity) if isinstance(calibration, Mapping) else None
    if not isinstance(profile, Mapping):
        return round(0.5 + 0.85 * (raw - 0.5), 6)
    reliability = max(0.0, min(1.0, float(profile.get("reliability", 0.5) or 0.5)))
    bias = max(-0.25, min(0.25, float(profile.get("confidence_bias", 0.0) or 0.0)))
    adjusted = raw - bias
    return round(0.5 + reliability * (adjusted - 0.5), 6)


def build_teacher_calibration(
    judgments: Iterable[Mapping[str, Any]],
    references: Mapping[str, str],
) -> dict[str, dict[str, float | int]]:
    """Estimate teacher reliability from held-out human/reference choices."""
    buckets: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for row in judgments:
        scenario_id = str(row.get("scenario_id") or "")
        reference = str(references.get(scenario_id) or "")
        choice = str(row.get("choice") or "")
        if not reference or not choice:
            continue
        buckets[teacher_identity(row)].append(
            (choice == reference, max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))))
        )

    output: dict[str, dict[str, float | int]] = {}
    for identity, items in buckets.items():
        accuracy = sum(1.0 for correct, _ in items if correct) / len(items)
        mean_confidence = sum(confidence for _, confidence in items) / len(items)
        # Bayesian shrinkage avoids granting extreme reliability from tiny samples.
        reliability = (accuracy * len(items) + 1.0) / (len(items) + 2.0)
        output[identity] = {
            "samples": len(items),
            "accuracy": round(accuracy, 6),
            "mean_confidence": round(mean_confidence, 6),
            "confidence_bias": round(mean_confidence - accuracy, 6),
            "reliability": round(reliability, 6),
        }
    return output


def scenario_stratum(scenario: Mapping[str, Any]) -> str:
    context = scenario.get("context") if isinstance(scenario.get("context"), Mapping) else {}
    candidates = scenario.get("candidates") if isinstance(scenario.get("candidates"), list) else []
    techniques = sorted(
        {
            str(item.get("technique") or "unknown")
            for item in candidates
            if isinstance(item, Mapping)
        }
    )
    styles = sorted(
        {
            str(item.get("style") or "unknown")
            for item in candidates
            if isinstance(item, Mapping)
        }
    )
    return "|".join(
        (
            str(scenario.get("task") or "unknown"),
            str(context.get("journey") or context.get("mode") or "unknown"),
            ",".join(styles) or "unknown",
            ",".join(techniques) or "unknown",
        )
    )


def balance_scenarios(
    scenarios: Iterable[Mapping[str, Any]],
    *,
    maximum_per_stratum: int = 200,
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for scenario in sorted(
        (dict(item) for item in scenarios),
        key=lambda item: str(item.get("scenario_id") or ""),
    ):
        stratum = scenario_stratum(scenario)
        if counts[stratum] >= max(1, int(maximum_per_stratum)):
            continue
        counts[stratum] += 1
        scenario["data_quality_stratum"] = stratum
        output.append(scenario)
    return output


@dataclass(frozen=True, slots=True)
class HumanReviewItem:
    scenario_id: str
    priority: float
    reasons: tuple[str, ...]
    task: str
    consensus_choice: str | None
    agreement: float
    calibrated_confidence: float
    disagreement_entropy: float
    distinct_providers: int
    distinct_models: int
    hard_negative_candidate: bool
    schema_version: str = TEACHER_QUALITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def human_review_item(consensus: Mapping[str, Any]) -> HumanReviewItem:
    agreement = float(consensus.get("agreement", 0.0) or 0.0)
    confidence = float(consensus.get("calibrated_mean_confidence", consensus.get("mean_confidence", 0.0)) or 0.0)
    entropy = float(consensus.get("disagreement_entropy", 0.0) or 0.0)
    providers = int(consensus.get("distinct_providers", 0) or 0)
    models = int(consensus.get("distinct_models", 0) or 0)
    decisive = bool(consensus.get("decisive"))
    reasons: list[str] = []
    priority = 0.0

    if not decisive:
        reasons.append("teacher_consensus_uncertain")
        priority += 0.40
    if entropy >= 0.55:
        reasons.append("high_teacher_disagreement")
        priority += 0.25 * entropy
    if confidence < 0.62:
        reasons.append("low_calibrated_confidence")
        priority += 0.15
    if providers < 2:
        reasons.append("insufficient_provider_diversity")
        priority += 0.12
    if models < 2:
        reasons.append("insufficient_model_diversity")
        priority += 0.08

    reason_codes = {str(item) for item in (consensus.get("reason_codes") or [])}
    hard_negative = bool(
        consensus.get("consensus_choice") == "neither"
        or reason_codes.intersection({"beats", "phrasing", "vocals", "bass", "harmony"})
    )
    if hard_negative:
        reasons.append("hard_negative_candidate")
        priority += 0.08

    return HumanReviewItem(
        scenario_id=str(consensus.get("scenario_id") or ""),
        priority=round(min(1.0, priority), 6),
        reasons=tuple(dict.fromkeys(reasons)),
        task=str(consensus.get("task") or "unknown"),
        consensus_choice=(
            str(consensus.get("consensus_choice"))
            if consensus.get("consensus_choice") is not None
            else None
        ),
        agreement=round(agreement, 6),
        calibrated_confidence=round(confidence, 6),
        disagreement_entropy=round(entropy, 6),
        distinct_providers=providers,
        distinct_models=models,
        hard_negative_candidate=hard_negative,
    )


def build_human_review_queue(
    consensus_path: str | Path,
    output_path: str | Path,
    *,
    minimum_priority: float = 0.10,
) -> Path:
    source = Path(consensus_path).expanduser().resolve()
    rows: list[HumanReviewItem] = []
    if source.exists():
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                item = human_review_item(payload)
                if item.priority >= minimum_priority:
                    rows.append(item)
    rows.sort(key=lambda item: (-item.priority, item.scenario_id))

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination
