from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..models import TransitionPlan

MEMORY_SCHEMA_VERSION = "1.0"
DEFAULT_REJECTION_PATH = "ratings/rejected-transitions.jsonl"


@dataclass(slots=True)
class RejectedTransition:
    recorded_at_utc: str
    track_a: str
    track_b: str
    transition_signature: str
    exit_bar: int
    entry_bar: int
    technique: str
    intent: str
    transition_bars: int
    section_pair: str
    severity: float = 1.0
    reasons: list[str] = field(default_factory=list)
    source: str = "human"
    listener: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity(path: str) -> str:
    # Keep the raw path for explainability but compare a stable normalized hash.
    normalized = str(Path(path).expanduser()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _plan_bar(plan: TransitionPlan, key: str, fallback: float) -> int:
    try:
        return int(round(float(plan.metadata.get(key, fallback))))
    except (TypeError, ValueError):
        return int(round(fallback))


def rejection_from_plan(
    plan: TransitionPlan,
    *,
    reasons: Iterable[str] = (),
    severity: float = 1.0,
    source: str = "human",
    listener: str = "local",
    metadata: dict[str, Any] | None = None,
) -> RejectedTransition:
    return RejectedTransition(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        track_a=plan.track_a,
        track_b=plan.track_b,
        transition_signature=plan.execution_signature,
        exit_bar=_plan_bar(plan, "exit_bar_index", plan.exit_time_a),
        entry_bar=_plan_bar(plan, "entry_bar_index", plan.entry_time_b),
        technique=plan.technique,
        intent=plan.intent,
        transition_bars=int(plan.transition_bars),
        section_pair=f"{plan.exit_section}->{plan.entry_section}",
        severity=max(0.0, min(1.0, float(severity))),
        reasons=[str(value).strip() for value in reasons if str(value).strip()],
        source=source,
        listener=listener,
        metadata=dict(metadata or {}),
    )


def append_rejection(
    plan: TransitionPlan,
    *,
    reasons: Iterable[str] = (),
    severity: float = 1.0,
    source: str = "human",
    listener: str = "local",
    metadata: dict[str, Any] | None = None,
    path: str | Path = DEFAULT_REJECTION_PATH,
) -> Path:
    record = rejection_from_plan(
        plan,
        reasons=reasons,
        severity=severity,
        source=source,
        listener=listener,
        metadata=metadata,
    )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    load_rejections.cache_clear()
    return destination


def append_rejection_payload(
    payload: dict[str, Any],
    *,
    reasons: Iterable[str] = (),
    severity: float = 1.0,
    source: str = "human",
    listener: str = "local",
    path: str | Path = DEFAULT_REJECTION_PATH,
) -> Path | None:
    """Record a rejection from an API/feedback serialized TransitionPlan."""
    try:
        from ..serialization import transition_plan_from_dict

        plan = transition_plan_from_dict(payload)
    except (TypeError, ValueError, KeyError):
        return None
    return append_rejection(
        plan,
        reasons=reasons,
        severity=severity,
        source=source,
        listener=listener,
        path=path,
    )


def append_feature_rejection(
    feature: dict[str, Any],
    *,
    reasons: Iterable[str] = (),
    severity: float = 1.0,
    source: str = "human",
    listener: str = "local",
    path: str | Path = DEFAULT_REJECTION_PATH,
) -> Path:
    """Record rejection from the compact feature payload used by Taste Lab."""
    record = RejectedTransition(
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        track_a=str(feature.get("track_a", "")),
        track_b=str(feature.get("track_b", "")),
        transition_signature=str(feature.get("execution_signature", "")),
        exit_bar=int(feature.get("exit_bar", 0) or 0),
        entry_bar=int(feature.get("entry_bar", 0) or 0),
        technique=str(feature.get("technique", "unknown")),
        intent=str(feature.get("intent", "unknown")),
        transition_bars=int(feature.get("transition_bars", 0) or 0),
        section_pair=str(feature.get("section_pair", f"{feature.get('exit_section', 'unknown')}->{feature.get('entry_section', 'unknown')}")),
        severity=max(0.0, min(1.0, float(severity))),
        reasons=[str(value).strip() for value in reasons if str(value).strip()],
        source=source,
        listener=listener,
        metadata={"taste_lab": True},
    )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
    load_rejections.cache_clear()
    return destination


@lru_cache(maxsize=8)
def load_rejections(path: str | Path = DEFAULT_REJECTION_PATH) -> tuple[RejectedTransition, ...]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return ()
    records: list[RejectedTransition] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            records.append(
                RejectedTransition(
                    **{
                        key: value
                        for key, value in payload.items()
                        if key in RejectedTransition.__dataclass_fields__
                    }
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(records)


def rejection_similarity(plan: TransitionPlan, rejected: RejectedTransition) -> float:
    if plan.execution_signature and plan.execution_signature == rejected.transition_signature:
        return 1.0
    if _identity(plan.track_a) != _identity(rejected.track_a) or _identity(plan.track_b) != _identity(rejected.track_b):
        return 0.0
    exit_bar = _plan_bar(plan, "exit_bar_index", plan.exit_time_a)
    entry_bar = _plan_bar(plan, "entry_bar_index", plan.entry_time_b)
    cue_similarity = math.exp(-abs(exit_bar - rejected.exit_bar) / 4.0) * math.exp(-abs(entry_bar - rejected.entry_bar) / 4.0)
    length_similarity = math.exp(-abs(int(plan.transition_bars) - rejected.transition_bars) / 5.0)
    technique_similarity = 1.0 if plan.technique == rejected.technique else 0.28
    intent_similarity = 1.0 if plan.intent == rejected.intent else 0.42
    section_similarity = 1.0 if f"{plan.exit_section}->{plan.entry_section}" == rejected.section_pair else 0.55
    return max(
        0.0,
        min(
            1.0,
            0.34 * cue_similarity
            + 0.20 * technique_similarity
            + 0.16 * intent_similarity
            + 0.12 * length_similarity
            + 0.18 * section_similarity,
        ),
    )


def rejection_penalty(
    plan: TransitionPlan,
    *,
    path: str | Path = DEFAULT_REJECTION_PATH,
    listener: str | None = None,
) -> tuple[float, RejectedTransition | None]:
    best_score = 0.0
    best: RejectedTransition | None = None
    for rejected in load_rejections(path):
        if listener is not None and rejected.listener not in {listener, "global", "local"}:
            continue
        similarity = rejection_similarity(plan, rejected)
        weighted = similarity * max(0.0, min(1.0, rejected.severity))
        if weighted > best_score:
            best_score, best = weighted, rejected
    return best_score, best
