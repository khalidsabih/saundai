from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..models import TransitionPlan

PREFERENCE_SCHEMA_VERSION = "1.1"
VALID_CHOICES = {"a", "b", "tie", "neither"}


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    schema_version: str
    comparison_id: str
    recorded_at_utc: str
    listener: str
    choice: str
    weight: float
    candidate_a: dict[str, Any]
    candidate_b: dict[str, Any]
    context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_features(plan: TransitionPlan) -> dict[str, Any]:
    compatibility = plan.compatibility
    measurement = plan.measurement
    metadata = dict(plan.metadata or {})
    return {
        "execution_signature": plan.execution_signature,
        "track_a": plan.track_a,
        "track_b": plan.track_b,
        "style": plan.style,
        "intent": plan.intent,
        "technique": plan.technique,
        "transition_bars": plan.transition_bars,
        "transition_seconds": plan.transition_seconds,
        "exit_bar": int(plan.metadata.get("exit_bar_index", 0) or 0),
        "entry_bar": int(plan.metadata.get("entry_bar_index", 0) or 0),
        "section_pair": f"{plan.exit_section}->{plan.entry_section}",
        "exit_section": plan.exit_section,
        "entry_section": plan.entry_section,
        "cue_confidence": plan.cue_confidence,
        "analytical_score": plan.analytical_score or plan.score,
        "measured_score": plan.measured_score,
        "measurement_passed": plan.measurement_passed,
        "intent_score": plan.intent_score,
        "technique_score": plan.technique_score,
        "tempo_difference_percent": compatibility.tempo_difference_percent,
        "energy_delta": compatibility.energy_delta,
        "compatibility": {
            "tempo": compatibility.tempo,
            "harmonic": compatibility.harmonic,
            "energy": compatibility.energy,
            "texture": compatibility.texture,
            "beat_grid": compatibility.beat_grid,
            "phrase": compatibility.phrase,
            "sections": compatibility.sections,
            "genre": compatibility.genre,
            "vocal_overlap": compatibility.vocal_overlap,
            "bass_conflict": compatibility.bass_conflict,
            "boundary": compatibility.boundary,
            "section_fit": compatibility.section_fit,
        },
        "measurement": measurement.to_dict() if measurement else None,
        "embedding_similarity": float(metadata.get("embedding_similarity", 0.5)),
        "rendered_transition_embedding": list(metadata.get("rendered_transition_embedding", [])),
        "rendered_transition_embedding_backend": str(metadata.get("rendered_transition_embedding_backend", "none")),
        "rejection_penalty": float(metadata.get("rejection_penalty", 0.0)),
        "set_context_score": float(metadata.get("set_context_score", 0.5)),
        "professional_judge_score": float(metadata.get("professional_judge_score", 0.0)),
        "dj_state": dict(metadata.get("dj_state_taste", {})) if isinstance(metadata.get("dj_state_taste"), dict) else None,
        "dj_move": dict(metadata.get("dj_move_taste", {})) if isinstance(metadata.get("dj_move_taste"), dict) else None,
        "state_context_id": str(metadata.get("state_context_id", "")) or None,
        "state_move_taste_shadow": dict(metadata.get("state_move_taste_shadow", {})) if isinstance(metadata.get("state_move_taste_shadow"), dict) else None,
    }


def comparison_id(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> str:
    left = str(candidate_a.get("execution_signature", ""))
    right = str(candidate_b.get("execution_signature", ""))
    digest = hashlib.sha256("|".join(sorted((left, right))).encode("utf-8")).hexdigest()[:18]
    return f"cmp-{digest}"


def append_preference(
    *,
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
    choice: str,
    listener: str = "local",
    context: dict[str, Any] | None = None,
    weight: float = 1.0,
    path: str | Path = "ratings/transition-preferences.jsonl",
) -> Path:
    normalized = choice.strip().lower()
    if normalized not in VALID_CHOICES:
        raise ValueError("Preference choice must be a, b, tie, or neither.")
    if candidate_a.get("execution_signature") == candidate_b.get("execution_signature"):
        raise ValueError("A/B preference candidates must be different transitions.")
    record = PreferenceRecord(
        schema_version=PREFERENCE_SCHEMA_VERSION,
        comparison_id=comparison_id(candidate_a, candidate_b),
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        listener=listener.strip() or "local",
        choice=normalized,
        weight=max(0.05, min(10.0, float(weight))),
        candidate_a=dict(candidate_a),
        candidate_b=dict(candidate_b),
        context=dict(context or {}),
    )
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    # 1.8.2 Crate Memory learns only from explicit human preference evidence.
    # Snapshot/test preference files with other names remain side-effect free.
    try:
        from ..brain.crate_memory import record_preference_to_crate_memory

        record_preference_to_crate_memory(
            candidate_a=record.candidate_a,
            candidate_b=record.candidate_b,
            choice=record.choice,
            listener=record.listener,
            context=record.context,
            weight=record.weight,
            preferences_path=destination,
        )
    except Exception:
        # Preference capture must remain available even if the local memory DB
        # is temporarily unavailable or from an older workspace.
        pass
    return destination


def load_preferences(path: str | Path) -> list[dict[str, Any]]:
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
            if isinstance(payload, dict) and payload.get("choice") in VALID_CHOICES:
                rows.append(payload)
    return rows
