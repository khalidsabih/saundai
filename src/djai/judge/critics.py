from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..brain.director import SetIntent, direct_set_intent, intent_track_score
from ..brain.set_state import SetState
from ..models import TrackAnalysis, TransitionMeasurement, TransitionPlan

CRITIC_ENSEMBLE_VERSION = "6.0"


@dataclass(frozen=True, slots=True)
class CriticResult:
    name: str
    score: float
    blocking: bool = False
    observations: tuple[str, ...] = ()
    remedies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observations"] = list(self.observations)
        payload["remedies"] = list(self.remedies)
        return payload


@dataclass(frozen=True, slots=True)
class CriticEnsemble:
    version: str
    critics: tuple[CriticResult, ...]
    failure_domains: tuple[str, ...] = ()
    remedies: tuple[str, ...] = ()
    set_intent: dict[str, Any] | None = None

    @property
    def scores(self) -> dict[str, float]:
        return {item.name: round(float(item.score), 4) for item in self.critics}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "critics": [item.to_dict() for item in self.critics],
            "scores": self.scores,
            "failure_domains": list(self.failure_domains),
            "remedies": list(self.remedies),
            "set_intent": self.set_intent,
        }


def _clamp(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _domain_from_failure(message: str) -> str:
    value = message.casefold()
    if "clip" in value or "artifact" in value or "nan" in value or "inf" in value:
        return "technical"
    if "phase" in value:
        return "phase"
    if "silence" in value:
        return "silence"
    if "vocal" in value:
        return "vocals"
    if "bass" in value or "low-frequency" in value:
        return "bass"
    if "rhythm" in value or "beat" in value or "groove" in value:
        return "rhythm"
    if "harmon" in value or "key" in value:
        return "harmony"
    return "technical"


def _remedies_for(domain: str) -> tuple[str, ...]:
    return {
        "technical": ("simplify_program", "shorten_overlap", "reduce_source_level"),
        "phase": ("shorten_overlap", "prefer_cut_or_stem_handoff"),
        "silence": ("reselect_entry_cue", "reselect_exit_cue"),
        "rhythm": ("reselect_phrase_aligned_cues", "shorten_overlap", "prefer_rhythmic_switch"),
        "harmony": ("shorten_overlap", "prefer_breakdown_or_cut", "reselect_local_key_pair"),
        "vocals": ("delay_incoming_vocal", "use_instrumental_or_stems", "shorten_overlap"),
        "bass": ("enforce_single_bass_owner", "move_bass_handoff", "shorten_overlap"),
        "phrasing": ("reselect_phrase_boundary", "change_transition_length"),
        "energy": ("change_transition_length", "choose_alternate_entry", "retarget_set_intent"),
        "set_context": ("choose_alternate_track", "change_transition_family"),
    }.get(domain, ("change_transition_family",))


def evaluate_critic_ensemble(
    plan: TransitionPlan,
    *,
    measurement: TransitionMeasurement | None,
    track_b: TrackAnalysis | None = None,
    set_state: SetState | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
) -> CriticEnsemble:
    hard_failures = list(measurement.hard_failures if measurement else [])
    technical = _clamp(measurement.overall if measurement else (plan.measured_score or plan.score))
    rhythm = _clamp(
        (
            (measurement.rhythm_coherence + measurement.transient_clarity + measurement.groove_continuity) / 3.0
            if measurement
            else plan.compatibility.beat_grid
        )
    )
    harmony = _clamp(
        0.55 * (measurement.harmonic_continuity if measurement else plan.compatibility.harmonic)
        + 0.45 * plan.compatibility.harmonic
    )
    if measurement and measurement.stem_evidence:
        vocals = _clamp(measurement.vocal_cleanliness)
        bass = _clamp(0.55 * measurement.stem_bass_cleanliness + 0.45 * measurement.bass_cleanliness)
    else:
        vocals = _clamp(plan.compatibility.vocal_overlap)
        bass = _clamp(
            0.58 * (measurement.bass_cleanliness if measurement else plan.compatibility.bass_conflict)
            + 0.42 * plan.compatibility.bass_conflict
        )
    phrase = _clamp(
        0.48 * plan.compatibility.phrase
        + 0.30 * plan.compatibility.boundary
        + 0.22 * plan.cue_confidence
    )
    energy = _clamp(
        measurement.energy_intent_fit if measurement else plan.compatibility.energy
    )
    set_intent: SetIntent | None = None
    set_context = 0.5
    if set_state is not None and track_b is not None:
        set_intent = direct_set_intent(journey, journey_progress, set_state)
        set_context, _ = intent_track_score(set_intent, set_state, track_b)

    scores = {
        "technical": technical,
        "rhythm": rhythm,
        "harmony": harmony,
        "vocals": vocals,
        "bass": bass,
        "phrasing": phrase,
        "energy": energy,
        "set_context": _clamp(set_context),
    }
    failure_domains: list[str] = []
    remedies: list[str] = []
    for failure in hard_failures:
        domain = _domain_from_failure(failure)
        if domain not in failure_domains:
            failure_domains.append(domain)
    thresholds = {
        "rhythm": 0.34,
        "harmony": 0.30,
        "vocals": 0.30,
        "bass": 0.30,
        "phrasing": 0.34,
        "energy": 0.30,
        "set_context": 0.28,
    }
    for name, threshold in thresholds.items():
        if scores[name] < threshold and name not in failure_domains:
            failure_domains.append(name)
    for domain in failure_domains:
        for remedy in _remedies_for(domain):
            if remedy not in remedies:
                remedies.append(remedy)

    critics: list[CriticResult] = []
    for name, score in scores.items():
        domain_failures = [value for value in hard_failures if _domain_from_failure(value) == name]
        observations: list[str] = []
        if domain_failures:
            observations.extend(domain_failures)
        elif name in failure_domains:
            observations.append(f"{name.replace('_', ' ').title()} critic is below the 6.0 replanning floor.")
        critics.append(
            CriticResult(
                name=name,
                score=round(score, 4),
                blocking=bool(domain_failures),
                observations=tuple(observations),
                remedies=_remedies_for(name) if name in failure_domains else (),
            )
        )

    return CriticEnsemble(
        version=CRITIC_ENSEMBLE_VERSION,
        critics=tuple(critics),
        failure_domains=tuple(failure_domains),
        remedies=tuple(remedies),
        set_intent=set_intent.to_dict() if set_intent is not None else None,
    )
