from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..models import TrackAnalysis, TransitionPlan

LayerOwner = Literal["A", "B", "mixed", "none"]
LayerSourceMode = Literal["full", "instrumental", "vocal", "silent"]

LAYER_INTELLIGENCE_VERSION = "1.0"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class LayerBrainConfig:
    """Conservative gates for full/vocal/instrumental brain moves.

    Layer moves are opportunistic: they may use only material that is already
    prepared.  They never trigger separation in the planning hot path and the
    normal full-master candidate remains the deterministic fallback.
    """

    enabled: bool = True
    minimum_quality: float = 0.55
    minimum_harmonic: float = 0.42
    minimum_phrase: float = 0.48
    minimum_beat_grid: float = 0.40
    minimum_incoming_vocal: float = 0.12

    @classmethod
    def from_environment(cls) -> "LayerBrainConfig":
        return cls(
            enabled=_enabled("DJAI_LAYER_INTELLIGENCE", "1"),
            minimum_quality=_clamp(_float_env("DJAI_LAYER_MIN_QUALITY", 0.55)),
            minimum_harmonic=_clamp(_float_env("DJAI_LAYER_MIN_HARMONIC", 0.42)),
            minimum_phrase=_clamp(_float_env("DJAI_LAYER_MIN_PHRASE", 0.48)),
            minimum_beat_grid=_clamp(_float_env("DJAI_LAYER_MIN_BEAT_GRID", 0.40)),
            minimum_incoming_vocal=_clamp(_float_env("DJAI_LAYER_MIN_INCOMING_VOCAL", 0.12)),
        )


@dataclass(frozen=True, slots=True)
class LayerState:
    """Authoritative layer ownership represented inside DJState.

    V1 deliberately models the musically meaningful ownership boundary rather
    than every DSP bus.  The Executive can later evolve this into a live
    per-layer state without changing the semantics of vocal/instrumental owner.
    """

    vocal_owner: LayerOwner = "A"
    instrumental_owner: LayerOwner = "A"
    deck_a_source: LayerSourceMode = "full"
    deck_b_source: LayerSourceMode = "silent"
    current_track_full_available: bool = True
    current_track_vocal_available: bool = False
    current_track_instrumental_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerSafetyAssessment:
    feasible: bool
    score: float
    harmonic: float
    phrase: float
    beat_grid: float
    vocal_collision_safety: float
    source_quality: float
    analytical_headroom: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["rendered_headroom_authoritative"] = True
        return payload


def layer_state_for_track(track: TrackAnalysis, *, owner: str = "A") -> LayerState:
    active: LayerOwner = "B" if str(owner).upper() == "B" else "A"
    info = dict(track.performance_layer_analysis or {})
    available = float(info.get("available", 0.0)) > 0.0
    complete = float(info.get("complete", 0.0)) > 0.0
    return LayerState(
        vocal_owner=active,
        instrumental_owner=active,
        deck_a_source="full" if active == "A" else "silent",
        deck_b_source="full" if active == "B" else "silent",
        current_track_full_available=True,
        current_track_vocal_available=available,
        current_track_instrumental_available=complete,
    )


def incoming_vocal_over_instrumental_choreography() -> dict[str, Any]:
    """Describe B-vocal-over-A-instrumental as a first-class brain action."""
    return {
        "version": LAYER_INTELLIGENCE_VERSION,
        "strategy": "incoming_vocal_over_outgoing_instrumental",
        "start": {
            "vocal_owner": "A",
            "instrumental_owner": "A",
            "deck_a_source": "full",
            "deck_b_source": "silent",
        },
        "phases": [
            {
                "name": "clear_outgoing_vocal",
                "range": [0.00, 0.24],
                "deck_a": "full_to_instrumental",
                "deck_b": "silent",
            },
            {
                "name": "introduce_incoming_vocal",
                "range": [0.12, 0.50],
                "deck_a": "instrumental",
                "deck_b": "vocal",
                "vocal_owner": "B",
                "instrumental_owner": "A",
            },
            {
                "name": "instrumental_handoff",
                "range": [0.46, 0.86],
                "deck_a": "instrumental_fade_out",
                "deck_b": "vocal_plus_instrumental",
                "vocal_owner": "B",
                "instrumental_owner": "mixed",
            },
            {
                "name": "full_incoming_ownership",
                "range": [0.82, 1.00],
                "deck_a": "silent",
                "deck_b": "layers_to_full",
                "vocal_owner": "B",
                "instrumental_owner": "B",
            },
        ],
        "end": {
            "vocal_owner": "B",
            "instrumental_owner": "B",
            "deck_a_source": "silent",
            "deck_b_source": "full",
        },
        "fallback": "full_master_transition",
    }


def assess_incoming_vocal_over_instrumental(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    quality_a: float,
    quality_b: float,
    config: LayerBrainConfig | None = None,
) -> LayerSafetyAssessment:
    """Cheap structural safety gate before a layer move reaches rehearsal.

    Actual peak/clipping/headroom safety remains authoritative on the rendered
    candidate in the existing measurement pipeline.  This gate only prevents
    obviously unsuitable layer combinations from consuming rehearsal budget.
    """
    cfg = config or LayerBrainConfig.from_environment()
    compatibility = plan.compatibility
    harmonic = _clamp(
        float(compatibility.local_harmonic)
        if float(compatibility.local_harmonic) > 0.0
        else float(compatibility.harmonic)
    )
    phrase = _clamp(float(compatibility.phrase))
    beat_grid = _clamp(float(compatibility.beat_grid))
    quality = _clamp(min(float(quality_a), float(quality_b)))
    incoming_vocal = _clamp(float(plan.metadata.get("entry_vocal_probability", track_b.vocal_probability)))

    # The outgoing lead is removed before B's lead becomes authoritative, so
    # the remaining collision risk is predominantly harmonic/arrangement risk.
    vocal_collision_safety = _clamp(
        0.55 + 0.20 * harmonic + 0.15 * phrase + 0.10 * (1.0 - float(compatibility.vocal_overlap))
    )
    # V1 intentionally keeps simultaneous layer target levels conservative.
    # The real rendered peak/clipping measurement later remains the hard gate.
    analytical_headroom = 0.84

    reasons: list[str] = []
    if not cfg.enabled:
        reasons.append("Layer Intelligence V1 is disabled.")
    if not plan.beatmatched:
        reasons.append("Incoming-vocal handoff requires a beatmatched transition base.")
    if quality < cfg.minimum_quality:
        reasons.append(f"Layer source quality {quality:.2f} is below {cfg.minimum_quality:.2f}.")
    if harmonic < cfg.minimum_harmonic:
        reasons.append(f"Harmonic safety {harmonic:.2f} is below {cfg.minimum_harmonic:.2f}.")
    if phrase < cfg.minimum_phrase:
        reasons.append(f"Phrase fit {phrase:.2f} is below {cfg.minimum_phrase:.2f}.")
    if beat_grid < cfg.minimum_beat_grid:
        reasons.append(f"Beat-grid safety {beat_grid:.2f} is below {cfg.minimum_beat_grid:.2f}.")
    if incoming_vocal < cfg.minimum_incoming_vocal:
        reasons.append(
            f"Incoming vocal activity {incoming_vocal:.2f} is too low to justify a vocal-led handoff."
        )

    feasible = bool(
        cfg.enabled
        and plan.beatmatched
        and quality >= cfg.minimum_quality
        and harmonic >= cfg.minimum_harmonic
        and phrase >= cfg.minimum_phrase
        and beat_grid >= cfg.minimum_beat_grid
        and incoming_vocal >= cfg.minimum_incoming_vocal
    )
    score = _clamp(
        0.25 * harmonic
        + 0.22 * phrase
        + 0.16 * beat_grid
        + 0.18 * vocal_collision_safety
        + 0.14 * quality
        + 0.05 * analytical_headroom
    )
    if feasible:
        reasons.append("Outgoing vocal can clear before B vocal takes ownership over A instrumental.")
        reasons.append("Rendered clipping/headroom and artifact safety remain authoritative in AI Headphones.")

    return LayerSafetyAssessment(
        feasible=feasible,
        score=round(score, 4),
        harmonic=round(harmonic, 4),
        phrase=round(phrase, 4),
        beat_grid=round(beat_grid, 4),
        vocal_collision_safety=round(vocal_collision_safety, 4),
        source_quality=round(quality, 4),
        analytical_headroom=round(analytical_headroom, 4),
        reasons=tuple(reasons),
    )
