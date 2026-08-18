from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any

from .models import TrackAnalysis, TransitionPlan

NATURALNESS_VERSION = "1.0"

_TECHNIQUE_INTRUSION: dict[str, float] = {
    "equal_power_blend": 0.04,
    "eq_bass_swap": 0.06,
    "layer_instrumental_handoff": 0.08,
    "layer_incoming_vocal_handoff": 0.09,
    "layer_vocal_bridge": 0.10,
    "stem_choreography": 0.09,
    "stem_vocal_overlay": 0.12,
    "filter_build": 0.13,
    "quick_cut": 0.28,
    "echo_out_drop_in": 0.30,
    "loop_roll": 0.34,
    "double_drop": 0.38,
    "backspin": 0.48,
}

_INTENT_TARGET_BARS: dict[str, int] = {
    "smooth_continuation": 16,
    "groove_handoff": 16,
    "energy_lift": 8,
    "energy_release": 8,
    "breakdown_to_drop": 4,
    "vocal_handoff": 8,
    "dramatic_switch": 4,
    "emergency_escape": 2,
}

_SECTION_TARGET_BARS: dict[tuple[str, str], int] = {
    ("outro", "intro"): 16,
    ("outro", "groove"): 8,
    ("groove", "intro"): 8,
    ("groove", "groove"): 16,
    ("groove", "build"): 8,
    ("groove", "breakdown"): 8,
    ("build", "drop"): 4,
    ("build", "groove"): 8,
    ("breakdown", "drop"): 4,
    ("breakdown", "groove"): 8,
    ("drop", "groove"): 8,
    ("drop", "breakdown"): 4,
    ("drop", "intro"): 4,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _target_bars(plan: TransitionPlan) -> float:
    intent_target = float(_INTENT_TARGET_BARS.get(plan.intent, 8))
    section_target = float(_SECTION_TARGET_BARS.get((plan.exit_section, plan.entry_section), 8))
    # Geometric mean prevents one uncertain label from dominating the duration.
    return max(2.0, min(24.0, math.sqrt(intent_target * section_target)))


def _duration_fit(plan: TransitionPlan) -> tuple[float, float]:
    target = _target_bars(plan)
    actual = max(1.0, float(plan.transition_bars))
    octaves = abs(math.log2(actual / target))
    return _clamp(math.exp(-((octaves / 0.82) ** 2))), target


def _tempo_fit(plan: TransitionPlan) -> tuple[float, float]:
    compatibility = _clamp(plan.compatibility.tempo)
    rates = [max(1e-6, _safe_float(plan.render_rate_a, 1.0)), max(1e-6, _safe_float(plan.render_rate_b, 1.0))]
    strain = max(abs(math.log(rate)) for rate in rates)
    stretch_fit = math.exp(-((strain / 0.085) ** 2))
    if not plan.beatmatched:
        # Native-tempo structural handoffs can still sound intentional, but a
        # long overlap should not receive the same continuity credit as a lock.
        stretch_fit = max(0.48, stretch_fit)
    return _clamp(0.64 * compatibility + 0.36 * stretch_fit), strain


def _rendered_fit(plan: TransitionPlan) -> float:
    measurement = plan.measurement
    if measurement is None:
        return 0.50
    return _clamp(
        0.34 * _safe_float(measurement.musical_overall, 0.5)
        + 0.24 * _safe_float(measurement.groove_continuity, 0.5)
        + 0.20 * _safe_float(measurement.harmonic_continuity, 0.5)
        + 0.14 * _safe_float(measurement.energy_intent_fit, 0.5)
        + 0.08 * _safe_float(measurement.transient_clarity, 0.5)
    )


@dataclass(frozen=True, slots=True)
class NaturalnessAssessment:
    score: float
    phrase_alignment: float
    section_flow: float
    duration_fit: float
    recommended_bars: float
    tempo_fit: float
    tempo_strain: float
    harmonic_fit: float
    source_cleanliness: float
    rendered_musicality: float
    simplicity: float
    penalties: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = NATURALNESS_VERSION
        return payload


def assess_transition_naturalness(
    plan: TransitionPlan,
    *,
    track_a: TrackAnalysis | None = None,
    track_b: TrackAnalysis | None = None,
) -> NaturalnessAssessment:
    """Score how intentional a technically valid transition is likely to feel.

    This is deliberately not a safety gate. Hard rendered-audio validation stays
    authoritative. Naturalness only distinguishes *among* executable choices and
    tells the planner when a shorter, simpler handoff is preferable to a forced
    long blend.
    """

    del track_a, track_b  # reserved for future learned/context features
    compatibility = plan.compatibility
    phrase = _clamp(
        0.36 * compatibility.phrase
        + 0.24 * compatibility.boundary
        + 0.22 * plan.cue_confidence
        + 0.18 * compatibility.section_fit
    )
    section_flow = _clamp(0.56 * compatibility.sections + 0.44 * compatibility.section_fit)
    duration_fit, target_bars = _duration_fit(plan)
    tempo_fit, tempo_strain = _tempo_fit(plan)
    harmonic = _clamp(compatibility.harmonic)
    cleanliness = _clamp(0.56 * compatibility.vocal_overlap + 0.44 * compatibility.bass_conflict)
    rendered = _rendered_fit(plan)
    intrusion = _clamp(_TECHNIQUE_INTRUSION.get(plan.technique, 0.22))
    simplicity = _clamp(1.0 - intrusion)

    score = (
        0.18 * phrase
        + 0.14 * section_flow
        + 0.14 * duration_fit
        + 0.13 * tempo_fit
        + 0.08 * harmonic
        + 0.12 * cleanliness
        + 0.14 * rendered
        + 0.07 * simplicity
    )

    penalties = 0.0
    reasons: list[str] = []
    if plan.transition_bars >= 8 and (cleanliness < 0.50 or harmonic < 0.44):
        penalties += 0.075
        reasons.append("Long overlap is discouraged because the selected regions compete harmonically, vocally, or in the bass.")
    if plan.transition_bars >= 16 and tempo_fit < 0.68:
        penalties += 0.055
        reasons.append("Long blend is discouraged because tempo continuity is not strong enough.")
    if plan.transition_mode == "emergency_cut":
        penalties += 0.055
        reasons.append("Emergency routing is intentionally less natural than a planned phrase-to-phrase handoff.")
    if plan.technique in {"quick_cut", "backspin", "echo_out_drop_in"}:
        penalties += 0.045
        reasons.append("Reset-style technique carries an explicit naturalness cost in automatic sets.")
    if plan.forced:
        penalties += 0.025
    if phrase >= 0.70:
        reasons.append("Selected cues align strongly with musical boundaries and phrase evidence.")
    if duration_fit >= 0.72:
        reasons.append(f"{plan.transition_bars}-bar duration matches the structural target of about {target_bars:.1f} bars.")
    if cleanliness >= 0.68:
        reasons.append("Vocal and bass ownership are clean enough for an audible overlap.")
    if tempo_fit < 0.50:
        reasons.append("Tempo relationship favors a compact structural handoff over a long beat-locked blend.")

    return NaturalnessAssessment(
        score=round(_clamp(score - penalties), 6),
        phrase_alignment=round(phrase, 6),
        section_flow=round(section_flow, 6),
        duration_fit=round(duration_fit, 6),
        recommended_bars=round(target_bars, 3),
        tempo_fit=round(tempo_fit, 6),
        tempo_strain=round(tempo_strain, 6),
        harmonic_fit=round(harmonic, 6),
        source_cleanliness=round(cleanliness, 6),
        rendered_musicality=round(rendered, 6),
        simplicity=round(simplicity, 6),
        penalties=round(penalties, 6),
        reasons=tuple(reasons),
    )


def annotate_transition_naturalness(
    plan: TransitionPlan,
    *,
    track_a: TrackAnalysis | None = None,
    track_b: TrackAnalysis | None = None,
) -> NaturalnessAssessment:
    assessment = assess_transition_naturalness(plan, track_a=track_a, track_b=track_b)
    plan.metadata = {
        **plan.metadata,
        "transition_naturalness": assessment.to_dict(),
        "transition_naturalness_score": assessment.score,
    }
    return assessment


def naturalness_weight() -> float:
    try:
        value = float(os.environ.get("DJAI_NATURALNESS_WEIGHT", "0.22"))
    except ValueError:
        value = 0.22
    return max(0.08, min(0.35, value))


def naturalness_blended_score(base_score: float, assessment: NaturalnessAssessment) -> float:
    weight = naturalness_weight()
    return _clamp((1.0 - weight) * _clamp(base_score) + weight * assessment.score)
