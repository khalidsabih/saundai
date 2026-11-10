from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from .compatibility import match_bpm
from .execution import RENDER_ENGINE_VERSION, bind_execution_assets, verify_execution_assets
from .models import (
    CompatibilityBreakdown,
    QualityClass,
    TempoPolicy,
    TempoStrategy,
    TechniqueAssessment,
    TrackAnalysis,
    TransitionCandidate,
    TransitionIntent,
    TransitionMode,
    TransitionPlan,
    TransitionStyle,
    TransitionTechnique,
)
from .version import DECISION_VERSION

CUT_TECHNIQUES: set[str] = {"echo_out_drop_in", "quick_cut", "backspin"}
BLEND_TECHNIQUES: tuple[TransitionTechnique, ...] = (
    "eq_bass_swap",
    "equal_power_blend",
    "filter_build",
    "loop_roll",
    "double_drop",
    "stem_vocal_overlay",
)
ALL_TECHNIQUES: tuple[TransitionTechnique, ...] = (*BLEND_TECHNIQUES, "echo_out_drop_in", "quick_cut", "backspin")

SAFE_TEMPO_DIFFERENCE = 0.10
SAFE_RATE_RANGE = (0.88, 1.12)
STRETCH_TEMPO_DIFFERENCE = 0.20
STRETCH_RATE_RANGE = (0.75, 1.33)


@dataclass(slots=True)
class TransitionDecision:
    intent: TransitionIntent
    intent_score: float
    assessment: TechniqueAssessment
    assessments: list[TechniqueAssessment]
    final_score: float
    target_bpm: float
    render_rate_a: float
    render_rate_b: float
    effective_bpm_a: float
    effective_bpm_b: float
    transition_seconds: float
    source_duration_a: float
    source_duration_b: float
    time_stretch_applied: bool
    warning: str | None
    requires_confirmation: bool
    quality: QualityClass
    rationale: list[str]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _gaussian(value: float, target: float, width: float) -> float:
    return _clamp(math.exp(-(((value - target) / max(width, 1e-9)) ** 2)))


def cue_confidence(breakdown: CompatibilityBreakdown) -> float:
    return _clamp(
        0.32 * (breakdown.beat_grid or 0.55)
        + 0.28 * (breakdown.boundary or 0.45)
        + 0.22 * (breakdown.phrase or 0.55)
        + 0.18 * (breakdown.section_fit or 0.55)
    )


_SECTION_INTENT: dict[TransitionIntent, dict[tuple[str, str], float]] = {
    "smooth_continuation": {
        ("groove", "groove"): 1.00,
        ("outro", "groove"): 0.90,
        ("breakdown", "groove"): 0.84,
        ("groove", "intro"): 0.82,
        ("outro", "intro"): 0.72,
        ("breakdown", "intro"): 0.68,
    },
    "groove_handoff": {
        ("groove", "groove"): 1.00,
        ("build", "groove"): 0.90,
        ("outro", "groove"): 0.86,
        ("groove", "build"): 0.84,
        ("groove", "intro"): 0.78,
        ("outro", "intro"): 0.66,
    },
    "energy_lift": {
        ("breakdown", "drop"): 1.00,
        ("build", "drop"): 0.98,
        ("groove", "drop"): 0.90,
        ("groove", "build"): 0.84,
        ("breakdown", "build"): 0.80,
        ("outro", "drop"): 0.70,
    },
    "energy_release": {
        ("drop", "breakdown"): 0.96,
        ("groove", "breakdown"): 0.90,
        ("groove", "groove"): 0.82,
        ("breakdown", "intro"): 0.76,
        ("outro", "intro"): 0.68,
    },
    "breakdown_to_drop": {
        ("breakdown", "drop"): 1.00,
        ("build", "drop"): 0.94,
        ("groove", "drop"): 0.76,
    },
    "vocal_handoff": {
        ("breakdown", "groove"): 0.94,
        ("groove", "groove"): 0.88,
        ("breakdown", "intro"): 0.84,
        ("groove", "intro"): 0.78,
        ("outro", "groove"): 0.76,
        ("outro", "intro"): 0.66,
    },
    "dramatic_switch": {
        ("breakdown", "drop"): 1.00,
        ("build", "drop"): 0.92,
        ("drop", "drop"): 0.88,
        ("groove", "drop"): 0.84,
        ("outro", "drop"): 0.76,
    },
    "emergency_escape": {
        ("groove", "groove"): 0.94,
        ("breakdown", "drop"): 0.92,
        ("groove", "drop"): 0.90,
        ("outro", "groove"): 0.86,
        ("outro", "drop"): 0.84,
        ("groove", "intro"): 0.74,
        ("outro", "intro"): 0.68,
    },
}



def section_intent_score(intent: TransitionIntent, exit_section: str, entry_section: str) -> float:
    pair = (exit_section, entry_section)
    if pair in _SECTION_INTENT[intent]:
        return _SECTION_INTENT[intent][pair]
    if exit_section == "unknown" or entry_section == "unknown":
        return 0.46
    if pair in {("outro", "outro"), ("intro", "outro")}:
        return 0.18
    if intent in {"energy_lift", "breakdown_to_drop"} and entry_section != "drop":
        return 0.32
    if intent == "smooth_continuation" and pair in {("drop", "drop"), ("drop", "breakdown")}:
        return 0.30
    return 0.52


def candidate_intents(
    style: TransitionStyle,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    breakdown: CompatibilityBreakdown,
) -> list[TransitionIntent]:
    if style == "emergency":
        return ["emergency_escape"]

    intents: list[TransitionIntent]
    if style == "creative":
        intents = ["breakdown_to_drop", "energy_lift", "dramatic_switch", "vocal_handoff"]
    else:
        intents = ["smooth_continuation", "groove_handoff", "energy_release", "vocal_handoff"]

    pair = (exit_candidate.section_label, entry_candidate.section_label)
    if pair == ("breakdown", "drop"):
        intents.insert(0, "breakdown_to_drop")
    if breakdown.energy_delta >= 0.10:
        intents.insert(0, "energy_lift")
    elif breakdown.energy_delta <= -0.10:
        intents.insert(0, "energy_release")
    return list(dict.fromkeys(intents))


def intent_suitability(
    intent: TransitionIntent,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    breakdown: CompatibilityBreakdown,
) -> float:
    section = section_intent_score(intent, exit_candidate.section_label, entry_candidate.section_label)
    delta = float(breakdown.energy_delta)
    if intent in {"smooth_continuation", "groove_handoff"}:
        energy = _gaussian(delta, 0.02, 0.24)
    elif intent in {"energy_lift", "breakdown_to_drop"}:
        energy = _gaussian(delta, 0.16, 0.25)
    elif intent == "energy_release":
        energy = _gaussian(delta, -0.14, 0.25)
    elif intent == "dramatic_switch":
        energy = _clamp(0.42 + abs(delta) / 0.35)
    elif intent == "vocal_handoff":
        energy = breakdown.vocal_overlap
    else:
        energy = _clamp(0.45 + 0.55 * entry_candidate.boundary_strength)

    source = breakdown.vocal_overlap if intent == "vocal_handoff" else (
        0.5 * breakdown.vocal_overlap + 0.5 * breakdown.bass_conflict
    )
    structure = _clamp(
        0.5 * min(exit_candidate.boundary_strength, entry_candidate.boundary_strength)
        + 0.5 * min(exit_candidate.phrase_strength, entry_candidate.phrase_strength)
    )
    return _clamp(0.55 * section + 0.20 * energy + 0.10 * source + 0.15 * structure)


_TECHNIQUE_INTENT: dict[TransitionIntent, dict[str, float]] = {
    "smooth_continuation": {"eq_bass_swap": 1.0, "equal_power_blend": 0.92, "filter_build": 0.62, "stem_vocal_overlay": 0.74},
    "groove_handoff": {"eq_bass_swap": 1.0, "equal_power_blend": 0.86, "filter_build": 0.70, "loop_roll": 0.62},
    "energy_lift": {"filter_build": 1.0, "loop_roll": 0.92, "double_drop": 0.86, "eq_bass_swap": 0.58},
    "energy_release": {"filter_build": 0.86, "eq_bass_swap": 0.72, "equal_power_blend": 0.68, "echo_out_drop_in": 0.66},
    "breakdown_to_drop": {"filter_build": 1.0, "double_drop": 0.96, "loop_roll": 0.88, "quick_cut": 0.68},
    "vocal_handoff": {"stem_vocal_overlay": 1.0, "filter_build": 0.82, "eq_bass_swap": 0.64, "quick_cut": 0.56},
    "dramatic_switch": {"double_drop": 1.0, "backspin": 0.88, "loop_roll": 0.84, "quick_cut": 0.80, "filter_build": 0.74},
    "emergency_escape": {"echo_out_drop_in": 1.0, "quick_cut": 0.92, "backspin": 0.78},
}

_ALLOWED_BARS: dict[str, tuple[int, ...]] = {
    "eq_bass_swap": (32, 16, 8, 4),
    "equal_power_blend": (16, 8, 4),
    "filter_build": (16, 8, 4),
    "loop_roll": (8, 4, 2),
    "double_drop": (4, 2, 1),
    "stem_vocal_overlay": (16, 8, 4),
    "echo_out_drop_in": (2, 1),
    "quick_cut": (1,),
    "backspin": (1,),
}


_FLEXIBLE_DURATION_TECHNIQUES: set[str] = {
    "eq_bass_swap",
    "equal_power_blend",
    "filter_build",
    "stem_vocal_overlay",
}


def _choose_bars(
    technique: str,
    maximum: int,
    style: TransitionStyle,
    score_hint: float,
    requested_bars: int | None,
    *,
    exact_bars: bool = False,
) -> int | None:
    available = [bars for bars in _ALLOWED_BARS[technique] if bars <= maximum]
    if requested_bars is not None:
        if exact_bars:
            if requested_bars < 1 or requested_bars > 64:
                raise ValueError("Exact transition duration must be between 1 and 64 bars")
            if technique in _FLEXIBLE_DURATION_TECHNIQUES:
                return requested_bars if 4 <= requested_bars <= maximum else None
            return requested_bars if requested_bars in _ALLOWED_BARS[technique] and requested_bars <= maximum else None
        if requested_bars not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("--bars must be one of 1, 2, 4, 8, 16, or 32")
        available = [bars for bars in available if bars <= requested_bars]
    if not available:
        return None
    if style == "creative" and technique not in CUT_TECHNIQUES:
        target = 8 if score_hint >= 0.70 else 4
        valid = [bars for bars in available if bars <= target]
        return max(valid) if valid else min(available)
    if technique in CUT_TECHNIQUES:
        return max(available)
    if score_hint >= 0.84:
        return max(available)
    if score_hint >= 0.66:
        valid = [bars for bars in available if bars <= 16]
        return max(valid) if valid else min(available)
    valid = [bars for bars in available if bars <= 8]
    return max(valid) if valid else min(available)


def _tempo_context(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    breakdown: CompatibilityBreakdown,
    tempo_policy: TempoPolicy,
    effective_bpm_a: float,
) -> tuple[bool, float, float, list[str]]:
    matched_b, difference, rate_b = match_bpm(effective_bpm_a, track_b.bpm)
    warnings: list[str] = []
    meter_a = int(track_a.time_signature or 0)
    meter_b = int(track_b.time_signature or 0)
    meter_ok = meter_a in {3, 4, 6} and meter_a == meter_b
    grid_ok = (
        breakdown.beat_grid >= 0.38
        and not track_a.variable_tempo
        and not track_b.variable_tempo
    )
    if tempo_policy == "safe":
        range_ok = SAFE_RATE_RANGE[0] <= rate_b <= SAFE_RATE_RANGE[1]
        blend_allowed = difference <= SAFE_TEMPO_DIFFERENCE and range_ok and meter_ok and grid_ok
    elif tempo_policy == "stretch":
        range_ok = STRETCH_RATE_RANGE[0] <= rate_b <= STRETCH_RATE_RANGE[1]
        blend_allowed = difference <= STRETCH_TEMPO_DIFFERENCE and range_ok and meter_ok and grid_ok
    else:
        blend_allowed = False

    if not meter_ok:
        warnings.append("The tracks use unknown or incompatible estimated meters, so transparent beatmatching is disabled.")
    if track_a.variable_tempo:
        warnings.append("The outgoing track has variable tempo, so a constant beatmatched blend is unsafe.")
    if track_b.variable_tempo:
        warnings.append("The incoming track has variable tempo, so a constant beatmatched blend is unsafe.")
    if breakdown.beat_grid < 0.38:
        warnings.append("The local beat grid is too uncertain for a transparent blend.")
    if difference > SAFE_TEMPO_DIFFERENCE:
        warnings.append(f"The tempo adjustment is large ({difference * 100.0:.1f}%).")
    elif difference > 0.06:
        warnings.append(f"The tempo adjustment is noticeable ({difference * 100.0:.1f}%).")
    return blend_allowed, difference, rate_b, warnings


def _technique_failures(
    technique: TransitionTechnique,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    breakdown: CompatibilityBreakdown,
    blend_allowed: bool,
    tempo_difference: float,
    stems_available: bool,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    if technique not in CUT_TECHNIQUES and not blend_allowed:
        failures.append("The tempo/meter/beat-grid state cannot support a beatmatched blend.")

    if technique == "eq_bass_swap":
        if breakdown.beat_grid < 0.45:
            failures.append("EQ bass swap needs a stronger local beat grid.")
        if breakdown.phrase < 0.38:
            failures.append("EQ bass swap needs clearer phrase alignment.")
        if breakdown.harmonic < 0.42:
            failures.append("The selected local harmony is too weak for a long EQ blend.")
        if breakdown.bass_conflict < 0.36:
            failures.append("Both selected sections contain too much bass for a clean bass swap.")
        if breakdown.vocal_overlap < 0.28:
            failures.append("Both selected sections contain prominent vocals.")
    elif technique == "equal_power_blend":
        if breakdown.beat_grid < 0.48:
            failures.append("A full-spectrum blend needs a stable local beat grid.")
        if breakdown.phrase < 0.42:
            failures.append("A full-spectrum blend needs clearer phrase alignment.")
        if breakdown.harmonic < 0.55:
            failures.append("A full-spectrum blend would expose a harmonic clash.")
        if breakdown.vocal_overlap < 0.50:
            failures.append("A full-spectrum blend would overlap too much vocal content.")
        if breakdown.bass_conflict < 0.45:
            failures.append("A full-spectrum blend would overlap both basslines.")
    elif technique == "filter_build":
        if breakdown.beat_grid < 0.38:
            failures.append("Filter build still needs a usable local beat grid.")
        if breakdown.phrase < 0.30:
            failures.append("Filter build needs a clearer phrase boundary.")
        if breakdown.harmonic < 0.26:
            failures.append("The local harmonic clash is too severe even for a filtered overlap.")
        if breakdown.vocal_overlap < 0.24:
            warnings.append("The filter transition may still expose competing vocals.")
    elif technique == "loop_roll":
        if breakdown.beat_grid < 0.55:
            failures.append("Loop roll needs a very stable beat grid.")
        if breakdown.phrase < 0.44:
            failures.append("Loop roll needs a strong phrase boundary.")
        if exit_candidate.boundary_strength < 0.38:
            failures.append("The outgoing cue is not strong enough for a loop roll.")
    elif technique == "double_drop":
        if track_a.genre.label != "drum_and_bass" or track_b.genre.label != "drum_and_bass":
            failures.append("Double drop is currently restricted to drum-and-bass pairs.")
        if entry_candidate.section_label != "drop":
            failures.append("Double drop requires an incoming drop.")
        if exit_candidate.section_label not in {"build", "breakdown", "drop"}:
            failures.append("Double drop needs an outgoing build, breakdown, or drop.")
        if breakdown.beat_grid < 0.62 or tempo_difference > 0.04:
            failures.append("Double drop needs tightly matched tempo and highly stable beat grids.")
    elif technique == "stem_vocal_overlay":
        if not stems_available:
            failures.append("Stem vocal overlay requires cached stems for both tracks.")
        if breakdown.harmonic < 0.50:
            failures.append("Stem vocal overlay needs compatible local harmony.")
        if breakdown.beat_grid < 0.40:
            failures.append("Stem vocal overlay needs a stable local beat grid.")
    elif technique == "backspin":
        if not ({track_a.genre.label, track_b.genre.label} & {"hip_hop", "pop"}):
            warnings.append("Backspin is unusual for this genre pair.")
    elif technique == "quick_cut":
        if entry_candidate.boundary_strength < 0.28:
            warnings.append("The incoming event is not a particularly strong cut point.")
    elif technique == "echo_out_drop_in":
        if exit_candidate.boundary_strength < 0.25:
            warnings.append("The outgoing echo point is structurally weak.")
    return failures, warnings


def _technique_score(
    technique: TransitionTechnique,
    intent: TransitionIntent,
    style: TransitionStyle,
    breakdown: CompatibilityBreakdown,
    intent_score: float,
    cue_score: float,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
) -> float:
    fit = _TECHNIQUE_INTENT[intent].get(technique, 0.28)
    style_fit = 0.55
    if style == "safe":
        style_fit = 1.0 if technique in {"eq_bass_swap", "equal_power_blend"} else 0.72 if technique == "filter_build" else 0.35
    elif style == "creative":
        style_fit = 1.0 if technique in {"filter_build", "loop_roll", "double_drop", "stem_vocal_overlay", "backspin"} else 0.62
    else:
        style_fit = 1.0 if technique in CUT_TECHNIQUES else 0.18

    if technique in CUT_TECHNIQUES:
        incoming_event = _clamp(0.55 * entry_candidate.boundary_strength + 0.45 * entry_candidate.phrase_strength)
        return _clamp(
            0.18 * breakdown.overall
            + 0.25 * intent_score
            + 0.22 * fit
            + 0.12 * style_fit
            + 0.13 * cue_score
            + 0.10 * incoming_event
        )
    return _clamp(
        0.30 * breakdown.overall
        + 0.24 * intent_score
        + 0.17 * fit
        + 0.09 * style_fit
        + 0.10 * cue_score
        + 0.04 * breakdown.harmonic
        + 0.03 * breakdown.vocal_overlap
        + 0.03 * breakdown.bass_conflict
    )


def assess_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    breakdown: CompatibilityBreakdown,
    *,
    style: TransitionStyle,
    tempo_policy: TempoPolicy,
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    requested_bars: int | None = None,
    stems_available: bool = False,
    allow_fallback: bool = False,
    exact_bars: bool = False,
) -> TransitionDecision | None:
    effective_a = float(effective_bpm_a or track_a.bpm)
    render_rate_a = max(0.01, float(render_rate_a))
    blend_allowed, tempo_difference, rate_b, tempo_warnings = _tempo_context(
        track_a, track_b, breakdown, tempo_policy, effective_a
    )
    cue_score = cue_confidence(breakdown)
    maximum_bars = min(exit_candidate.available_bars, entry_candidate.available_bars)
    intents = candidate_intents(style, exit_candidate, entry_candidate, breakdown)

    assessments: list[TechniqueAssessment] = []
    best_payload: tuple[float, TransitionIntent, float, TechniqueAssessment, float, float, float, float] | None = None

    for intent in intents:
        intent_score = intent_suitability(intent, exit_candidate, entry_candidate, breakdown)
        for technique in ALL_TECHNIQUES:
            if style == "emergency" and technique not in CUT_TECHNIQUES:
                continue
            if style != "emergency" and technique in CUT_TECHNIQUES and not allow_fallback:
                continue
            bars = _choose_bars(
                technique,
                maximum_bars,
                style,
                breakdown.overall,
                requested_bars,
                exact_bars=exact_bars,
            )
            failures, warnings = _technique_failures(
                technique,
                track_a,
                track_b,
                exit_candidate,
                entry_candidate,
                breakdown,
                blend_allowed,
                tempo_difference,
                stems_available,
            )
            if bars is None:
                failures.append("The selected cues do not contain enough bars for this technique.")
                bars = 1

            beatmatched = technique not in CUT_TECHNIQUES
            if beatmatched:
                transition_mode: TransitionMode = "beatmatched_blend"
                tempo_strategy: TempoStrategy = "constant_stretch" if abs(rate_b - 1.0) >= 0.002 else "native_blend"
                actual_rate_b = rate_b
                effective_b = effective_a
            else:
                transition_mode = "emergency_cut"
                tempo_strategy = "native_cut"
                actual_rate_b = 1.0
                effective_b = track_b.bpm

            beats_per_bar = track_a.time_signature if track_a.time_signature in {3, 4, 6} else 4
            transition_seconds = bars * beats_per_bar * 60.0 / max(effective_a, 1.0)
            source_duration_a = transition_seconds * render_rate_a
            if technique == "echo_out_drop_in":
                # The incoming track enters only on the final native-tempo beat.
                source_duration_b = min(
                    transition_seconds,
                    60.0 / max(track_b.bpm, 1.0),
                )
            elif technique == "quick_cut":
                # The cut pivots at roughly 41% of the rendered window.
                source_duration_b = transition_seconds * 0.59 * actual_rate_b
            elif technique == "backspin":
                # The incoming track becomes audible at roughly 48%.
                source_duration_b = transition_seconds * 0.52 * actual_rate_b
            else:
                source_duration_b = transition_seconds * actual_rate_b
            if exit_candidate.time_seconds + source_duration_a > track_a.duration_seconds + 0.10:
                failures.append("The outgoing track does not contain enough source audio after this cue.")
            if entry_candidate.time_seconds + source_duration_b > track_b.duration_seconds + 0.10:
                failures.append("The incoming track does not contain enough source audio after this cue.")

            score = _technique_score(
                technique, intent, style, breakdown, intent_score, cue_score, exit_candidate, entry_candidate
            )
            if warnings:
                score -= min(0.12, 0.025 * len(warnings))
            feasible = not failures
            assessment = TechniqueAssessment(
                technique=technique,
                intent=intent,
                feasible=feasible,
                score=round(_clamp(score) if feasible else 0.0, 4),
                transition_bars=bars,
                beatmatched=beatmatched,
                tempo_strategy=tempo_strategy,
                transition_mode=transition_mode,
                hard_failures=failures,
                warnings=[*tempo_warnings, *warnings] if beatmatched else warnings,
                reasons=[
                    f"Intent fit is {intent_score:.0%} for {exit_candidate.section_label} -> {entry_candidate.section_label}.",
                    f"Technique suitability is {_TECHNIQUE_INTENT[intent].get(technique, 0.28):.0%} for {intent.replace('_', ' ')}.",
                ],
            )
            assessments.append(assessment)
            if not feasible:
                continue
            payload = (
                assessment.score,
                intent,
                intent_score,
                assessment,
                transition_seconds,
                source_duration_a,
                source_duration_b,
                actual_rate_b,
            )
            if best_payload is None or payload[0] > best_payload[0]:
                best_payload = payload

    if best_payload is None:
        return None

    final_score, intent, intent_score, chosen, transition_seconds, source_duration_a, source_duration_b, actual_rate_b = best_payload
    if chosen.beatmatched:
        effective_b = effective_a
    else:
        effective_b = track_b.bpm
    time_stretch_applied = chosen.beatmatched and abs(actual_rate_b - 1.0) >= 0.002

    warnings = list(dict.fromkeys(chosen.warnings))
    large_stretch = chosen.beatmatched and tempo_difference > SAFE_TEMPO_DIFFERENCE
    requires_confirmation = (
        chosen.transition_mode == "emergency_cut"
        or final_score < 0.72
        or cue_score < 0.52
        or large_stretch
        or tempo_policy in {"stretch", "cut"}
    )
    quality: QualityClass = (
        "good"
        if final_score >= 0.72 and not requires_confirmation
        else "risky"
        if final_score >= 0.48 or chosen.transition_mode == "emergency_cut"
        else "poor"
    )
    warning = warnings[0] if warnings else (
        "This transition needs review before rendering." if requires_confirmation else None
    )

    rationale = [
        f"Chosen intent: {intent.replace('_', ' ')} ({intent_score:.0%} suitability).",
        f"Selected {exit_candidate.section_label} -> {entry_candidate.section_label} at strong local cue points.",
        f"Technique: {chosen.technique.replace('_', ' ')} over {chosen.transition_bars} bar(s).",
    ]
    if chosen.beatmatched:
        rationale.append(
            f"The incoming track is rendered at {effective_a:.2f} BPM with rate {actual_rate_b:.5f}."
            if time_stretch_applied
            else f"The tracks blend at {effective_a:.2f} BPM without material time stretching."
        )
    else:
        rationale.append("The transition uses a short native-tempo cut and does not claim transparent beatmatching.")
    if breakdown.local_key_a and breakdown.local_key_b:
        rationale.append(
            f"Local harmonic evidence compares {breakdown.local_key_a} -> {breakdown.local_key_b} at {breakdown.harmonic:.0%}."
        )
    if breakdown.vocal_overlap < 0.55:
        rationale.append(f"Vocal safety is only {breakdown.vocal_overlap:.0%}; keep the overlap controlled.")
    if breakdown.bass_conflict < 0.55:
        rationale.append(f"Bass safety is only {breakdown.bass_conflict:.0%}; the technique limits low-end overlap.")

    return TransitionDecision(
        intent=intent,
        intent_score=round(intent_score, 4),
        assessment=chosen,
        assessments=assessments,
        final_score=round(final_score, 4),
        target_bpm=round(effective_a, 4),
        render_rate_a=round(render_rate_a, 6),
        render_rate_b=round(actual_rate_b, 6),
        effective_bpm_a=round(effective_a, 4),
        effective_bpm_b=round(effective_b, 4),
        transition_seconds=round(transition_seconds, 6),
        source_duration_a=round(source_duration_a, 6),
        source_duration_b=round(source_duration_b, 6),
        time_stretch_applied=time_stretch_applied,
        warning=warning,
        requires_confirmation=requires_confirmation,
        quality=quality,
        rationale=rationale,
    )


def execution_signature(plan: TransitionPlan) -> str:
    """Hash the complete immutable execution contract, including audio assets."""
    payload = {
        "track_a": plan.track_a,
        "track_b": plan.track_b,
        "exit_time_a": round(plan.exit_time_a, 6),
        "entry_time_b": round(plan.entry_time_b, 6),
        "intent": plan.intent,
        "technique": plan.technique,
        "transition_bars": plan.transition_bars,
        "transition_seconds": round(plan.transition_seconds, 6),
        "transition_mode": plan.transition_mode,
        "tempo_strategy": plan.tempo_strategy,
        "effective_bpm_a": round(plan.effective_bpm_a, 4),
        "effective_bpm_b": round(plan.effective_bpm_b, 4),
        "render_rate_a": round(plan.render_rate_a, 6),
        "render_rate_b": round(plan.render_rate_b, 6),
        "phase_nudge_seconds_b": round(plan.phase_nudge_seconds_b, 6),
        "tempo_release_seconds_b": round(plan.tempo_release_seconds_b, 6),
        "tempo_release_source_duration_b": round(plan.tempo_release_source_duration_b, 6),
        "tempo_release_start_rate_b": round(plan.tempo_release_start_rate_b, 6),
        "tempo_release_end_rate_b": round(plan.tempo_release_end_rate_b, 6),
        "post_transition_rate_b": round(plan.post_transition_rate_b, 6),
        "post_transition_bpm_b": round(plan.post_transition_bpm_b, 4),
        "source_duration_a": round(plan.source_duration_a, 6),
        "source_duration_b": round(plan.source_duration_b, 6),
        "performance_program_fingerprint": plan.performance_program_fingerprint,
        "execution_asset_paths": dict(sorted(plan.execution_asset_paths.items())),
        "execution_asset_fingerprints": dict(sorted(plan.execution_asset_fingerprints.items())),
        "renderer_version": plan.renderer_version,
        "decision_version": plan.decision_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def seal_execution_plan(plan: TransitionPlan) -> TransitionPlan:
    """Bind all audio inputs and renderer semantics, then sign the plan."""
    bind_execution_assets(plan)
    plan.execution_signature = execution_signature(plan)
    return plan


def validate_execution_plan(plan: TransitionPlan, *, verify_assets: bool = True) -> None:
    if not plan.compiled_execution:
        raise ValueError("Transition plan is not compiled for execution.")
    if plan.decision_version != DECISION_VERSION:
        raise ValueError(
            f"Transition decision version {plan.decision_version!r} is incompatible with {DECISION_VERSION!r}."
        )
    if plan.renderer_version != RENDER_ENGINE_VERSION:
        raise ValueError(
            f"Transition renderer version {plan.renderer_version!r} is incompatible with {RENDER_ENGINE_VERSION!r}."
        )
    if plan.execution_signature != execution_signature(plan):
        raise ValueError("Transition execution signature is invalid; re-plan before rendering.")
    if verify_assets:
        verify_execution_assets(plan)
    if plan.render_rate_a <= 0.0 or plan.render_rate_b <= 0.0 or plan.post_transition_rate_b <= 0.0:
        raise ValueError("Transition render rates must be positive.")
    if plan.tempo_release_seconds_b < 0.0 or plan.tempo_release_source_duration_b < 0.0:
        raise ValueError("Deck tempo-release durations must be non-negative.")
    if plan.tempo_strategy == "deck_handoff":
        if plan.tempo_release_seconds_b <= 0.0:
            raise ValueError("Deck handoff tempo strategy requires a positive tempo release.")
        if abs(plan.tempo_release_start_rate_b - plan.render_rate_b) > 1e-5:
            raise ValueError("Deck handoff release must begin at the rehearsed incoming match rate.")
    if plan.transition_seconds <= 0.0:
        raise ValueError("Transition duration must be positive.")
    if plan.source_duration_a < 0.0 or plan.source_duration_b < 0.0:
        raise ValueError("Transition source durations must be non-negative.")
    if plan.technique == "stem_choreography":
        required = ("vocals", "drums", "bass", "other")
        if not all(plan.stem_paths_a.get(name) and plan.stem_paths_b.get(name) for name in required):
            raise ValueError("Stem choreography requires vocals/drums/bass/other stems for both tracks.")
        profile = str(dict(plan.metadata.get("stem_choreography", {})).get("profile", ""))
        if not profile:
            raise ValueError("Stem choreography is missing its musical-role ownership profile.")
    if plan.technique == "layer_vocal_bridge":
        outgoing_vocal = dict(plan.performance_layers_a or {}).get("vocals")
        incoming_instrumental = dict(plan.performance_layers_b or {}).get("instrumental")
        if not isinstance(outgoing_vocal, dict) or not outgoing_vocal.get("path"):
            raise ValueError("Layer vocal bridge is missing the outgoing vocal performance-layer source.")
        if not isinstance(incoming_instrumental, dict) or not incoming_instrumental.get("path"):
            raise ValueError("Layer vocal bridge is missing the incoming instrumental performance-layer source.")
    if plan.technique == "layer_incoming_vocal_handoff":
        outgoing_instrumental = dict(plan.performance_layers_a or {}).get("instrumental")
        incoming_vocal = dict(plan.performance_layers_b or {}).get("vocals")
        incoming_instrumental = dict(plan.performance_layers_b or {}).get("instrumental")
        if not isinstance(outgoing_instrumental, dict) or not outgoing_instrumental.get("path"):
            raise ValueError("Incoming-vocal layer handoff is missing the outgoing instrumental source.")
        if not isinstance(incoming_vocal, dict) or not incoming_vocal.get("path"):
            raise ValueError("Incoming-vocal layer handoff is missing the incoming vocal source.")
        if not isinstance(incoming_instrumental, dict) or not incoming_instrumental.get("path"):
            raise ValueError("Incoming-vocal layer handoff is missing the incoming instrumental source.")
        layer_safety = dict(plan.metadata.get("layer_safety", {}) or {})
        if not bool(layer_safety.get("feasible", False)):
            raise ValueError("Incoming-vocal layer handoff is missing a passing Layer Intelligence safety assessment.")
    if plan.technique == "layer_instrumental_handoff":
        incoming_instrumental = dict(plan.performance_layers_b or {}).get("instrumental")
        if not isinstance(incoming_instrumental, dict) or not incoming_instrumental.get("path"):
            raise ValueError("Layer instrumental handoff is missing the incoming instrumental performance-layer source.")
    if plan.performance_program_fingerprint:
        from .performance.program import transition_program_from_dict
        program = transition_program_from_dict(dict(plan.performance_program or {}))
        if program.fingerprint != plan.performance_program_fingerprint:
            raise ValueError("Transition performance-program fingerprint is invalid; re-plan before rendering.")
        if program.technique != plan.technique or program.intent != plan.intent:
            raise ValueError("Transition performance program does not match the immutable plan.")
