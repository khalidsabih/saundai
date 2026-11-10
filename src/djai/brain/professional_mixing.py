from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from ..decision import seal_execution_plan
from ..models import TrackAnalysis, TransitionPlan
from ..performance.program import attach_performance_program
from .transition_router import TransitionRoute

PROFESSIONAL_MIXING_VERSION = "1.0.0"

MixFamily = Literal[
    "full_blend",
    "layer_blend",
    "stem_performance",
    "structural_bridge",
    "phrase_switch",
    "structural_reset",
]

RESET_TECHNIQUES = {"echo_out_drop_in", "quick_cut", "backspin"}
CONTINUOUS_TECHNIQUES = {
    "eq_bass_swap",
    "equal_power_blend",
    "filter_build",
    "loop_roll",
    "stem_vocal_overlay",
    "stem_choreography",
    "layer_vocal_bridge",
    "layer_incoming_vocal_handoff",
    "layer_instrumental_handoff",
}
STEM_TECHNIQUES = {"stem_vocal_overlay", "stem_choreography"}
LAYER_TECHNIQUES = {
    "layer_vocal_bridge",
    "layer_incoming_vocal_handoff",
    "layer_instrumental_handoff",
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _layer_capability(track: TrackAnalysis) -> float:
    layers = dict(track.performance_layer_analysis or {})
    stems = dict(track.stem_analysis or {})
    layer_value = 0.0
    if float(layers.get("available", 0.0) or 0.0) > 0.0:
        layer_value = _clamp(
            float(layers.get("quality_confidence", layers.get("alignment_confidence", 0.72)) or 0.72)
        )
    stem_value = 0.0
    if float(stems.get("available", 0.0) or 0.0) > 0.0:
        stem_value = _clamp(float(stems.get("quality_confidence", 0.72) or 0.72))
    return max(layer_value, stem_value)


def _complete_stem_plan(plan: TransitionPlan) -> bool:
    required = ("vocals", "drums", "bass", "other")
    return all(plan.stem_paths_a.get(name) and plan.stem_paths_b.get(name) for name in required)


def _performance_layer_plan(plan: TransitionPlan) -> bool:
    return bool(plan.performance_layers_a or plan.performance_layers_b)


def _local_vocal_conflict(plan: TransitionPlan, track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    outgoing = float(plan.metadata.get("exit_vocal_probability", track_a.vocal_probability) or 0.0)
    incoming = float(plan.metadata.get("entry_vocal_probability", track_b.vocal_probability) or 0.0)
    compatibility_risk = 1.0 - _clamp(plan.compatibility.vocal_overlap)
    local_overlap = _clamp(min(outgoing, incoming) * 1.35)
    return _clamp(0.62 * compatibility_risk + 0.38 * local_overlap)


def _bass_conflict(plan: TransitionPlan) -> float:
    return _clamp(1.0 - plan.compatibility.bass_conflict)


def _melodic_conflict(plan: TransitionPlan) -> float:
    harmonic_risk = 1.0 - _clamp(plan.compatibility.harmonic)
    texture_risk = 1.0 - _clamp(plan.compatibility.texture)
    return _clamp(0.72 * harmonic_risk + 0.28 * texture_risk)


def _tempo_distance(plan: TransitionPlan) -> float:
    value = plan.compatibility.local_tempo_difference_percent or plan.compatibility.tempo_difference_percent
    return _clamp(abs(float(value)) / 18.0)


@dataclass(frozen=True, slots=True)
class ProfessionalMixProfile:
    version: str
    family: MixFamily
    continuous_mix_required: bool
    structural_reset: bool
    reset_justified: bool
    bridge_preferred: bool
    tempo_distance: float
    rhythm_compatibility: float
    harmonic_compatibility: float
    phrase_compatibility: float
    vocal_conflict: float
    bass_conflict: float
    melodic_conflict: float
    layer_capability: float
    recommended_source_mode: str
    ownership: dict[str, str]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def classify_professional_mix(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    route: TransitionRoute,
) -> ProfessionalMixProfile:
    tempo_distance = _tempo_distance(plan)
    rhythm = _clamp(0.58 * plan.compatibility.beat_grid + 0.42 * plan.compatibility.phrase)
    harmonic = _clamp(plan.compatibility.harmonic)
    phrase = _clamp(plan.compatibility.phrase)
    vocal_conflict = _local_vocal_conflict(plan, track_a, track_b)
    bass_conflict = _bass_conflict(plan)
    melodic_conflict = _melodic_conflict(plan)
    layer_capability = max(_layer_capability(track_a), _layer_capability(track_b))
    has_stems = _complete_stem_plan(plan) or plan.technique in STEM_TECHNIQUES
    has_layers = _performance_layer_plan(plan) or plan.technique in LAYER_TECHNIQUES
    structural_reset = plan.technique in RESET_TECHNIQUES or plan.transition_mode == "emergency_cut"
    reasons: list[str] = []

    close_pair = (
        tempo_distance <= 0.34
        and rhythm >= 0.67
        and harmonic >= 0.55
        and vocal_conflict <= 0.38
        and bass_conflict <= 0.42
    )
    conflict_heavy = max(vocal_conflict, bass_conflict, melodic_conflict) >= 0.46

    if close_pair and not structural_reset:
        family: MixFamily = "full_blend"
        reasons.append("The selected sections are close enough for a phrase-aligned continuous blend.")
    elif has_stems and conflict_heavy and not structural_reset:
        family = "stem_performance"
        reasons.append("Four-stem control can remove the musical elements that would otherwise collide.")
    elif has_layers and (vocal_conflict >= 0.34 or melodic_conflict >= 0.42) and not structural_reset:
        family = "layer_blend"
        reasons.append("Supplied vocal/instrumental layers can preserve one lead while the accompaniment changes owner.")
    elif (
        not structural_reset
        and rhythm >= 0.48
        and phrase >= 0.48
        and (tempo_distance <= 0.72 or route.tempo.relation != "direct")
    ):
        family = "structural_bridge"
        reasons.append("The full records are not naturally close, so the mix should bridge through compatible rhythmic or instrumental material.")
    elif plan.intent in {"breakdown_to_drop", "dramatic_switch"} and plan.technique in {"double_drop", "loop_roll"}:
        family = "phrase_switch"
        reasons.append("The move is an intentional phrase/drop switch rather than a long full-master blend.")
    elif structural_reset:
        family = "structural_reset"
        reasons.append("The proposal is an explicit structural reset rather than a continuous mix.")
    else:
        family = "structural_bridge"
        reasons.append(
            "The pair is too incompatible for a full blend; keep it as a bridge candidate only if rehearsal proves a valid musical handoff."
        )

    bridge_preferred = bool(route.family == "bridge_preferred" or route.bridge_preference >= 0.66)
    reset_justified = bool(
        structural_reset
        and not bridge_preferred
        and route.direct_transition_confidence <= 0.48
        and route.family in {"hard_reset", "rhythmic_switch", "breakdown_drop"}
        and layer_capability < 0.55
        and plan.style in {"creative", "emergency"}
    )
    continuous_required = family not in {"phrase_switch", "structural_reset"}

    if family == "stem_performance":
        recommended_source = "stems"
    elif family == "layer_blend":
        recommended_source = "layers"
    elif family == "structural_bridge" and has_stems:
        recommended_source = "stems"
    elif family == "structural_bridge" and has_layers:
        recommended_source = "layers"
    else:
        recommended_source = "master"

    if family in {"stem_performance", "layer_blend", "structural_bridge"}:
        ownership = {
            "drums": "a_to_b_overlap",
            "bass": "single_owner_a_to_b",
            "vocals": "single_lead_a_then_b",
            "melody": "one_dominant_bed_at_a_time",
            "master": "outgoing_owner_until_incoming_established",
        }
    elif family == "full_blend":
        ownership = {
            "drums": "shared_phrase_locked",
            "bass": "single_owner_a_to_b",
            "vocals": "avoid_dual_lead",
            "melody": "compatible_overlap",
            "master": "progressive_a_to_b",
        }
    elif family == "phrase_switch":
        ownership = {
            "drums": "pivot_at_phrase_boundary",
            "bass": "single_owner_at_pivot",
            "vocals": "clear_before_pivot",
            "melody": "clear_before_pivot",
            "master": "intentional_phrase_switch",
        }
    else:
        ownership = {
            "drums": "reset_boundary",
            "bass": "no_overlap_required",
            "vocals": "no_overlap_required",
            "melody": "no_overlap_required",
            "master": "structural_reset",
        }

    if bridge_preferred:
        reasons.append("Set-level routing prefers another track over forcing this pair when no professional continuous program passes rehearsal.")
    if structural_reset and not reset_justified:
        reasons.append("Reset-style effects are not considered justified by the current compatibility/capability context.")

    return ProfessionalMixProfile(
        version=PROFESSIONAL_MIXING_VERSION,
        family=family,
        continuous_mix_required=continuous_required,
        structural_reset=structural_reset,
        reset_justified=reset_justified,
        bridge_preferred=bridge_preferred,
        tempo_distance=round(tempo_distance, 4),
        rhythm_compatibility=round(rhythm, 4),
        harmonic_compatibility=round(harmonic, 4),
        phrase_compatibility=round(phrase, 4),
        vocal_conflict=round(vocal_conflict, 4),
        bass_conflict=round(bass_conflict, 4),
        melodic_conflict=round(melodic_conflict, 4),
        layer_capability=round(layer_capability, 4),
        recommended_source_mode=recommended_source,
        ownership=ownership,
        reasons=tuple(reasons),
    )


def _policy_adjustment(plan: TransitionPlan, profile: ProfessionalMixProfile) -> float:
    adjustment = 0.0
    if profile.family == "full_blend":
        if plan.technique == "eq_bass_swap":
            adjustment += 0.07
        elif plan.technique == "stem_choreography":
            adjustment += 0.06
        elif plan.technique == "equal_power_blend":
            adjustment += 0.025
    elif profile.family == "stem_performance":
        adjustment += 0.10 if plan.technique == "stem_choreography" else 0.04 if plan.technique in STEM_TECHNIQUES else -0.04
    elif profile.family == "layer_blend":
        adjustment += 0.09 if plan.technique in LAYER_TECHNIQUES else -0.035
    elif profile.family == "structural_bridge":
        if plan.technique == "stem_choreography":
            adjustment += 0.09
        elif plan.technique in LAYER_TECHNIQUES:
            adjustment += 0.08
        elif plan.technique in {"filter_build", "loop_roll", "eq_bass_swap"}:
            adjustment += 0.035
    elif profile.family == "phrase_switch":
        adjustment += 0.055 if plan.technique in {"double_drop", "loop_roll"} else 0.0

    if profile.structural_reset:
        adjustment += 0.035 if profile.reset_justified else -0.36
    if profile.bridge_preferred and profile.structural_reset:
        adjustment -= 0.18
    if profile.continuous_mix_required and plan.transition_mode == "emergency_cut":
        adjustment -= 0.30
    return max(-0.60, min(0.16, adjustment))


def apply_professional_mixing_policy(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    *,
    route: TransitionRoute,
) -> list[TransitionPlan]:
    """Rerank complete executable ideas by how a professional DJ would combine the material.

    The function does not choose the winner and does not bypass Professional Brain.
    It annotates each already-feasible plan with a compatibility-driven mix family,
    musical ownership intent, and a bounded utility adjustment. Performance programs
    are then recompiled so the immutable execution IR carries the same contract that
    the Professional Brain judged.
    """
    output: list[TransitionPlan] = []
    for plan in plans:
        profile = classify_professional_mix(track_a, track_b, plan, route=route)
        before = float(plan.metadata.get("selection_utility", plan.score))
        adjustment = _policy_adjustment(plan, profile)
        after = _clamp(before + adjustment)
        plan.metadata = {
            **plan.metadata,
            "selection_utility": round(after, 8),
            "professional_mix": {
                **profile.to_dict(),
                "utility_before": round(before, 8),
                "utility_adjustment": round(adjustment, 8),
                "utility_after": round(after, 8),
                "principle": "mix_compatible_parts_not_necessarily_complete_records",
                "reset_is_last_resort": True,
                "professional_brain_may_reject_pair": True,
            },
        }
        plan.rationale = [*plan.rationale, *profile.reasons]
        # The program fingerprint is part of the execution signature. Rebuild and
        # reseal after the higher-level ownership contract becomes known.
        attach_performance_program(plan)
        seal_execution_plan(plan)
        output.append(plan)
    return sorted(
        output,
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            float(plan.score),
            float(plan.cue_confidence),
            str(plan.execution_signature or ""),
        ),
        reverse=True,
    )


def professional_mixing_status() -> dict[str, Any]:
    from .pair_adaptive_mixing import pair_adaptive_status

    return {
        "version": PROFESSIONAL_MIXING_VERSION,
        "compatibility_driven": True,
        "families": [
            "full_blend",
            "layer_blend",
            "stem_performance",
            "structural_bridge",
            "phrase_switch",
            "structural_reset",
        ],
        "ownership_roles": ["drums", "bass", "vocals", "melody", "master"],
        "single_bass_owner": True,
        "avoid_dual_lead_vocals": True,
        "outgoing_owner_until_incoming_established": True,
        "reset_is_last_resort": True,
        "pair_may_be_rejected": True,
        "pair_adaptive_mixing": pair_adaptive_status(),
    }
