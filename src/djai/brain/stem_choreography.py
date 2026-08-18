from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

from ..decision import execution_signature, seal_execution_plan
from ..models import TechniqueAssessment, TrackAnalysis, TransitionPlan
from ..performance.program import attach_performance_program
from ..stems import STEM_NAMES, find_cached_stems, load_stem_manifest
from .transition_router import TransitionRoute


@dataclass(frozen=True, slots=True)
class StemChoreographyProfile:
    name: str
    intent: str
    rationale: str
    complexity: float


_PROFILES = {
    "deck_vocal_exchange": StemChoreographyProfile(
        name="deck_vocal_exchange",
        intent="vocal_handoff",
        rationale="Use deck-style ownership: lock the incoming drums to the outgoing groove, remove the outgoing lead vocal before revealing the incoming lead, swap bass ownership once, then let the incoming instrumental take the room.",
        complexity=0.075,
    ),
    "progressive_handoff": StemChoreographyProfile(
        name="progressive_handoff",
        intent="groove_handoff",
        rationale="Introduce the incoming drums first, hand bass ownership over once, preserve useful outgoing musical material, and reveal incoming vocals late.",
        complexity=0.08,
    ),
    "outgoing_vocal_bridge": StemChoreographyProfile(
        name="outgoing_vocal_bridge",
        intent="vocal_handoff",
        rationale="Keep the outgoing vocal as the continuity layer while incoming drums, bass and instrumental material establish the next track; incoming vocal waits until the outgoing phrase clears.",
        complexity=0.10,
    ),
    "instrumental_reset": StemChoreographyProfile(
        name="instrumental_reset",
        intent="groove_handoff",
        rationale="Remove lead vocals through the middle of the overlap, exchange drums/bass/instrumental ownership, then reveal the incoming vocal after the new groove is established.",
        complexity=0.09,
    ),
    "drum_first_build": StemChoreographyProfile(
        name="drum_first_build",
        intent="energy_lift",
        rationale="Establish incoming drums early, swap bass near the structural pivot, then reveal melodic material and vocals toward the incoming payoff.",
        complexity=0.11,
    ),
    "bass_ownership_swap": StemChoreographyProfile(
        name="bass_ownership_swap",
        intent="smooth_continuation",
        rationale="Keep groove continuity while enforcing one bass owner at a time and handing vocal ownership from the outgoing to incoming track late in the transition.",
        complexity=0.07,
    ),
}


def _complete_stems(track: TrackAnalysis) -> tuple[dict[str, str], float]:
    mapping = find_cached_stems(track.path)
    if not mapping or not all(mapping.get(name) for name in STEM_NAMES):
        return {}, 0.0
    manifest = load_stem_manifest(track.path)
    quality = dict(manifest.get("quality", {}))
    confidence = max(0.0, min(1.0, float(quality.get("quality_confidence", 0.72))))
    return mapping, confidence


def _profiles_for(track_a: TrackAnalysis, track_b: TrackAnalysis, plan: TransitionPlan, route: TransitionRoute) -> tuple[list[StemChoreographyProfile], dict[str, object]]:
    compatibility = plan.compatibility
    values: list[str] = []

    # A real deck handoff is the default when both tracks contain meaningful
    # vocal material: never let two lead vocals simply crossfade by accident.
    local_vocal = max(
        float(plan.metadata.get("exit_vocal_probability", track_a.vocal_probability)),
        float(plan.metadata.get("entry_vocal_probability", track_b.vocal_probability)),
    )
    if local_vocal >= 0.28 or compatibility.vocal_overlap < 0.78:
        values.append("deck_vocal_exchange")

    # Compatibility fields are safety scores: lower means more collision risk.
    if compatibility.vocal_overlap < 0.60 and compatibility.harmonic >= 0.44:
        values.append("outgoing_vocal_bridge")
    if compatibility.bass_conflict < 0.62:
        values.append("bass_ownership_swap")
    if route.family in {"breakdown_drop", "hard_reset"} or compatibility.energy_delta >= 0.10:
        values.append("drum_first_build")
    if compatibility.vocal_overlap < 0.48 or compatibility.harmonic < 0.44:
        values.append("instrumental_reset")
    if route.family in {"direct_blend", "layer_handoff"}:
        values.append("progressive_handoff")

    if not values:
        values.append("progressive_handoff")
    heuristic=list(dict.fromkeys(values)); neural: dict[str,object]={}
    try:
        from ..intelligence.neural_policy import rank_stem_profiles
        ordered,neural=rank_stem_profiles(track_a,track_b,route,plan,heuristic); values=[name for name in ordered if name in _PROFILES]
    except Exception as exc:
        values=heuristic; neural={"available":False,"reason":f"stem choreography policy unavailable: {exc}"}
    return [_PROFILES[name] for name in values[:2]], neural


def _profile_advantage(plan: TransitionPlan, profile: StemChoreographyProfile, confidence: float) -> float:
    compatibility = plan.compatibility
    vocal_need = 1.0 - max(0.0, min(1.0, compatibility.vocal_overlap))
    bass_need = 1.0 - max(0.0, min(1.0, compatibility.bass_conflict))
    structural = max(0.0, min(1.0, 0.5 * compatibility.phrase + 0.5 * compatibility.beat_grid))
    if profile.name == "deck_vocal_exchange":
        benefit = 0.055 + 0.12 * vocal_need + 0.05 * structural + 0.035 * bass_need
    elif profile.name == "outgoing_vocal_bridge":
        benefit = 0.04 + 0.11 * vocal_need + 0.03 * structural
    elif profile.name == "bass_ownership_swap":
        benefit = 0.035 + 0.12 * bass_need + 0.025 * structural
    elif profile.name == "instrumental_reset":
        benefit = 0.035 + 0.08 * vocal_need + 0.04 * (1.0 - compatibility.harmonic)
    elif profile.name == "drum_first_build":
        benefit = 0.035 + 0.06 * max(0.0, compatibility.energy_delta) + 0.04 * structural
    else:
        benefit = 0.025 + 0.035 * structural + 0.025 * max(vocal_need, bass_need)
    return max(0.0, min(0.18, benefit * (0.72 + 0.28 * confidence)))


def augment_stem_choreography_plans(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    *,
    route: TransitionRoute,
) -> list[TransitionPlan]:
    """Generate stem-ownership programs as first-class transition candidates.

    This intentionally does *not* treat stems as a decorative effect. Each
    candidate carries a complete ownership choreography for drums, bass,
    vocals and other musical material. The deterministic renderer executes
    those curves exactly and the normal rendered-audio judge decides whether
    the choreography earns a place in the final set.
    """
    stems_a, confidence_a = _complete_stems(track_a)
    stems_b, confidence_b = _complete_stems(track_b)
    if not stems_a or not stems_b:
        return list(plans)
    confidence = min(confidence_a, confidence_b)
    if confidence < 0.38:
        return list(plans)

    output = list(plans)
    seen = {plan.execution_signature for plan in output}
    # Use the strongest distinct cue/duration bases. Stem intelligence changes
    # source ownership; it should not invent weaker cue points merely to use stems.
    base_keys: set[tuple[float, float, int]] = set()
    bases: list[TransitionPlan] = []
    for plan in plans:
        if not plan.beatmatched or plan.transition_mode == "emergency_cut":
            continue
        key = (round(plan.exit_time_a, 3), round(plan.entry_time_b, 3), int(plan.transition_bars))
        if key in base_keys:
            continue
        base_keys.add(key)
        bases.append(plan)
        if len(bases) >= 5:
            break

    for base in bases:
        profiles,neural_policy=_profiles_for(track_a,track_b,base,route)
        for profile in profiles:
            plan = copy.deepcopy(base)
            plan.technique = "stem_choreography"
            plan.intent = profile.intent  # type: ignore[assignment]
            plan.style = "safe" if profile.name in {"deck_vocal_exchange", "progressive_handoff", "bass_ownership_swap", "instrumental_reset"} else "creative"
            plan.alternative_id = f"stem:{profile.name}"
            plan.stem_paths_a = dict(stems_a)
            plan.stem_paths_b = dict(stems_b)
            advantage = _profile_advantage(plan, profile, confidence)
            base_utility = float(base.metadata.get("selection_utility", base.score))
            utility = min(0.995, base_utility + advantage)
            plan.score = round(min(0.99, max(base.score, utility)), 4)
            plan.analytical_score = plan.score
            plan.technique_score = round(max(0.0, min(0.99, 0.76 + advantage)), 4)
            plan.rationale = [
                *base.rationale,
                profile.rationale,
                "Four separated stems are treated as controllable musical roles, not as a single full-master crossfade.",
            ]
            plan.technique_assessments = [
                *base.technique_assessments,
                TechniqueAssessment(
                    technique="stem_choreography",  # type: ignore[arg-type]
                    intent=plan.intent,
                    feasible=True,
                    score=plan.technique_score,
                    transition_bars=plan.transition_bars,
                    beatmatched=True,
                    tempo_strategy=plan.tempo_strategy,
                    transition_mode=plan.transition_mode,
                    reasons=[profile.rationale],
                ),
            ]
            plan.metadata = {
                **plan.metadata,
                "selection_utility": round(utility, 8),
                "stem_choreography": {
                    "profile": profile.name,
                    "quality_confidence": round(confidence, 4),
                    "principle": "musical_role_ownership_before_effects",
                    "single_bass_owner": True,
                    "avoid_dual_lead_vocals": True,
                    "roles": ["drums", "bass", "vocals", "other"],
                    "neural_policy": dict(neural_policy),
                },
            }
            attach_performance_program(plan)
            seal_execution_plan(plan)
            if plan.execution_signature not in seen:
                output.append(plan)
                seen.add(plan.execution_signature)
    return output
