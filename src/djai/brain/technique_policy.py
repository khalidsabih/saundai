from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from ..models import TrackAnalysis, TransitionPlan
from ..planning_context import PairEvidenceMatrix
from .transition_router import TransitionRoute, route_fit


@dataclass(frozen=True, slots=True)
class TechniquePolicy:
    """Intent-first performance policy for one routed transition family.

    Effects are deliberately downstream of the musical family.  The policy
    says which execution techniques are credible, which intents are preferred,
    and how much intervention each technique costs before rendered-audio
    judging.  The cost is a tie-breaker, not a prohibition: complex moves can
    still win when they deliver a materially better result.
    """

    family: str
    allowed_techniques: tuple[str, ...]
    preferred_intents: tuple[str, ...]
    duration_priority: tuple[int, ...]
    minimum_route_fit: float
    complexity_weight: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed_techniques"] = list(self.allowed_techniques)
        payload["preferred_intents"] = list(self.preferred_intents)
        payload["duration_priority"] = list(self.duration_priority)
        payload["reasons"] = list(self.reasons)
        return payload


# Intervention cost is intentionally about *how much the box is doing*, not
# whether a technique is fashionable.  A clean blend should beat a complex
# effect when their musical outcomes are otherwise similar.
_INTERVENTION_COST: dict[str, float] = {
    "equal_power_blend": 0.04,
    "eq_bass_swap": 0.06,
    "layer_instrumental_handoff": 0.08,
    "layer_incoming_vocal_handoff": 0.09,
    "layer_vocal_bridge": 0.10,
    "stem_vocal_overlay": 0.12,
    "stem_choreography": 0.09,
    "filter_build": 0.13,
    "quick_cut": 0.14,
    "echo_out_drop_in": 0.16,
    "loop_roll": 0.20,
    "double_drop": 0.22,
    "backspin": 0.30,
}


_FAMILY_POLICY: dict[str, dict[str, object]] = {
    "direct_blend": {
        "techniques": (
            "stem_choreography",
            "eq_bass_swap",
            "equal_power_blend",
            "layer_instrumental_handoff",
            "layer_incoming_vocal_handoff",
            "stem_vocal_overlay",
            "layer_vocal_bridge",
            "filter_build",
        ),
        "intents": ("smooth_continuation", "groove_handoff", "vocal_handoff", "energy_release"),
        "durations": (32, 16, 8, 4),
        "minimum_route_fit": 0.58,
        "complexity_weight": 0.18,
        "reason": "Preserve groove and phrase continuity; use the least intrusive program that solves bass/vocal overlap.",
    },
    "layer_handoff": {
        "techniques": (
            "stem_choreography",
            "layer_vocal_bridge",
            "layer_incoming_vocal_handoff",
            "layer_instrumental_handoff",
            "stem_vocal_overlay",
            "eq_bass_swap",
            "filter_build",
        ),
        "intents": ("vocal_handoff", "groove_handoff", "smooth_continuation", "energy_lift"),
        "durations": (16, 32, 8, 4),
        "minimum_route_fit": 0.60,
        "complexity_weight": 0.16,
        "reason": "Use isolation to remove a specific vocal/bass conflict rather than decorating an otherwise clean blend.",
    },
    "breakdown_drop": {
        "techniques": (
            "stem_choreography",
            "filter_build",
            "layer_vocal_bridge",
            "layer_incoming_vocal_handoff",
            "layer_instrumental_handoff",
            "double_drop",
            "echo_out_drop_in",
            "quick_cut",
        ),
        "intents": ("breakdown_to_drop", "energy_lift", "dramatic_switch", "vocal_handoff"),
        "durations": (16, 8, 4, 2),
        "minimum_route_fit": 0.60,
        "complexity_weight": 0.14,
        "reason": "Use structural tension/release; do not attempt a long full-spectrum blend across a meaningful BPM/style gap.",
    },
    "rhythmic_switch": {
        "techniques": (
            "stem_choreography",
            "echo_out_drop_in",
            "quick_cut",
            "layer_vocal_bridge",
            "layer_incoming_vocal_handoff",
            "layer_instrumental_handoff",
            "backspin",
        ),
        "intents": ("dramatic_switch", "vocal_handoff", "emergency_escape"),
        "durations": (4, 2, 1, 8),
        "minimum_route_fit": 0.64,
        "complexity_weight": 0.13,
        "reason": "Reinterpret or reset the pulse deliberately; avoid pretending a half/double-time relationship is a transparent long blend.",
    },
    "hard_reset": {
        "techniques": (
            "stem_choreography",
            "echo_out_drop_in",
            "quick_cut",
            "layer_vocal_bridge",
            "layer_incoming_vocal_handoff",
            "layer_instrumental_handoff",
            "backspin",
        ),
        "intents": ("dramatic_switch", "vocal_handoff", "emergency_escape"),
        "durations": (4, 2, 1, 8),
        "minimum_route_fit": 0.64,
        "complexity_weight": 0.13,
        "reason": "Remove rhythmic/harmonic commitment before establishing the new track; short intentional resets beat forced blends.",
    },
    "bridge_preferred": {
        "techniques": (
            "stem_choreography",
            "echo_out_drop_in",
            "quick_cut",
            "layer_vocal_bridge",
            "layer_incoming_vocal_handoff",
            "layer_instrumental_handoff",
        ),
        "intents": ("dramatic_switch", "vocal_handoff", "emergency_escape"),
        "durations": (2, 1, 4),
        "minimum_route_fit": 0.68,
        "complexity_weight": 0.12,
        "reason": "The direct edge is a last-choice reset.  Offline planning should prefer another track/bridge unless the rendered reset is exceptional.",
    },
}


def technique_policy(route: TransitionRoute) -> TechniquePolicy:
    payload = _FAMILY_POLICY.get(route.family, _FAMILY_POLICY["hard_reset"])
    return TechniquePolicy(
        family=route.family,
        allowed_techniques=tuple(payload["techniques"]),  # type: ignore[arg-type]
        preferred_intents=tuple(payload["intents"]),  # type: ignore[arg-type]
        duration_priority=tuple(payload["durations"]),  # type: ignore[arg-type]
        minimum_route_fit=float(payload["minimum_route_fit"]),
        complexity_weight=float(payload["complexity_weight"]),
        reasons=(str(payload["reason"]),),
    )


def intervention_cost(technique: str) -> float:
    return max(0.0, min(1.0, float(_INTERVENTION_COST.get(technique, 0.18))))


def _intent_fit(policy: TechniquePolicy, intent: str) -> float:
    if intent not in policy.preferred_intents:
        return 0.34
    index = policy.preferred_intents.index(intent)
    return max(0.62, 1.0 - 0.10 * index)


def apply_technique_policy(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    route: TransitionRoute,
    plans: Sequence[TransitionPlan],
) -> list[TransitionPlan]:
    """Filter/score generated programs after the musical route is known."""
    from ..naturalness import annotate_transition_naturalness

    policy = technique_policy(route)
    scored: list[TransitionPlan] = []
    for plan in plans:
        fit = route_fit(route, plan)
        allowed = plan.technique in policy.allowed_techniques
        # Keep an off-policy idea only when the route fit itself is very strong;
        # this preserves experimentation without a random effect lottery.
        if not allowed and fit < max(0.76, policy.minimum_route_fit + 0.12):
            continue
        intent_fit = _intent_fit(policy, plan.intent)
        cost = intervention_cost(plan.technique)
        utility = float(plan.metadata.get("selection_utility", plan.score))
        utility += 0.16 * (fit - 0.5)
        utility += 0.08 * (intent_fit - 0.5)
        utility -= policy.complexity_weight * cost
        if allowed:
            utility += 0.018
        naturalness = annotate_transition_naturalness(plan, track_a=track_a, track_b=track_b)
        # Naturalness is a bounded ranking signal, not a safety gate. It helps
        # phrase-aligned, appropriately short/simple programs reach rehearsal
        # before flashy or over-long alternatives with similar feasibility.
        utility += 0.18 * (naturalness.score - 0.5)
        plan.metadata = {
            **plan.metadata,
            "selection_utility": round(utility, 8),
            "pre_render_naturalness_score": naturalness.score,
            "technique_policy": {
                **policy.to_dict(),
                "allowed": allowed,
                "route_fit": round(fit, 4),
                "intent_fit": round(intent_fit, 4),
                "intervention_cost": round(cost, 4),
                "principle": "musical_intent_before_dsp",
            },
        }
        scored.append(plan)

    if not scored:
        # Do not turn a strict policy filter into a new technical failure.  The
        # rendered judge can still reject these; this branch merely keeps the
        # most route-compatible generated ideas available for rehearsal.
        fallback = sorted(
            plans,
            key=lambda plan: (route_fit(route, plan), plan.score, plan.execution_signature),
            reverse=True,
        )[: max(1, min(4, len(plans)))]
        for plan in fallback:
            plan.metadata = {
                **plan.metadata,
                "technique_policy": {
                    **policy.to_dict(),
                    "allowed": False,
                    "route_fit": round(route_fit(route, plan), 4),
                    "intent_fit": round(_intent_fit(policy, plan.intent), 4),
                    "intervention_cost": round(intervention_cost(plan.technique), 4),
                    "policy_relaxed": True,
                    "principle": "musical_intent_before_dsp",
                },
            }
        scored = fallback

    return sorted(
        scored,
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            -intervention_cost(plan.technique),
            plan.score,
            plan.execution_signature,
        ),
        reverse=True,
    )


def duration_candidates(
    route: TransitionRoute,
    evidence_matrix: PairEvidenceMatrix | None,
    *,
    maximum_candidates: int = 6,
) -> tuple[int, ...]:
    """Return standard plus musically-derived transition lengths.

    Non-standard lengths are only proposed when the cue pair itself exposes
    that many bars and both boundaries carry useful phrase evidence.  This is
    how an 18-bar candidate can exist without making 18 a random preset.
    """
    policy = technique_policy(route)
    values: list[int] = list(policy.duration_priority)
    if evidence_matrix is not None:
        phrase_lengths: list[tuple[float, int]] = []
        for evidence in evidence_matrix.pairs:
            available = min(
                int(evidence.exit_candidate.available_bars),
                int(evidence.entry_candidate.available_bars),
            )
            if not 4 <= available <= 32 or available in {4, 8, 16, 32}:
                continue
            phrase_strength = min(
                float(evidence.exit_candidate.phrase_strength),
                float(evidence.entry_candidate.phrase_strength),
            )
            boundary = min(
                float(evidence.exit_candidate.boundary_strength),
                float(evidence.entry_candidate.boundary_strength),
            )
            if phrase_strength < 0.52 or boundary < 0.34:
                continue
            score = 0.62 * phrase_strength + 0.38 * boundary
            phrase_lengths.append((score, available))
        for _, bars in sorted(phrase_lengths, reverse=True):
            if bars not in values:
                # Put structural durations after the first policy default but
                # before the shorter fallbacks.
                values.insert(min(1, len(values)), bars)

    unique: list[int] = []
    for value in values:
        if value > 0 and value not in unique:
            unique.append(value)
        if len(unique) >= max(1, maximum_candidates):
            break
    return tuple(unique)
