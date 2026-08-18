from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from ..models import TrackAnalysis, TransitionPlan
from .candidate_generator import CandidateGenerationConfig, generate_transition_ideas
from .crowd_state import CrowdDirective, crowd_directive_from_mapping

TRANSITION_POLICY_VERSION = "2.1.0"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class TransitionPolicyContext:
    journey: str = "balanced"
    journey_progress: float = 0.5
    tempo_policy: str = "safe"
    effective_bpm_a: float | None = None
    render_rate_a: float = 1.0
    stems_available: bool = False


@dataclass(frozen=True, slots=True)
class TransitionPolicyConfig:
    enabled: bool = True

    @classmethod
    def from_environment(cls) -> "TransitionPolicyConfig":
        return cls(enabled=_env_bool("DJAI_LIVE_TRANSITION_POLICY", True))


def _base_policy_payload(
    *,
    context: TransitionPolicyContext,
    authority: str,
) -> dict[str, Any]:
    return {
        "version": TRANSITION_POLICY_VERSION,
        "journey": context.journey,
        "journey_progress": round(context.journey_progress, 6),
        "tempo_policy": context.tempo_policy,
        "effective_bpm_a": (
            round(float(context.effective_bpm_a), 4)
            if context.effective_bpm_a is not None
            else None
        ),
        "render_rate_a": round(float(context.render_rate_a), 6),
        "stems_available": bool(context.stems_available),
        "policy": "executive_transition_policy_boundary_v2_1",
        "authority": authority,
    }


def annotate_transition_policy(
    plans: list[TransitionPlan],
    *,
    context: TransitionPolicyContext | None = None,
    authority: str = "observe",
) -> list[TransitionPlan]:
    """Attach/refresh the formal transition-policy contract.

    Existing bounded-policy details are preserved so the final selected plan
    keeps the exact adjustment/Executive directive that affected ranking.
    """
    ctx = context or TransitionPolicyContext()
    for plan in plans:
        previous = dict(plan.metadata.get("transition_policy", {}) or {})
        plan.metadata = {
            **plan.metadata,
            "transition_policy": {
                **previous,
                **_base_policy_payload(context=ctx, authority=authority),
                "ranking_changed": bool(previous.get("ranking_changed", False)),
            },
        }
    return plans


def _policy_raw_score(plan: TransitionPlan, directive: CrowdDirective) -> float:
    """Return -1..1 preference without bypassing feasibility/safety."""
    bars = max(1, int(plan.transition_bars))
    duration_position = _clamp((bars - 4.0) / 28.0)
    short_fit = (1.0 - 2.0 * duration_position) * directive.prefer_short_transition

    style = str(plan.style or "").strip().lower()
    if style == "creative":
        style_axis = 1.0
    elif style == "safe":
        style_axis = -0.20
    else:
        style_axis = -0.65
    creative_fit = style_axis * directive.prefer_creative_transition

    technique = str(plan.technique or "").strip().lower()
    vocal_heavy = any(token in technique for token in ("vocal", "overlay", "double_drop"))
    vocal_fit = -directive.vocal_simplicity if vocal_heavy else 0.25 * directive.vocal_simplicity

    raw = 0.45 * short_fit + 0.40 * creative_fit + 0.15 * vocal_fit

    # Crowd preference must never promote an emergency/cut or a plan that still
    # requires human confirmation. Hard execution/safety semantics dominate.
    if str(plan.transition_mode or "") == "emergency_cut" or plan.requires_confirmation:
        raw = min(0.0, raw)
    return max(-1.0, min(1.0, raw))


def apply_bounded_transition_policy(
    plans: list[TransitionPlan],
    *,
    context: TransitionPolicyContext | None = None,
    crowd_directive: CrowdDirective | None = None,
    crowd_state: Mapping[str, Any] | None = None,
) -> list[TransitionPlan]:
    """Boundedly rerank already-feasible transition programs for live FUTURE.

    This policy never creates a new technique, bypasses Professional Brain, or
    overrides hard safety. It only nudges ``selection_utility`` within the
    Executive-provided maximum and records the exact adjustment on each plan.
    """
    ctx = context or TransitionPolicyContext()
    cfg = TransitionPolicyConfig.from_environment()
    directive = crowd_directive or crowd_directive_from_mapping(crowd_state)
    if not cfg.enabled or not directive.active:
        return annotate_transition_policy(plans, context=ctx, authority="observe")

    for plan in plans:
        before = _clamp(float(plan.metadata.get("selection_utility", plan.score)))
        raw = _policy_raw_score(plan, directive)
        adjustment = max(
            -directive.max_transition_adjustment,
            min(
                directive.max_transition_adjustment,
                directive.max_transition_adjustment * directive.strength * raw,
            ),
        )
        after = _clamp(before + adjustment)
        plan.metadata = {
            **plan.metadata,
            "selection_utility": round(after, 8),
            "transition_policy": {
                **_base_policy_payload(context=ctx, authority="bounded_advisory"),
                "ranking_changed": abs(adjustment) > 1e-12,
                "utility_before": round(before, 8),
                "utility_after": round(after, 8),
                "adjustment": round(adjustment, 8),
                "raw_preference": round(raw, 6),
                "crowd_directive": directive.to_dict(),
                "hard_safety_may_override": True,
                "professional_brain_may_reject": True,
            },
        }
    plans.sort(
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            float(plan.score),
            str(plan.execution_signature or ""),
        ),
        reverse=True,
    )
    return plans


def propose_transition_programs(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    context: TransitionPolicyContext | None = None,
    generation: CandidateGenerationConfig | None = None,
    evidence_matrix=None,
) -> list[TransitionPlan]:
    """Policy boundary: propose DJ moves without deciding which one wins."""
    ctx = context or TransitionPolicyContext()
    plans = generate_transition_ideas(
        track_a,
        track_b,
        config=generation,
        tempo_policy=ctx.tempo_policy,
        effective_bpm_a=ctx.effective_bpm_a,
        render_rate_a=ctx.render_rate_a,
        stems_available=ctx.stems_available,
        evidence_matrix=evidence_matrix,
    )
    return annotate_transition_policy(plans, context=ctx, authority="proposal")
