from __future__ import annotations

"""Professional Brain sequence-resolution helper.

This module invokes the existing Professional Brain resolver and interprets the
accepted/replan outcome. Emergency escape and deterministic fallback remain
outside this module.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .candidate_renderer import QualityLoopConfig
from .journey import JourneyMode
from .models import TrackAnalysis, TransitionPlan
from .planning_context import PairEvidenceMatrix
from .execution import TransitionRenderContext


@dataclass(frozen=True, slots=True)
class ProfessionalResolutionOutcome:
    selected: TransitionPlan | None
    measurement_summary: dict[str, object] | None
    replan_action: str | None = None
    replan_reason: str | None = None
    replan_resolution: dict[str, object] | None = None
    needs_emergency_escape: bool = False


def resolve_professional_sequence_transition(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    matrix: PairEvidenceMatrix,
    professional_config: Any,
    tempo_policy: str,
    preferred_style: str,
    current_effective_bpm: float,
    current_rate: float,
    stems_available: bool,
    set_state: Any,
    dj_state: Any,
    journey: JourneyMode,
    journey_progress: float,
    listener: str | None,
    quality_loop: QualityLoopConfig,
    transition_policy_context: Any,
    crowd_directive: Any,
    thinking_budget: Any,
    global_position: int,
    from_index: int,
    to_index: int,
    render_context: TransitionRenderContext | None,
    replan_on_review: bool,
    allow_emergency_escape: bool,
    resolve_transition: Callable[..., Any],
) -> ProfessionalResolutionOutcome:
    resolution = resolve_transition(
        track_a,
        track_b,
        evidence_matrix=matrix,
        professional_config=professional_config,
        tempo_policy=tempo_policy,
        preferred_style=preferred_style,
        effective_bpm_a=current_effective_bpm,
        render_rate_a=current_rate,
        stems_available=stems_available,
        set_state=set_state,
        dj_state=dj_state,
        journey=journey,
        journey_progress=journey_progress,
        listener=listener,
        measurement_log_path=quality_loop.log_path,
        transition_policy_context=transition_policy_context,
        crowd_directive=crowd_directive,
        context={
            "mode": "set_sequence",
            "sequence_position": global_position,
            "from_index": from_index,
            "to_index": to_index,
            "journey": journey,
            "preferred_style": preferred_style,
            "adaptive_thinking": (
                thinking_budget.to_dict()
                if thinking_budget is not None
                else None
            ),
        },
        render_context=render_context,
    )

    if resolution.accepted:
        assert resolution.selected is not None
        if (
            replan_on_review
            and not allow_emergency_escape
            and resolution.selected.transition_mode == "emergency_cut"
        ):
            payload = {
                **resolution.to_dict(),
                "reason": "emergency_cut_not_production_route",
            }
            return ProfessionalResolutionOutcome(
                selected=None,
                measurement_summary=None,
                replan_action="replan_track",
                replan_reason=(
                    "Professional rehearsal found only a structural reset for this pair; "
                    "offline automatic delivery must route through another track before using emergency escape."
                ),
                replan_resolution=payload,
            )
        return ProfessionalResolutionOutcome(
            selected=resolution.selected,
            measurement_summary=resolution.to_dict(),
        )

    if replan_on_review and not allow_emergency_escape:
        return ProfessionalResolutionOutcome(
            selected=None,
            measurement_summary=None,
            replan_action=resolution.action,
            replan_reason=(
                resolution.reasons[-1]
                if resolution.reasons
                else "Rendered transition did not clear the professional acceptance gate."
            ),
            replan_resolution=resolution.to_dict(),
        )

    return ProfessionalResolutionOutcome(
        selected=None,
        measurement_summary=None,
        needs_emergency_escape=True,
    )


@dataclass(frozen=True, slots=True)
class EmergencyResolutionOutcome:
    selected: TransitionPlan | None
    measurement_summary: dict[str, object] | None
    emergency_resolution: dict[str, object]


def resolve_professional_emergency_escape(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    matrix: PairEvidenceMatrix,
    professional_config: Any,
    current_effective_bpm: float,
    current_rate: float,
    stems_available: bool,
    set_state: Any,
    journey: JourneyMode,
    journey_progress: float,
    listener: str | None,
    quality_loop: QualityLoopConfig,
    global_position: int,
    from_index: int,
    to_index: int,
    render_context: TransitionRenderContext | None,
    resolve_emergency_escape: Callable[..., Any],
) -> EmergencyResolutionOutcome:
    emergency = resolve_emergency_escape(
        track_a,
        track_b,
        evidence_matrix=matrix,
        professional_config=professional_config,
        effective_bpm_a=current_effective_bpm,
        render_rate_a=current_rate,
        stems_available=stems_available,
        set_state=set_state,
        journey=journey,
        journey_progress=journey_progress,
        listener=listener,
        measurement_log_path=quality_loop.log_path,
        context={
            "mode": "set_sequence_final_escape",
            "sequence_position": global_position,
            "from_index": from_index,
            "to_index": to_index,
        },
        render_context=render_context,
    )
    payload = emergency.to_dict()
    return EmergencyResolutionOutcome(
        selected=(
            emergency.selected
            if emergency.accepted
            else None
        ),
        measurement_summary=(
            payload
            if emergency.accepted
            else None
        ),
        emergency_resolution=payload,
    )
