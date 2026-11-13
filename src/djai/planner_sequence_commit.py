from __future__ import annotations

"""Final sequence edge safety and deterministic fallback plumbing.

The caller retains Executive replan decisions and chooses whether fallback is
permitted. These helpers preserve deterministic rescue, program validation,
edge assembly, and post-transition deck state.
"""

import os
from typing import Any, Callable

from .models import TrackAnalysis, TransitionPlan
from .decision_trace import rejected_phase_trace
from .planning_context import PairEvidenceMatrix


def deterministic_final_escape(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    current_effective_bpm: float,
    current_rate: float,
    source_cursor: float,
    required_exit: float,
    render_context: Any,
    emergency_resolution: dict[str, object],
    build_last_resort: Callable[..., TransitionPlan],
) -> tuple[TransitionPlan, dict[str, object]]:
    chosen = build_last_resort(
        track_a,
        track_b,
        effective_bpm_a=current_effective_bpm,
        render_rate_a=current_rate,
        source_cursor=source_cursor,
        preferred_exit_time=required_exit,
        reason=(
            "All alternate track routes and separately rehearsed emergency escapes were exhausted; "
            "deterministic phrase-aligned final escape used to preserve delivery."
        ),
        render_context=render_context,
        verify_audio=True,
    )
    chosen.metadata = {
        **chosen.metadata,
        "unified_transition_final_escape": True,
        "emergency_resolution": emergency_resolution,
    }
    return chosen, {
        **emergency_resolution,
        "deterministic_final_escape": True,
    }


def finalize_sequence_edge(
    *,
    chosen: TransitionPlan,
    plans: list[TransitionPlan],
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    analyses: list[TrackAnalysis],
    from_index: int,
    to_index: int,
    dj_state: Any,
    automatic_delivery: bool,
    quality_loop_enabled: bool,
    current_effective_bpm: float,
    current_rate: float,
    source_cursor: float,
    required_exit: float,
    render_context: Any,
    sequence_evidence_reused: bool,
    matrix: PairEvidenceMatrix | None,
    policy_context: Any,
    transition_policy_authority: str,
    journey: Any,
    listener_profile: Any,
    rank_as_moves: Callable[..., list[TransitionPlan]],
    build_last_resort: Callable[..., TransitionPlan],
    annotate_policy: Callable[..., Any],
    edge_objective: Callable[..., float],
    edge_from_plan: Callable[..., Any],
    validate_plan: Callable[[TransitionPlan], Any],
) -> tuple[Any, float, float, float]:
    if "dj_move" not in chosen.metadata:
        chosen = rank_as_moves(
            track_a, track_b, [chosen], state=dj_state
        )[0]
    chosen.metadata = {
        **chosen.metadata,
        "dj_state_before_move": dj_state.to_dict(),
    }

    if automatic_delivery:
        allow_hard_cuts = os.environ.get(
            "DJAI_ALLOW_HARD_CUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        deck_performance = dict(
            chosen.metadata.get("deck_performance", {}) or {}
        )
        phase_integrity = dict(
            deck_performance.get("phase_integrity", {}) or {}
        )
        if (
            chosen.beatmatched
            and phase_integrity
            and not bool(phase_integrity.get("safe", True))
        ):
            rejected_signature = chosen.execution_signature
            rejected_detail = rejected_phase_trace(chosen, phase_integrity) if os.environ.get("DJAI_DECISION_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"} else None
            previous_move = chosen.metadata.get("dj_move")
            previous_state = chosen.metadata.get(
                "dj_state_before_move"
            )
            chosen = build_last_resort(
                track_a,
                track_b,
                effective_bpm_a=current_effective_bpm,
                render_rate_a=current_rate,
                source_cursor=source_cursor,
                preferred_exit_time=required_exit,
                reason=(
                    "Beat Integrity rejected the selected rhythmic overlap because "
                    "bounded continuous phase correction could not hold lock."
                ),
                render_context=render_context,
                verify_audio=quality_loop_enabled,
            )
            chosen.metadata = {
                **chosen.metadata,
                "beat_integrity_replanned": True,
                "rejected_phase_integrity_signature": rejected_signature,
                **({"decision_trace_rejected_phase": rejected_detail} if rejected_detail is not None else {}),
                **(
                    {"dj_move": previous_move}
                    if previous_move is not None
                    else {}
                ),
                **(
                    {"dj_state_before_move": previous_state}
                    if previous_state is not None
                    else {}
                ),
            }
        if not allow_hard_cuts and chosen.technique in {
            "quick_cut", "backspin"
        }:
            rejected_technique = chosen.technique
            chosen = build_last_resort(
                track_a,
                track_b,
                effective_bpm_a=current_effective_bpm,
                render_rate_a=current_rate,
                source_cursor=source_cursor,
                preferred_exit_time=required_exit,
                reason=(
                    f"Automatic policy re-evaluated {rejected_technique} through the compatibility-aware rescue hierarchy."
                ),
                render_context=render_context,
                verify_audio=quality_loop_enabled,
            )
            rescue_kind = str(
                chosen.metadata.get(
                    "rescue_transition_kind", "unknown"
                )
            )
            chosen.metadata = {
                **chosen.metadata,
                "replaced_automatic_cut": rejected_technique,
                "continuous_mix_policy": (
                    rescue_kind == "filtered_overlap"
                ),
                "automatic_rescue_policy": "compatibility_aware",
            }
        chosen.requires_confirmation = False
        chosen.auto_approved = True
        chosen.metadata = {
            **chosen.metadata, "automatic_delivery": True
        }

    chosen.metadata = {
        **chosen.metadata,
        "sequence_evidence_reused": sequence_evidence_reused,
        "sequence_matrix_size": (
            matrix.compatibility_evaluations
            if matrix is not None
            else int(
                chosen.metadata.get(
                    "sequence_matrix_size", 0
                ) or 0
            )
        ),
    }
    annotate_policy(
        [chosen],
        context=policy_context,
        authority=transition_policy_authority,
    )

    alternatives: list[TransitionPlan] = []
    for style in ("safe", "creative", "emergency"):
        candidates = [
            plan for plan in plans if plan.style == style
        ]
        if candidates:
            alternatives.append(
                max(
                    candidates,
                    key=lambda value: edge_objective(
                        value,
                        track_a,
                        track_b,
                        journey=journey,
                        listener_profile=listener_profile,
                    ),
                )
            )
    edge = edge_from_plan(
        from_index,
        to_index,
        chosen,
        alternatives,
        analyses,
        journey=journey,
        listener_profile=listener_profile,
        extra_reason=(
            "The transition was compiled from the track's actual rendered tempo and consumed source position."
        ),
    )
    validate_plan(edge.plan)
    next_source_cursor = (
        edge.plan.post_transition_source_end_b
    )
    next_rate = edge.plan.post_transition_rate_b
    next_bpm = (
        edge.plan.post_transition_bpm_b
        or (track_b.bpm * next_rate)
    )
    return edge, next_source_cursor, next_rate, next_bpm
