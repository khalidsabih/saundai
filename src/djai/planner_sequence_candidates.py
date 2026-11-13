from __future__ import annotations

"""Sequence evidence preparation and candidate-pool assembly.

This module reuses/builds pair evidence, generates transition candidates, and
attaches execution-program metadata. It does not rehearse, select, fallback,
or advance sequence state.
"""

from dataclasses import dataclass
from typing import Sequence

from .candidate_renderer import QualityLoopConfig
from .decision import seal_execution_plan
from .models import TrackAnalysis, TransitionCandidate, TransitionPlan
from .performance.program import attach_performance_program
from .planner import DifficultTransitionError, plan_transition_candidate_pool
from .planning_context import PairEvidenceMatrix, PlanningSession


@dataclass(frozen=True, slots=True)
class SequenceCandidatePool:
    plans: tuple[TransitionPlan, ...]
    matrix: PairEvidenceMatrix | None
    sequence_evidence_reused: bool
    stems_available: bool


def build_sequence_candidate_pool(
    *,
    from_index: int,
    to_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    selected_exits: Sequence[TransitionCandidate],
    selected_entries: Sequence[TransitionCandidate],
    session: PlanningSession,
    stem_paths_a: dict[str, str],
    stem_paths_b: dict[str, str],
    preferred_style: str,
    quality_loop: QualityLoopConfig | None,
    tempo_policy: str,
    current_effective_bpm: float,
    current_rate: float,
    global_position: int,
    source_cursor: float,
    required_source: float,
    target_exit: float,
    soft_latest_exit: float,
) -> SequenceCandidatePool:
    styles = (
        [preferred_style]
        if preferred_style in {"safe", "creative", "emergency"}
        else ["safe", "creative", "emergency"]
    )
    matrix: PairEvidenceMatrix | None = None
    sequence_evidence_reused = False
    stems_available = bool(stem_paths_a and stem_paths_b)
    plans: list[TransitionPlan] = []

    if selected_exits:
        matrix = session.get((from_index, to_index))
        if matrix is not None:
            sequence_evidence_reused = True
            matrix = matrix.subset(
                exit_candidates=list(selected_exits),
                entry_candidates=list(selected_entries) or None,
            )
            if (
                len(matrix.exit_candidates) < len(selected_exits)
                or (
                    selected_entries
                    and len(matrix.entry_candidates) < len(selected_entries)
                )
            ):
                matrix = None
                sequence_evidence_reused = False

        if matrix is None or not matrix.pairs:
            sequence_evidence_reused = False
            from .planning_context import build_pair_evidence_matrix

            matrix = build_pair_evidence_matrix(
                track_a,
                track_b,
                exit_candidates=list(selected_exits),
                entry_candidates=list(selected_entries) or None,
                scoring_model=session.scoring_model,
            )
            session.store((from_index, to_index), matrix)

        try:
            candidate_plans = plan_transition_candidate_pool(
                track_a,
                track_b,
                styles=styles,
                per_style=(
                    max(2, quality_loop.top_k)
                    if quality_loop is not None and quality_loop.enabled
                    else 1
                ),
                force=True,
                tempo_policy=tempo_policy,
                effective_bpm_a=current_effective_bpm,
                render_rate_a=current_rate,
                stems_available=stems_available,
                evidence_matrix=matrix,
            )
        except (DifficultTransitionError, RuntimeError, ValueError):
            candidate_plans = []

        for plan in candidate_plans:
            plan.stem_paths_a = dict(stem_paths_a)
            plan.stem_paths_b = dict(stem_paths_b)
            plan.metadata = {
                **plan.metadata,
                "sequence_position": global_position,
                "source_cursor_before_body": round(source_cursor, 6),
                "minimum_source_body": round(required_source, 6),
                "preferred_exit_time": round(target_exit, 6),
                "soft_latest_exit_time": round(soft_latest_exit, 6),
                "selected_entry_pool": [
                    round(item.time_seconds, 6)
                    for item in selected_entries
                ],
                "sequence_evidence_reused": sequence_evidence_reused,
                "sequence_matrix_size": matrix.compatibility_evaluations,
                "stem_rendering": bool(
                    stems_available
                    and plan.technique
                    in {
                        "eq_bass_swap",
                        "filter_build",
                        "double_drop",
                        "stem_vocal_overlay",
                        "stem_choreography",
                    }
                ),
                "performance_layer_rendering": plan.technique
                in {
                    "layer_vocal_bridge",
                    "layer_incoming_vocal_handoff",
                    "layer_instrumental_handoff",
                },
            }
            attach_performance_program(plan)
            seal_execution_plan(plan)
            plans.append(plan)

    return SequenceCandidatePool(
        plans=tuple(plans),
        matrix=matrix,
        sequence_evidence_reused=sequence_evidence_reused,
        stems_available=stems_available,
    )
