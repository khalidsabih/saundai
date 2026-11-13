from __future__ import annotations

"""Non-automatic sequence quality selection.

This module owns the manual/non-automatic measured-selection wrapper and
quality-loop metadata packaging. Automatic delivery, Professional Brain
resolution, emergency fallback, and safety replacement remain outside.
"""

from typing import Any, Callable, Sequence

from .candidate_renderer import QualityLoopConfig
from .journey import JourneyMode
from .models import TrackAnalysis, TransitionCandidate, TransitionPlan
from .planning_context import PairEvidenceMatrix
from .execution import TransitionRenderContext


def select_nonautomatic_sequence_transition(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: list[TransitionPlan],
    quality_loop: QualityLoopConfig | None,
    preferred_complete_decision: Any,
    preferred_style: str,
    journey: JourneyMode,
    listener_profile: Any,
    global_position: int,
    global_track_count: int,
    source_cursor: float,
    required_source: float,
    target_exit: float,
    soft_latest_exit: float,
    selected_entries: Sequence[TransitionCandidate],
    sequence_evidence_reused: bool,
    matrix: PairEvidenceMatrix | None,
    render_context: TransitionRenderContext | None,
    progress: Callable[[str, int, int, str], None] | None,
    measure_and_select: Callable[..., Any],
    select_alternative: Callable[..., tuple[TransitionPlan, float]],
) -> tuple[TransitionPlan, dict[str, object] | None]:
    measurement_summary: dict[str, object] | None = None

    if quality_loop is not None and quality_loop.enabled:
        if progress:
            progress(
                "measure",
                global_position,
                max(1, global_track_count - 1),
                f"{track_a.title} -> {track_b.title}",
            )

        measured = measure_and_select(
            track_a,
            track_b,
            plans,
            config=quality_loop,
            context={
                "sequence_position": global_position,
                "from_index": None,
                "to_index": None,
                "journey": journey,
                "preferred_style": preferred_style,
            },
            render_context=render_context,
        )
        chosen = measured.selected
        measurement_summary = measured.to_dict()
        chosen.metadata = {
            **chosen.metadata,
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
            "sequence_matrix_size": (
                matrix.compatibility_evaluations
                if matrix is not None
                else 0
            ),
            "quality_loop_summary": measurement_summary,
        }
        if progress:
            progress(
                "measure",
                global_position + 1,
                max(1, global_track_count - 1),
                (
                    f"{track_a.title} -> {track_b.title}: "
                    f"{chosen.measured_score or 0.0:.0%}"
                ),
            )
        return chosen, measurement_summary

    if preferred_complete_decision is not None:
        chosen = max(
            plans,
            key=lambda candidate: (
                float(
                    candidate.metadata.get(
                        "selection_utility",
                        candidate.score,
                    )
                ),
                float(candidate.score),
                candidate.execution_signature,
            ),
        )
        return chosen, None

    chosen, _ = select_alternative(
        plans,
        track_a,
        track_b,
        preferred_style=preferred_style,
        journey=journey,
        listener_profile=listener_profile,
    )
    return chosen, None
