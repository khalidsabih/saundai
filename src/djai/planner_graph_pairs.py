from __future__ import annotations

"""Graph-pair planning mechanics for set planning.

This module owns worker-local graph state and pair-plan construction. Musical
objective scoring stays in set_planner and is injected as callbacks so this
extraction does not change planner policy.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .fallback import build_last_resort_plan
from .journey import JourneyMode
from .models import QualityClass, TrackAnalysis, TransitionPlan
from .parallel import worker_initializer
from .planner import (
    DifficultTransitionError,
    plan_transition_alternatives,
    plan_transition_styles,
)
from .planner_candidates import graph_scout_candidates
from .planner_graph_cache import GraphPairResult
from .planner_large_crate import graph_scout_cue_limit
from .planning_context import build_pair_evidence_matrix
from .set_models import TransitionEdge


@dataclass(frozen=True, slots=True)
class GraphPairTask:
    from_index: int
    to_index: int


SelectAlternative = Callable[
    [list[TransitionPlan], TrackAnalysis, TrackAnalysis],
    tuple[TransitionPlan, float],
]
ClassifyQuality = Callable[[TransitionPlan], QualityClass]
TransitionWarnings = Callable[[TransitionPlan], list[str]]


_GRAPH_WORKER_ANALYSES: tuple[TrackAnalysis, ...] | None = None
_GRAPH_WORKER_TEMPO_POLICY = "safe"
_GRAPH_WORKER_PREFERRED_STYLE = "auto"
_GRAPH_WORKER_JOURNEY: JourneyMode = "balanced"
_GRAPH_WORKER_STEM_MAPS: dict[int, dict[str, str]] = {}
_GRAPH_WORKER_SCORING_MODEL: dict[str, Any] | None = None
_GRAPH_WORKER_SELECT_ALTERNATIVE: SelectAlternative | None = None
_GRAPH_WORKER_CLASSIFY_QUALITY: ClassifyQuality | None = None
_GRAPH_WORKER_TRANSITION_WARNINGS: TransitionWarnings | None = None


def graph_worker_initializer(
    analyses: tuple[TrackAnalysis, ...],
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    stem_maps: dict[int, dict[str, str]],
    scoring_model: dict[str, Any] | None,
    select_alternative: SelectAlternative,
    classify_quality: ClassifyQuality,
    transition_warnings: TransitionWarnings,
) -> None:
    """Load immutable graph-planning inputs once per worker process."""
    worker_initializer()

    global _GRAPH_WORKER_ANALYSES
    global _GRAPH_WORKER_TEMPO_POLICY
    global _GRAPH_WORKER_PREFERRED_STYLE
    global _GRAPH_WORKER_JOURNEY
    global _GRAPH_WORKER_STEM_MAPS
    global _GRAPH_WORKER_SCORING_MODEL
    global _GRAPH_WORKER_SELECT_ALTERNATIVE
    global _GRAPH_WORKER_CLASSIFY_QUALITY
    global _GRAPH_WORKER_TRANSITION_WARNINGS

    _GRAPH_WORKER_ANALYSES = analyses
    _GRAPH_WORKER_TEMPO_POLICY = tempo_policy
    _GRAPH_WORKER_PREFERRED_STYLE = preferred_style
    _GRAPH_WORKER_JOURNEY = journey
    _GRAPH_WORKER_STEM_MAPS = stem_maps
    _GRAPH_WORKER_SCORING_MODEL = scoring_model
    _GRAPH_WORKER_SELECT_ALTERNATIVE = select_alternative
    _GRAPH_WORKER_CLASSIFY_QUALITY = classify_quality
    _GRAPH_WORKER_TRANSITION_WARNINGS = transition_warnings


def plan_graph_pair_core(
    task: GraphPairTask,
    *,
    analyses: tuple[TrackAnalysis, ...] | list[TrackAnalysis],
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    stem_maps: dict[int, dict[str, str]],
    scoring_model: dict[str, Any] | None,
    select_alternative: SelectAlternative,
    classify_quality: ClassifyQuality,
    transition_warnings: TransitionWarnings,
) -> GraphPairResult:
    track_a = analyses[task.from_index]
    track_b = analyses[task.to_index]
    stems_a = stem_maps[task.from_index]
    stems_b = stem_maps[task.to_index]

    scout_cues = graph_scout_cue_limit(len(analyses))
    scout_exits = (
        graph_scout_candidates(
            track_a.exit_candidates,
            kind="exit",
            limit=scout_cues,
        )
        if scout_cues
        else list(track_a.exit_candidates)
    )
    scout_entries = (
        graph_scout_candidates(
            track_b.entry_candidates,
            kind="entry",
            limit=scout_cues,
        )
        if scout_cues
        else list(track_b.entry_candidates)
    )

    matrix = build_pair_evidence_matrix(
        track_a,
        track_b,
        exit_candidates=scout_exits,
        entry_candidates=scout_entries,
        scoring_model=scoring_model,
    )

    try:
        if scout_cues:
            alternatives = plan_transition_styles(
                track_a,
                track_b,
                styles=("safe", "creative"),
                force=True,
                tempo_policy=tempo_policy,
                stems_available=bool(stems_a and stems_b),
                evidence_matrix=matrix,
                diversify_cues=True,
            )
        else:
            alternatives = plan_transition_alternatives(
                track_a,
                track_b,
                force=True,
                tempo_policy=tempo_policy,
                stems_available=bool(stems_a and stems_b),
                evidence_matrix=matrix,
            )
    except (DifficultTransitionError, RuntimeError, ValueError) as exc:
        fallback = build_last_resort_plan(
            track_a,
            track_b,
            preferred_exit_time=(
                track_a.exit_candidates[0].time_seconds
                if track_a.exit_candidates
                else track_a.duration_seconds * 0.65
            ),
            reason=f"Graph edge used an automatic safe fallback: {exc}",
        )
        alternatives = [fallback]

    for alternative in alternatives:
        alternative.stem_paths_a = dict(stems_a)
        alternative.stem_paths_b = dict(stems_b)

    selected, objective = select_alternative(
        alternatives,
        track_a,
        track_b,
    )
    edge = TransitionEdge(
        from_index=task.from_index,
        to_index=task.to_index,
        selected_style=selected.style,
        objective_score=round(objective, 6),
        quality=classify_quality(selected),
        plan=selected,
        alternatives=alternatives,
        warnings=transition_warnings(selected),
        reasons=list(selected.rationale),
    )
    return GraphPairResult(
        from_index=task.from_index,
        to_index=task.to_index,
        label=f"{track_a.title} -> {track_b.title}",
        edge=edge,
        evidence_matrix=matrix,
    )


def plan_graph_pair_worker(task: GraphPairTask) -> GraphPairResult:
    analyses = _GRAPH_WORKER_ANALYSES
    select_alternative = _GRAPH_WORKER_SELECT_ALTERNATIVE
    classify_quality = _GRAPH_WORKER_CLASSIFY_QUALITY
    transition_warnings = _GRAPH_WORKER_TRANSITION_WARNINGS
    if (
        analyses is None
        or select_alternative is None
        or classify_quality is None
        or transition_warnings is None
    ):
        raise RuntimeError("Graph worker was not initialized.")

    return plan_graph_pair_core(
        task,
        analyses=analyses,
        tempo_policy=_GRAPH_WORKER_TEMPO_POLICY,
        preferred_style=_GRAPH_WORKER_PREFERRED_STYLE,
        journey=_GRAPH_WORKER_JOURNEY,
        stem_maps=_GRAPH_WORKER_STEM_MAPS,
        scoring_model=_GRAPH_WORKER_SCORING_MODEL,
        select_alternative=select_alternative,
        classify_quality=classify_quality,
        transition_warnings=transition_warnings,
    )
