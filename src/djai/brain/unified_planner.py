from __future__ import annotations

"""SAUND.AI 1.0 unified set-order planner.

This planner is deliberately a *candidate producer*.  The canonical set
objective in ``set_planner`` still decides which complete order wins and the
render/rehearsal loop still owns technical feasibility.  That separation lets
learned taste improve search without becoming an audio-safety authority.
"""

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Mapping, Sequence

import numpy as np

from ..intelligence.hybrid_next_track import HybridNextTrackPrediction, score_hybrid_next_track
from ..journey import curve_fit_score
from ..models import TrackAnalysis, TransitionPlan
from .set_state import build_set_state

UNIFIED_PLANNER_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class UnifiedPlannerConfig:
    beam_width: int = 8
    candidate_width: int = 8
    confidence_bonus: float = 0.035
    story_weight: float = 0.28
    diversity_weight: float = 0.06

    @classmethod
    def from_environment(cls) -> "UnifiedPlannerConfig":
        def integer(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default

        def number(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return default

        return cls(
            beam_width=max(2, min(24, integer("SAUND_UNIFIED_PLANNER_BEAM", 8))),
            candidate_width=max(2, min(24, integer("SAUND_UNIFIED_PLANNER_CANDIDATES", 8))),
            confidence_bonus=max(0.0, min(0.12, number("SAUND_UNIFIED_CONFIDENCE_BONUS", 0.035))),
            story_weight=max(0.0, min(0.60, number("SAUND_UNIFIED_STORY_WEIGHT", 0.28))),
            diversity_weight=max(0.0, min(0.20, number("SAUND_UNIFIED_DIVERSITY_WEIGHT", 0.06))),
        )


@dataclass(frozen=True, slots=True)
class PlannerStep:
    position: int
    from_index: int
    to_index: int
    score: float
    confidence: float
    deterministic_score: float
    learned_authority: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: round(value, 6) if isinstance(value, float) else value for key, value in payload.items()}


@dataclass(frozen=True, slots=True)
class UnifiedPlannerResult:
    order: tuple[int, ...]
    search_score: float
    story_score: float
    diversity_score: float
    trace: tuple[PlannerStep, ...] = field(default_factory=tuple)
    version: str = UNIFIED_PLANNER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "order": list(self.order),
            "search_score": round(self.search_score, 6),
            "story_score": round(self.story_score, 6),
            "diversity_score": round(self.diversity_score, 6),
            "trace": [item.to_dict() for item in self.trace],
        }


@dataclass(frozen=True, slots=True)
class _BeamState:
    order: tuple[int, ...]
    score: float
    trace: tuple[PlannerStep, ...]


def _edge_value(edge: object) -> float:
    try:
        return float(getattr(edge, "objective_score"))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _edge_plan(edge: object) -> TransitionPlan | None:
    plan = getattr(edge, "plan", None)
    return plan if isinstance(plan, TransitionPlan) else None


def _prefix_state(order: Sequence[int], analyses: Sequence[TrackAnalysis], graph: Mapping[tuple[int, int], object]):
    tracks = [analyses[index] for index in order]
    transitions: list[TransitionPlan] = []
    for left, right in zip(order, order[1:]):
        edge = graph.get((left, right))
        if edge is not None:
            plan = _edge_plan(edge)
            if plan is not None:
                transitions.append(plan)
    return build_set_state(tracks, transitions)


def _transition_diversity(order: Sequence[int], graph: Mapping[tuple[int, int], object]) -> float:
    techniques: list[str] = []
    intents: list[str] = []
    for left, right in zip(order, order[1:]):
        edge = graph.get((left, right))
        plan = _edge_plan(edge) if edge is not None else None
        if plan is None:
            continue
        techniques.append(str(plan.technique))
        intents.append(str(plan.intent))
    if not techniques:
        return 0.5
    technique_ratio = len(set(techniques)) / len(techniques)
    intent_ratio = len(set(intents)) / max(1, len(intents))
    return float(np.clip(0.65 * technique_ratio + 0.35 * intent_ratio, 0.0, 1.0))


def _final_value(
    state: _BeamState,
    analyses: Sequence[TrackAnalysis],
    graph: Mapping[tuple[int, int], object],
    *,
    journey: str,
    config: UnifiedPlannerConfig,
) -> tuple[float, float, float]:
    story = float(curve_fit_score(list(state.order), list(analyses), journey=journey))
    diversity = _transition_diversity(state.order, graph)
    edge_count = max(1, len(state.order) - 1)
    normalized_search = state.score / edge_count
    total = (
        normalized_search
        + config.story_weight * (story - 0.5)
        + config.diversity_weight * (diversity - 0.5)
    )
    return float(total), story, diversity


def build_unified_ai_order(
    graph: Mapping[tuple[int, int], object],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str = "balanced",
    start: int | None = None,
    end: int | None = None,
    blocked_pairs: set[tuple[int, int]] | None = None,
    config: UnifiedPlannerConfig | None = None,
) -> UnifiedPlannerResult:
    """Beam-search complete set orders using the hybrid next-track brain.

    The function never invents a missing edge and never crosses a blocked edge.
    When the search cannot form a complete Hamiltonian route it raises instead
    of silently weakening safety constraints.
    """
    cfg = config or UnifiedPlannerConfig.from_environment()
    count = len(analyses)
    if count == 0:
        raise ValueError("Unified planner requires at least one track.")
    if count == 1:
        only = 0 if start is None else start
        return UnifiedPlannerResult(order=(only,), search_score=0.0, story_score=1.0, diversity_score=1.0)
    if start is not None and not 0 <= start < count:
        raise ValueError("start is outside the track list")
    if end is not None and not 0 <= end < count:
        raise ValueError("end is outside the track list")
    if start is not None and end is not None and start == end:
        raise ValueError("The opener and closer must be different tracks.")

    blocked = blocked_pairs or set()
    starts = [start] if start is not None else [index for index in range(count) if index != end]
    beams = [_BeamState(order=(index,), score=0.0, trace=()) for index in starts if index is not None]
    target_without_end = count - (1 if end is not None else 0)

    while beams and len(beams[0].order) < target_without_end:
        expanded: list[_BeamState] = []
        for beam in beams:
            current = beam.order[-1]
            used = set(beam.order)
            remaining = [index for index in range(count) if index not in used and index != end]
            state = _prefix_state(beam.order, analyses, graph)
            progress = min(1.0, len(beam.order) / max(1, count - 1))
            candidates: list[tuple[float, int, HybridNextTrackPrediction]] = []
            for candidate in remaining:
                if (current, candidate) in blocked:
                    continue
                edge = graph.get((current, candidate))
                if edge is None:
                    continue
                prediction = score_hybrid_next_track(
                    analyses[current],
                    analyses[candidate],
                    edge_score=_edge_value(edge),
                    set_state=state,
                    journey=journey,
                    progress=progress,
                    plan=_edge_plan(edge),
                )
                if prediction.safety.get("hard_failures"):
                    continue
                utility = prediction.score + cfg.confidence_bonus * (prediction.confidence - 0.5)
                candidates.append((utility, candidate, prediction))
            candidates.sort(key=lambda item: (item[0], item[2].confidence, -item[1]), reverse=True)
            for utility, candidate, prediction in candidates[: cfg.candidate_width]:
                step = PlannerStep(
                    position=len(beam.order),
                    from_index=current,
                    to_index=candidate,
                    score=prediction.score,
                    confidence=prediction.confidence,
                    deterministic_score=prediction.deterministic_score,
                    learned_authority=prediction.learned_authority,
                )
                expanded.append(
                    _BeamState(
                        order=(*beam.order, candidate),
                        score=beam.score + utility,
                        trace=(*beam.trace, step),
                    )
                )
        expanded.sort(key=lambda item: (item.score, item.order), reverse=True)
        beams = expanded[: cfg.beam_width]

    completed: list[_BeamState] = []
    for beam in beams:
        if len(beam.order) != target_without_end:
            continue
        if end is None:
            completed.append(beam)
            continue
        current = beam.order[-1]
        if (current, end) in blocked:
            continue
        edge = graph.get((current, end))
        if edge is None:
            continue
        state = _prefix_state(beam.order, analyses, graph)
        prediction = score_hybrid_next_track(
            analyses[current],
            analyses[end],
            edge_score=_edge_value(edge),
            set_state=state,
            journey=journey,
            progress=1.0,
            plan=_edge_plan(edge),
        )
        if prediction.safety.get("hard_failures"):
            continue
        step = PlannerStep(
            position=len(beam.order),
            from_index=current,
            to_index=end,
            score=prediction.score,
            confidence=prediction.confidence,
            deterministic_score=prediction.deterministic_score,
            learned_authority=prediction.learned_authority,
        )
        completed.append(
            _BeamState(
                order=(*beam.order, end),
                score=beam.score + prediction.score + cfg.confidence_bonus * (prediction.confidence - 0.5),
                trace=(*beam.trace, step),
            )
        )

    if not completed:
        raise RuntimeError("Unified planner could not construct a complete safe order.")

    ranked: list[tuple[float, float, float, _BeamState]] = []
    for beam in completed:
        value, story, diversity = _final_value(beam, analyses, graph, journey=journey, config=cfg)
        ranked.append((value, story, diversity, beam))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3].order), reverse=True)
    total, story, diversity, winner = ranked[0]
    return UnifiedPlannerResult(
        order=winner.order,
        search_score=total,
        story_score=story,
        diversity_score=diversity,
        trace=winner.trace,
    )
