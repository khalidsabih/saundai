from __future__ import annotations

"""Recursive complete-move lookahead selection.

This module owns only route lookahead mechanics. Candidate retrieval, complete
move generation, timing adjustment, and DJ-state construction remain the same
authoritative implementations used elsewhere.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .brain.candidate_retrieval import CandidateRetrievalResult, retrieve_candidate_tracks
from .brain.crowd_state import CrowdDirective
from .brain.dj_move import DJState, build_dj_state
from .brain.move_generator import CompleteMoveConfig, generate_complete_moves
from .brain.set_state import build_set_state
from .brain.timing import SetTimingIntent, build_timing_state, timing_candidate_adjustment
from .journey import JourneyMode
from .models import TrackAnalysis, TransitionPlan
from .set_models import TransitionEdge


@dataclass(frozen=True, slots=True)
class LookaheadChoice:
    winner: Any
    candidates: tuple[Any, ...]
    search_score: float
    projected_path: tuple[int, ...]
    lookahead_depth: int
    retrieval: CandidateRetrievalResult


def choose_complete_move_with_lookahead(
    *,
    current: int,
    remaining: set[int],
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    dj_state: DJState,
    played_order: list[int],
    played_plans: list[TransitionPlan],
    journey: JourneyMode,
    end_index: int | None,
    blocked_pairs: set[tuple[int, int]],
    config: CompleteMoveConfig,
    ensure_pairs: Callable[[Iterable[tuple[int, int]]], None] | None,
    depth: int,
    crowd_directive: CrowdDirective | None = None,
    timing_intent: SetTimingIntent | None = None,
) -> LookaheadChoice | None:
    available = sorted(remaining)
    if end_index is not None and end_index in available and len(available) > 1:
        available = [index for index in available if index != end_index]
    if not available:
        return None

    fallback_progress = len(played_order) / max(1, len(analyses) - 1)
    timing_state = build_timing_state(
        [analyses[index] for index in played_order],
        played_plans,
        intent=timing_intent,
        fallback_progress=fallback_progress,
    )
    progress = timing_state.progress
    set_state = build_set_state(
        [analyses[index] for index in played_order],
        played_plans,
    )
    set_state.elapsed_seconds = timing_state.elapsed_seconds
    width = min(len(available), config.lazy_candidate_width)

    while True:
        retrieval = retrieve_candidate_tracks(
            current_index=current,
            candidate_indices=available,
            analyses=analyses,
            set_state=set_state,
            journey=journey,
            progress=progress,
            width=width,
            crowd_directive=crowd_directive,
        )
        shortlisted = list(retrieval.selected)
        if ensure_pairs is not None:
            ensure_pairs((current, index) for index in shortlisted)

        generated = generate_complete_moves(
            current_index=current,
            candidate_indices=shortlisted,
            analyses=analyses,
            graph=graph,
            dj_state=dj_state,
            set_state=set_state,
            journey=journey,
            progress=progress,
            blocked_pairs=blocked_pairs,
            config=config,
            crowd_directive=crowd_directive,
        )
        if generated or width >= len(available):
            break
        width = min(len(available), width + config.lazy_expansion_width)

    if not generated:
        return None

    best: LookaheadChoice | None = None
    branch = generated[: max(1, config.search_branch_width)]
    for candidate in branch:
        next_remaining = set(remaining)
        next_remaining.discard(candidate.next_index)
        projected_path = (candidate.next_index,)
        realized_depth = 1

        timing_adjustment, timing_evidence = timing_candidate_adjustment(
            timing_state,
            analyses[candidate.next_index],
        )
        score = float(candidate.selection_score) + timing_adjustment
        candidate.plan.metadata = {
            **candidate.plan.metadata,
            "set_timing": {
                **timing_state.to_dict(),
                "candidate_adjustment": round(timing_adjustment, 6),
                "candidate_evidence": timing_evidence,
            },
        }

        if depth > 1 and next_remaining:
            projected = candidate.move.projected_state
            next_dj_state = build_dj_state(
                analyses[candidate.next_index],
                source_cursor=projected.source_cursor_seconds,
                effective_bpm=projected.effective_bpm,
                render_rate=projected.render_rate,
                set_state=build_set_state(
                    [analyses[index] for index in played_order + [candidate.next_index]],
                    played_plans + [candidate.plan],
                ),
                tempo_owner=projected.tempo_owner,
            )
            future = choose_complete_move_with_lookahead(
                current=candidate.next_index,
                remaining=next_remaining,
                analyses=analyses,
                graph=graph,
                dj_state=next_dj_state,
                played_order=played_order + [candidate.next_index],
                played_plans=played_plans + [candidate.plan],
                journey=journey,
                end_index=end_index,
                blocked_pairs=blocked_pairs,
                config=config,
                ensure_pairs=ensure_pairs,
                depth=depth - 1,
                crowd_directive=crowd_directive,
                timing_intent=timing_intent,
            )
            if future is not None:
                discount = float(config.lookahead_discount)
                score = (score + discount * future.search_score) / (1.0 + discount)
                projected_path = projected_path + future.projected_path
                realized_depth = 1 + future.lookahead_depth

        choice = LookaheadChoice(
            winner=candidate,
            candidates=tuple(generated),
            search_score=round(score, 6),
            projected_path=projected_path,
            lookahead_depth=realized_depth,
            retrieval=retrieval,
        )
        if best is None or (
            choice.search_score,
            candidate.selection_score,
            candidate.move.future_optionality,
            tuple(-index for index in choice.projected_path),
        ) > (
            best.search_score,
            best.winner.selection_score,
            best.winner.move.future_optionality,
            tuple(-index for index in best.projected_path),
        ):
            best = choice

    return best
