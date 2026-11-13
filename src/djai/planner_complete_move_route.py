from __future__ import annotations

"""Complete-move route orchestration.

This module owns the outer move-authoritative route loop: adaptive lookahead,
Headphones rehearsal, move-intent binding, and DJ-state advancement. Sequence
compilation and Executive commit remain outside this module.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from .brain.dj_move import build_dj_state
from .brain.move_generator import (
    COMPLETE_MOVE_VERSION,
    CompleteMoveConfig,
    CompleteMoveDecision,
)
from .brain.move_intent import build_dj_move_intent
from .brain.professional import ProfessionalBrainConfig
from .brain.set_state import build_set_state
from .brain.thinking_budget import (
    AdaptiveThinkingConfig,
    assess_context_preflight,
    assess_move_brittleness,
    build_thinking_budget,
    merge_brittleness,
)
from .brain.timing import SetTimingIntent, build_timing_state
from .journey import JourneyMode
from .models import TrackAnalysis, TransitionPlan
from .planner_complete_move_lookahead import choose_complete_move_with_lookahead
from .rehearsal.headphones import AIHeadphonesConfig, listen_to_complete_moves
from .set_models import TransitionEdge


@dataclass(frozen=True, slots=True)
class CompleteMoveRoute:
    order: tuple[int, ...]
    decisions: tuple[CompleteMoveDecision, ...]
    score: float
    completed: bool
    reason: str = ""

    def to_dict(self, *, top_k: int = 5) -> dict[str, Any]:
        return {
            "version": COMPLETE_MOVE_VERSION,
            "order": list(self.order),
            "score": round(self.score, 6),
            "completed": self.completed,
            "reason": self.reason,
            "decisions": [
                decision.to_dict(top_k=top_k)
                for decision in self.decisions
            ],
        }


def complete_move_route_from_opener(
    opener: int,
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    end_index: int | None,
    journey: JourneyMode,
    blocked_pairs: set[tuple[int, int]],
    config: CompleteMoveConfig,
    headphones_config: AIHeadphonesConfig,
    listener: str | None = None,
    ensure_pairs: Callable[[Iterable[tuple[int, int]]], None] | None = None,
    timing_intent: SetTimingIntent | None = None,
    listen_to_moves: Callable[..., Any] = listen_to_complete_moves,
) -> CompleteMoveRoute:
    order = [opener]
    plans: list[TransitionPlan] = []
    decisions: list[CompleteMoveDecision] = []
    remaining = set(range(len(analyses))) - {opener}
    current = opener

    set_state = build_set_state([analyses[opener]])
    dj_state = build_dj_state(
        analyses[opener],
        source_cursor=0.0,
        effective_bpm=analyses[opener].bpm,
        render_rate=1.0,
        set_state=set_state,
    )

    adaptive_cfg = AdaptiveThinkingConfig.from_environment()
    professional_pilot = (
        ProfessionalBrainConfig.from_environment().adaptive_pilot_finalists
    )

    while remaining:
        preflight_indices = sorted(remaining)
        if (
            end_index is not None
            and end_index in preflight_indices
            and len(preflight_indices) > 1
        ):
            preflight_indices = [
                index
                for index in preflight_indices
                if index != end_index
            ]

        preflight = assess_context_preflight(
            track_a=analyses[current],
            candidate_tracks=[
                analyses[index]
                for index in preflight_indices
            ],
            dj_state=dj_state,
            config=adaptive_cfg,
        )
        initial_depth = (
            min(
                config.lookahead_moves,
                max(1, adaptive_cfg.easy_lookahead),
            )
            if adaptive_cfg.enabled
            and preflight.easy_candidate_context
            else config.lookahead_moves
        )

        choice = choose_complete_move_with_lookahead(
            current=current,
            remaining=remaining,
            analyses=analyses,
            graph=graph,
            dj_state=dj_state,
            played_order=order,
            played_plans=plans,
            journey=journey,
            end_index=end_index,
            blocked_pairs=blocked_pairs,
            config=config,
            ensure_pairs=ensure_pairs,
            depth=initial_depth,
            timing_intent=timing_intent,
        )
        if choice is None:
            return CompleteMoveRoute(
                order=tuple(order),
                decisions=tuple(decisions),
                score=(
                    sum(
                        item.commit_score
                        or item.search_score
                        or item.winner.selection_score
                        for item in decisions
                    )
                    / max(1, len(decisions))
                ),
                completed=False,
                reason=(
                    "no executable complete move from track index "
                    f"{current}"
                ),
            )

        initial_assessment = assess_move_brittleness(
            track_a=analyses[current],
            track_b=analyses[choice.winner.next_index],
            winner=choice.winner,
            candidates=choice.candidates,
            config=adaptive_cfg,
        )
        provisional_budget = build_thinking_budget(
            assessment=initial_assessment,
            configured_lookahead=config.lookahead_moves,
            preflight=preflight,
            headphones_serious_pool=headphones_config.serious_pool,
            headphones_minimum_renders=headphones_config.minimum_renders,
            headphones_maximum_renders=headphones_config.maximum_renders,
            professional_pilot_finalists=professional_pilot,
            config=adaptive_cfg,
            lookahead_initial=initial_depth,
            lookahead_used=initial_depth,
        )

        search_depth_used = initial_depth
        lookahead_escalated = False
        if (
            provisional_budget.lookahead_target > initial_depth
            and len(remaining) > 1
        ):
            deeper = choose_complete_move_with_lookahead(
                current=current,
                remaining=remaining,
                analyses=analyses,
                graph=graph,
                dj_state=dj_state,
                played_order=order,
                played_plans=plans,
                journey=journey,
                end_index=end_index,
                blocked_pairs=blocked_pairs,
                config=config,
                ensure_pairs=ensure_pairs,
                depth=provisional_budget.lookahead_target,
                timing_intent=timing_intent,
            )
            if deeper is not None:
                choice = deeper
                search_depth_used = (
                    provisional_budget.lookahead_target
                )
                lookahead_escalated = True

        final_assessment = assess_move_brittleness(
            track_a=analyses[current],
            track_b=analyses[choice.winner.next_index],
            winner=choice.winner,
            candidates=choice.candidates,
            config=adaptive_cfg,
        )
        assessment = (
            merge_brittleness(
                initial_assessment,
                final_assessment,
                config=adaptive_cfg,
            )
            if lookahead_escalated
            else final_assessment
        )
        thinking_budget = build_thinking_budget(
            assessment=assessment,
            configured_lookahead=config.lookahead_moves,
            preflight=preflight,
            headphones_serious_pool=headphones_config.serious_pool,
            headphones_minimum_renders=headphones_config.minimum_renders,
            headphones_maximum_renders=headphones_config.maximum_renders,
            professional_pilot_finalists=professional_pilot,
            config=adaptive_cfg,
            lookahead_initial=initial_depth,
            lookahead_used=search_depth_used,
            lookahead_escalated=lookahead_escalated,
        )

        winner = choice.winner
        decision_lookahead_depth = choice.lookahead_depth
        decision_projected_path = choice.projected_path
        commit_score = float(
            choice.search_score or winner.selection_score
        )
        headphones_report: dict[str, Any] | None = None

        if (
            config.mode == "move_authoritative"
            and headphones_config.enabled
        ):
            current_set_state = build_set_state(
                [analyses[index] for index in order],
                plans,
            )
            decision_headphones_config = replace(
                headphones_config,
                serious_pool=thinking_budget.headphones_serious_pool,
                minimum_renders=thinking_budget.headphones_minimum_renders,
                maximum_renders=thinking_budget.headphones_maximum_renders,
            ).normalized()

            headphones = listen_to_moves(
                preferred=choice.winner,
                candidates=choice.candidates,
                analyses=analyses,
                set_state=current_set_state,
                journey=journey,
                progress=build_timing_state(
                    [analyses[index] for index in order],
                    plans,
                    intent=timing_intent,
                    fallback_progress=(
                        len(order)
                        / max(1, len(analyses) - 1)
                    ),
                ).progress,
                listener=listener,
                config=decision_headphones_config,
            )
            winner = headphones.winner
            headphones_report = {
                **headphones.to_dict(),
                "adaptive_thinking": thinking_budget.to_dict(),
            }
            if headphones.authoritative:
                commit_score = float(headphones.winner_score)
                if (
                    winner.plan.execution_signature
                    != choice.winner.plan.execution_signature
                ):
                    decision_lookahead_depth = 1
                    decision_projected_path = (
                        winner.next_index,
                    )

        if (
            winner.plan.execution_signature
            != choice.winner.plan.execution_signature
        ):
            listened_assessment = assess_move_brittleness(
                track_a=analyses[current],
                track_b=analyses[winner.next_index],
                winner=winner,
                candidates=choice.candidates,
                config=adaptive_cfg,
            )
            assessment = merge_brittleness(
                assessment,
                listened_assessment,
                config=adaptive_cfg,
            )
            thinking_budget = build_thinking_budget(
                assessment=assessment,
                configured_lookahead=config.lookahead_moves,
                preflight=preflight,
                headphones_serious_pool=headphones_config.serious_pool,
                headphones_minimum_renders=headphones_config.minimum_renders,
                headphones_maximum_renders=headphones_config.maximum_renders,
                professional_pilot_finalists=professional_pilot,
                config=adaptive_cfg,
                lookahead_initial=initial_depth,
                lookahead_used=search_depth_used,
                lookahead_escalated=lookahead_escalated,
            )

        same_target_candidates = tuple(
            candidate
            for candidate in choice.candidates
            if int(candidate.next_index) == int(winner.next_index)
        )
        move_intent = build_dj_move_intent(
            current_index=current,
            next_index=winner.next_index,
            track_a=analyses[current],
            track_b=analyses[winner.next_index],
            winner_move=winner.move,
            winner_plan=winner.plan,
            candidate_moves=tuple(
                candidate.move
                for candidate in same_target_candidates
            ),
            candidate_plans=tuple(
                candidate.plan
                for candidate in same_target_candidates
            ),
        )
        decision = CompleteMoveDecision(
            current_index=current,
            winner=winner,
            candidates=choice.candidates,
            search_score=choice.search_score,
            lookahead_depth=decision_lookahead_depth,
            projected_path=decision_projected_path,
            retrieval=choice.retrieval,
            commit_score=round(commit_score, 6),
            headphones=headphones_report,
            move_intent=move_intent,
            thinking_budget=thinking_budget,
        )
        decisions.append(decision)
        order.append(winner.next_index)
        remaining.remove(winner.next_index)
        plans.append(winner.plan)
        current = winner.next_index

        set_state = build_set_state(
            [analyses[index] for index in order],
            plans,
        )
        projected = winner.move.projected_state
        dj_state = build_dj_state(
            analyses[current],
            source_cursor=projected.source_cursor_seconds,
            effective_bpm=projected.effective_bpm,
            render_rate=projected.render_rate,
            set_state=set_state,
            tempo_owner=projected.tempo_owner,
        )

    completed = (
        len(order) == len(analyses)
        and (
            end_index is None
            or order[-1] == end_index
        )
    )
    return CompleteMoveRoute(
        order=tuple(order),
        decisions=tuple(decisions),
        score=(
            sum(
                item.commit_score
                or item.search_score
                or item.winner.selection_score
                for item in decisions
            )
            / max(1, len(decisions))
        ),
        completed=completed,
        reason=(
            ""
            if completed
            else "fixed closer constraint could not be satisfied"
        ),
    )
