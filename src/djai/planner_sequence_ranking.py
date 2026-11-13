from __future__ import annotations

"""Sequence candidate ranking preparation.

This module deduplicates generated plans, applies the existing musical selection
utility through an injected scorer, ranks plans as DJ moves through an injected
ranker, and applies complete-move preference metadata. It does not rehearse,
quality-gate, fallback, or commit a transition.
"""

from typing import Callable

from .brain.move_generator import COMPLETE_MOVE_VERSION, CompleteMoveDecision
from .decision import seal_execution_plan
from .models import TrackAnalysis, TransitionPlan
from .performance.program import attach_performance_program


def dedupe_plans(plans: list[TransitionPlan]) -> list[TransitionPlan]:
    unique: dict[
        tuple[str, str, float, float, int, str],
        TransitionPlan,
    ] = {}
    for plan in plans:
        key = (
            plan.style,
            plan.technique,
            round(plan.exit_time_a, 4),
            round(plan.entry_time_b, 4),
            plan.transition_bars,
            plan.execution_signature,
        )
        unique[key] = plan
    return list(unique.values())


def complete_move_preference_score(
    plan: TransitionPlan,
    decision: CompleteMoveDecision,
) -> float:
    target = decision.winner.move
    exit_bar = int(
        plan.metadata.get("exit_bar_index", -10_000)
    )
    entry_bar = int(
        plan.metadata.get("entry_bar_index", -10_000)
    )
    exit_fit = max(
        0.0,
        1.0
        - abs(exit_bar - int(target.exit_bar)) / 16.0,
    )
    entry_fit = max(
        0.0,
        1.0
        - abs(entry_bar - int(target.entry_bar)) / 16.0,
    )
    duration_fit = max(
        0.0,
        1.0
        - abs(
            int(plan.transition_bars)
            - int(target.transition_bars)
        )
        / 16.0,
    )
    technique_fit = (
        1.0
        if str(plan.technique) == str(target.technique)
        else 0.0
    )
    intent_fit = (
        1.0
        if str(plan.intent) == str(target.intent)
        else 0.0
    )
    tempo_fit = (
        1.0
        if str(plan.tempo_strategy)
        == str(target.tempo_strategy)
        else 0.35
    )
    return max(
        0.0,
        min(
            1.0,
            0.22 * exit_fit
            + 0.18 * entry_fit
            + 0.15 * duration_fit
            + 0.20 * technique_fit
            + 0.12 * intent_fit
            + 0.13 * tempo_fit,
        ),
    )


def prepare_sequence_candidates(
    plans: list[TransitionPlan],
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    dj_state: object,
    preferred_complete_decision: CompleteMoveDecision | None,
    selection_utility: Callable[[TransitionPlan], float],
    rank_as_moves: Callable[
        [TrackAnalysis, TrackAnalysis, list[TransitionPlan], object],
        list[TransitionPlan],
    ],
    state_taste_payload: Callable[[object], dict],
    move_taste_payload: Callable[[object], dict],
    state_context_id: Callable[[object], str],
) -> list[TransitionPlan]:
    prepared = dedupe_plans(plans)

    for candidate in prepared:
        candidate.metadata = {
            **candidate.metadata,
            "selection_utility": round(
                float(selection_utility(candidate)),
                8,
            ),
        }

    prepared = rank_as_moves(
        track_a,
        track_b,
        prepared,
        dj_state,
    )

    if preferred_complete_decision is None:
        return prepared

    for candidate in prepared:
        match = complete_move_preference_score(
            candidate,
            preferred_complete_decision,
        )
        previous = float(
            candidate.metadata.get(
                "selection_utility",
                candidate.score,
            )
        )
        preferred_utility = (
            max(previous, 0.985)
            if match >= 0.94
            else 0.78 * previous + 0.22 * match
        )
        candidate.metadata = {
            **candidate.metadata,
            "selection_utility": round(
                preferred_utility,
                8,
            ),
            "complete_move_preference": {
                "version": COMPLETE_MOVE_VERSION,
                "match": round(match, 6),
                "target": (
                    preferred_complete_decision
                    .winner.move.to_dict()
                ),
                "generator_score": (
                    preferred_complete_decision
                    .winner.selection_score
                ),
                "safety_may_override": True,
            },
            "dj_state_taste": (
                state_taste_payload(
                    preferred_complete_decision
                    .winner.state
                )
                if preferred_complete_decision
                .winner.state is not None
                else None
            ),
            "dj_move_taste": move_taste_payload(
                preferred_complete_decision
                .winner.move
            ),
            "state_context_id": (
                state_context_id(
                    preferred_complete_decision
                    .winner.state
                )
                if preferred_complete_decision
                .winner.state is not None
                else None
            ),
            "state_move_taste_shadow": (
                preferred_complete_decision
                .winner.taste_shadow
            ),
            "crate_memory": (
                preferred_complete_decision
                .winner.crate_memory
            ),
            "adaptive_thinking": (
                preferred_complete_decision
                .thinking_budget.to_dict()
                if preferred_complete_decision
                .thinking_budget is not None
                else None
            ),
        }

        if (
            match >= 0.94
            and preferred_complete_decision
            .winner.move.deck_operations
        ):
            candidate.metadata = {
                **candidate.metadata,
                "deck_operations": [
                    operation.to_dict()
                    for operation
                    in preferred_complete_decision
                    .winner.move.deck_operations
                ],
                "structural_operation_authority": {
                    "version": "1.7.2",
                    "source": (
                        "complete_djmove_exact_match"
                    ),
                    "match": round(match, 6),
                    "deterministic_renderer_authoritative": True,
                },
            }
            attach_performance_program(candidate)
            seal_execution_plan(candidate)

    prepared.sort(
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
        reverse=True,
    )
    return prepared
