from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from ..decision import seal_execution_plan
from ..models import TrackAnalysis, TransitionPlan
from ..performance.program import attach_performance_program
from .candidate_retrieval import CandidateRetrievalResult
from .crowd_state import CrowdDirective, crowd_alignment_adjustment
from .director import direct_set_intent, intent_track_score
from .dj_move import DJMove, DJState, MoveBrainConfig, score_transition_plan_as_move
from .move_intent import DJMoveIntent
from .thinking_budget import ThinkingBudget
from .set_state import SetState, next_track_context_score, track_state_vector

COMPLETE_MOVE_VERSION = "1.8.0"
PlannerMode = Literal["legacy", "move_shadow", "move_authoritative"]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class CompleteMoveConfig:
    """Controls the 1.7 complete-move planner without replacing safety gates.

    1.7.1 adds sparse candidate retrieval and bounded receding-horizon search
    while keeping the same complete-move execution primitive and safety gates.
    """

    mode: PlannerMode = "move_shadow"
    move_weight: float = 0.50
    context_weight: float = 0.20
    intent_weight: float = 0.12
    edge_weight: float = 0.10
    plan_weight: float = 0.08
    minimum_advance_bars: int = 8
    candidates_per_track: int = 4
    report_top_k: int = 5
    lookahead_moves: int = 3
    search_branch_width: int = 3
    lookahead_discount: float = 0.72
    lazy_candidate_width: int = 10
    lazy_expansion_width: int = 10
    opener_candidates: int = 5

    @classmethod
    def from_environment(cls) -> "CompleteMoveConfig":
        raw = os.environ.get("DJAI_PLANNER_MODE", "move_authoritative").strip().lower()
        mode: PlannerMode = raw if raw in {"legacy", "move_shadow", "move_authoritative"} else "move_shadow"  # type: ignore[assignment]
        weights = [
            max(0.0, _float_env("DJAI_COMPLETE_MOVE_WEIGHT", 0.50)),
            max(0.0, _float_env("DJAI_COMPLETE_CONTEXT_WEIGHT", 0.20)),
            max(0.0, _float_env("DJAI_COMPLETE_INTENT_WEIGHT", 0.12)),
            max(0.0, _float_env("DJAI_COMPLETE_EDGE_WEIGHT", 0.10)),
            max(0.0, _float_env("DJAI_COMPLETE_PLAN_WEIGHT", 0.08)),
        ]
        total = sum(weights) or 1.0
        move, context, intent, edge, plan = [value / total for value in weights]
        return cls(
            mode=mode,
            move_weight=move,
            context_weight=context,
            intent_weight=intent,
            edge_weight=edge,
            plan_weight=plan,
            minimum_advance_bars=max(1, min(32, _int_env("DJAI_COMPLETE_MIN_ADVANCE_BARS", 8))),
            candidates_per_track=max(1, min(12, _int_env("DJAI_COMPLETE_CANDIDATES_PER_TRACK", 4))),
            report_top_k=max(1, min(12, _int_env("DJAI_COMPLETE_REPORT_TOP_K", 5))),
            lookahead_moves=max(2, min(3, _int_env("DJAI_MOVE_LOOKAHEAD", 3))),
            search_branch_width=max(1, min(6, _int_env("DJAI_MOVE_SEARCH_BRANCH_WIDTH", 3))),
            lookahead_discount=max(0.35, min(0.95, _float_env("DJAI_MOVE_LOOKAHEAD_DISCOUNT", 0.72))),
            lazy_candidate_width=max(3, min(32, _int_env("DJAI_LAZY_CANDIDATE_WIDTH", 10))),
            lazy_expansion_width=max(2, min(32, _int_env("DJAI_LAZY_EXPANSION_WIDTH", 10))),
            opener_candidates=max(1, min(12, _int_env("DJAI_LAZY_OPENER_CANDIDATES", 5))),
        )


@dataclass(frozen=True, slots=True)
class CompleteMoveCandidate:
    current_index: int
    next_index: int
    move: DJMove
    plan: TransitionPlan = field(compare=False, repr=False)
    selection_score: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    state: DJState | None = field(default=None, compare=False, repr=False)
    taste_shadow: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    crate_memory: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    def to_dict(self, *, include_plan: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": COMPLETE_MOVE_VERSION,
            "current_index": self.current_index,
            "next_index": self.next_index,
            "selection_score": round(self.selection_score, 6),
            "components": {key: round(float(value), 6) for key, value in self.components.items()},
            "move": self.move.to_dict(),
            "execution_signature": self.plan.execution_signature,
            "dj_state": None,
            "state_move_taste_shadow": self.taste_shadow,
            "crate_memory": self.crate_memory,
        }
        if self.state is not None:
            from ..training.state_move_taste import state_taste_payload

            payload["dj_state"] = state_taste_payload(self.state)
        if include_plan:
            payload["plan"] = self.plan.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class CompleteMoveDecision:
    current_index: int
    winner: CompleteMoveCandidate
    candidates: tuple[CompleteMoveCandidate, ...]
    search_score: float = 0.0
    lookahead_depth: int = 1
    projected_path: tuple[int, ...] = ()
    retrieval: CandidateRetrievalResult | None = field(default=None, compare=False, repr=False)
    commit_score: float = 0.0
    headphones: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    move_intent: DJMoveIntent | None = field(default=None, compare=False, repr=False)
    thinking_budget: ThinkingBudget | None = field(default=None, compare=False, repr=False)

    def to_dict(self, *, top_k: int = 5) -> dict[str, Any]:
        payload = {
            "version": COMPLETE_MOVE_VERSION,
            "current_index": self.current_index,
            "winner": self.winner.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates[: max(1, top_k)]],
            "search_score": round(float(self.search_score or self.winner.selection_score), 6),
            "lookahead_depth": int(self.lookahead_depth),
            "projected_path": list(self.projected_path),
            "commit_score": round(float(self.commit_score or self.search_score or self.winner.selection_score), 6),
        }
        if self.retrieval is not None:
            payload["retrieval"] = self.retrieval.to_dict()
        if self.headphones is not None:
            payload["ai_headphones"] = self.headphones
        if self.move_intent is not None:
            payload["move_intent"] = self.move_intent.to_dict()
        if self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget.to_dict()
        return payload


def _edge_plans(edge: object, *, limit: int) -> list[TransitionPlan]:
    plans: list[TransitionPlan] = []
    selected = getattr(edge, "plan", None)
    if isinstance(selected, TransitionPlan):
        plans.append(selected)
    for item in getattr(edge, "alternatives", ()) or ():
        if isinstance(item, TransitionPlan):
            plans.append(item)
    unique: dict[tuple[Any, ...], TransitionPlan] = {}
    for plan in plans:
        key = (
            plan.execution_signature,
            round(float(plan.exit_time_a), 4),
            round(float(plan.entry_time_b), 4),
            int(plan.transition_bars),
            str(plan.technique),
            str(plan.intent),
        )
        unique[key] = plan
    ranked = sorted(
        unique.values(),
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            float(plan.score),
            float(plan.cue_confidence),
        ),
        reverse=True,
    )
    return ranked[: max(1, limit)]


def generate_complete_moves(
    *,
    current_index: int,
    candidate_indices: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    graph: Mapping[tuple[int, int], object],
    dj_state: DJState,
    set_state: SetState,
    journey: str,
    progress: float,
    blocked_pairs: set[tuple[int, int]] | None = None,
    config: CompleteMoveConfig | None = None,
    crowd_directive: CrowdDirective | None = None,
) -> list[CompleteMoveCandidate]:
    """Generate fully specified next actions across candidate tracks.

    Each returned item binds the *track choice* to the concrete executable
    transition plan that supplies exit/entry bars, wait duration, transition
    length, tempo strategy, phase behavior, stem behavior and technique. This
    is the complete-move execution primitive used by the 1.7.x search layer.
    """

    cfg = config or CompleteMoveConfig.from_environment()
    blocked = blocked_pairs or set()
    move_cfg = MoveBrainConfig.from_environment()
    current_track = analyses[current_index]
    intent = direct_set_intent(journey, progress, set_state, crowd_directive=crowd_directive)
    generated: list[CompleteMoveCandidate] = []

    for next_index in candidate_indices:
        if next_index == current_index or (current_index, next_index) in blocked:
            continue
        edge = graph.get((current_index, next_index))
        if edge is None:
            continue
        next_track = analyses[next_index]
        context_score, _ = next_track_context_score(
            set_state,
            next_track,
            journey=journey,
            progress=progress,
        )
        intent_score, _ = intent_track_score(intent, set_state, next_track)
        edge_score = _clamp(float(getattr(edge, "objective_score", 0.0)))

        for plan in _edge_plans(edge, limit=cfg.candidates_per_track):
            move = score_transition_plan_as_move(
                current_track,
                next_track,
                plan,
                state=dj_state,
                config=move_cfg,
            )
            # A complete move must still be executable from the current deck
            # position. Static graph plans that point behind the playhead are
            # observations, not valid actions.
            required_exit_bar = int(dj_state.source_bar) + int(cfg.minimum_advance_bars)
            if int(move.exit_bar) < required_exit_bar:
                continue
            if move.protected_payoff_violation:
                if move.protected_commitment_kind in {"first_body_payoff", "protected_hook"}:
                    continue
                if move.intent not in {"breakdown_to_drop", "dramatic_switch"}:
                    continue
            plan_score = _clamp(float(plan.metadata.get("selection_utility", plan.score)))
            score = _clamp(
                cfg.move_weight * move.score
                + cfg.context_weight * context_score
                + cfg.intent_weight * intent_score
                + cfg.edge_weight * edge_score
                + cfg.plan_weight * plan_score
            )
            if plan.requires_confirmation:
                score = _clamp(score - 0.08)
            if plan.transition_mode == "emergency_cut":
                score = _clamp(score - 0.12)
            try:
                from .crate_memory import crate_memory_hint

                crate_memory = crate_memory_hint(current_track, next_track, move)
            except Exception as exc:
                crate_memory = {
                    "version": "1.8.2",
                    "available": False,
                    "authority": 0.0,
                    "score": 0.5,
                    "adjustment": 0.0,
                    "reason": f"crate memory unavailable: {type(exc).__name__}: {exc}",
                    "hard_safety_authoritative": True,
                }
            # Record/pair memory is intentionally a small residual. It can break
            # ties between already-feasible complete moves but cannot resurrect
            # a move rejected above for protected-payoff or executability reasons.
            memory_adjustment = float(crate_memory.get("adjustment", 0.0) or 0.0)
            score = _clamp(score + memory_adjustment)
            vector = track_state_vector(next_track)
            crowd_adjustment = crowd_alignment_adjustment(
                current_energy=set_state.energy,
                candidate_energy=vector.energy,
                candidate_vocal_density=vector.vocal_density,
                directive=crowd_directive,
            )
            score = _clamp(score + crowd_adjustment)
            try:
                from ..training.state_move_taste import (
                    move_taste_payload,
                    shadow_state_move_prediction,
                    state_context_id,
                    state_taste_payload,
                )

                taste_shadow = shadow_state_move_prediction(dj_state, move)
                taste_state_payload = state_taste_payload(dj_state)
                taste_move_payload = move_taste_payload(move)
                taste_state_id = state_context_id(dj_state)
            except Exception as exc:
                taste_shadow = {
                    "available": False,
                    "authority": 0.0,
                    "reason": f"state-move taste shadow unavailable: {type(exc).__name__}: {exc}",
                }
                taste_state_payload = None
                taste_move_payload = None
                taste_state_id = None
            candidate_plan = copy.deepcopy(plan)
            candidate_plan.metadata = {
                **candidate_plan.metadata,
                "state_move_taste_shadow": taste_shadow,
                "dj_state_taste": taste_state_payload,
                "dj_move_taste": taste_move_payload,
                "state_context_id": taste_state_id,
                "crate_memory": crate_memory,
            }
            if crowd_directive is not None and crowd_directive.active:
                candidate_plan.metadata = {
                    **candidate_plan.metadata,
                    "crowd_directive": crowd_directive.to_dict(),
                }
            if cfg.mode == "move_authoritative" and move.deck_operations:
                candidate_plan.metadata = {
                    **candidate_plan.metadata,
                    "deck_operations": [operation.to_dict() for operation in move.deck_operations],
                    "structural_operation_authority": {
                        "version": "1.7.2",
                        "source": "complete_djmove_generator",
                        "deterministic_renderer_authoritative": True,
                    },
                }
                attach_performance_program(candidate_plan)
                seal_execution_plan(candidate_plan)
            generated.append(
                CompleteMoveCandidate(
                    current_index=current_index,
                    next_index=next_index,
                    move=move,
                    plan=candidate_plan,
                    selection_score=round(score, 6),
                    components={
                        "move": move.score,
                        "set_context": context_score,
                        "director_intent": intent_score,
                        "edge": edge_score,
                        "plan": plan_score,
                        "crate_memory_adjustment": memory_adjustment,
                        "crowd_adjustment": crowd_adjustment,
                    },
                    state=dj_state,
                    taste_shadow=taste_shadow,
                    crate_memory=crate_memory,
                )
            )

    generated.sort(
        key=lambda item: (
            item.selection_score,
            item.move.score,
            item.move.future_optionality,
            -item.move.wait_bars,
            -item.next_index,
        ),
        reverse=True,
    )
    return generated


def complete_move_status() -> dict[str, Any]:
    cfg = CompleteMoveConfig.from_environment()
    try:
        from ..rehearsal.headphones import ai_headphones_status

        headphones = ai_headphones_status(planner_mode=cfg.mode)
    except Exception as exc:
        headphones = {"version": "unavailable", "enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        from .crate_memory import crate_memory_status

        crate_memory = crate_memory_status()
    except Exception as exc:
        crate_memory = {"version": "unavailable", "available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "version": COMPLETE_MOVE_VERSION,
        "planner_mode": cfg.mode,
        "planning_primitive": "DJState + candidate tracks -> complete DJMove",
        "joint_decisions": [
            "next_track",
            "exit_bar",
            "entry_bar",
            "wait_bars",
            "transition_bars",
            "tempo_strategy",
            "phase_strategy",
            "stem_strategy",
            "technique",
            "deck_operations",
        ],
        "lookahead_moves": cfg.lookahead_moves,
        "commit_moves": 1,
        "receding_horizon": True,
        "lazy_candidate_retrieval": True,
        "lazy_candidate_width": cfg.lazy_candidate_width,
        "lazy_expansion_width": cfg.lazy_expansion_width,
        "search_branch_width": cfg.search_branch_width,
        "opener_candidates": cfg.opener_candidates,
        "lookahead_discount": cfg.lookahead_discount,
        "legacy_planner_preserved": True,
        "shadow_comparison": cfg.mode == "move_shadow",
        "authoritative_route": cfg.mode == "move_authoritative",
        "hard_audio_safety_authoritative": True,
        "structural_deck_operations": True,
        "true_wait_compiled": True,
        "minimum_advance_bars": cfg.minimum_advance_bars,
        "candidates_per_track": cfg.candidates_per_track,
        "ai_headphones": headphones,
        "crate_memory": crate_memory,
    }
