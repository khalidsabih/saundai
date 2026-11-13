from __future__ import annotations

"""Set-plan reporting and metadata assembly.

This module packages already-decided planner results. It does not select routes,
compile transitions, run quality gates, finalize measurements, or create
confirmation tokens.
"""

import hashlib
import json
from typing import Any, Sequence

from .brain.dj_memory import build_dj_memory
from .brain.move_generator import COMPLETE_MOVE_VERSION, CompleteMoveConfig
from .brain.set_state import set_story_report
from .brain.thinking_budget import adaptive_thinking_status
from .brain.timing import SetTimingIntent, build_timing_state
from .brain.director import set_director_report
from .evaluation.full_set import FULL_SET_BENCHMARK_VERSION
from .journey import JourneyMode, journey_metadata
from .planner_complete_move_route import CompleteMoveRoute
from .planner_large_crate import env_int, large_set_threshold
from .planning_context import PlanningSession
from .rehearsal.headphones import ai_headphones_status
from .set_models import TransitionEdge
from .models import TrackAnalysis
from .version import MEASUREMENT_VERSION
from .brain.unified_planner import UNIFIED_PLANNER_VERSION


def build_complete_move_report(
    *,
    analyses: list[TrackAnalysis],
    order: list[int],
    transitions: list[TransitionEdge],
    complete_move_runs: Sequence[CompleteMoveRoute],
    complete_move_cfg: CompleteMoveConfig,
    blocked_pairs: set[tuple[int, int]],
    graph_pair_count: int,
    manual: bool,
    lazy_graph_authority: bool,
) -> dict[str, Any]:
    latest_complete_route = complete_move_runs[-1] if complete_move_runs else None
    complete_route_order = (
        list(latest_complete_route.order)
        if latest_complete_route is not None and latest_complete_route.completed
        else []
    )
    route_divergence = [
        position
        for position, (actual, proposed) in enumerate(
            zip(order, complete_route_order)
        )
        if actual != proposed
    ]
    generated_signatures = (
        [
            decision.winner.plan.execution_signature
            for decision in latest_complete_route.decisions
        ]
        if latest_complete_route is not None
        else []
    )
    compiled_signatures = [
        edge.plan.execution_signature
        for edge in transitions
    ]
    signature_matches = sum(
        1
        for generated, compiled in zip(
            generated_signatures,
            compiled_signatures,
        )
        if generated and generated == compiled
    )

    final_adjacent_pairs = set(zip(order, order[1:]))
    training_route_attempts: list[dict[str, Any]] = []
    for run_index, run in enumerate(complete_move_runs):
        decision_rows: list[dict[str, Any]] = []
        for decision_index, decision in enumerate(run.decisions):
            state = decision.winner.state
            candidate_by_target: dict[int, Any] = {}
            for candidate in decision.candidates:
                previous = candidate_by_target.get(int(candidate.next_index))
                if (
                    previous is None
                    or float(candidate.selection_score)
                    > float(previous.selection_score)
                ):
                    candidate_by_target[int(candidate.next_index)] = candidate

            decision_rows.append(
                {
                    "decision_index": decision_index,
                    "current_index": int(decision.current_index),
                    "winner_index": int(decision.winner.next_index),
                    "search_score": round(float(decision.search_score), 6),
                    "commit_score": round(float(decision.commit_score), 6),
                    "lookahead_depth": int(decision.lookahead_depth),
                    "projected_path": [
                        int(value)
                        for value in decision.projected_path
                    ],
                    "state": (
                        {
                            "energy": round(float(state.set_energy), 6),
                            "tension": round(float(state.set_tension), 6),
                            "groove": round(float(state.set_groove), 6),
                            "vocal_density": round(
                                float(state.set_vocal_density),
                                6,
                            ),
                            "source_cursor_seconds": round(
                                float(state.source_cursor_seconds),
                                6,
                            ),
                            "source_bar": int(state.source_bar),
                            "effective_bpm": round(
                                float(state.effective_bpm),
                                6,
                            ),
                        }
                        if state is not None
                        else None
                    ),
                    "candidates": [
                        {
                            "next_index": int(candidate.next_index),
                            "selection_score": round(
                                float(candidate.selection_score),
                                6,
                            ),
                            "move_score": round(
                                float(candidate.move.score),
                                6,
                            ),
                            "future_optionality": round(
                                float(candidate.move.future_optionality),
                                6,
                            ),
                            "components": {
                                key: round(float(value), 6)
                                for key, value
                                in candidate.components.items()
                            },
                            "won_attempt": (
                                int(candidate.next_index)
                                == int(decision.winner.next_index)
                            ),
                            "blocked_by_executive": (
                                int(decision.current_index),
                                int(candidate.next_index),
                            ) in blocked_pairs,
                            "survived_final_adjacency": (
                                int(decision.current_index),
                                int(candidate.next_index),
                            ) in final_adjacent_pairs,
                        }
                        for candidate in sorted(
                            candidate_by_target.values(),
                            key=lambda item: (
                                float(item.selection_score),
                                float(item.move.future_optionality),
                                -int(item.next_index),
                            ),
                            reverse=True,
                        )
                    ],
                }
            )

        training_route_attempts.append(
            {
                "attempt_index": run_index,
                "completed": bool(run.completed),
                "reason": str(run.reason or ""),
                "score": round(float(run.score), 6),
                "order": [int(value) for value in run.order],
                "decision_count": len(run.decisions),
                "decisions": decision_rows,
            }
        )

    crate_track_fingerprints: list[str] = []
    for track in analyses:
        identity_payload = track.to_dict(include_beats=False)
        identity_payload.pop("path", None)
        crate_track_fingerprints.append(
            hashlib.sha256(
                json.dumps(
                    identity_payload,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
    crate_fingerprint = hashlib.sha256(
        json.dumps(sorted(crate_track_fingerprints)).encode("utf-8")
    ).hexdigest()[:16]

    latest_decisions = (
        latest_complete_route.decisions
        if latest_complete_route is not None
        else ()
    )

    return {
        "version": COMPLETE_MOVE_VERSION,
        "mode": complete_move_cfg.mode,
        "active": bool(
            not manual
            and complete_move_cfg.mode != "legacy"
        ),
        "legacy_planner_preserved": True,
        "audio_safety_compiler_authoritative": True,
        "route_authoritative": bool(
            not manual
            and complete_move_cfg.mode == "move_authoritative"
            and complete_route_order == order
        ),
        "shadow_only": bool(
            not manual
            and complete_move_cfg.mode == "move_shadow"
        ),
        "lookahead_moves": complete_move_cfg.lookahead_moves,
        "adaptive_thinking": adaptive_thinking_status(),
        "commit_moves": 1,
        "receding_horizon": True,
        "structural_deck_operations": True,
        "true_wait_compiled": True,
        "advanced_structural_operations_opt_in": True,
        "ai_headphones": ai_headphones_status(
            planner_mode=complete_move_cfg.mode
        ),
        "ai_headphones_authoritative_decisions": sum(
            1
            for decision in latest_decisions
            if isinstance(decision.headphones, dict)
            and bool(decision.headphones.get("authoritative"))
        ),
        "ai_headphones_rendered_candidates": sum(
            int(decision.headphones.get("rendered_candidates", 0))
            for decision in latest_decisions
            if isinstance(decision.headphones, dict)
        ),
        "thinking_budget_levels": {
            level: sum(
                1
                for decision in latest_decisions
                if decision.thinking_budget is not None
                and decision.thinking_budget.brittleness.level == level
            )
            for level in ("easy", "normal", "brittle")
        },
        "thinking_budget_deep_search_escalations": sum(
            1
            for decision in latest_decisions
            if decision.thinking_budget is not None
            and decision.thinking_budget.lookahead_escalated
        ),
        "complete_move_generated_candidates": sum(
            len(decision.candidates)
            for decision in latest_decisions
        ),
        "crate_memory_informed_decisions": sum(
            1
            for decision in latest_decisions
            if isinstance(decision.winner.crate_memory, dict)
            and abs(
                float(
                    decision.winner.crate_memory.get(
                        "adjustment",
                        0.0,
                    )
                    or 0.0
                )
            ) > 1e-9
        ),
        "crate_memory_total_adjustment": round(
            sum(
                float(
                    decision.winner.crate_memory.get(
                        "adjustment",
                        0.0,
                    )
                    or 0.0
                )
                for decision in latest_decisions
                if isinstance(
                    decision.winner.crate_memory,
                    dict,
                )
            ),
            6,
        ),
        "lazy_candidate_retrieval": True,
        "lazy_candidate_width": complete_move_cfg.lazy_candidate_width,
        "lazy_expansion_width": complete_move_cfg.lazy_expansion_width,
        "search_branch_width": complete_move_cfg.search_branch_width,
        "opener_candidates": complete_move_cfg.opener_candidates,
        "sparse_graph_authority": lazy_graph_authority,
        "materialized_graph_pairs": graph_pair_count,
        "possible_graph_pairs": (
            len(analyses)
            * max(0, len(analyses) - 1)
        ),
        "run_count": len(complete_move_runs),
        "training_evidence": {
            "version": "1.0",
            "crate_fingerprint": crate_fingerprint,
            "attempt_count": len(training_route_attempts),
            "candidate_decision_count": sum(
                len(attempt["decisions"])
                for attempt in training_route_attempts
            ),
            "candidate_target_count": sum(
                len(decision["candidates"])
                for attempt in training_route_attempts
                for decision in attempt["decisions"]
            ),
            "attempts": training_route_attempts,
            "final_adjacent_pairs": [
                [int(left), int(right)]
                for left, right
                in sorted(final_adjacent_pairs)
            ],
            "blocked_pairs": [
                [int(left), int(right)]
                for left, right
                in sorted(blocked_pairs)
            ],
            "training_only": True,
            "execution_authority": False,
        },
        "proposed_order": complete_route_order,
        "final_order": list(order),
        "order_matches": bool(
            complete_route_order
            and complete_route_order == list(order)
        ),
        "divergence_positions": route_divergence,
        "generated_transition_signature_matches": signature_matches,
        "compiled_transition_count": len(compiled_signatures),
        "latest_route": (
            latest_complete_route.to_dict(
                top_k=complete_move_cfg.report_top_k
            )
            if latest_complete_route is not None
            else None
        ),
    }


def build_set_plan_metadata(
    *,
    analyses: list[TrackAnalysis],
    order: list[int],
    transitions: list[TransitionEdge],
    timing_intent: SetTimingIntent | None,
    estimated_duration_seconds: float,
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    listener: str | None,
    listener_profile_confidence: float,
    automatic_delivery: bool,
    complete_move_report: dict[str, Any],
    dj_executive_report: dict[str, Any],
    baseline_score: float,
    optimized_score: float,
    compiled_transition_score: float,
    minimum_body_seconds: float,
    minimum_body_fraction: float,
    graph_pair_count: int,
    session: PlanningSession,
    route_search_seconds: float,
    route_search_calls: int,
    compile_seconds: float,
    compile_calls: int,
    executive_seconds: float,
    complete_move_runs: Sequence[CompleteMoveRoute],
    broad_graph_materialized: bool,
    max_replans: int,
    quality_loop: Any,
    blocked_pairs: set[tuple[int, int]],
    replan_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    final_timing_state = build_timing_state(
        [analyses[index] for index in order],
        [edge.plan for edge in transitions],
        intent=timing_intent,
        projected_total_seconds=estimated_duration_seconds,
        fallback_progress=1.0,
    )
    threshold = large_set_threshold()
    is_large_set = len(analyses) >= threshold

    return {
        "tempo_policy": tempo_policy,
        "preferred_style": preferred_style,
        "journey": journey,
        "timing": {
            "intent": (
                timing_intent or SetTimingIntent()
            ).normalized().to_dict(),
            "state": final_timing_state.to_dict(),
            "time_aware_progress": (
                timing_intent is not None
                and timing_intent.target_duration_seconds is not None
            ),
        },
        "journey_arc": journey_metadata(
            order,
            analyses,
            journey=journey,
        ),
        "professional_story": set_story_report(
            order,
            analyses,
            journey=journey,
        ),
        "set_director_story": set_director_report(
            order,
            analyses,
            transitions,
            journey=journey,
        ),
        "dj_memory": build_dj_memory(
            [analyses[index] for index in order],
            [edge.plan for edge in transitions],
        ).to_dict(),
        "listener": listener,
        "listener_profile_confidence": listener_profile_confidence,
        "automatic_delivery": automatic_delivery,
        "complete_move_planner": complete_move_report,
        "dj_executive": dj_executive_report,
        "canonical_set_score": {
            "baseline": round(baseline_score, 6),
            "optimized": round(optimized_score, 6),
            "compiled_transition_objective_sum": round(
                compiled_transition_score,
                6,
            ),
            "comparison_basis": "order_score",
        },
        "fallback_transition_count": sum(
            1
            for edge in transitions
            if edge.plan.fallback_level > 0
        ),
        "minimum_body_seconds": minimum_body_seconds,
        "minimum_body_fraction": minimum_body_fraction,
        "directed_edge_count": graph_pair_count,
        "performance": {
            "planning": {
                **session.to_dict(),
                "route_search_seconds": round(
                    route_search_seconds,
                    6,
                ),
                "route_search_calls": route_search_calls,
                "compile_rehearsal_seconds": round(
                    compile_seconds,
                    6,
                ),
                "compile_rehearsal_calls": compile_calls,
                "executive_seconds": round(
                    executive_seconds,
                    6,
                ),
            }
        },
        "large_crate_planning": {
            "enabled": is_large_set,
            "track_count": len(analyses),
            "threshold": threshold,
            "graph_mode": session.graph_mode,
            "graph_pair_count": session.graph_pair_count,
            "graph_scout_cues": session.graph_scout_cues,
            "graph_build_calls": session.graph_build_calls,
            "graph_cache_hits": session.graph_cache_hits,
            "graph_cache_misses": session.graph_cache_misses,
            "graph_cache_hit_rate": round(
                session.graph_cache_hits
                / max(
                    1,
                    session.graph_cache_hits
                    + session.graph_cache_misses,
                ),
                6,
            ),
            "effective_sparse_attempts": len(
                complete_move_runs
            ),
            "best_sparse_partial_tracks": max(
                (
                    len(item.order)
                    for item in complete_move_runs
                ),
                default=0,
            ),
            "completed_sparse_attempts": sum(
                1
                for item in complete_move_runs
                if item.completed
            ),
            "broad_graph_materialized": (
                broad_graph_materialized
            ),
            "policy_start_limit": (
                env_int(
                    "DJAI_LARGE_SET_POLICY_STARTS",
                    4,
                    minimum=2,
                    maximum=8,
                )
                if is_large_set
                else len(analyses)
            ),
            "local_improve_passes": (
                env_int(
                    "DJAI_LARGE_SET_LOCAL_PASSES",
                    2,
                    minimum=1,
                    maximum=4,
                )
                if is_large_set
                else 8
            ),
            "maximum_replans": max_replans,
            "final_edges_restore_full_cues": True,
        },
        "quality_loop": {
            "enabled": bool(
                quality_loop is not None
                and quality_loop.enabled
            ),
            "top_k": (
                quality_loop.top_k
                if quality_loop is not None
                else 0
            ),
            "sample_rate": (
                quality_loop.sample_rate
                if quality_loop is not None
                else None
            ),
            "checked_transitions": sum(
                1
                for edge in transitions
                if edge.plan.quality_checked
            ),
            "passed_transitions": sum(
                1
                for edge in transitions
                if bool(
                    dict(
                        edge.plan.metadata.get(
                            "quality_loop",
                            {},
                        )
                    ).get(
                        "selection_gate_passed",
                        dict(
                            edge.plan.metadata.get(
                                "quality_loop",
                                {},
                            )
                        ).get(
                            "gate_passed",
                            False,
                        ),
                    )
                )
            ),
            "measurement_version": (
                MEASUREMENT_VERSION
                if quality_loop is not None
                else None
            ),
        },
        "compiled_execution": True,
        "unified_transition_intelligence": {
            "enabled": True,
            "simple_pro_shared_brain": True,
            "review_needed_is_replan": True,
            "blocked_edges": [
                list(pair)
                for pair in sorted(blocked_pairs)
            ],
            "replan_count": len(replan_trace),
            "replan_trace": replan_trace,
            "maximum_replans": max_replans,
        },
        "saund_ai_1": {
            "unified_planner_version": (
                UNIFIED_PLANNER_VERSION
            ),
            "full_set_benchmark_version": (
                FULL_SET_BENCHMARK_VERSION
            ),
            "learned_authority_policy": (
                "bounded_and_safety_gated"
            ),
        },
    }
