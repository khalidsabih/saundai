from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .cache import AnalysisBatchStats, analyze_folder, analyze_tracks
from .brain.set_state import SetState, build_set_state, next_track_context_score, set_story_report
from .brain.timing import SetTimingIntent, build_timing_state, timing_candidate_adjustment
from .brain.candidate_retrieval import CandidateRetrievalResult, retrieve_candidate_tracks
from .brain.crowd_state import CrowdDirective, crowd_directive_from_payload
from .brain.director import direct_set_intent, intent_track_score, set_director_report
from .brain.orchestrator import ExecutiveOrchestrator, ExecutiveReplanRequest
from .brain.live_orchestrator import (
    LiveExecutiveConfig,
    LiveFutureCandidate,
    LiveReplanRequest,
)
from .brain.dj_memory import build_dj_memory
from .brain.dj_move import (
    DJState,
    MoveBrainConfig,
    MusicalTimeline,
    build_dj_state,
    build_musical_timeline,
    entry_candidate_musical_score,
    exit_candidate_musical_score,
    move_opportunity_score,
    rank_transition_plans_as_moves,
)
from .brain.professional import ProfessionalBrainConfig, rehearse_candidate_pool
from .brain.move_generator import (
    COMPLETE_MOVE_VERSION,
    CompleteMoveConfig,
    CompleteMoveDecision,
    generate_complete_moves,
)
from .brain.move_intent import build_dj_move_intent
from .brain.thinking_budget import (
    AdaptiveThinkingConfig,
    adaptive_thinking_status,
    assess_context_preflight,
    assess_move_brittleness,
    build_thinking_budget,
    merge_brittleness,
)
from .brain.transition_policy import (
    TransitionPolicyContext,
    annotate_transition_policy,
    apply_bounded_transition_policy,
)
from .brain.set_policy import SetPolicyConfig, build_receding_horizon_order
from .brain.unified_planner import UNIFIED_PLANNER_VERSION, build_unified_ai_order
from .candidate_renderer import QualityLoopConfig, measure_and_select_transition
from .fallback import build_last_resort_plan, select_automatic_transition
from .confirmation import create_confirmation_token
from .decision import CUT_TECHNIQUES, seal_execution_plan, validate_execution_plan
from .execution import TransitionRenderContext, professional_normalization_gain
from .environment import current_environment
from .evaluation.full_set import FULL_SET_BENCHMARK_VERSION, evaluate_full_set
from .models import CompatibilityBreakdown, QualityClass, TrackAnalysis, TransitionCandidate, TransitionPlan
from .measurement_store import finalize_measurement_run, measurement_log_offset
from .music.timeline import earliest_safe_song_form_exit_bar
from .journey import JourneyMode, curve_fit_score, journey_metadata, normalize_journey
from .performance.program import attach_performance_program
from .rehearsal.headphones import AIHeadphonesConfig, ai_headphones_status, listen_to_complete_moves
from .personalization import ListenerProfile, get_listener_profile, plan_personalization_adjustment
from .parallel import resolve_worker_selection, worker_initializer
from .planner_graph_workers import iter_graph_worker_results
from .planner_graph_pairs import (
    GraphPairTask as _GraphPairTask,
    graph_worker_initializer as _graph_worker_initializer,
    plan_graph_pair_core as _plan_graph_pair_core,
    plan_graph_pair_worker as _plan_graph_pair_worker,
)
from .planner_candidates import (
    candidate_diverse_by_bar as _candidate_diverse_by_bar,
    graph_scout_candidates as _graph_scout_candidates,
)
from .planner_large_crate import (
    env_int as _env_int,
    graph_scout_cue_limit as _graph_scout_cue_limit,
    large_crate_runtime_status,
    large_set_threshold as _large_set_threshold,
)
from .planner_route_search import (
    greedy_orders as _greedy_orders,
    local_improve as _route_local_improve,
    valid_order as _valid_order,
)

from .planner_order_optimizer import optimize_order_candidates
from .planner_complete_move_lookahead import (
    LookaheadChoice as _LookaheadChoice,
    choose_complete_move_with_lookahead as _choose_complete_move_with_lookahead,
)
from .planner_complete_move_route import (
    CompleteMoveRoute as _CompleteMoveRoute,
    complete_move_route_from_opener as _complete_move_route_from_opener,
)
from .planner_reporting import (
    build_complete_move_report as _build_complete_move_report,
    build_set_plan_metadata as _build_set_plan_metadata,
)
from .planner_quality_loop import (
    finalize_quality_measurement_run as _finalize_quality_measurement_run,
    measurement_run_start_offset as _measurement_run_start_offset,
    professional_brain_config_for_transition as _professional_brain_config_for_transition,
)
from .planner_sequence_cues import (
    required_source_body as _required_source_body,
    mixing_window as _mixing_window,
    sequence_exit_candidates as _sequence_exit_candidates_impl,
    sequence_entry_candidates as _sequence_entry_candidates_impl,
)
from .planner_sequence_candidates import (
    build_sequence_candidate_pool as _build_sequence_candidate_pool,
)
from .planner_sequence_ranking import (
    prepare_sequence_candidates as _prepare_sequence_candidates,
)
from .planner_sequence_policy import (
    apply_sequence_transition_policy as _apply_sequence_transition_policy,
)
from .planner import (
    DifficultTransitionError,
    plan_transition,
    plan_transition_alternatives,
    plan_transition_candidate_pool,
    plan_transition_styles,
)
from .planning_context import CandidatePairEvidence, PairEvidenceMatrix, PlanningSession, build_pair_evidence_matrix, scoring_model_fingerprint
from .renderer import DEFAULT_RENDER_SAMPLE_RATE
from .serialization import track_analysis_from_dict, transition_candidate_from_dict, transition_plan_from_dict
from .set_models import SetPlan, TransitionEdge
from .stems import enrich_analyses_with_cached_stems, find_cached_stems
from .training.state_move_taste import move_taste_payload, state_context_id, state_taste_payload
from .layers import enrich_analyses_with_performance_layers
from .version import MEASUREMENT_VERSION, PLAN_SCHEMA_VERSION, SET_PLAN_VERSION

ProgressCallback = Callable[[str, int, int, str], None]


from .planner_graph_cache import (
    GRAPH_PAIR_CACHE_VERSION,
    GraphPairResult as _GraphPairResult,
    graph_pair_cache_key as _graph_pair_cache_key,
    load_graph_pair_cache as _load_graph_pair_cache,
    store_graph_pair_cache as _store_graph_pair_cache,
)


class TransitionReplanRequired(RuntimeError):
    """A rehearsed edge is not good enough to enter an automatic final set."""

    def __init__(
        self,
        from_index: int,
        to_index: int,
        *,
        action: str,
        reason: str,
        resolution: dict[str, Any] | None = None,
        compiled_prefix: Sequence[TransitionEdge] | None = None,
    ) -> None:
        super().__init__(reason)
        self.from_index = from_index
        self.to_index = to_index
        self.action = action
        self.reason = reason
        self.resolution = dict(resolution or {})
        self.compiled_prefix = list(compiled_prefix or ())


def classify_quality(plan: TransitionPlan) -> QualityClass:
    if plan.quality in {"good", "risky", "poor"}:
        return plan.quality
    if plan.score >= 0.72 and not plan.requires_confirmation and plan.transition_mode != "emergency_cut":
        return "good"
    if plan.score >= 0.48 or plan.transition_mode == "emergency_cut":
        return "risky"
    return "poor"


def _edge_objective(
    plan: TransitionPlan,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    journey: JourneyMode,
    listener_profile: ListenerProfile | None = None,
) -> float:
    value = float(plan.score)
    if plan.transition_mode == "emergency_cut":
        value -= 0.15
    if plan.requires_confirmation:
        value -= 0.08
    value -= 0.11 * max(0.0, 0.58 - plan.cue_confidence)
    value -= 0.05 * max(0.0, 0.55 - plan.intent_score)
    if track_a.variable_tempo or track_b.variable_tempo:
        value -= 0.04
    meter_a = int(track_a.time_signature or 0)
    meter_b = int(track_b.time_signature or 0)
    if meter_a not in {3, 4, 6} or meter_b not in {3, 4, 6}:
        value -= 0.04
    elif meter_a != meter_b:
        value -= 0.08

    router = plan.metadata.get("transition_router", {})
    if isinstance(router, dict):
        route_fit = float(router.get("route_fit", 0.5))
        bridge_preference = float(router.get("bridge_preference", 0.0))
        value += 0.08 * (route_fit - 0.5)
        # This is the Set Brain's bridge-track mechanism: musically distant
        # direct edges remain executable, but are expensive enough that an
        # intermediate track wins when the crate contains a credible route.
        value -= 0.38 * max(0.0, min(1.0, bridge_preference))

    delta = track_b.energy - track_a.energy
    if journey == "build":
        value += 0.06 * math.exp(-(((delta - 0.08) / 0.18) ** 2))
        value -= 0.05 * max(0.0, -delta - 0.10)
    elif journey == "smooth":
        value -= 0.09 * max(0.0, abs(delta) - 0.18)
    elif journey == "warmup":
        value += 0.035 * math.exp(-(((delta - 0.05) / 0.20) ** 2))
        value -= 0.045 * max(0.0, -delta - 0.12)
    elif journey == "peak":
        value += 0.025 * min(track_a.energy, track_b.energy)
        value -= 0.055 * max(0.0, -delta - 0.20)
    elif journey == "sunset":
        value -= 0.04 * max(0.0, abs(delta) - 0.24)
    else:
        value -= 0.04 * max(0.0, abs(delta) - 0.28)

    move_payload = plan.metadata.get("dj_move", {})
    if isinstance(move_payload, dict) and move_payload:
        move_score = max(0.0, min(1.0, float(move_payload.get("score", 0.5))))
        value += 0.28 * (move_score - 0.5)
        if bool(move_payload.get("protected_payoff_violation", False)):
            commitment_kind = str(move_payload.get("protected_commitment_kind") or "")
            if commitment_kind in {"first_body_payoff", "protected_hook"}:
                value -= 1.0
            elif str(plan.intent) not in {"breakdown_to_drop", "dramatic_switch"}:
                value -= 0.32

    adjustment = plan_personalization_adjustment(plan, listener_profile)
    plan.listener_adjustment = round(adjustment, 6)
    value += adjustment
    try:
        from .intelligence.neural_policy import next_track_prediction
        prediction=next_track_prediction(track_a,track_b); plan.metadata={**plan.metadata,"neural_next_track":prediction.to_dict()}; value += 0.16*prediction.authority*(2.0*prediction.score-1.0)
    except Exception as exc:
        plan.metadata={**plan.metadata,"neural_next_track":{"available":False,"reason":f"next-track policy unavailable: {exc}"}}
    return float(value)


def _select_alternative(
    alternatives: list[TransitionPlan],
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    preferred_style: str,
    journey: JourneyMode,
    listener_profile: ListenerProfile | None = None,
) -> tuple[TransitionPlan, float]:
    candidates = alternatives
    if preferred_style in {"safe", "creative", "emergency"}:
        preferred = [plan for plan in alternatives if plan.style == preferred_style]
        if preferred:
            candidates = preferred
    ranked: list[tuple[float, TransitionPlan]] = []
    for plan in candidates:
        utility = _edge_objective(
            plan, track_a, track_b, journey=journey, listener_profile=listener_profile
        )
        if preferred_style == "auto":
            if plan.style == "safe":
                utility += 0.025
            elif plan.style == "creative":
                utility += 0.006
        ranked.append((utility, plan))
    ranked.sort(key=lambda item: (item[0], item[1].score), reverse=True)
    return ranked[0][1], ranked[0][0]


def _transition_warnings(plan: TransitionPlan) -> list[str]:
    warnings: list[str] = []
    if plan.warning:
        warnings.append(plan.warning)
    if plan.compatibility.harmonic < 0.45:
        warnings.append("Local harmonic compatibility is weak.")
    if plan.compatibility.beat_grid < 0.42 and plan.beatmatched:
        warnings.append("One or both local beat grids are uncertain.")
    if plan.compatibility.vocal_overlap < 0.45:
        warnings.append("Both selected sections may contain prominent vocals.")
    if plan.compatibility.bass_conflict < 0.45:
        warnings.append("Both selected sections may contain strong bass content.")
    return list(dict.fromkeys(warnings))


def _graph_select_alternative(
    alternatives: list[TransitionPlan],
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    preferred_style: str,
    journey: JourneyMode,
) -> tuple[TransitionPlan, float]:
    return _select_alternative(
        alternatives,
        track_a,
        track_b,
        preferred_style=preferred_style,
        journey=journey,
    )


def _edge_from_plan(
    from_index: int,
    to_index: int,
    plan: TransitionPlan,
    alternatives: list[TransitionPlan],
    analyses: list[TrackAnalysis],
    *,
    journey: JourneyMode,
    listener_profile: ListenerProfile | None = None,
    extra_reason: str | None = None,
) -> TransitionEdge:
    quality = classify_quality(plan)
    plan.quality = quality
    if "dj_move_opportunity" not in plan.metadata:
        move_score, move_meta = move_opportunity_score(
            analyses[from_index], analyses[to_index]
        )
        plan.metadata = {
            **plan.metadata,
            "dj_move_opportunity": {**move_meta, "score": round(move_score, 6)},
        }
    reasons = list(plan.rationale)
    if extra_reason:
        reasons.append(extra_reason)
    return TransitionEdge(
        from_index=from_index,
        to_index=to_index,
        selected_style=plan.style,
        objective_score=round(
            _edge_objective(
                plan,
                analyses[from_index],
                analyses[to_index],
                journey=journey,
                listener_profile=listener_profile,
            ),
            6,
        ),
        quality=quality,
        plan=plan,
        alternatives=alternatives,
        warnings=_transition_warnings(plan),
        reasons=list(dict.fromkeys(reasons)),
    )


def build_transition_graph(
    analyses: list[TrackAnalysis],
    *,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    journey: JourneyMode = "balanced",
    progress: ProgressCallback | None = None,
    workers: int | None = None,
    session: PlanningSession | None = None,
    pair_indices: Iterable[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], TransitionEdge]:
    """Build native-state directional edges, optionally for a sparse pair set.

    Cached pair evidence is resolved in the parent process before worker
    selection. Only true cache misses are submitted to graph workers, and
    worker results are persisted back into the same pair cache.
    """
    if len(analyses) < 2:
        raise ValueError("A set needs at least two tracks.")

    active_session = session if session is not None else PlanningSession()
    active_session.graph_build_calls += 1
    stem_maps = {index: find_cached_stems(track.path) for index, track in enumerate(analyses)}
    if pair_indices is None:
        pairs = [
            (from_index, to_index)
            for from_index in range(len(analyses))
            for to_index in range(len(analyses))
            if from_index != to_index
        ]
    else:
        pairs = sorted({
            (int(from_index), int(to_index))
            for from_index, to_index in pair_indices
            if 0 <= int(from_index) < len(analyses)
            and 0 <= int(to_index) < len(analyses)
            and int(from_index) != int(to_index)
        })
    tasks = [_GraphPairTask(from_index=a, to_index=b) for a, b in pairs]
    total = len(tasks)
    if total == 0:
        return {}

    scout_cues = _graph_scout_cue_limit(len(analyses))
    active_session.graph_mode = "lazy" if pair_indices is not None else ("scout" if scout_cues else "full")
    active_session.graph_scout_cues = scout_cues
    started = time.perf_counter()
    results: dict[tuple[int, int], _GraphPairResult] = {}
    cache_keys: dict[tuple[int, int], str] = {}
    missing_tasks: list[_GraphPairTask] = []
    model_fingerprint = scoring_model_fingerprint(active_session.scoring_model)

    # Cache lookup is intentionally outside the serial/parallel split. Previous
    # behavior bypassed the cache entirely when a process pool was used.
    for task in tasks:
        track_a = analyses[task.from_index]
        track_b = analyses[task.to_index]
        key = (task.from_index, task.to_index)
        cache_key = _graph_pair_cache_key(
            track_a,
            track_b,
            tempo_policy=tempo_policy,
            preferred_style=preferred_style,
            journey=journey,
            scout_cues=scout_cues,
            model_fingerprint=model_fingerprint,
        )
        cache_keys[key] = cache_key
        result = _load_graph_pair_cache(
            cache_key,
            from_index=task.from_index,
            to_index=task.to_index,
            track_a=track_a,
            track_b=track_b,
        )
        if result is None:
            missing_tasks.append(task)
            active_session.graph_cache_misses += 1
        else:
            results[key] = result
            active_session.graph_cache_hits += 1

    completed = len(results)
    if progress and completed:
        progress(
            "graph",
            completed,
            total,
            f"{completed}/{total} transition pairs restored from cache",
        )

    requested_selection = resolve_worker_selection("graph", total, requested=workers)
    active_session.graph_workers = max(
        active_session.graph_workers,
        requested_selection.workers,
    )
    execution_selection = resolve_worker_selection(
        "graph",
        len(missing_tasks),
        requested=workers,
    )

    if missing_tasks and not execution_selection.parallel:
        for task in missing_tasks:
            track_a = analyses[task.from_index]
            track_b = analyses[task.to_index]
            if progress:
                progress("graph", completed, total, f"{track_a.title} -> {track_b.title}")
            result = _plan_graph_pair_core(
                task,
                analyses=analyses,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                stem_maps=stem_maps,
                scoring_model=active_session.scoring_model,
                select_alternative=_graph_select_alternative,
                classify_quality=classify_quality,
                transition_warnings=_transition_warnings,
            )
            key = (result.from_index, result.to_index)
            results[key] = result
            _store_graph_pair_cache(cache_keys[key], result)
            completed += 1
            if progress:
                progress("graph", completed, total, result.label)
    elif missing_tasks:
        if progress:
            mode = f"scout {scout_cues}x{scout_cues}" if scout_cues else "full cue graph"
            progress(
                "graph",
                completed,
                total,
                f"{execution_selection.workers} graph workers · {len(missing_tasks)} cache misses · {mode}",
            )
        for result in iter_graph_worker_results(
            missing_tasks,
            workers=execution_selection.workers,
            initializer=_graph_worker_initializer,
            initargs=(
                tuple(analyses),
                tempo_policy,
                preferred_style,
                journey,
                stem_maps,
                active_session.scoring_model,
                _graph_select_alternative,
                classify_quality,
                _transition_warnings,
            ),
            worker=_plan_graph_pair_worker,
        ):
            key = (result.from_index, result.to_index)
            results[key] = result
            _store_graph_pair_cache(cache_keys[key], result)
            completed += 1
            if progress:
                progress("graph", completed, total, result.label)

    graph: dict[tuple[int, int], TransitionEdge] = {}
    move_cfg = MoveBrainConfig.from_environment()
    timelines = [build_musical_timeline(track) for track in analyses] if move_cfg.enabled else []
    for task in tasks:
        key = (task.from_index, task.to_index)
        result = results[key]
        if move_cfg.enabled:
            move_score, move_meta = move_opportunity_score(
                analyses[task.from_index],
                analyses[task.to_index],
                timeline_a=timelines[task.from_index],
                timeline_b=timelines[task.to_index],
            )
            payload = {**move_meta, "score": round(move_score, 6), "graph_cached": True}
            result.edge.plan.metadata = {
                **result.edge.plan.metadata,
                "dj_move_opportunity": payload,
            }
            for alternative in result.edge.alternatives:
                alternative.metadata = {**alternative.metadata, "dj_move_opportunity": payload}
        graph[key] = result.edge
        active_session.store(key, result.evidence_matrix)

    active_session.graph_pair_count = len(active_session.pair_evidence)
    active_session.graph_seconds += time.perf_counter() - started
    return graph


def _order_score(
    order: list[int],
    graph: dict[tuple[int, int], TransitionEdge],
    analyses: list[TrackAnalysis],
    *,
    journey: JourneyMode,
    listener_profile: ListenerProfile | None = None,
    blocked_pairs: set[tuple[int, int]] | None = None,
) -> float:
    if len(order) < 2:
        return 0.0
    blocked = blocked_pairs or set()
    if any((a, b) in blocked for a, b in zip(order, order[1:])):
        return -1.0e12
    total = sum(
        _edge_objective(
            graph[(a, b)].plan,
            analyses[a],
            analyses[b],
            journey=journey,
            listener_profile=listener_profile,
        )
        for a, b in zip(order, order[1:])
    )
    # SAUND 1.4: a route is not good merely because its tracks are
    # individually desirable.  Every edge must offer a plausible phrase-level
    # DJ move.  Reward the average opportunity and explicitly protect the
    # weakest edge so a locally excellent first choice cannot create a musical
    # dead-end one track later.
    move_cfg = MoveBrainConfig.from_environment()
    if move_cfg.enabled:
        move_scores: list[float] = []
        for a, b in zip(order, order[1:]):
            cached = graph[(a, b)].plan.metadata.get("dj_move_opportunity", {})
            if isinstance(cached, dict) and cached.get("score") is not None:
                try:
                    move_scores.append(float(cached["score"]))
                    continue
                except (TypeError, ValueError):
                    pass
            move_scores.append(move_opportunity_score(analyses[a], analyses[b])[0])
        if move_scores:
            centered = sum(score - 0.5 for score in move_scores)
            weakest = min(move_scores) - 0.5
            total += move_cfg.order_weight * (0.74 * centered + 0.26 * weakest)
    energies = [analyses[index].energy for index in order]
    bpms = [analyses[index].bpm for index in order]
    if journey == "build":
        positive = sum(max(0.0, b - a) for a, b in zip(energies, energies[1:]))
        severe_drops = sum(max(0.0, a - b - 0.18) for a, b in zip(energies, energies[1:]))
        total += 0.08 * positive - 0.12 * severe_drops
    elif journey == "smooth":
        total -= 0.07 * sum(max(0.0, abs(b - a) - 0.18) for a, b in zip(energies, energies[1:]))
        total -= 0.015 * sum(max(0.0, abs(b - a) - 8.0) / 8.0 for a, b in zip(bpms, bpms[1:]))
    total += 0.55 * curve_fit_score(order, analyses, journey=journey)

    # Set-state contextual scoring: evaluate each candidate in the context of
    # musical history, not only as an isolated A->B edge. This adds a bounded
    # multi-dimensional story score (energy/tension/groove/vocals/brightness/
    # novelty) while the directed edge graph remains the hard local backbone.
    prefix: list[TrackAnalysis] = []
    prefix_transitions: list[TransitionPlan] = []
    context_total = 0.0
    denominator = max(1, len(order) - 1)
    for position, track_index in enumerate(order):
        candidate = analyses[track_index]
        if prefix:
            state = build_set_state(prefix, prefix_transitions)
            progress = position / denominator
            context_score, _ = next_track_context_score(
                state,
                candidate,
                journey=journey,
                progress=progress,
            )
            set_intent = direct_set_intent(journey, progress, state)
            director_score, _ = intent_track_score(set_intent, state, candidate)
            context_total += 0.72 * context_score + 0.28 * director_score - 0.5
            previous_index = order[position - 1]
            prefix_transitions.append(graph[(previous_index, track_index)].plan)
        prefix.append(candidate)
    total += 0.24 * context_total

    if listener_profile is not None:
        total += listener_profile.confidence * listener_profile.journey_bias.get(journey, 0.0)
    return float(total)


def _local_improve(
    order: list[int],
    graph: dict[tuple[int, int], TransitionEdge],
    analyses: list[TrackAnalysis],
    *,
    journey: JourneyMode,
    listener_profile: ListenerProfile | None,
    start: int | None,
    end: int | None,
    blocked_pairs: set[tuple[int, int]] | None = None,
    max_passes: int = 8,
) -> list[int]:
    blocked = blocked_pairs or set()
    return _route_local_improve(
        order,
        score_order=lambda candidate: _order_score(
            candidate,
            graph,
            analyses,
            journey=journey,
            listener_profile=listener_profile,
            blocked_pairs=blocked,
        ),
        start=start,
        end=end,
        max_passes=max_passes,
    )


def optimize_order(
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    journey: JourneyMode = "balanced",
    listener_profile: ListenerProfile | None = None,
    blocked_pairs: set[tuple[int, int]] | None = None,
) -> list[int]:
    blocked = blocked_pairs or set()
    return optimize_order_candidates(
        analyses,
        graph,
        score_order=lambda candidate: _order_score(
            candidate,
            graph,
            analyses,
            journey=journey,
            listener_profile=listener_profile,
            blocked_pairs=blocked,
        ),
        start_index=start_index,
        end_index=end_index,
        journey=journey,
        blocked_pairs=blocked,
    )


class SparseRouteExhaustedError(RuntimeError):
    """Sparse retrieval could not construct a complete route.

    Diagnostics are deliberately attached to the failure so Phase16 and other
    research harnesses retain useful evidence even when no SetPlan is emitted.
    """

    def __init__(self, message: str, *, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class _BeamRouteState:
    order: tuple[int, ...]
    plans: tuple[TransitionPlan, ...]
    decisions: tuple[CompleteMoveDecision, ...]
    dj_state: DJState
    score_total: float


def _build_filtered_beam_route(
    opener: int | Sequence[int],
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    end_index: int | None,
    journey: JourneyMode,
    blocked_pairs: set[tuple[int, int]],
    config: CompleteMoveConfig,
    ensure_pairs: Callable[[Iterable[tuple[int, int]]], None] | None,
    timing_intent: SetTimingIntent | None = None,
) -> _CompleteMoveRoute:
    """Search several model-filtered partial routes instead of greedily committing.

    This is used only by the explicit neural-retrieval experiment. Retrieval
    stays cheap and broad, while expensive graph materialization is batched once
    per beam depth across all surviving partial routes.
    """
    opener_values = (
        [int(opener)]
        if isinstance(opener, int)
        else [int(value) for value in opener]
    )
    if not opener_values:
        return _CompleteMoveRoute(
            order=(),
            decisions=(),
            score=0.0,
            completed=False,
            reason="no opener candidates",
        )

    beam_width = max(
        2,
        min(
            8,
            int(os.environ.get("DJAI_NEURAL_ROUTE_BEAM_WIDTH", "4")),
        ),
    )
    branch_width = max(2, min(6, int(config.search_branch_width)))
    beam: list[_BeamRouteState] = []
    for opener_index in opener_values:
        initial_state = build_set_state([analyses[opener_index]])
        initial_dj = build_dj_state(
            analyses[opener_index],
            source_cursor=0.0,
            effective_bpm=analyses[opener_index].bpm,
            render_rate=1.0,
            set_state=initial_state,
        )
        beam.append(
            _BeamRouteState(
                order=(opener_index,),
                plans=(),
                decisions=(),
                dj_state=initial_dj,
                score_total=0.0,
            )
        )

    for _step in range(1, len(analyses)):
        expanded: list[_BeamRouteState] = []
        prepared_states: list[tuple[Any, ...]] = []
        pair_batch: set[tuple[int, int]] = set()

        # First pass: do only cheap route-state/retrieval work. Collect every
        # shortlisted pair needed by the whole beam before materializing graph
        # edges. This avoids repeated graph setup for each beam state.
        for state in beam:
            current = state.order[-1]
            remaining = set(range(len(analyses))) - set(state.order)
            if not remaining:
                expanded.append(state)
                continue
            available = sorted(remaining)
            if end_index is not None and end_index in available and len(available) > 1:
                available = [index for index in available if index != end_index]
            if not available:
                continue

            fallback_progress = len(state.order) / max(1, len(analyses) - 1)
            timing_state = build_timing_state(
                [analyses[index] for index in state.order],
                list(state.plans),
                intent=timing_intent,
                fallback_progress=fallback_progress,
            )
            set_state = build_set_state(
                [analyses[index] for index in state.order],
                list(state.plans),
            )
            set_state.elapsed_seconds = timing_state.elapsed_seconds
            retrieval = retrieve_candidate_tracks(
                current_index=current,
                candidate_indices=available,
                analyses=analyses,
                set_state=set_state,
                journey=journey,
                progress=timing_state.progress,
                width=min(len(available), config.lazy_candidate_width),
            )
            shortlisted = list(retrieval.selected)
            prepared_states.append(
                (state, current, timing_state, set_state, retrieval, shortlisted)
            )
            pair_batch.update((current, index) for index in shortlisted)

        if ensure_pairs is not None and pair_batch:
            ensure_pairs(pair_batch)

        # Second pass: score complete moves only after all graph edges for this
        # beam depth are available. Safety/executability gates are unchanged.
        for state, current, timing_state, set_state, retrieval, shortlisted in prepared_states:
            generated = generate_complete_moves(
                current_index=current,
                candidate_indices=shortlisted,
                analyses=analyses,
                graph=graph,
                dj_state=state.dj_state,
                set_state=set_state,
                journey=journey,
                progress=timing_state.progress,
                blocked_pairs=blocked_pairs,
                config=config,
            )
            if not generated:
                continue

            retrieval_by_index = {item.index: item for item in retrieval.ranked}
            target_best: dict[int, tuple[float, Any]] = {}
            for candidate in generated:
                retrieved = retrieval_by_index.get(candidate.next_index)
                survivability = (
                    float(retrieved.components.get("future_survivability", 0.5))
                    if retrieved is not None
                    else 0.5
                )
                route_score = float(candidate.selection_score) + 0.12 * (survivability - 0.5)
                previous = target_best.get(candidate.next_index)
                if previous is None or route_score > previous[0]:
                    target_best[candidate.next_index] = (route_score, candidate)

            branch = sorted(
                target_best.values(),
                key=lambda item: (
                    item[0],
                    float(item[1].move.future_optionality),
                    -int(item[1].next_index),
                ),
                reverse=True,
            )[:branch_width]

            for route_score, candidate in branch:
                retrieved = retrieval_by_index.get(candidate.next_index)
                survivability = (
                    float(retrieved.components.get("future_survivability", 0.5))
                    if retrieved is not None
                    else 0.5
                )
                timing_adjustment, timing_evidence = timing_candidate_adjustment(
                    timing_state,
                    analyses[candidate.next_index],
                )
                step_score = route_score + timing_adjustment
                candidate.plan.metadata = {
                    **candidate.plan.metadata,
                    "set_timing": {
                        **timing_state.to_dict(),
                        "candidate_adjustment": round(timing_adjustment, 6),
                        "candidate_evidence": timing_evidence,
                    },
                    "route_survivability": round(survivability, 6),
                    "neural_route_beam": True,
                }
                candidate.components["future_survivability"] = round(survivability, 6)
                candidate.components["route_survivability_adjustment"] = round(
                    0.12 * (survivability - 0.5),
                    6,
                )
                projected = candidate.move.projected_state
                next_order = state.order + (candidate.next_index,)
                next_plans = state.plans + (candidate.plan,)
                next_set_state = build_set_state(
                    [analyses[index] for index in next_order],
                    list(next_plans),
                )
                next_dj_state = build_dj_state(
                    analyses[candidate.next_index],
                    source_cursor=projected.source_cursor_seconds,
                    effective_bpm=projected.effective_bpm,
                    render_rate=projected.render_rate,
                    set_state=next_set_state,
                    tempo_owner=projected.tempo_owner,
                )
                decision = CompleteMoveDecision(
                    current_index=current,
                    winner=candidate,
                    candidates=tuple(generated),
                    search_score=round(step_score, 6),
                    lookahead_depth=1,
                    projected_path=(candidate.next_index,),
                    retrieval=retrieval,
                    commit_score=round(step_score, 6),
                )
                expanded.append(
                    _BeamRouteState(
                        order=next_order,
                        plans=next_plans,
                        decisions=state.decisions + (decision,),
                        dj_state=next_dj_state,
                        score_total=state.score_total + step_score,
                    )
                )

        if not expanded:
            break
        expanded.sort(
            key=lambda item: (
                len(item.order),
                item.score_total / max(1, len(item.decisions)),
                sum(decision.winner.move.future_optionality for decision in item.decisions),
                tuple(-index for index in item.order),
            ),
            reverse=True,
        )
        beam = expanded[:beam_width]

    completed = [
        state for state in beam
        if len(state.order) == len(analyses)
        and (end_index is None or state.order[-1] == end_index)
    ]
    if not completed:
        best_partial = max(
            beam,
            key=lambda item: (
                len(item.order),
                item.score_total / max(1, len(item.decisions)),
            ),
        )
        return _CompleteMoveRoute(
            order=best_partial.order,
            decisions=best_partial.decisions,
            score=best_partial.score_total / max(1, len(best_partial.decisions)),
            completed=False,
            reason=(
                f"filtered beam search exhausted after {len(best_partial.order)}/"
                f"{len(analyses)} tracks"
            ),
        )
    best = max(
        completed,
        key=lambda item: (
            item.score_total / max(1, len(item.decisions)),
            sum(decision.winner.move.future_optionality for decision in item.decisions),
            tuple(-index for index in item.order),
        ),
    )
    return _CompleteMoveRoute(
        order=best.order,
        decisions=best.decisions,
        score=best.score_total / max(1, len(best.decisions)),
        completed=True,
        reason="",
    )


def _large_crate_complete_move_config(
    config: CompleteMoveConfig,
    track_count: int,
    *,
    experimental_retrieval: bool | None = None,
) -> CompleteMoveConfig:
    """Bound structural search cost for large crates without touching safety gates.

    During the explicit neural-retrieval experiment, caller-provided widening
    tiers are intentional and must survive this helper. Normal production
    planning keeps the historical conservative caps.
    """
    if track_count < _large_set_threshold():
        return config
    if experimental_retrieval is None:
        experimental_retrieval = os.environ.get(
            "DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    if experimental_retrieval:
        return replace(
            config,
            lookahead_moves=min(config.lookahead_moves, 2),
            search_branch_width=min(config.search_branch_width, 4),
            lazy_candidate_width=min(config.lazy_candidate_width, max(6, track_count - 1)),
            lazy_expansion_width=min(config.lazy_expansion_width, max(4, track_count - 1)),
            opener_candidates=min(config.opener_candidates, 4),
            candidates_per_track=min(config.candidates_per_track, 4),
        )
    return replace(
        config,
        lookahead_moves=min(config.lookahead_moves, 2),
        search_branch_width=min(config.search_branch_width, 2),
        lazy_candidate_width=min(config.lazy_candidate_width, 6),
        lazy_expansion_width=min(config.lazy_expansion_width, 6),
        opener_candidates=min(config.opener_candidates, 2),
        candidates_per_track=min(config.candidates_per_track, 3),
    )


def _route_search_signature(
    config: CompleteMoveConfig,
    track_count: int,
    *,
    experimental_retrieval: bool | None = None,
) -> tuple[int, ...]:
    effective = _large_crate_complete_move_config(
        config,
        track_count,
        experimental_retrieval=experimental_retrieval,
    )
    return (
        int(effective.lookahead_moves),
        int(effective.search_branch_width),
        int(effective.lazy_candidate_width),
        int(effective.lazy_expansion_width),
        int(effective.opener_candidates),
        int(effective.candidates_per_track),
    )


def _progressive_sparse_configs(
    config: CompleteMoveConfig,
    track_count: int,
    *,
    experimental_retrieval: bool,
) -> tuple[CompleteMoveConfig, ...]:
    """Return only widening configs that materially change effective search.

    Large-crate caps can make nominal tiers equivalent. Re-running an
    equivalent tier repeats expensive sparse graph work without increasing
    route coverage.
    """
    requested_tiers = (
        ((8, 3, 3), (10, 4, 3), (12, 4, 4))
        if experimental_retrieval
        else ((8, 4, 3),)
    )
    seen_signatures = {
        _route_search_signature(
            config,
            track_count,
            experimental_retrieval=experimental_retrieval,
        )
    }
    result: list[CompleteMoveConfig] = []
    for candidate_width, opener_count, branch_width in requested_tiers:
        candidate_cfg = replace(
            config,
            opener_candidates=min(
                track_count,
                max(config.opener_candidates, opener_count),
            ),
            search_branch_width=min(
                candidate_width,
                max(config.search_branch_width, branch_width),
            ),
            lazy_candidate_width=min(
                max(1, track_count - 1),
                max(config.lazy_candidate_width, candidate_width),
            ),
            lazy_expansion_width=min(
                max(2, track_count - 1),
                max(config.lazy_expansion_width, candidate_width),
            ),
        )
        signature = _route_search_signature(
            candidate_cfg,
            track_count,
            experimental_retrieval=experimental_retrieval,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        result.append(candidate_cfg)
    return tuple(result)


def build_complete_move_route(
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    journey: JourneyMode = "balanced",
    blocked_pairs: set[tuple[int, int]] | None = None,
    config: CompleteMoveConfig | None = None,
    headphones_config: AIHeadphonesConfig | None = None,
    listener: str | None = None,
    ensure_pairs: Callable[[Iterable[tuple[int, int]]], None] | None = None,
    timing_intent: SetTimingIntent | None = None,
) -> _CompleteMoveRoute:
    """Build a route with lazy retrieval and 2-3 complete-move lookahead.

    Search advances only the winning first move in a non-authoritative projected
    route, then replans from that simulated DJState. DJ Executive V2 owns final
    commitment after exact program preparation. ``ensure_pairs`` lets authoritative mode
    materialize expensive transition edges only when search actually reaches
    that neighbourhood.
    """
    cfg = _large_crate_complete_move_config(
        config or CompleteMoveConfig.from_environment(),
        len(analyses),
    )
    headphones_cfg = headphones_config or AIHeadphonesConfig.from_environment()
    blocked = blocked_pairs or set()
    if len(analyses) < 2:
        raise ValueError("A complete-move route needs at least two tracks.")
    if start_index is not None:
        openers = [start_index]
    else:
        opener_pool = [index for index in range(len(analyses)) if index != end_index]
        empty_state = SetState()
        ranked_openers = sorted(
            opener_pool,
            key=lambda index: (
                next_track_context_score(
                    empty_state, analyses[index], journey=journey, progress=0.0
                )[0],
                -index,
            ),
            reverse=True,
        )
        openers = ranked_openers[: min(len(ranked_openers), cfg.opener_candidates)]
    # Opener/root comparison remains structural and cheap. Rendering 2-4 private
    # rehearsals at every step under every possible opener would multiply audio
    # work before the system has even chosen which route to commit. Once the
    # strongest opener route is known, rebuild that one projected route with
    # Headphones at each current-state search boundary.
    scouting_headphones = (
        replace(headphones_cfg, enabled=False)
        if cfg.mode == "move_authoritative" and headphones_cfg.enabled and len(openers) > 1
        else headphones_cfg
    )
    experimental_retrieval = os.environ.get(
        "DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if experimental_retrieval and cfg.mode == "move_authoritative":
        # Compare opener candidates inside one shared beam. Running a complete
        # beam independently for each opener multiplies sparse graph work while
        # preventing candidate-pair batching across opener hypotheses.
        routes = [
            _build_filtered_beam_route(
                tuple(openers),
                analyses,
                graph,
                end_index=end_index,
                journey=journey,
                blocked_pairs=blocked,
                config=cfg,
                ensure_pairs=ensure_pairs,
                timing_intent=timing_intent,
            )
        ]
    else:
        routes = [
            _complete_move_route_from_opener(
                opener,
                analyses,
                graph,
                end_index=end_index,
                journey=journey,
                blocked_pairs=blocked,
                config=cfg,
                headphones_config=scouting_headphones,
                listener=listener,
                ensure_pairs=ensure_pairs,
                timing_intent=timing_intent,
                listen_to_moves=listen_to_complete_moves,
            )
            for opener in openers
        ]
    complete = [route for route in routes if route.completed]
    if not complete:
        reason = "; ".join(route.reason for route in routes if route.reason) or "no complete route"
        partial = max(
            routes,
            key=lambda route: (
                len(route.order),
                route.score,
                sum(decision.winner.move.future_optionality for decision in route.decisions),
            ),
        )
        return _CompleteMoveRoute(
            order=partial.order,
            decisions=partial.decisions,
            score=partial.score,
            completed=False,
            reason=reason,
        )
    best_scout = max(
        complete,
        key=lambda route: (
            route.score,
            sum(decision.winner.move.future_optionality for decision in route.decisions),
            tuple(-index for index in route.order),
        ),
    )
    if experimental_retrieval:
        # Beam decisions are structural preferences only. Final compilation,
        # professional rehearsal, rendered judging, and safety still decide
        # whether each edge may commit.
        return best_scout
    if scouting_headphones.enabled == headphones_cfg.enabled:
        return best_scout
    committed = _complete_move_route_from_opener(
        best_scout.order[0],
        analyses,
        graph,
        end_index=end_index,
        journey=journey,
        blocked_pairs=blocked,
        config=cfg,
        headphones_config=headphones_cfg,
        listener=listener,
        ensure_pairs=ensure_pairs,
        timing_intent=timing_intent,
        listen_to_moves=listen_to_complete_moves,
    )
    return committed if committed.completed else best_scout

def _sequence_exit_candidates(
    track: TrackAnalysis,
    *,
    source_cursor: float,
    current_rate: float,
    required_source: float,
    dj_state: DJState | None = None,
    timeline: MusicalTimeline | None = None,
    limit: int = 10,
) -> tuple[list[TransitionCandidate], float, float]:
    return _sequence_exit_candidates_impl(
        track,
        source_cursor=source_cursor,
        current_rate=current_rate,
        required_source=required_source,
        dj_state=dj_state,
        timeline=timeline,
        limit=limit,
        candidate_diverse_by_bar=_candidate_diverse_by_bar,
    )


def _sequence_entry_candidates(
    track: TrackAnalysis,
    *,
    limit: int = 10,
) -> list[TransitionCandidate]:
    return _sequence_entry_candidates_impl(
        track,
        limit=limit,
        candidate_diverse_by_bar=_candidate_diverse_by_bar,
    )

def _compile_sequence_edges(
    order: list[int],
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    minimum_body_seconds: float,
    minimum_body_fraction: float,
    session: PlanningSession | None = None,
    quality_loop: QualityLoopConfig | None = None,
    automatic_delivery: bool = False,
    listener_profile: ListenerProfile | None = None,
    progress: ProgressCallback | None = None,
    replan_on_review: bool = True,
    allow_emergency_escape: bool = False,
    preferred_complete_moves: dict[tuple[int, int], CompleteMoveDecision] | None = None,
    initial_source_cursor: float = 0.0,
    initial_rate: float = 1.0,
    initial_effective_bpm: float | None = None,
    history_order: Sequence[int] | None = None,
    history_plans: Sequence[TransitionPlan] | None = None,
    position_offset: int = 0,
    total_track_count: int | None = None,
    transition_policy_authority: str = "observe",
    crowd_directive: CrowdDirective | None = None,
) -> list[TransitionEdge]:
    """Compile path-dependent plans while reusing graph-stage cue evidence.

    The optional initial/history arguments are used by the live Executive when
    compiling a FUTURE suffix from the exact post-NEXT deck state. Defaults
    preserve the prepared-set behavior byte-for-byte.
    """
    del graph
    if not order:
        return []

    active_session = session if session is not None else PlanningSession()
    compiled: list[TransitionEdge] = []
    source_cursor = max(0.0, float(initial_source_cursor))
    current_rate = max(1e-9, float(initial_rate))
    current_effective_bpm = float(
        initial_effective_bpm
        if initial_effective_bpm is not None
        else analyses[order[0]].bpm * current_rate
    )
    history_indices = [int(item) for item in (history_order or ())]
    history_plan_list = list(history_plans or ())
    if history_indices and history_indices[-1] == int(order[0]):
        history_prefix = history_indices[:-1]
    else:
        history_prefix = history_indices
    global_track_count = max(
        1,
        int(total_track_count)
        if total_track_count is not None
        else len(history_prefix) + len(order),
    )
    stem_maps = {index: find_cached_stems(track.path) for index, track in enumerate(analyses)}

    for position, (from_index, to_index) in enumerate(zip(order, order[1:])):
        global_position = max(0, int(position_offset) + position)
        journey_progress = global_position / max(1, global_track_count - 1)
        track_a = analyses[from_index]
        track_b = analyses[to_index]
        edge_render_context = (
            TransitionRenderContext(
                sample_rate=quality_loop.sample_rate,
                gain_a=professional_normalization_gain(integrated_lufs=track_a.integrated_lufs, true_peak_dbtp=track_a.true_peak_dbtp, legacy_loudness_dbfs=track_a.loudness_dbfs),
                gain_b=professional_normalization_gain(integrated_lufs=track_b.integrated_lufs, true_peak_dbtp=track_b.true_peak_dbtp, legacy_loudness_dbfs=track_b.loudness_dbfs),
            )
            if quality_loop is not None and quality_loop.enabled
            else None
        )
        required_source = _required_source_body(
            track_a,
            source_cursor,
            current_rate,
            minimum_body_seconds=minimum_body_seconds,
            minimum_body_fraction=minimum_body_fraction,
        )
        required_exit = source_cursor + required_source
        prefix_tracks = [analyses[index] for index in history_prefix + order[: position + 1]]
        move_set_state = build_set_state(
            prefix_tracks,
            history_plan_list + [edge.plan for edge in compiled],
        )
        move_timeline_a = build_musical_timeline(track_a)
        source_bar = move_timeline_a.bar_for_time(source_cursor)
        song_form_safe_bar = earliest_safe_song_form_exit_bar(move_timeline_a, source_bar)
        song_form_safe_time = move_timeline_a.bar_start_seconds(song_form_safe_bar)
        if song_form_safe_time > required_exit:
            required_source = max(required_source, song_form_safe_time - source_cursor)
            required_exit = source_cursor + required_source
        dj_state = build_dj_state(
            track_a,
            source_cursor=source_cursor,
            effective_bpm=current_effective_bpm,
            render_rate=current_rate,
            set_state=move_set_state,
            timeline=move_timeline_a,
        )
        selected_exits, target_exit, soft_latest_exit = _sequence_exit_candidates(
            track_a,
            source_cursor=source_cursor,
            current_rate=current_rate,
            required_source=required_source,
            dj_state=dj_state,
            timeline=move_timeline_a,
            limit=10,
        )
        selected_entries = _sequence_entry_candidates(track_b, limit=10)

        candidate_pool = _build_sequence_candidate_pool(
            from_index=from_index,
            to_index=to_index,
            track_a=track_a,
            track_b=track_b,
            selected_exits=selected_exits,
            selected_entries=selected_entries,
            session=active_session,
            stem_paths_a=stem_maps[from_index],
            stem_paths_b=stem_maps[to_index],
            preferred_style=preferred_style,
            quality_loop=quality_loop,
            tempo_policy=tempo_policy,
            current_effective_bpm=current_effective_bpm,
            current_rate=current_rate,
            global_position=global_position,
            source_cursor=source_cursor,
            required_source=required_source,
            target_exit=target_exit,
            soft_latest_exit=soft_latest_exit,
        )
        all_plans = list(candidate_pool.plans)
        matrix = candidate_pool.matrix
        sequence_evidence_reused = candidate_pool.sequence_evidence_reused
        stems_available = candidate_pool.stems_available

        if not all_plans:
            if automatic_delivery and replan_on_review and not allow_emergency_escape:
                raise TransitionReplanRequired(
                    from_index,
                    to_index,
                    action="replan_track",
                    reason="No technique-feasible candidate survived source/body/cue constraints; automatic mode must replan rather than expose a blind fade.",
                    resolution={"action": "replan_track", "reason": "no_candidate_pool"},
                    compiled_prefix=list(compiled),
                )
            feasible_downbeats = [
                value for value in track_a.downbeats_seconds
                if max(source_cursor + 0.15, required_exit) <= value <= track_a.duration_seconds - 0.10
            ]
            if automatic_delivery:
                emergency = build_last_resort_plan(
                    track_a,
                    track_b,
                    effective_bpm_a=current_effective_bpm,
                    render_rate_a=current_rate,
                    source_cursor=source_cursor,
                    preferred_exit_time=required_exit,
                    reason=(
                        "No technique-feasible plan remained after preserving the minimum audible track body."
                    ),
                )
            else:
                if not feasible_downbeats:
                    raise RuntimeError(
                        f"Track '{track_a.title}' cannot support another transition after its consumed source position."
                    )
                emergency = plan_transition(
                    track_a,
                    track_b,
                    force=True,
                    tempo_policy="cut",
                    style="emergency",
                    requested_bars=1,
                    exit_time_a=feasible_downbeats[-1],
                    effective_bpm_a=current_effective_bpm,
                    render_rate_a=current_rate,
                    stems_available=stems_available,
                )
            emergency.metadata = {
                **emergency.metadata,
                "sequence_position": global_position,
                "source_cursor_before_body": round(source_cursor, 6),
                "minimum_source_body": round(required_source, 6),
                "last_resort": True,
                "automatic_delivery": automatic_delivery,
            }
            all_plans = [emergency]

        preferred_complete_decision = (
            preferred_complete_moves.get((from_index, to_index))
            if preferred_complete_moves is not None
            else None
        )
        all_plans = _prepare_sequence_candidates(
            all_plans,
            track_a=track_a,
            track_b=track_b,
            dj_state=dj_state,
            preferred_complete_decision=preferred_complete_decision,
            selection_utility=lambda candidate: _edge_objective(
                candidate,
                track_a,
                track_b,
                journey=journey,
                listener_profile=listener_profile,
            ),
            rank_as_moves=lambda left, right, candidates, state: rank_transition_plans_as_moves(
                left,
                right,
                candidates,
                state=state,
            ),
            state_taste_payload=state_taste_payload,
            move_taste_payload=move_taste_payload,
            state_context_id=state_context_id,
        )

        policy_context = _apply_sequence_transition_policy(
            all_plans,
            journey=journey,
            journey_progress=journey_progress,
            tempo_policy=tempo_policy,
            current_effective_bpm=current_effective_bpm,
            current_rate=current_rate,
            stems_available=stems_available,
            authority=transition_policy_authority,
            crowd_directive=crowd_directive,
            apply_bounded_policy=apply_bounded_transition_policy,
        )

        measurement_summary: dict[str, object] | None = None
        if quality_loop is not None and quality_loop.enabled:
            if progress:
                progress(
                    "measure",
                    position,
                    max(1, global_track_count - 1),
                    f"{track_a.title} -> {track_b.title}",
                )
            if automatic_delivery:
                brain_enabled = os.environ.get("DJAI_PROFESSIONAL_BRAIN", "1").strip().lower() not in {"0", "false", "no", "off"}
                if brain_enabled:
                    if matrix is None or not matrix.pairs:
                        if replan_on_review and not allow_emergency_escape:
                            raise TransitionReplanRequired(
                                from_index,
                                to_index,
                                action="replan_track",
                                reason="No pair-evidence matrix remained for a rehearsed automatic transition.",
                                resolution={"action": "replan_track", "reason": "no_evidence_matrix"},
                                compiled_prefix=list(compiled),
                            )
                        chosen = build_last_resort_plan(
                            track_a,
                            track_b,
                            effective_bpm_a=current_effective_bpm,
                            render_rate_a=current_rate,
                            source_cursor=source_cursor,
                            preferred_exit_time=required_exit,
                            reason="No evidence matrix remained after global replanning; deterministic final escape used.",
                            render_context=edge_render_context,
                            verify_audio=True,
                        )
                        measurement_summary = {
                            "action": "emergency_escape",
                            "fallback_used": True,
                            "reason": "no_evidence_matrix",
                        }
                    else:
                        from .brain.unified_transition import resolve_emergency_escape, resolve_transition

                        state = move_set_state
                        thinking_budget = (
                            preferred_complete_decision.thinking_budget
                            if preferred_complete_decision is not None
                            else None
                        )
                        brain_cfg = _professional_brain_config_for_transition(
                            quality_loop, preferred_complete_decision
                        )
                        resolution = resolve_transition(
                            track_a,
                            track_b,
                            evidence_matrix=matrix,
                            professional_config=brain_cfg,
                            tempo_policy=tempo_policy,
                            preferred_style=preferred_style,
                            effective_bpm_a=current_effective_bpm,
                            render_rate_a=current_rate,
                            stems_available=stems_available,
                            set_state=state,
                            dj_state=dj_state,
                            journey=journey,
                            journey_progress=journey_progress,
                            listener=listener_profile.listener if listener_profile is not None else None,
                            measurement_log_path=quality_loop.log_path,
                            transition_policy_context=(
                                policy_context if transition_policy_authority == "bounded_advisory" else None
                            ),
                            crowd_directive=(
                                crowd_directive if transition_policy_authority == "bounded_advisory" else None
                            ),
                            context={
                                "mode": "set_sequence",
                                "sequence_position": global_position,
                                "from_index": from_index,
                                "to_index": to_index,
                                "journey": journey,
                                "preferred_style": preferred_style,
                                "adaptive_thinking": (
                                    thinking_budget.to_dict() if thinking_budget is not None else None
                                ),
                            },
                            render_context=edge_render_context,
                        )
                        if resolution.accepted:
                            assert resolution.selected is not None
                            if (
                                automatic_delivery
                                and replan_on_review
                                and not allow_emergency_escape
                                and resolution.selected.transition_mode == "emergency_cut"
                            ):
                                raise TransitionReplanRequired(
                                    from_index,
                                    to_index,
                                    action="replan_track",
                                    reason=(
                                        "Professional rehearsal found only a structural reset for this pair; "
                                        "offline automatic delivery must route through another track before using emergency escape."
                                    ),
                                    resolution={
                                        **resolution.to_dict(),
                                        "reason": "emergency_cut_not_production_route",
                                    },
                                    compiled_prefix=list(compiled),
                                )
                            chosen = resolution.selected
                            measurement_summary = resolution.to_dict()
                        elif replan_on_review and not allow_emergency_escape:
                            raise TransitionReplanRequired(
                                from_index,
                                to_index,
                                action=resolution.action,
                                reason=(
                                    resolution.reasons[-1]
                                    if resolution.reasons
                                    else "Rendered transition did not clear the professional acceptance gate."
                                ),
                                resolution=resolution.to_dict(),
                                compiled_prefix=list(compiled),
                            )
                        else:
                            emergency = resolve_emergency_escape(
                                track_a,
                                track_b,
                                evidence_matrix=matrix,
                                professional_config=brain_cfg,
                                effective_bpm_a=current_effective_bpm,
                                render_rate_a=current_rate,
                                stems_available=stems_available,
                                set_state=state,
                                journey=journey,
                                journey_progress=journey_progress,
                                listener=listener_profile.listener if listener_profile is not None else None,
                                measurement_log_path=quality_loop.log_path,
                                context={
                                    "mode": "set_sequence_final_escape",
                                    "sequence_position": global_position,
                                    "from_index": from_index,
                                    "to_index": to_index,
                                },
                                render_context=edge_render_context,
                            )
                            if emergency.accepted:
                                assert emergency.selected is not None
                                chosen = emergency.selected
                                measurement_summary = emergency.to_dict()
                            else:
                                chosen = build_last_resort_plan(
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
                                    render_context=edge_render_context,
                                    verify_audio=True,
                                )
                                chosen.metadata = {
                                    **chosen.metadata,
                                    "unified_transition_final_escape": True,
                                    "emergency_resolution": emergency.to_dict(),
                                }
                                measurement_summary = {
                                    **emergency.to_dict(),
                                    "deterministic_final_escape": True,
                                }
                else:
                    scores = {
                        plan.execution_signature: float(plan.metadata.get("selection_utility", plan.score))
                        for plan in all_plans
                    }
                    chosen = select_automatic_transition(
                        track_a, track_b, all_plans, analytical_scores=scores, quality_config=quality_loop,
                        context={"sequence_position": global_position, "from_index": from_index, "to_index": to_index, "journey": journey, "preferred_style": preferred_style},
                        render_context=edge_render_context,
                        source_cursor=source_cursor, preferred_exit_time=required_exit, effective_bpm_a=current_effective_bpm, render_rate_a=current_rate,
                    )
                    measurement_summary = {
                        "automatic_delivery": True, "fallback_level": chosen.fallback_level,
                        "selected_measured_score": chosen.measured_score, "selected_final_score": chosen.score,
                        "gate_passed": chosen.measurement_passed,
                    }
            else:
                measured = measure_and_select_transition(
                    track_a,
                    track_b,
                    all_plans,
                    config=quality_loop,
                    context={
                        "sequence_position": global_position,
                        "from_index": from_index,
                        "to_index": to_index,
                        "journey": journey,
                        "preferred_style": preferred_style,
                    },
                    render_context=edge_render_context,
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
                "selected_entry_pool": [round(item.time_seconds, 6) for item in selected_entries],
                "sequence_evidence_reused": sequence_evidence_reused,
                "sequence_matrix_size": (matrix.compatibility_evaluations if matrix is not None else 0),
                "quality_loop_summary": measurement_summary,
            }
            if progress:
                progress(
                    "measure",
                    position + 1,
                    max(1, global_track_count - 1),
                    f"{track_a.title} -> {track_b.title}: {chosen.measured_score or 0.0:.0%}",
                )
        else:
            if preferred_complete_decision is not None:
                chosen = max(
                    all_plans,
                    key=lambda candidate: (
                        float(candidate.metadata.get("selection_utility", candidate.score)),
                        float(candidate.score),
                        candidate.execution_signature,
                    ),
                )
            else:
                chosen, _ = _select_alternative(
                    all_plans,
                    track_a,
                    track_b,
                    preferred_style=preferred_style,
                    journey=journey,
                    listener_profile=listener_profile,
                )

        if "dj_move" not in chosen.metadata:
            chosen = rank_transition_plans_as_moves(
                track_a, track_b, [chosen], state=dj_state
            )[0]
        chosen.metadata = {
            **chosen.metadata,
            "dj_state_before_move": dj_state.to_dict(),
        }

        if automatic_delivery:
            allow_hard_cuts = os.environ.get("DJAI_ALLOW_HARD_CUTS", "0").strip().lower() in {"1", "true", "yes", "on"}
            deck_performance = dict(chosen.metadata.get("deck_performance", {}) or {})
            phase_integrity = dict(deck_performance.get("phase_integrity", {}) or {})
            if chosen.beatmatched and phase_integrity and not bool(phase_integrity.get("safe", True)):
                rejected_signature = chosen.execution_signature
                previous_move = chosen.metadata.get("dj_move")
                previous_state = chosen.metadata.get("dj_state_before_move")
                chosen = build_last_resort_plan(
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
                    render_context=edge_render_context,
                    verify_audio=bool(quality_loop is not None and quality_loop.enabled),
                )
                chosen.metadata = {
                    **chosen.metadata,
                    "beat_integrity_replanned": True,
                    "rejected_phase_integrity_signature": rejected_signature,
                    **({"dj_move": previous_move} if previous_move is not None else {}),
                    **({"dj_state_before_move": previous_state} if previous_state is not None else {}),
                }
            if not allow_hard_cuts and chosen.technique in {"quick_cut", "backspin"}:
                rejected_technique = chosen.technique
                chosen = build_last_resort_plan(
                    track_a,
                    track_b,
                    effective_bpm_a=current_effective_bpm,
                    render_rate_a=current_rate,
                    source_cursor=source_cursor,
                    preferred_exit_time=required_exit,
                    reason=(
                        f"Automatic policy re-evaluated {rejected_technique} through the compatibility-aware rescue hierarchy."
                    ),
                    render_context=edge_render_context,
                    verify_audio=bool(quality_loop is not None and quality_loop.enabled),
                )
                rescue_kind = str(chosen.metadata.get("rescue_transition_kind", "unknown"))
                chosen.metadata = {
                    **chosen.metadata,
                    "replaced_automatic_cut": rejected_technique,
                    "continuous_mix_policy": rescue_kind == "filtered_overlap",
                    "automatic_rescue_policy": "compatibility_aware",
                }
            chosen.requires_confirmation = False
            chosen.auto_approved = True
            chosen.metadata = {**chosen.metadata, "automatic_delivery": True}

        # A deterministic rescue may replace the candidate after graph evidence
        # informed sequence compilation. Preserve that provenance on the final
        # executable plan instead of making the replacement look evidence-free.
        chosen.metadata = {
            **chosen.metadata,
            "sequence_evidence_reused": sequence_evidence_reused,
            "sequence_matrix_size": (
                matrix.compatibility_evaluations
                if matrix is not None
                else int(chosen.metadata.get("sequence_matrix_size", 0) or 0)
            ),
        }

        # V6 connects the formal transition-policy boundary without changing
        # candidate ranking in this behavior-preserving migration. The
        # Executive owns when this policy is consulted; the selected program
        # remains the exact program already chosen by the current brain.
        annotate_transition_policy(
            [chosen],
            context=policy_context,
            authority=transition_policy_authority,
        )

        alternatives: list[TransitionPlan] = []
        for style in ("safe", "creative", "emergency"):
            candidates = [plan for plan in all_plans if plan.style == style]
            if candidates:
                alternatives.append(
                    max(
                        candidates,
                        key=lambda value: _edge_objective(
                            value,
                            track_a,
                            track_b,
                            journey=journey,
                            listener_profile=listener_profile,
                        ),
                    )
                )
        edge = _edge_from_plan(
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
        validate_execution_plan(edge.plan)
        compiled.append(edge)
        source_cursor = edge.plan.post_transition_source_end_b
        current_rate = edge.plan.post_transition_rate_b
        current_effective_bpm = edge.plan.post_transition_bpm_b or (track_b.bpm * current_rate)
    return compiled


def estimate_set_duration(
    order: list[int],
    analyses: list[TrackAnalysis],
    transitions: list[TransitionEdge],
    *,
    tempo_policy: str = "safe",
) -> float:
    del tempo_policy  # The compiled plans already contain the exact execution state.
    if not order:
        return 0.0
    if not transitions:
        return analyses[order[0]].duration_seconds
    duration = 0.0
    source_cursor = 0.0
    current_rate = 1.0
    for edge in transitions:
        plan = edge.plan
        validate_execution_plan(plan)
        body_source = max(0.0, plan.exit_time_a - source_cursor)
        duration += body_source / max(current_rate, 1e-9)
        duration += plan.transition_seconds
        duration += max(0.0, plan.tempo_release_seconds_b)
        source_cursor = plan.post_transition_source_end_b
        current_rate = plan.post_transition_rate_b
    final_track = analyses[order[-1]]
    duration += max(0.0, final_track.duration_seconds - source_cursor) / max(current_rate, 1e-9)
    return float(duration)


def _plan_identity(
    paths: list[str],
    order: list[int],
    manual: bool,
    *,
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
) -> str:
    payload = {
        "paths": paths,
        "order": order,
        "manual": manual,
        "tempo_policy": tempo_policy,
        "preferred_style": preferred_style,
        "journey": journey,
        "version": SET_PLAN_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def confirmation_payload(
    plan_id: str,
    order: list[int],
    transitions: list[TransitionEdge],
) -> dict[str, object]:
    return {
        "plan_id": plan_id,
        "order": order,
        "qualities": [edge.quality for edge in transitions],
        "scores": [edge.plan.score for edge in transitions],
        "execution_signatures": [edge.plan.execution_signature for edge in transitions],
    }



def _build_live_complete_move_route(
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    request: LiveReplanRequest,
    history_plans: list[TransitionPlan],
    blocked_pairs: set[tuple[int, int]],
    config: CompleteMoveConfig,
    headphones_config: AIHeadphonesConfig,
    ensure_pairs: Callable[[Iterable[tuple[int, int]]], None] | None,
) -> _CompleteMoveRoute:
    """Recompute only FUTURE from the exact post-NEXT projected deck state."""

    route_order = [int(request.anchor_track)]
    route_plans: list[TransitionPlan] = []
    decisions: list[CompleteMoveDecision] = []
    remaining = set(int(item) for item in request.remaining_indices)
    current = int(request.anchor_track)
    history_order = [int(item) for item in request.history_order]
    if not history_order or history_order[-1] != current:
        raise ValueError("Live replan history must end at the locked NEXT anchor track.")
    if len(history_plans) != max(0, len(history_order) - 1):
        raise ValueError("Live replan history order/transition count is inconsistent.")

    set_state = build_set_state(
        [analyses[index] for index in history_order],
        history_plans,
    )
    dj_state = build_dj_state(
        analyses[current],
        source_cursor=request.anchor_source_cursor,
        effective_bpm=request.anchor_effective_bpm,
        render_rate=request.anchor_render_rate,
        set_state=set_state,
        tempo_owner="deck_b",
    )
    adaptive_cfg = AdaptiveThinkingConfig.from_environment()
    professional_pilot = ProfessionalBrainConfig.from_environment().adaptive_pilot_finalists
    live_cfg = LiveExecutiveConfig.from_environment()
    crowd_directive = crowd_directive_from_payload(request.crowd_directive)

    while remaining:
        played_order = history_order[:-1] + route_order
        played_plans = history_plans + route_plans
        preflight_indices = sorted(remaining)
        if request.fixed_closer is not None and request.fixed_closer in preflight_indices and len(preflight_indices) > 1:
            preflight_indices = [index for index in preflight_indices if index != request.fixed_closer]
        preflight = assess_context_preflight(
            track_a=analyses[current],
            candidate_tracks=[analyses[index] for index in preflight_indices],
            dj_state=dj_state,
            config=adaptive_cfg,
        )
        configured_depth = max(1, min(config.lookahead_moves, live_cfg.future_depth))
        initial_depth = (
            min(configured_depth, max(1, adaptive_cfg.easy_lookahead))
            if adaptive_cfg.enabled and preflight.easy_candidate_context
            else configured_depth
        )
        choice = _choose_complete_move_with_lookahead(
            current=current,
            remaining=remaining,
            analyses=analyses,
            graph=graph,
            dj_state=dj_state,
            played_order=played_order,
            played_plans=played_plans,
            journey=normalize_journey(request.journey),
            end_index=request.fixed_closer,
            blocked_pairs=blocked_pairs,
            config=config,
            ensure_pairs=ensure_pairs,
            depth=initial_depth,
            crowd_directive=crowd_directive,
        )
        if choice is None:
            return _CompleteMoveRoute(
                order=tuple(route_order),
                decisions=tuple(decisions),
                score=(
                    sum(
                        item.commit_score or item.search_score or item.winner.selection_score
                        for item in decisions
                    )
                    / max(1, len(decisions))
                ),
                completed=False,
                reason=f"no executable live complete move from track index {current}",
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
            configured_lookahead=configured_depth,
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
        deeper_target = min(configured_depth, provisional_budget.lookahead_target)
        if deeper_target > initial_depth and len(remaining) > 1:
            deeper = _choose_complete_move_with_lookahead(
                current=current,
                remaining=remaining,
                analyses=analyses,
                graph=graph,
                dj_state=dj_state,
                played_order=played_order,
                played_plans=played_plans,
                journey=normalize_journey(request.journey),
                end_index=request.fixed_closer,
                blocked_pairs=blocked_pairs,
                config=config,
                ensure_pairs=ensure_pairs,
                depth=deeper_target,
                crowd_directive=crowd_directive,
            )
            if deeper is not None:
                choice = deeper
                search_depth_used = deeper_target
                lookahead_escalated = True

        final_assessment = assess_move_brittleness(
            track_a=analyses[current],
            track_b=analyses[choice.winner.next_index],
            winner=choice.winner,
            candidates=choice.candidates,
            config=adaptive_cfg,
        )
        assessment = (
            merge_brittleness(initial_assessment, final_assessment, config=adaptive_cfg)
            if lookahead_escalated
            else final_assessment
        )
        thinking_budget = build_thinking_budget(
            assessment=assessment,
            configured_lookahead=configured_depth,
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
        commit_score = float(choice.search_score or winner.selection_score)
        headphones_report: dict[str, Any] | None = None
        if config.mode == "move_authoritative" and headphones_config.enabled:
            current_set_state = build_set_state(
                [analyses[index] for index in played_order],
                played_plans,
            )
            decision_headphones_config = replace(
                headphones_config,
                serious_pool=thinking_budget.headphones_serious_pool,
                minimum_renders=thinking_budget.headphones_minimum_renders,
                maximum_renders=thinking_budget.headphones_maximum_renders,
            ).normalized()
            headphones = listen_to_complete_moves(
                preferred=choice.winner,
                candidates=choice.candidates,
                analyses=analyses,
                set_state=current_set_state,
                journey=normalize_journey(request.journey),
                progress=len(played_order) / max(1, len(analyses) - 1),
                listener=request.listener,
                config=decision_headphones_config,
            )
            winner = headphones.winner
            headphones_report = {
                **headphones.to_dict(),
                "adaptive_thinking": thinking_budget.to_dict(),
                "live_executive_cycle": request.cycle,
            }
            if headphones.authoritative:
                commit_score = float(headphones.winner_score)
                if winner.plan.execution_signature != choice.winner.plan.execution_signature:
                    decision_lookahead_depth = 1
                    decision_projected_path = (winner.next_index,)

        if winner.plan.execution_signature != choice.winner.plan.execution_signature:
            listened_assessment = assess_move_brittleness(
                track_a=analyses[current],
                track_b=analyses[winner.next_index],
                winner=winner,
                candidates=choice.candidates,
                config=adaptive_cfg,
            )
            assessment = merge_brittleness(assessment, listened_assessment, config=adaptive_cfg)
            thinking_budget = build_thinking_budget(
                assessment=assessment,
                configured_lookahead=configured_depth,
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
            candidate_moves=tuple(candidate.move for candidate in same_target_candidates),
            candidate_plans=tuple(candidate.plan for candidate in same_target_candidates),
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
        route_order.append(winner.next_index)
        remaining.remove(winner.next_index)
        route_plans.append(winner.plan)
        current = winner.next_index
        played_order = history_order[:-1] + route_order
        played_plans = history_plans + route_plans
        set_state = build_set_state(
            [analyses[index] for index in played_order],
            played_plans,
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

    completed = len(route_order) == len(request.remaining_indices) + 1
    return _CompleteMoveRoute(
        order=tuple(route_order),
        decisions=tuple(decisions),
        score=(
            sum(
                item.commit_score or item.search_score or item.winner.selection_score
                for item in decisions
            )
            / max(1, len(decisions))
        ),
        completed=completed,
        reason="" if completed else "live future route did not consume every remaining track",
    )


def replan_live_future_candidate(
    plan: SetPlan,
    request: LiveReplanRequest,
    *,
    history_transitions: Sequence[TransitionEdge],
) -> LiveFutureCandidate | None:
    """Plan and professionally compile a FUTURE-only replacement candidate.

    The current NEXT transition is already locked and represented by the last
    history transition. Planning starts from its exact projected post-release
    state. No mutation is performed here; DJ Executive signs the returned
    candidate separately and the deterministic Executor validates that commit
    at the safe boundary.
    """

    analyses = plan.tracks
    history_edges = list(history_transitions)
    history_signatures = tuple(str(edge.plan.execution_signature or "") for edge in history_edges)
    if history_signatures != tuple(request.history_execution_signatures):
        return None
    if not request.remaining_indices:
        return None
    if request.anchor_track in request.remaining_indices:
        return None

    complete_cfg = CompleteMoveConfig.from_environment()
    if complete_cfg.mode == "legacy":
        return None
    headphones_cfg = AIHeadphonesConfig.from_environment()
    live_cfg = LiveExecutiveConfig.from_environment()
    graph: dict[tuple[int, int], TransitionEdge] = {}
    session = PlanningSession()
    journey = normalize_journey(request.journey)
    listener_profile = get_listener_profile(request.listener)
    tempo_policy = str(plan.metadata.get("tempo_policy", "safe"))
    preferred_style = str(plan.metadata.get("preferred_style", "auto"))
    minimum_body_seconds = float(plan.metadata.get("minimum_body_seconds", 30.0))
    minimum_body_fraction = float(plan.metadata.get("minimum_body_fraction", 0.35))

    quality_meta = dict(plan.metadata.get("quality_loop", {}) or {})
    quality_enabled = bool(quality_meta.get("enabled", False))
    quality_loop = QualityLoopConfig.from_environment(
        enabled=quality_enabled,
        top_k=int(quality_meta.get("top_k", 4) or 4),
        sample_rate=int(quality_meta.get("sample_rate") or DEFAULT_RENDER_SAMPLE_RATE),
    )

    def ensure_pairs(pairs: Iterable[tuple[int, int]]) -> None:
        requested = {tuple(map(int, pair)) for pair in pairs}
        missing = sorted(pair for pair in requested if pair not in graph)
        if not missing:
            return
        graph.update(
            build_transition_graph(
                analyses,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                workers=None,
                session=session,
                pair_indices=missing,
            )
        )

    blocked_pairs: set[tuple[int, int]] = set()
    history_plans = [edge.plan for edge in history_edges]
    for revision in range(live_cfg.max_replans + 1):
        route = _build_live_complete_move_route(
            analyses,
            graph,
            request=request,
            history_plans=history_plans,
            blocked_pairs=blocked_pairs,
            config=complete_cfg,
            headphones_config=headphones_cfg,
            ensure_pairs=ensure_pairs,
        )
        if not route.completed:
            return None
        order_suffix = list(route.order)
        preferred_moves = {
            (decision.current_index, decision.winner.next_index): decision
            for decision in route.decisions
        }
        try:
            compiled = _compile_sequence_edges(
                order_suffix,
                analyses,
                graph,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                minimum_body_seconds=minimum_body_seconds,
                minimum_body_fraction=minimum_body_fraction,
                session=session,
                quality_loop=quality_loop,
                automatic_delivery=True,
                listener_profile=listener_profile,
                replan_on_review=True,
                allow_emergency_escape=False,
                preferred_complete_moves=preferred_moves,
                initial_source_cursor=request.anchor_source_cursor,
                initial_rate=request.anchor_render_rate,
                initial_effective_bpm=request.anchor_effective_bpm,
                history_order=request.history_order,
                history_plans=history_plans,
                position_offset=request.replace_from_edge,
                total_track_count=len(analyses),
                transition_policy_authority="bounded_advisory",
                crowd_directive=crowd_directive_from_payload(request.crowd_directive),
            )
        except TransitionReplanRequired as exc:
            blocked_pairs.add((int(exc.from_index), int(exc.to_index)))
            continue
        if not compiled:
            return None
        return LiveFutureCandidate(
            cycle=request.cycle,
            schedule_generation=request.schedule_generation,
            replace_from_edge=request.replace_from_edge,
            anchor_track=request.anchor_track,
            anchor_source_cursor=request.anchor_source_cursor,
            order_suffix=tuple(order_suffix),
            transitions=tuple(compiled),
            planning_revision=revision,
            blocked_pairs=tuple(sorted(blocked_pairs)),
            decisions=tuple(
                decision.to_dict(top_k=min(complete_cfg.report_top_k, 3))
                for decision in route.decisions
            ),
        )
    return None

def make_set_plan(
    analyses: list[TrackAnalysis],
    *,
    source: str,
    manual_order: list[int] | None = None,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    journey: JourneyMode = "balanced",
    start_index: int | None = None,
    end_index: int | None = None,
    minimum_body_seconds: float = 30.0,
    minimum_body_fraction: float = 0.35,
    progress: ProgressCallback | None = None,
    graph_workers: int | None = None,
    planning_session: PlanningSession | None = None,
    quality_loop: QualityLoopConfig | None = None,
    automatic_delivery: bool = True,
    listener: str = "local",
    timing_intent: SetTimingIntent | None = None,
) -> SetPlan:
    journey = normalize_journey(journey)
    listener_profile = get_listener_profile(listener)
    session = planning_session if planning_session is not None else PlanningSession()
    measurement_run_offset = _measurement_run_start_offset(quality_loop)

    complete_move_cfg = CompleteMoveConfig.from_environment()
    headphones_cfg = AIHeadphonesConfig.from_environment()
    lazy_graph_authority = complete_move_cfg.mode == "move_authoritative" and manual_order is None
    if lazy_graph_authority:
        # Authoritative complete-move search materializes graph edges only as needed.
        graph: dict[tuple[int, int], TransitionEdge] = {}
        session.graph_mode = "lazy"
    else:
        # Shadow/legacy still need the historical graph for an apples-to-apples
        # route comparison. Manual order also keeps the established path.
        graph = build_transition_graph(
            analyses,
            tempo_policy=tempo_policy,
            preferred_style=preferred_style,
            journey=journey,
            progress=progress,
            workers=graph_workers,
            session=session,
        )

    def _ensure_graph_pairs(pairs: Iterable[tuple[int, int]]) -> None:
        requested = {tuple(map(int, pair)) for pair in pairs}
        missing = sorted(pair for pair in requested if pair not in graph)
        if not missing:
            return
        graph.update(
            build_transition_graph(
                analyses,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                progress=progress,
                workers=graph_workers,
                session=session,
                pair_indices=missing,
            )
        )

    broad_graph_materialized = False

    def _ensure_full_graph() -> None:
        nonlocal broad_graph_materialized
        if broad_graph_materialized:
            return
        all_pairs = (
            (from_index, to_index)
            for from_index in range(len(analyses))
            for to_index in range(len(analyses))
            if from_index != to_index
        )
        _ensure_graph_pairs(all_pairs)
        broad_graph_materialized = True

    baseline_order = list(range(len(analyses)))
    blocked_pairs: set[tuple[int, int]] = set()
    replan_trace: list[dict[str, Any]] = []
    default_replans = max(3, min(8, (len(analyses) + 1) // 2))
    max_replans = max(1, min(48, int(os.environ.get("DJAI_MAX_SET_REPLANS", str(default_replans)))))
    complete_move_runs: list[_CompleteMoveRoute] = []
    route_search_seconds = 0.0
    route_search_calls = 0
    compile_seconds = 0.0
    compile_calls = 0

    def _choose_automatic_order_impl(active_blocked: set[tuple[int, int]]) -> list[int]:
        route: _CompleteMoveRoute | None = None
        if broad_graph_materialized:
            return optimize_order(
                analyses,
                graph,
                start_index=start_index,
                end_index=end_index,
                journey=journey,
                listener_profile=listener_profile,
                blocked_pairs=active_blocked,
            )
        if complete_move_cfg.mode != "legacy":
            route = build_complete_move_route(
                analyses,
                graph,
                start_index=start_index,
                end_index=end_index,
                journey=journey,
                blocked_pairs=active_blocked,
                config=complete_move_cfg,
                headphones_config=headphones_cfg,
                listener=listener,
                ensure_pairs=_ensure_graph_pairs if lazy_graph_authority else None,
                timing_intent=timing_intent,
            )
            complete_move_runs.append(route)
            if complete_move_cfg.mode == "move_authoritative" and route.completed:
                return list(route.order)

            if lazy_graph_authority and len(analyses) >= _large_set_threshold():
                # Learned retrieval should widen locally before the planner is
                # allowed to consider the historical O(N^2) graph. Each tier
                # reuses already materialized/cached pair evidence.
                experimental_retrieval = os.environ.get(
                    "DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL", ""
                ).strip().lower() in {"1", "true", "yes", "on"}
                expansion_tiers = _progressive_sparse_configs(
                    complete_move_cfg,
                    len(analyses),
                    experimental_retrieval=experimental_retrieval,
                )
                total_tiers = len(expansion_tiers)
                for tier_index, expanded_cfg in enumerate(expansion_tiers, start=1):
                    if progress:
                        progress(
                            "route-search",
                            tier_index,
                            total_tiers + 1,
                            (
                                f"Sparse route incomplete; widening learned retrieval "
                                f"to top {expanded_cfg.lazy_candidate_width}"
                            ),
                        )
                    expanded_route = build_complete_move_route(
                        analyses,
                        graph,
                        start_index=start_index,
                        end_index=end_index,
                        journey=journey,
                        blocked_pairs=active_blocked,
                        config=expanded_cfg,
                        headphones_config=headphones_cfg,
                        listener=listener,
                        ensure_pairs=_ensure_graph_pairs,
                        timing_intent=timing_intent,
                    )
                    complete_move_runs.append(expanded_route)
                    if expanded_route.completed:
                        if progress:
                            progress(
                                "route-search",
                                total_tiers + 1,
                                total_tiers + 1,
                                "Progressive sparse retrieval completed; broad graph avoided",
                            )
                        return list(expanded_route.order)
        if lazy_graph_authority:
            # Neural retrieval experiments deliberately test whether bounded
            # candidate filtering can complete the route without falling back
            # to O(N^2) graph construction. Historical broad-graph recovery
            # remains available outside the experiment, or when explicitly
            # re-enabled for diagnosis.
            experimental_retrieval = os.environ.get(
                "DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            allow_broad = os.environ.get(
                "DJAI_ALLOW_BROAD_GRAPH_FALLBACK",
                "0" if experimental_retrieval else "1",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not allow_broad:
                if progress:
                    effective_attempts = max(1, len(complete_move_runs))
                    best_partial_depth = max(
                        (len(item.order) for item in complete_move_runs),
                        default=0,
                    )
                    progress(
                        "route-search",
                        effective_attempts,
                        effective_attempts,
                        (
                            "Progressive learned retrieval exhausted after "
                            f"{effective_attempts} effective sparse attempt(s); "
                            f"best partial route reached {best_partial_depth}/{len(analyses)} tracks; "
                            "broad graph disabled for this experiment"
                        ),
                    )
                raise SparseRouteExhaustedError(
                    "Progressive learned retrieval could not complete a safe route without broad-graph fallback.",
                    diagnostics={
                        "effective_sparse_attempts": len(complete_move_runs),
                        "best_sparse_partial_tracks": max(
                            (len(item.order) for item in complete_move_runs),
                            default=0,
                        ),
                        "materialized_graph_pairs": len(graph),
                        "possible_graph_pairs": len(analyses) * max(0, len(analyses) - 1),
                        "graph": session.to_dict(),
                        "attempts": [
                            {
                                "completed": bool(item.completed),
                                "tracks": len(item.order),
                                "score": round(float(item.score), 6),
                                "reason": str(item.reason or ""),
                            }
                            for item in complete_move_runs
                        ],
                    },
                )
            if progress:
                effective_attempts = max(1, len(complete_move_runs))
                progress(
                    "route-search",
                    effective_attempts,
                    effective_attempts + 1,
                    (
                        f"Sparse route incomplete after {effective_attempts} effective attempt(s); "
                        "materializing broad graph fallback"
                    ),
                )
            _ensure_full_graph()
        return optimize_order(
            analyses,
            graph,
            start_index=start_index,
            end_index=end_index,
            journey=journey,
            listener_profile=listener_profile,
            blocked_pairs=active_blocked,
        )

    def _choose_automatic_order(active_blocked: set[tuple[int, int]]) -> list[int]:
        nonlocal route_search_seconds
        nonlocal route_search_calls
        started = time.perf_counter()
        route_search_calls += 1
        sparse_failure = False
        try:
            return _choose_automatic_order_impl(active_blocked)
        except SparseRouteExhaustedError as exc:
            sparse_failure = True
            route_search_seconds += time.perf_counter() - started
            exc.diagnostics["route_search_seconds"] = round(route_search_seconds, 6)
            exc.diagnostics["route_search_calls"] = route_search_calls
            raise
        finally:
            if not sparse_failure:
                route_search_seconds += time.perf_counter() - started

    def _complete_move_preferences(selected_order: list[int]) -> dict[tuple[int, int], CompleteMoveDecision] | None:
        if complete_move_cfg.mode != "move_authoritative" or not complete_move_runs:
            return None
        route = complete_move_runs[-1]
        if not route.completed or tuple(selected_order) != route.order:
            return None
        return {
            (decision.current_index, decision.winner.next_index): decision
            for decision in route.decisions
        }

    manual = manual_order is not None
    if manual:
        assert manual_order is not None
        if not _valid_order(list(manual_order), len(analyses), None, None):
            raise ValueError("Manual order must contain every track exactly once.")
        algorithm = "client_supplied_order_with_unified_transition_intelligence"
    elif complete_move_cfg.mode == "move_authoritative":
        algorithm = "saund_ai_1_8_2_crate_memory_ai_headphones_complete_move_route"
    elif complete_move_cfg.mode == "move_shadow":
        algorithm = "saund_ai_1_unified_planner_with_1_8_2_crate_memory_shadow"
    else:
        algorithm = "saund_ai_1_unified_planner_with_rehearsal_feedback_and_edge_replanning"

    prepared_prefix_cache: dict[tuple[int, ...], list[TransitionEdge]] = {}

    def _remember_prepared_prefix(order_value: list[int], edges: Sequence[TransitionEdge]) -> None:
        for count in range(1, len(edges) + 1):
            key = tuple(order_value[: count + 1])
            prepared_prefix_cache[key] = list(edges[:count])

    def _compile_for_executive_impl(
        selected_order: list[int],
        replan_on_review: bool,
        allow_emergency_escape: bool,
    ) -> list[TransitionEdge]:
        cached_edges: list[TransitionEdge] = []
        cached_track_count = 1
        for track_count in range(len(selected_order), 1, -1):
            key = tuple(selected_order[:track_count])
            if key in prepared_prefix_cache:
                cached_edges = list(prepared_prefix_cache[key])
                cached_track_count = track_count
                break

        suffix_order = selected_order[cached_track_count - 1 :] if cached_edges else selected_order
        history_order = selected_order[:cached_track_count] if cached_edges else None
        history_plans = [edge.plan for edge in cached_edges] if cached_edges else None
        initial_cursor = cached_edges[-1].plan.post_transition_source_end_b if cached_edges else 0.0
        initial_rate = cached_edges[-1].plan.post_transition_rate_b if cached_edges else 1.0
        initial_bpm = (
            cached_edges[-1].plan.post_transition_bpm_b
            if cached_edges
            else None
        )
        if cached_edges and progress:
            progress(
                "prefix-reuse",
                len(cached_edges),
                max(1, len(selected_order) - 1),
                f"Reusing {len(cached_edges)} already-rehearsed transition(s)",
            )

        try:
            suffix_edges = _compile_sequence_edges(
                suffix_order,
                analyses,
                graph,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                minimum_body_seconds=minimum_body_seconds,
                minimum_body_fraction=minimum_body_fraction,
                session=session,
                quality_loop=quality_loop,
                automatic_delivery=automatic_delivery,
                listener_profile=listener_profile,
                progress=progress,
                replan_on_review=replan_on_review,
                allow_emergency_escape=allow_emergency_escape,
                preferred_complete_moves=_complete_move_preferences(selected_order),
                initial_source_cursor=initial_cursor,
                initial_rate=initial_rate,
                initial_effective_bpm=initial_bpm,
                history_order=history_order,
                history_plans=history_plans,
                position_offset=len(cached_edges),
                total_track_count=len(selected_order),
            )
            result = cached_edges + suffix_edges
            _remember_prepared_prefix(selected_order, result)
            return result
        except TransitionReplanRequired as exc:
            partial = cached_edges + list(exc.compiled_prefix)
            if partial:
                _remember_prepared_prefix(selected_order, partial)
            raise ExecutiveReplanRequest(
                exc.from_index,
                exc.to_index,
                action=exc.action,
                reason=exc.reason,
                resolution=exc.resolution,
            ) from exc

    def _compile_for_executive(
        selected_order: list[int],
        replan_on_review: bool,
        allow_emergency_escape: bool,
    ) -> list[TransitionEdge]:
        nonlocal compile_seconds
        nonlocal compile_calls
        started = time.perf_counter()
        compile_calls += 1
        try:
            return _compile_for_executive_impl(
                selected_order,
                replan_on_review,
                allow_emergency_escape,
            )
        finally:
            compile_seconds += time.perf_counter() - started

    def _latest_complete_move_decisions() -> tuple[CompleteMoveDecision, ...]:
        if not complete_move_runs:
            return ()
        return tuple(complete_move_runs[-1].decisions)

    executive_orchestrator = ExecutiveOrchestrator(
        analyses=analyses,
        source=source,
        journey=journey,
        listener=listener,
        automatic_delivery=automatic_delivery,
        max_replans=max_replans,
        progress=progress,
    )
    executive_started = time.perf_counter()
    executive_result = executive_orchestrator.run_prepared_set(
        manual_order=(list(manual_order) if manual_order is not None else None),
        choose_order=_choose_automatic_order,
        compile_order=_compile_for_executive,
        decisions=_latest_complete_move_decisions,
        top_alternatives=complete_move_cfg.report_top_k,
    )
    executive_seconds = time.perf_counter() - executive_started
    order = list(executive_result.order)
    transitions = list(executive_result.transitions)
    blocked_pairs = set(executive_result.blocked_pairs)
    replan_trace = list(executive_result.replan_trace)
    dj_executive_report = dict(executive_result.executive_report)
    if lazy_graph_authority:
        # Scoring diagnostics need only the adjacent baseline/final edges, not
        # the full N*(N-1) graph. Keep authoritative search sparse even for reporting.
        _ensure_graph_pairs(
            list(zip(baseline_order, baseline_order[1:])) + list(zip(order, order[1:]))
        )
    baseline_score = _order_score(
        baseline_order,
        graph,
        analyses,
        journey=journey,
        listener_profile=listener_profile,
    )
    # Baseline and optimized scores must come from the same canonical ordering
    # objective; path-dependent compiled edge scores are reported separately.
    optimized_score = _order_score(
        order,
        graph,
        analyses,
        journey=journey,
        listener_profile=listener_profile,
    )
    compiled_transition_score = sum(edge.objective_score for edge in transitions)
    plan_id = _plan_identity(
        [analysis.path for analysis in analyses],
        order,
        manual,
        tempo_policy=tempo_policy,
        preferred_style=preferred_style,
        journey=journey,
    )
    warnings = [
        f"{analyses[edge.from_index].title} -> {analyses[edge.to_index].title}: {warning}"
        for edge in transitions
        for warning in edge.warnings
    ]
    estimated_duration_seconds = estimate_set_duration(
        order,
        analyses,
        transitions,
    )
    complete_move_report = _build_complete_move_report(
        analyses=analyses,
        order=order,
        transitions=transitions,
        complete_move_runs=complete_move_runs,
        complete_move_cfg=complete_move_cfg,
        blocked_pairs=blocked_pairs,
        graph_pair_count=len(graph),
        manual=manual,
        lazy_graph_authority=lazy_graph_authority,
    )
    metadata = _build_set_plan_metadata(
        analyses=analyses,
        order=order,
        transitions=transitions,
        timing_intent=timing_intent,
        estimated_duration_seconds=estimated_duration_seconds,
        tempo_policy=tempo_policy,
        preferred_style=preferred_style,
        journey=journey,
        listener=listener,
        listener_profile_confidence=(
            listener_profile.confidence
            if listener_profile is not None
            else 0.0
        ),
        automatic_delivery=automatic_delivery,
        complete_move_report=complete_move_report,
        dj_executive_report=dj_executive_report,
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        compiled_transition_score=compiled_transition_score,
        minimum_body_seconds=minimum_body_seconds,
        minimum_body_fraction=minimum_body_fraction,
        graph_pair_count=len(graph),
        session=session,
        route_search_seconds=route_search_seconds,
        route_search_calls=route_search_calls,
        compile_seconds=compile_seconds,
        compile_calls=compile_calls,
        executive_seconds=executive_seconds,
        complete_move_runs=complete_move_runs,
        broad_graph_materialized=broad_graph_materialized,
        max_replans=max_replans,
        quality_loop=quality_loop,
        blocked_pairs=blocked_pairs,
        replan_trace=replan_trace,
    )
    plan = SetPlan(
        plan_id=plan_id,
        tracks=analyses,
        order=order,
        transitions=transitions,
        baseline_order=baseline_order,
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        estimated_duration_seconds=estimated_duration_seconds,
        source=source,
        algorithm=algorithm,
        manual=manual,
        warnings=list(dict.fromkeys(warnings)),
        automatic_delivery=automatic_delivery,
        listener=listener,
        metadata=metadata,
    )
    plan.metadata["full_set_benchmark"] = evaluate_full_set(plan).to_dict()
    _finalize_quality_measurement_run(
        quality_loop,
        start_offset=measurement_run_offset,
        transitions=transitions,
    )
    if plan.requires_confirmation:
        plan.confirmation_token = create_confirmation_token(
            confirmation_payload(plan_id, order, transitions)
        )
    return plan


def plan_folder(
    folder: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    recursive: bool = False,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    journey: JourneyMode = "balanced",
    start_track: str | None = None,
    end_track: str | None = None,
    minimum_body_seconds: float = 30.0,
    minimum_body_fraction: float = 0.35,
    progress: ProgressCallback | None = None,
    analysis_workers: int | None = None,
    graph_workers: int | None = None,
    measure_transitions: bool = True,
    measurement_top_k: int = 4,
    measurement_log: str | None = "ratings/automatic-measurements.jsonl",
    automatic_delivery: bool = True,
    listener: str = "local",
    timing_intent: SetTimingIntent | None = None,
) -> SetPlan:
    analysis_stats = AnalysisBatchStats()
    analyses = analyze_folder(
        folder,
        cache_dir=cache_dir,
        use_cache=use_cache,
        recursive=recursive,
        progress=progress,
        workers=analysis_workers,
        stats=analysis_stats,
    )
    stem_enriched = enrich_analyses_with_cached_stems(analyses)
    layer_enriched = enrich_analyses_with_performance_layers(analyses)
    names = [Path(analysis.path).name for analysis in analyses]

    def resolve_name(value: str | None) -> int | None:
        if value is None:
            return None
        matches = [
            index for index, name in enumerate(names)
            if name == value or Path(name).stem == value
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not resolve track constraint '{value}' uniquely.")
        return matches[0]

    plan = make_set_plan(
        analyses,
        source=str(Path(folder).expanduser().resolve()),
        tempo_policy=tempo_policy,
        preferred_style=preferred_style,
        journey=journey,
        start_index=resolve_name(start_track),
        end_index=resolve_name(end_track),
        minimum_body_seconds=minimum_body_seconds,
        minimum_body_fraction=minimum_body_fraction,
        progress=progress,
        graph_workers=graph_workers,
        quality_loop=QualityLoopConfig.from_environment(
            enabled=measure_transitions,
            top_k=measurement_top_k,
            log_path=measurement_log,
        ),
        automatic_delivery=automatic_delivery,
        listener=listener,
        timing_intent=timing_intent,
    )
    plan.metadata["stem_enriched_tracks"] = stem_enriched
    plan.metadata["performance_layer_enriched_tracks"] = layer_enriched
    performance = dict(plan.metadata.get("performance", {}))
    performance["analysis"] = analysis_stats.to_dict()
    plan.metadata["performance"] = performance
    return plan


def plan_manual_paths(
    paths: Iterable[str | Path],
    *,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    journey: JourneyMode = "balanced",
    minimum_body_seconds: float = 30.0,
    minimum_body_fraction: float = 0.35,
    progress: ProgressCallback | None = None,
    analysis_workers: int | None = None,
    graph_workers: int | None = None,
    measure_transitions: bool = True,
    measurement_top_k: int = 4,
    measurement_log: str | None = "ratings/automatic-measurements.jsonl",
    automatic_delivery: bool = False,
    listener: str = "local",
    timing_intent: SetTimingIntent | None = None,
) -> SetPlan:
    from .layers import collapse_performance_layer_paths

    path_list = collapse_performance_layer_paths(paths, validate=True)
    analysis_stats = AnalysisBatchStats()
    analyses = analyze_tracks(
        path_list,
        cache_dir=cache_dir,
        use_cache=use_cache,
        progress=progress,
        workers=analysis_workers,
        stats=analysis_stats,
    )
    stem_enriched = enrich_analyses_with_cached_stems(analyses)
    layer_enriched = enrich_analyses_with_performance_layers(analyses)
    plan = make_set_plan(
        analyses,
        source="manual",
        manual_order=list(range(len(analyses))),
        tempo_policy=tempo_policy,
        preferred_style=preferred_style,
        journey=journey,
        minimum_body_seconds=minimum_body_seconds,
        minimum_body_fraction=minimum_body_fraction,
        progress=progress,
        graph_workers=graph_workers,
        quality_loop=QualityLoopConfig.from_environment(
            enabled=measure_transitions,
            top_k=measurement_top_k,
            log_path=measurement_log,
        ),
        automatic_delivery=automatic_delivery,
        listener=listener,
        timing_intent=timing_intent,
    )
    plan.metadata["stem_enriched_tracks"] = stem_enriched
    plan.metadata["performance_layer_enriched_tracks"] = layer_enriched
    performance = dict(plan.metadata.get("performance", {}))
    performance["analysis"] = analysis_stats.to_dict()
    plan.metadata["performance"] = performance
    return plan


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_set_plan(plan: SetPlan, output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    _atomic_write(path, json.dumps(plan.to_dict(include_analysis=True), indent=2) + "\n")
    return path


def load_set_plan(path: str | Path) -> SetPlan:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != PLAN_SCHEMA_VERSION:
        raise ValueError(
            "This set plan uses an older schema. Re-run DJAI planning before rendering."
        )
    tracks: list[TrackAnalysis] = []
    for item in payload["tracks"]:
        if "analysis" not in item:
            raise ValueError("Set plan does not contain embedded track analyses.")
        tracks.append(track_analysis_from_dict(item["analysis"]))
    transitions: list[TransitionEdge] = []
    for item in payload["transitions"]:
        plan = transition_plan_from_dict(item["plan"])
        validate_execution_plan(plan)
        alternatives = [
            transition_plan_from_dict(value) for value in item.get("alternatives", [])
        ]
        transitions.append(
            TransitionEdge(
                from_index=int(item["from_index"]),
                to_index=int(item["to_index"]),
                selected_style=str(item.get("selected_style", plan.style)),
                objective_score=float(item.get("objective_score", plan.score)),
                quality=str(item.get("quality", classify_quality(plan))),  # type: ignore[arg-type]
                plan=plan,
                alternatives=alternatives,
                warnings=list(item.get("warnings", [])),
                reasons=list(item.get("reasons", [])),
            )
        )
    return SetPlan(
        plan_id=str(payload["plan_id"]),
        tracks=tracks,
        order=[int(value) for value in payload["order"]],
        transitions=transitions,
        baseline_order=[int(value) for value in payload.get("baseline_order", range(len(tracks)))],
        baseline_score=float(payload.get("baseline_score", 0.0)),
        optimized_score=float(payload.get("optimized_score", 0.0)),
        estimated_duration_seconds=float(payload.get("estimated_duration_seconds", 0.0)),
        source=str(payload.get("source", "plan")),
        algorithm=str(payload.get("algorithm", "loaded")),
        manual=bool(payload.get("manual", False)),
        confirmed=bool(payload.get("confirmed", False)),
        confirmation_token=payload.get("confirmation_token"),
        warnings=list(payload.get("warnings", [])),
        created_at_utc=str(payload.get("created_at_utc", "")),
        version=str(payload.get("version", SET_PLAN_VERSION)),
        schema_version=str(payload.get("schema_version", PLAN_SCHEMA_VERSION)),
        metadata=dict(payload.get("metadata", {})),
        automatic_delivery=bool(payload.get("automatic_delivery", False)),
        listener=str(payload.get("listener", "local")),
    )
