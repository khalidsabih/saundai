from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from .parallel import prepare_process_environment, process_context, resolve_worker_selection, worker_initializer
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


@dataclass(frozen=True, slots=True)
class _GraphPairTask:
    from_index: int
    to_index: int


@dataclass(slots=True)
class _GraphPairResult:
    from_index: int
    to_index: int
    label: str
    edge: TransitionEdge
    evidence_matrix: PairEvidenceMatrix


GRAPH_PAIR_CACHE_VERSION = "1.0"

def _analysis_graph_fingerprint(track: TrackAnalysis) -> str:
    payload = track.to_dict(include_beats=False)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def _graph_pair_cache_key(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    scout_cues: int,
    model_fingerprint: str | None,
) -> str:
    payload = {
        "version": GRAPH_PAIR_CACHE_VERSION,
        "track_a": _analysis_graph_fingerprint(track_a),
        "track_b": _analysis_graph_fingerprint(track_b),
        "tempo_policy": tempo_policy,
        "preferred_style": preferred_style,
        "journey": journey,
        "scout_cues": int(scout_cues),
        "scoring_model_fingerprint": model_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

def _graph_pair_cache_path(key: str) -> Path:
    return current_environment().cache_dir / "graph-pairs" / f"{key}.json"

def _pair_evidence_to_dict(matrix: PairEvidenceMatrix) -> dict[str, Any]:
    return {
        "track_a_path": matrix.track_a_path,
        "track_b_path": matrix.track_b_path,
        "exit_candidates": [asdict(item) for item in matrix.exit_candidates],
        "entry_candidates": [asdict(item) for item in matrix.entry_candidates],
        "pairs": [
            {
                "exit_index": item.exit_index,
                "entry_index": item.entry_index,
                "exit_candidate": asdict(item.exit_candidate),
                "entry_candidate": asdict(item.entry_candidate),
                "breakdown": asdict(item.breakdown),
            }
            for item in matrix.pairs
        ],
        "scoring_model_fingerprint": matrix.scoring_model_fingerprint,
    }

def _pair_evidence_from_dict(payload: dict[str, Any]) -> PairEvidenceMatrix:
    return PairEvidenceMatrix(
        track_a_path=str(payload["track_a_path"]),
        track_b_path=str(payload["track_b_path"]),
        exit_candidates=tuple(
            transition_candidate_from_dict(dict(item))
            for item in payload.get("exit_candidates", [])
        ),
        entry_candidates=tuple(
            transition_candidate_from_dict(dict(item))
            for item in payload.get("entry_candidates", [])
        ),
        pairs=tuple(
            CandidatePairEvidence(
                exit_index=int(item["exit_index"]),
                entry_index=int(item["entry_index"]),
                exit_candidate=transition_candidate_from_dict(dict(item["exit_candidate"])),
                entry_candidate=transition_candidate_from_dict(dict(item["entry_candidate"])),
                breakdown=CompatibilityBreakdown(**dict(item["breakdown"])),
            )
            for item in payload.get("pairs", [])
        ),
        scoring_model_fingerprint=payload.get("scoring_model_fingerprint"),
    )

def _load_graph_pair_cache(
    key: str,
    *,
    from_index: int,
    to_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
) -> _GraphPairResult | None:
    path = _graph_pair_cache_path(key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != GRAPH_PAIR_CACHE_VERSION:
            return None
        if int(payload.get("from_index", -1)) != from_index or int(payload.get("to_index", -1)) != to_index:
            return None
        matrix = _pair_evidence_from_dict(dict(payload["evidence_matrix"]))
        matrix.validate_tracks(track_a, track_b)
        edge_payload = dict(payload["edge"])
        edge = TransitionEdge(
            from_index=from_index,
            to_index=to_index,
            selected_style=str(edge_payload["selected_style"]),
            objective_score=float(edge_payload["objective_score"]),
            quality=str(edge_payload["quality"]),
            plan=transition_plan_from_dict(dict(edge_payload["plan"])),
            alternatives=[
                transition_plan_from_dict(dict(item))
                for item in edge_payload.get("alternatives", [])
            ],
            warnings=[str(item) for item in edge_payload.get("warnings", [])],
            reasons=[str(item) for item in edge_payload.get("reasons", [])],
        )
        return _GraphPairResult(
            from_index=from_index,
            to_index=to_index,
            label=f"{track_a.title} -> {track_b.title}",
            edge=edge,
            evidence_matrix=matrix,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

def _store_graph_pair_cache(key: str, result: _GraphPairResult) -> None:
    path = _graph_pair_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GRAPH_PAIR_CACHE_VERSION,
            "from_index": result.from_index,
            "to_index": result.to_index,
            "edge": result.edge.to_dict(),
            "evidence_matrix": _pair_evidence_to_dict(result.evidence_matrix),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        return

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


_GRAPH_WORKER_ANALYSES: tuple[TrackAnalysis, ...] | None = None
_GRAPH_WORKER_TEMPO_POLICY = "safe"
_GRAPH_WORKER_PREFERRED_STYLE = "auto"
_GRAPH_WORKER_JOURNEY: JourneyMode = "balanced"
_GRAPH_WORKER_STEM_MAPS: dict[int, dict[str, str]] = {}
_GRAPH_WORKER_SCORING_MODEL: dict[str, Any] | None = None


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _large_set_threshold() -> int:
    return _env_int("DJAI_LARGE_SET_THRESHOLD", 8, minimum=4, maximum=64)


def _graph_scout_cue_limit(track_count: int) -> int:
    if track_count < _large_set_threshold():
        return 0
    return _env_int("DJAI_GRAPH_SCOUT_CUES", 5, minimum=3, maximum=8)


def large_crate_runtime_status() -> dict[str, Any]:
    try:
        embed_window = max(8.0, min(45.0, float(os.environ.get("DJAI_TRACK_EMBED_WINDOW_SECONDS", "24"))))
    except ValueError:
        embed_window = 24.0
    embed_windows = _env_int("DJAI_TRACK_EMBED_WINDOWS", 3, minimum=1, maximum=6)
    return {
        "schema_version": "1.5.1",
        "threshold_tracks": _large_set_threshold(),
        "graph_strategy": "bounded_scout_then_full_final_edges",
        "graph_scout_cues_per_side": _env_int("DJAI_GRAPH_SCOUT_CUES", 5, minimum=3, maximum=8),
        "receding_horizon_opener_limit": _env_int("DJAI_LARGE_SET_POLICY_STARTS", 4, minimum=2, maximum=8),
        "local_improve_passes": _env_int("DJAI_LARGE_SET_LOCAL_PASSES", 2, minimum=1, maximum=4),
        "track_embedding_windows": embed_windows,
        "track_embedding_window_seconds": round(embed_window, 3),
        "full_cue_pool_restored_for_final_edges": True,
        "rendered_quality_gate_unchanged": True,
        "hard_audio_safety_authoritative": True,
    }


def _graph_scout_candidates(
    candidates: list[TransitionCandidate] | tuple[TransitionCandidate, ...],
    *,
    kind: str,
    limit: int,
) -> list[TransitionCandidate]:
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    scored: list[tuple[float, TransitionCandidate]] = []
    for candidate in candidates:
        low_vocal = 1.0 - max(0.0, min(1.0, float(candidate.vocal_probability)))
        value = (
            0.44 * float(candidate.score)
            + 0.20 * float(candidate.boundary_strength)
            + 0.18 * float(candidate.phrase_strength)
            + 0.10 * float(candidate.beat_grid_confidence)
            + 0.08 * low_vocal
        )
        scored.append((value, candidate))
    selected = _candidate_diverse_by_bar(scored, limit=limit, minimum_bar_distance=4)
    if kind == "entry":
        earliest = min(candidates, key=lambda item: (float(item.time_seconds), -float(item.score)))
        if earliest.section_label == "intro" and earliest not in selected:
            if len(selected) >= limit:
                selected[-1] = earliest
            else:
                selected.append(earliest)
    return selected[:limit]


def _graph_worker_initializer(
    analyses: tuple[TrackAnalysis, ...],
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    stem_maps: dict[int, dict[str, str]],
    scoring_model: dict[str, Any] | None,
) -> None:
    """Load immutable graph-planning inputs once per worker process."""
    worker_initializer()
    global _GRAPH_WORKER_ANALYSES
    global _GRAPH_WORKER_TEMPO_POLICY
    global _GRAPH_WORKER_PREFERRED_STYLE
    global _GRAPH_WORKER_JOURNEY
    global _GRAPH_WORKER_STEM_MAPS
    global _GRAPH_WORKER_SCORING_MODEL
    _GRAPH_WORKER_ANALYSES = analyses
    _GRAPH_WORKER_TEMPO_POLICY = tempo_policy
    _GRAPH_WORKER_PREFERRED_STYLE = preferred_style
    _GRAPH_WORKER_JOURNEY = journey
    _GRAPH_WORKER_STEM_MAPS = stem_maps
    _GRAPH_WORKER_SCORING_MODEL = scoring_model


def _plan_graph_pair_core(
    task: _GraphPairTask,
    *,
    analyses: tuple[TrackAnalysis, ...] | list[TrackAnalysis],
    tempo_policy: str,
    preferred_style: str,
    journey: JourneyMode,
    stem_maps: dict[int, dict[str, str]],
    scoring_model: dict[str, Any] | None,
) -> _GraphPairResult:
    track_a = analyses[task.from_index]
    track_b = analyses[task.to_index]
    stems_a = stem_maps[task.from_index]
    stems_b = stem_maps[task.to_index]
    scout_cues = _graph_scout_cue_limit(len(analyses))
    scout_exits = _graph_scout_candidates(
        track_a.exit_candidates, kind="exit", limit=scout_cues
    ) if scout_cues else list(track_a.exit_candidates)
    scout_entries = _graph_scout_candidates(
        track_b.entry_candidates, kind="entry", limit=scout_cues
    ) if scout_cues else list(track_b.entry_candidates)
    matrix = build_pair_evidence_matrix(
        track_a,
        track_b,
        exit_candidates=scout_exits,
        entry_candidates=scout_entries,
        scoring_model=scoring_model,
    )
    try:
        if scout_cues:
            # Ordering only needs a faithful local feasibility estimate. The
            # final path-dependent compiler restores the full cue pool and
            # performs rendered judging on the transitions that survive.
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
    selected, objective = _select_alternative(
        alternatives,
        track_a,
        track_b,
        preferred_style=preferred_style,
        journey=journey,
    )
    edge = TransitionEdge(
        from_index=task.from_index,
        to_index=task.to_index,
        selected_style=selected.style,
        objective_score=round(objective, 6),
        quality=classify_quality(selected),
        plan=selected,
        alternatives=alternatives,
        warnings=_transition_warnings(selected),
        reasons=list(selected.rationale),
    )
    return _GraphPairResult(
        from_index=task.from_index,
        to_index=task.to_index,
        label=f"{track_a.title} -> {track_b.title}",
        edge=edge,
        evidence_matrix=matrix,
    )


def _plan_graph_pair_worker(task: _GraphPairTask) -> _GraphPairResult:
    analyses = _GRAPH_WORKER_ANALYSES
    if analyses is None:
        raise RuntimeError("Graph worker was not initialized.")
    return _plan_graph_pair_core(
        task,
        analyses=analyses,
        tempo_policy=_GRAPH_WORKER_TEMPO_POLICY,
        preferred_style=_GRAPH_WORKER_PREFERRED_STYLE,
        journey=_GRAPH_WORKER_JOURNEY,
        stem_maps=_GRAPH_WORKER_STEM_MAPS,
        scoring_model=_GRAPH_WORKER_SCORING_MODEL,
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

    1.7.1 uses ``pair_indices`` to materialize only the edges reached by lazy
    candidate retrieval. Omitting it preserves the historical all-pairs graph.
    """
    if len(analyses) < 2:
        raise ValueError("A set needs at least two tracks.")

    active_session = session if session is not None else PlanningSession()
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
    selection = resolve_worker_selection("graph", total, requested=workers)
    active_session.graph_workers = selection.workers
    active_session.graph_mode = "lazy" if pair_indices is not None else ("scout" if scout_cues else "full")
    active_session.graph_scout_cues = scout_cues
    started = time.perf_counter()
    results: dict[tuple[int, int], _GraphPairResult] = {}

    if not selection.parallel:
        for completed, task in enumerate(tasks, start=1):
            track_a = analyses[task.from_index]
            track_b = analyses[task.to_index]
            if progress:
                progress("graph", completed - 1, total, f"{track_a.title} -> {track_b.title}")
            cache_key = _graph_pair_cache_key(
                track_a,
                track_b,
                tempo_policy=tempo_policy,
                preferred_style=preferred_style,
                journey=journey,
                scout_cues=scout_cues,
                model_fingerprint=scoring_model_fingerprint(active_session.scoring_model),
            )
            result = _load_graph_pair_cache(
                cache_key,
                from_index=task.from_index,
                to_index=task.to_index,
                track_a=track_a,
                track_b=track_b,
            )
            cache_hit = result is not None
            if result is None:
                result = _plan_graph_pair_core(
                    task,
                    analyses=analyses,
                    tempo_policy=tempo_policy,
                    preferred_style=preferred_style,
                    journey=journey,
                    stem_maps=stem_maps,
                    scoring_model=active_session.scoring_model,
                )
                _store_graph_pair_cache(cache_key, result)
            results[(result.from_index, result.to_index)] = result
            if progress:
                detail = f"{result.label} · cached" if cache_hit else result.label
                progress("graph", completed, total, detail)
    else:
        prepare_process_environment()
        if progress:
            mode = f"scout {scout_cues}x{scout_cues}" if scout_cues else "full cue graph"
            progress("graph", 0, total, f"{selection.workers} graph workers · {mode}")
        with ProcessPoolExecutor(
            max_workers=selection.workers,
            mp_context=process_context(),
            initializer=_graph_worker_initializer,
            initargs=(
                tuple(analyses),
                tempo_policy,
                preferred_style,
                journey,
                stem_maps,
                active_session.scoring_model,
            ),
        ) as executor:
            future_map = {executor.submit(_plan_graph_pair_worker, task): task for task in tasks}
            completed = 0
            try:
                for future in as_completed(future_map):
                    result = future.result()
                    results[(result.from_index, result.to_index)] = result
                    completed += 1
                    if progress:
                        progress("graph", completed, total, result.label)
            except BaseException:
                for future in future_map:
                    future.cancel()
                raise

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

    # Professional Set Brain: evaluate each candidate in the context of the
    # musical history, not only as an isolated A->B edge.  This adds a bounded
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


def _valid_order(order: list[int], count: int, start: int | None, end: int | None) -> bool:
    return (
        len(order) == count
        and len(set(order)) == count
        and set(order) == set(range(count))
        and (start is None or order[0] == start)
        and (end is None or order[-1] == end)
    )


def _greedy_orders(
    graph: dict[tuple[int, int], TransitionEdge],
    count: int,
    *,
    start: int | None,
    end: int | None,
    blocked_pairs: set[tuple[int, int]] | None = None,
) -> Iterable[list[int]]:
    starts = [start] if start is not None else [index for index in range(count) if index != end]
    for first in starts:
        if first is None:
            continue
        order = [first]
        unused = set(range(count)) - {first}
        if end is not None:
            unused.discard(end)
        blocked = blocked_pairs or set()
        while unused:
            current = order[-1]
            allowed = [candidate for candidate in sorted(unused) if (current, candidate) not in blocked]
            if not allowed:
                order = []
                break
            next_index = max(
                allowed,
                key=lambda candidate: graph[(current, candidate)].objective_score,
            )
            order.append(next_index)
            unused.remove(next_index)
        if not order:
            continue
        if end is not None and end not in order:
            order.append(end)
        if _valid_order(order, count, start, end):
            yield order


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
    best = list(order)
    best_score = _order_score(
        best, graph, analyses, journey=journey, listener_profile=listener_profile, blocked_pairs=blocked_pairs
    )
    count = len(best)
    first_mutable = 1 if start is not None else 0
    last_exclusive = count - 1 if end is not None else count
    for _ in range(max_passes):
        improved = False
        for left in range(first_mutable, last_exclusive):
            for right in range(left + 1, last_exclusive):
                candidate = list(best)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                score = _order_score(
                    candidate,
                    graph,
                    analyses,
                    journey=journey,
                    listener_profile=listener_profile,
                    blocked_pairs=blocked_pairs,
                )
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True
        for source in range(first_mutable, last_exclusive):
            for target in range(first_mutable, last_exclusive):
                if source == target:
                    continue
                candidate = list(best)
                node = candidate.pop(source)
                candidate.insert(target, node)
                if not _valid_order(candidate, count, start, end):
                    continue
                score = _order_score(
                    candidate,
                    graph,
                    analyses,
                    journey=journey,
                    listener_profile=listener_profile,
                    blocked_pairs=blocked_pairs,
                )
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True
        for left in range(first_mutable, last_exclusive - 1):
            for right in range(left + 2, last_exclusive + 1):
                candidate = best[:left] + list(reversed(best[left:right])) + best[right:]
                if not _valid_order(candidate, count, start, end):
                    continue
                score = _order_score(
                    candidate,
                    graph,
                    analyses,
                    journey=journey,
                    listener_profile=listener_profile,
                    blocked_pairs=blocked_pairs,
                )
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True
        if not improved:
            break
    return best


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
    count = len(analyses)
    if start_index is not None and not 0 <= start_index < count:
        raise ValueError("start_index is outside the track list")
    if end_index is not None and not 0 <= end_index < count:
        raise ValueError("end_index is outside the track list")
    if start_index is not None and start_index == end_index and count > 1:
        raise ValueError("The same track cannot be fixed as both opener and closer.")
    blocked = blocked_pairs or set()

    # A two-track set has only two possible orders. Running receding-horizon,
    # unified beam search and local-improvement machinery cannot discover a
    # third option, so score the complete search space directly. This removes
    # avoidable model/context work from the most common quick-create case.
    if count == 2:
        exact = [
            order
            for order in ([0, 1], [1, 0])
            if _valid_order(order, count, start_index, end_index)
            and not any((a, b) in blocked for a, b in zip(order, order[1:]))
        ]
        if not exact:
            raise RuntimeError("The optimizer could not construct a valid two-track order.")
        return max(
            exact,
            key=lambda value: _order_score(
                value,
                graph,
                analyses,
                journey=journey,
                listener_profile=listener_profile,
                blocked_pairs=blocked,
            ),
        )

    candidates = list(
        _greedy_orders(
            graph,
            count,
            start=start_index,
            end=end_index,
            blocked_pairs=blocked,
        )
    )
    policy_config = SetPolicyConfig.from_environment()
    policy_starts = [start_index] if start_index is not None else [index for index in range(count) if index != end_index]
    large_set = count >= _large_set_threshold()
    if large_set and start_index is None and candidates:
        # The unified beam already explores opener diversity. Running a full
        # receding-horizon search from every possible opener is redundant and
        # became the dominant cost at 10-20 tracks. Keep the strongest greedy
        # opener seeds, then let beam search + local refinement challenge them.
        start_limit = _env_int("DJAI_LARGE_SET_POLICY_STARTS", 4, minimum=2, maximum=8)
        ranked_seeds = sorted(
            candidates,
            key=lambda order: sum(
                float(graph[(left, right)].objective_score)
                for left, right in zip(order, order[1:])
            ),
            reverse=True,
        )
        reduced: list[int] = []
        for order in ranked_seeds:
            if not order:
                continue
            opener = int(order[0])
            if opener not in reduced:
                reduced.append(opener)
            if len(reduced) >= start_limit:
                break
        if reduced:
            policy_starts = reduced
    for policy_start in policy_starts:
        if policy_start is None:
            continue
        try:
            policy_order, _ = build_receding_horizon_order(
                policy_start, graph, analyses, journey=journey, end=end_index, config=policy_config
            )
            if (
                _valid_order(policy_order, count, start_index, end_index)
                and not any((a, b) in blocked for a, b in zip(policy_order, policy_order[1:]))
            ):
                candidates.append(policy_order)
        except (RuntimeError, ValueError):
            pass
    try:
        unified = build_unified_ai_order(
            graph,
            analyses,
            journey=journey,
            start=start_index,
            end=end_index,
            blocked_pairs=blocked,
        )
        if _valid_order(unified.order, count, start_index, end_index):
            candidates.append(unified.order)
    except (RuntimeError, ValueError):
        # The unified search is additive. Existing deterministic/receding-horizon
        # candidates remain the fallback when it cannot complete a safe route.
        pass
    if not candidates:
        raise RuntimeError("The optimizer could not construct a valid set order.")
    best = max(
        candidates,
        key=lambda value: _order_score(
            value,
            graph,
            analyses,
            journey=journey,
            listener_profile=listener_profile,
            blocked_pairs=blocked,
        ),
    )
    local_passes = (
        _env_int("DJAI_LARGE_SET_LOCAL_PASSES", 2, minimum=1, maximum=4)
        if large_set
        else 8
    )
    return _local_improve(
        best,
        graph,
        analyses,
        journey=journey,
        listener_profile=listener_profile,
        start=start_index,
        end=end_index,
        blocked_pairs=blocked,
        max_passes=local_passes,
    )


@dataclass(frozen=True, slots=True)
class _CompleteMoveRoute:
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
            "decisions": [decision.to_dict(top_k=top_k) for decision in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class _LookaheadChoice:
    winner: Any
    candidates: tuple[Any, ...]
    search_score: float
    projected_path: tuple[int, ...]
    lookahead_depth: int
    retrieval: CandidateRetrievalResult


def _choose_complete_move_with_lookahead(
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
) -> _LookaheadChoice | None:
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
    set_state = build_set_state([analyses[index] for index in played_order], played_plans)
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

    best: _LookaheadChoice | None = None
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
            future = _choose_complete_move_with_lookahead(
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

        choice = _LookaheadChoice(
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


def _complete_move_route_from_opener(
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
) -> _CompleteMoveRoute:
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
    professional_pilot = ProfessionalBrainConfig.from_environment().adaptive_pilot_finalists

    while remaining:
        preflight_indices = sorted(remaining)
        if end_index is not None and end_index in preflight_indices and len(preflight_indices) > 1:
            preflight_indices = [index for index in preflight_indices if index != end_index]
        preflight = assess_context_preflight(
            track_a=analyses[current],
            candidate_tracks=[analyses[index] for index in preflight_indices],
            dj_state=dj_state,
            config=adaptive_cfg,
        )
        initial_depth = (
            min(config.lookahead_moves, max(1, adaptive_cfg.easy_lookahead))
            if adaptive_cfg.enabled and preflight.easy_candidate_context
            else config.lookahead_moves
        )
        choice = _choose_complete_move_with_lookahead(
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
            return _CompleteMoveRoute(
                order=tuple(order),
                decisions=tuple(decisions),
                score=(sum(item.commit_score or item.search_score or item.winner.selection_score for item in decisions) / max(1, len(decisions))),
                completed=False,
                reason=f"no executable complete move from track index {current}",
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
        if provisional_budget.lookahead_target > initial_depth and len(remaining) > 1:
            deeper = _choose_complete_move_with_lookahead(
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
                search_depth_used = provisional_budget.lookahead_target
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
        commit_score = float(choice.search_score or winner.selection_score)
        headphones_report: dict[str, Any] | None = None
        if config.mode == "move_authoritative" and headphones_config.enabled:
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
            headphones = listen_to_complete_moves(
                preferred=choice.winner,
                candidates=choice.candidates,
                analyses=analyses,
                set_state=current_set_state,
                journey=journey,
                progress=build_timing_state(
                    [analyses[index] for index in order],
                    plans,
                    intent=timing_intent,
                    fallback_progress=len(order) / max(1, len(analyses) - 1),
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
                if winner.plan.execution_signature != choice.winner.plan.execution_signature:
                    # Listening changed the first projected action. The prior
                    # future path was projected from a different DJState, so do
                    # not pretend it remains valid. Advance only the search
                    # projection and replan; DJ Executive V4 owns intent binding, compute budgeting, and final commit.
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
            candidate for candidate in choice.candidates if int(candidate.next_index) == int(winner.next_index)
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
        order.append(winner.next_index)
        remaining.remove(winner.next_index)
        plans.append(winner.plan)
        current = winner.next_index
        set_state = build_set_state([analyses[index] for index in order], plans)
        projected = winner.move.projected_state
        dj_state = build_dj_state(
            analyses[current],
            source_cursor=projected.source_cursor_seconds,
            effective_bpm=projected.effective_bpm,
            render_rate=projected.render_rate,
            set_state=set_state,
            tempo_owner=projected.tempo_owner,
        )

    completed = len(order) == len(analyses) and (end_index is None or order[-1] == end_index)
    return _CompleteMoveRoute(
        order=tuple(order),
        decisions=tuple(decisions),
        score=(sum(item.commit_score or item.search_score or item.winner.selection_score for item in decisions) / max(1, len(decisions))),
        completed=completed,
        reason="" if completed else "fixed closer constraint could not be satisfied",
    )


def _large_crate_complete_move_config(
    config: CompleteMoveConfig,
    track_count: int,
) -> CompleteMoveConfig:
    """Bound structural search cost for large crates without touching safety gates."""
    if track_count < _large_set_threshold():
        return config
    return replace(
        config,
        lookahead_moves=min(config.lookahead_moves, 2),
        search_branch_width=min(config.search_branch_width, 2),
        lazy_candidate_width=min(config.lazy_candidate_width, 6),
        lazy_expansion_width=min(config.lazy_expansion_width, 6),
        opener_candidates=min(config.opener_candidates, 2),
        candidates_per_track=min(config.candidates_per_track, 3),
    )


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
        )
        for opener in openers
    ]
    complete = [route for route in routes if route.completed]
    if not complete:
        reason = "; ".join(route.reason for route in routes if route.reason) or "no complete route"
        return _CompleteMoveRoute(order=(), decisions=(), score=0.0, completed=False, reason=reason)
    best_scout = max(
        complete,
        key=lambda route: (
            route.score,
            sum(decision.winner.move.future_optionality for decision in route.decisions),
            tuple(-index for index in route.order),
        ),
    )
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
    )
    return committed if committed.completed else best_scout

def _required_source_body(
    track: TrackAnalysis,
    current_cursor: float,
    current_rate: float,
    *,
    minimum_body_seconds: float,
    minimum_body_fraction: float,
) -> float:
    remaining = max(0.0, track.duration_seconds - current_cursor)
    move_cfg = MoveBrainConfig.from_environment()
    if move_cfg.enabled and track.bpm > 1.0:
        # 1.4 hard minimums are musical bars, not a percentage of the file.
        # Convert the user's pacing request to the next four-bar boundary and
        # guarantee a small phrase body. The historical fraction remains a UI
        # pacing hint elsewhere; it no longer forces SAUND to ignore a strong
        # early phrase opportunity on a long record.
        bar_source_seconds = max(0.20, (track.time_signature if track.time_signature in {3, 4, 6} else 4) * 60.0 / track.bpm)
        requested_source = minimum_body_seconds * max(current_rate, 1e-9)
        requested_bars = int(math.ceil(requested_source / bar_source_seconds))
        requested_bars = int(math.ceil(max(move_cfg.minimum_body_bars, requested_bars) / 4.0) * 4)
        desired = requested_bars * bar_source_seconds
    else:
        source_for_time = minimum_body_seconds * max(current_rate, 1e-9)
        source_for_fraction = track.duration_seconds * minimum_body_fraction
        desired = max(source_for_time, source_for_fraction)
    if track.duration_seconds < max(60.0, minimum_body_seconds * 2.0):
        desired = min(desired, max(0.75, remaining * 0.45))
    return max(0.75, min(desired, max(0.75, remaining - 0.5)))

def _mixing_window(
    track: TrackAnalysis,
    current_cursor: float,
    current_rate: float,
    required_source: float,
) -> tuple[float, float]:
    """Return a weak pacing target and latest window in source seconds.

    In 1.4 this is deliberately subordinate to phrase/payoff reasoning.  When
    the Move Brain is enabled the pacing clock is expressed in bars, not wall
    time; the candidate scorer gives this window only a small residual weight.
    """
    remaining = max(0.0, track.duration_seconds - current_cursor)
    if remaining <= 1.0:
        return current_cursor, current_cursor
    move_cfg = MoveBrainConfig.from_environment()
    if move_cfg.enabled and track.bpm > 1.0:
        bar_source_seconds = max(0.20, (track.time_signature if track.time_signature in {3, 4, 6} else 4) * 60.0 / track.bpm)
        target_cap = move_cfg.soft_target_bars * bar_source_seconds
        latest_cap = move_cfg.soft_latest_bars * bar_source_seconds
    else:
        rate = max(current_rate, 1e-9)
        target_cap = float(os.environ.get("DJAI_TARGET_SOURCE_PLAY_SECONDS", "150")) * rate
        latest_cap = float(os.environ.get("DJAI_SOFT_MAX_SOURCE_PLAY_SECONDS", "210")) * rate
    target_body = max(required_source, min(target_cap, remaining * 0.62))
    latest_body = max(target_body, min(latest_cap, remaining * 0.80))
    latest_body = min(latest_body, max(required_source, remaining - 0.5))
    return current_cursor + target_body, current_cursor + latest_body

def _candidate_diverse_by_bar(
    candidates: list[tuple[float, TransitionCandidate]],
    *,
    limit: int,
    minimum_bar_distance: int = 2,
) -> list[TransitionCandidate]:
    selected: list[TransitionCandidate] = []
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(abs(candidate.bar_index - existing.bar_index) >= minimum_bar_distance for existing in selected):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(candidates)):
        for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


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
    """Rank exits primarily by musical state; elapsed time is only a weak prior.

    SAUND 1.4 no longer becomes eager to leave merely because a fixed number
    of seconds elapsed.  Phrase readiness, protected payoffs and the current
    bar/phrase state dominate.  The legacy timing window remains as a bounded
    sanity prior so malformed structure analysis cannot strand the compiler.
    """
    target_exit, soft_latest = _mixing_window(track, source_cursor, current_rate, required_source)
    required_exit = source_cursor + required_source
    remaining = max(1.0, track.duration_seconds - source_cursor)
    spread = max(12.0, remaining * 0.24)
    section_value = {
        "breakdown": 1.00,
        "groove": 0.94,
        "build": 0.78,
        "outro": 0.86,
        "drop": 0.44,
        "intro": 0.18,
        "unknown": 0.52,
    }
    active_timeline = timeline or build_musical_timeline(track)
    active_state = dj_state or build_dj_state(
        track,
        source_cursor=source_cursor,
        effective_bpm=track.bpm * current_rate,
        render_rate=current_rate,
        timeline=active_timeline,
    )
    song_form_safe_bar = earliest_safe_song_form_exit_bar(active_timeline, active_state.source_bar)
    song_form_safe_time = active_timeline.bar_start_seconds(song_form_safe_bar)
    required_exit = max(required_exit, song_form_safe_time)
    target_exit = max(target_exit, required_exit)
    soft_latest = max(soft_latest, required_exit)
    scored: list[tuple[float, TransitionCandidate]] = []
    protected: list[tuple[float, TransitionCandidate]] = []
    for candidate in track.exit_candidates:
        if candidate.time_seconds < required_exit - 1e-6:
            continue
        musical, move_meta = exit_candidate_musical_score(
            track, candidate, active_state, timeline=active_timeline
        )
        position = math.exp(-(((candidate.time_seconds - target_exit) / spread) ** 2))
        late = max(0.0, candidate.time_seconds - soft_latest) / max(remaining * 0.20, 1.0)
        low_vocal = 1.0 - max(0.0, min(1.0, candidate.vocal_probability))
        value = (
            0.48 * musical
            + 0.18 * candidate.score
            + 0.10 * candidate.boundary_strength
            + 0.09 * candidate.phrase_strength
            + 0.06 * section_value.get(candidate.section_label, 0.5)
            + 0.04 * low_vocal
            + 0.05 * position
            - 0.07 * min(1.0, late)
        )
        item = (value, candidate)
        if bool(move_meta.get("protected_payoff_violation", False)):
            protected.append(item)
        else:
            scored.append(item)
    # Do not mix out before a musical promise if any valid post-payoff exit is
    # available. Protected exits survive only as a last-resort pool.
    if not scored:
        scored = protected
    if not scored:
        fallback = [
            candidate for candidate in track.exit_candidates
            if candidate.time_seconds >= max(source_cursor + 0.25, required_exit)
        ]
        scored = [(candidate.score, candidate) for candidate in fallback]
    return _candidate_diverse_by_bar(scored, limit=limit), target_exit, soft_latest

def _sequence_entry_candidates(track: TrackAnalysis, *, limit: int = 10) -> list[TransitionCandidate]:
    """Rank incoming musical opportunities by phrase role instead of clock time."""
    if not track.entry_candidates:
        return []
    move_cfg = MoveBrainConfig.from_environment()
    if not move_cfg.enabled:
        # Exact 1.3 compatibility path: prefer a mature cue and avoid literal
        # time zero when another usable incoming phrase exists.
        bar_duration = max(0.5, (track.time_signature if track.time_signature in {3, 4, 6} else 4) * 60.0 / max(track.bpm, 1.0))
        mature_after = min(4.0 * bar_duration, max(2.0, track.duration_seconds * 0.08))
        preferred_before = min(135.0, track.duration_seconds * 0.48)
        mature = [
            candidate for candidate in track.entry_candidates
            if candidate.time_seconds >= mature_after and candidate.time_seconds <= preferred_before
        ]
        pool = mature or [
            candidate for candidate in track.entry_candidates
            if candidate.time_seconds <= min(150.0, track.duration_seconds * 0.55)
        ] or list(track.entry_candidates)
        section_value = {
            "groove": 1.00,
            "build": 0.96,
            "drop": 0.94,
            "intro": 0.72,
            "breakdown": 0.70,
            "outro": 0.18,
            "unknown": 0.52,
        }
        target_time = min(18.0 * bar_duration, max(mature_after, track.duration_seconds * 0.16))
        spread = max(8.0 * bar_duration, track.duration_seconds * 0.18, 12.0)
        scored: list[tuple[float, TransitionCandidate]] = []
        for candidate in pool:
            position = math.exp(-(((candidate.time_seconds - target_time) / spread) ** 2))
            low_vocal = 1.0 - max(0.0, min(1.0, candidate.vocal_probability))
            value = (
                0.34 * candidate.score
                + 0.18 * candidate.boundary_strength
                + 0.17 * candidate.phrase_strength
                + 0.13 * section_value.get(candidate.section_label, 0.5)
                + 0.10 * low_vocal
                + 0.08 * position
            )
            scored.append((value, candidate))
        return _candidate_diverse_by_bar(scored, limit=limit)

    # Human-DJ mode: keep extreme late/outro cues available only when the
    # analysis offers nothing earlier. A real intro at bar zero is a valid
    # musical entry and must be allowed to compete with later cues.
    primary = [
        candidate for candidate in track.entry_candidates
        if candidate.section_label != "outro"
        and candidate.time_seconds <= max(24.0, track.duration_seconds * 0.68)
    ]
    pool = primary or list(track.entry_candidates)
    timeline = build_musical_timeline(track)
    scored: list[tuple[float, TransitionCandidate]] = []
    for candidate in pool:
        musical, _ = entry_candidate_musical_score(track, candidate, timeline=timeline)
        low_vocal = 1.0 - max(0.0, min(1.0, candidate.vocal_probability))
        value = (
            0.62 * musical
            + 0.14 * candidate.score
            + 0.09 * candidate.boundary_strength
            + 0.08 * candidate.phrase_strength
            + 0.07 * low_vocal
        )
        scored.append((value, candidate))
    return _candidate_diverse_by_bar(scored, limit=limit)

def _dedupe_plans(plans: list[TransitionPlan]) -> list[TransitionPlan]:
    unique: dict[tuple[str, str, float, float, int, str], TransitionPlan] = {}
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



def _complete_move_preference_score(plan: TransitionPlan, decision: CompleteMoveDecision) -> float:
    """Measure how closely a path-dependent plan preserves a generated 1.7 move.

    The final compiler is allowed to substitute a safer/rehearsed plan, but in
    move_authoritative mode it should preserve the generator's concrete timing
    and technique whenever those choices remain executable.
    """
    target = decision.winner.move
    exit_bar = int(plan.metadata.get("exit_bar_index", -10_000))
    entry_bar = int(plan.metadata.get("entry_bar_index", -10_000))
    exit_fit = max(0.0, 1.0 - abs(exit_bar - int(target.exit_bar)) / 16.0)
    entry_fit = max(0.0, 1.0 - abs(entry_bar - int(target.entry_bar)) / 16.0)
    duration_fit = max(0.0, 1.0 - abs(int(plan.transition_bars) - int(target.transition_bars)) / 16.0)
    technique_fit = 1.0 if str(plan.technique) == str(target.technique) else 0.0
    intent_fit = 1.0 if str(plan.intent) == str(target.intent) else 0.0
    tempo_fit = 1.0 if str(plan.tempo_strategy) == str(target.tempo_strategy) else 0.35
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


def _professional_brain_config_for_transition(
    quality_loop: QualityLoopConfig,
    preferred_complete_decision: CompleteMoveDecision | None,
) -> ProfessionalBrainConfig:
    """Apply V4 compute budgeting without changing Professional Brain gates."""

    thinking_budget = (
        preferred_complete_decision.thinking_budget
        if preferred_complete_decision is not None
        else None
    )
    return ProfessionalBrainConfig.from_environment(
        finalists=(
            int(os.environ["DJAI_BRAIN_FINALISTS"])
            if os.environ.get("DJAI_BRAIN_FINALISTS", "").isdigit()
            else max(6, quality_loop.top_k * 2)
        ),
        adaptive_pilot_finalists=(
            thinking_budget.professional_pilot_finalists
            if thinking_budget is not None
            else None
        ),
        adaptive_rehearsal=(
            thinking_budget.professional_adaptive_rehearsal
            if thinking_budget is not None
            else None
        ),
        measurement_sample_rate=quality_loop.sample_rate,
        measurement_context_seconds=quality_loop.context_seconds,
        measurement_minimum=quality_loop.minimum_measured_score,
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

        styles = (
            [preferred_style]
            if preferred_style in {"safe", "creative", "emergency"}
            else ["safe", "creative", "emergency"]
        )
        all_plans: list[TransitionPlan] = []
        matrix: PairEvidenceMatrix | None = None
        stems_available = bool(stem_maps[from_index] and stem_maps[to_index])
        if selected_exits:
            matrix = active_session.get((from_index, to_index))
            if matrix is not None:
                matrix = matrix.subset(
                    exit_candidates=selected_exits,
                    entry_candidates=selected_entries or None,
                )
                # Large-crate graph scouting deliberately stores only a small
                # cue matrix. Final path compilation must restore the complete
                # selected cue pool instead of silently inheriting the scout.
                if (
                    len(matrix.exit_candidates) < len(selected_exits)
                    or (selected_entries and len(matrix.entry_candidates) < len(selected_entries))
                ):
                    matrix = None
            if matrix is None or not matrix.pairs:
                matrix = build_pair_evidence_matrix(
                    track_a,
                    track_b,
                    exit_candidates=selected_exits,
                    entry_candidates=selected_entries or None,
                    scoring_model=active_session.scoring_model,
                )
                active_session.store((from_index, to_index), matrix)

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
                plan.stem_paths_a = dict(stem_maps[from_index])
                plan.stem_paths_b = dict(stem_maps[to_index])
                plan.metadata = {
                    **plan.metadata,
                    "sequence_position": global_position,
                    "source_cursor_before_body": round(source_cursor, 6),
                    "minimum_source_body": round(required_source, 6),
                    "preferred_exit_time": round(target_exit, 6),
                    "soft_latest_exit_time": round(soft_latest_exit, 6),
                    "selected_entry_pool": [round(item.time_seconds, 6) for item in selected_entries],
                    "sequence_evidence_reused": True,
                    "sequence_matrix_size": matrix.compatibility_evaluations,
                    "stem_rendering": bool(
                        stems_available
                        and plan.technique
                        in {"eq_bass_swap", "filter_build", "double_drop", "stem_vocal_overlay", "stem_choreography"}
                    ),
                    "performance_layer_rendering": plan.technique
                    in {
                        "layer_vocal_bridge",
                        "layer_incoming_vocal_handoff",
                        "layer_instrumental_handoff",
                    },
                }
                # Stem paths may change the executable program from master to
                # stem buses. Recompile before any rehearsal or execution.
                attach_performance_program(plan)
                seal_execution_plan(plan)
                all_plans.append(plan)

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

        all_plans = _dedupe_plans(all_plans)
        for candidate in all_plans:
            candidate.metadata = {
                **candidate.metadata,
                "selection_utility": round(
                    _edge_objective(
                        candidate,
                        track_a,
                        track_b,
                        journey=journey,
                        listener_profile=listener_profile,
                    ),
                    8,
                ),
            }

        all_plans = rank_transition_plans_as_moves(
            track_a,
            track_b,
            all_plans,
            state=dj_state,
        )

        preferred_complete_decision = (
            preferred_complete_moves.get((from_index, to_index))
            if preferred_complete_moves is not None
            else None
        )
        if preferred_complete_decision is not None:
            for candidate in all_plans:
                match = _complete_move_preference_score(candidate, preferred_complete_decision)
                previous = float(candidate.metadata.get("selection_utility", candidate.score))
                # Exact/near-exact complete moves own selection unless a later
                # rehearsal or deterministic safety gate rejects them. Looser
                # matches receive a bounded preference rather than a hard lock.
                preferred_utility = (
                    max(previous, 0.985)
                    if match >= 0.94
                    else 0.78 * previous + 0.22 * match
                )
                candidate.metadata = {
                    **candidate.metadata,
                    "selection_utility": round(preferred_utility, 8),
                    "complete_move_preference": {
                        "version": COMPLETE_MOVE_VERSION,
                        "match": round(match, 6),
                        "target": preferred_complete_decision.winner.move.to_dict(),
                        "generator_score": preferred_complete_decision.winner.selection_score,
                        "safety_may_override": True,
                    },
                    "dj_state_taste": (
                        state_taste_payload(preferred_complete_decision.winner.state)
                        if preferred_complete_decision.winner.state is not None
                        else None
                    ),
                    "dj_move_taste": move_taste_payload(preferred_complete_decision.winner.move),
                    "state_context_id": (
                        state_context_id(preferred_complete_decision.winner.state)
                        if preferred_complete_decision.winner.state is not None
                        else None
                    ),
                    "state_move_taste_shadow": preferred_complete_decision.winner.taste_shadow,
                    "crate_memory": preferred_complete_decision.winner.crate_memory,
                    "adaptive_thinking": (
                        preferred_complete_decision.thinking_budget.to_dict()
                        if preferred_complete_decision.thinking_budget is not None
                        else None
                    ),
                }
                if match >= 0.94 and preferred_complete_decision.winner.move.deck_operations:
                    candidate.metadata = {
                        **candidate.metadata,
                        "deck_operations": [
                            operation.to_dict()
                            for operation in preferred_complete_decision.winner.move.deck_operations
                        ],
                        "structural_operation_authority": {
                            "version": "1.7.2",
                            "source": "complete_djmove_exact_match",
                            "match": round(match, 6),
                            "deterministic_renderer_authoritative": True,
                        },
                    }
                    # Structural operations are sample-affecting executable IR.
                    # Recompile and reseal so the program fingerprint and final
                    # execution signature bind the exact operation payload.
                    attach_performance_program(candidate)
                    seal_execution_plan(candidate)
            all_plans.sort(
                key=lambda candidate: (
                    float(candidate.metadata.get("selection_utility", candidate.score)),
                    float(candidate.score),
                    candidate.execution_signature,
                ),
                reverse=True,
            )

        policy_context = TransitionPolicyContext(
            journey=journey,
            journey_progress=journey_progress,
            tempo_policy=tempo_policy,
            effective_bpm_a=current_effective_bpm,
            render_rate_a=current_rate,
            stems_available=stems_available,
        )
        if transition_policy_authority == "bounded_advisory":
            apply_bounded_transition_policy(
                all_plans,
                context=policy_context,
                crowd_directive=crowd_directive,
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
                "sequence_evidence_reused": bool(matrix is not None and matrix.pairs),
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
    measurement_run_offset = measurement_log_offset(
        quality_loop.log_path if quality_loop is not None and quality_loop.enabled else None
    )

    complete_move_cfg = CompleteMoveConfig.from_environment()
    headphones_cfg = AIHeadphonesConfig.from_environment()
    lazy_graph_authority = complete_move_cfg.mode == "move_authoritative" and manual_order is None
    if lazy_graph_authority:
        # 1.7.1 materializes graph edges only as the move search reaches them.
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

    def _choose_automatic_order(active_blocked: set[tuple[int, int]]) -> list[int]:
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
                # A bounded large-crate scout may dead-end after committing an
                # unfortunate opener even though a complete sparse route exists.
                # Escalate the *structural* search first and reuse every graph
                # pair already materialized. This is much cheaper than jumping
                # immediately to the historical broad graph; rendered rehearsal
                # and all hard safety gates remain unchanged.
                expanded_cfg = replace(
                    complete_move_cfg,
                    opener_candidates=min(4, max(complete_move_cfg.opener_candidates + 2, 3)),
                    search_branch_width=min(3, max(complete_move_cfg.search_branch_width + 1, 2)),
                    lazy_candidate_width=min(8, max(complete_move_cfg.lazy_candidate_width + 2, 5)),
                    lazy_expansion_width=min(8, max(complete_move_cfg.lazy_expansion_width + 2, 4)),
                )
                if progress:
                    progress(
                        "route-search",
                        1,
                        2,
                        "Sparse large-crate scout incomplete; expanding opener/branch search before broad graph fallback",
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
                            2,
                            2,
                            "Expanded sparse route completed; broad graph fallback avoided",
                        )
                    return list(expanded_route.order)
        if lazy_graph_authority:
            # Safe fallback only: if both bounded sparse complete-move searches
            # cannot finish, materialize the historical graph before invoking
            # legacy order.
            if progress:
                progress(
                    "route-search",
                    2,
                    2,
                    "Sparse route still incomplete; materializing broad graph fallback",
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

    def _compile_for_executive(
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
    executive_result = executive_orchestrator.run_prepared_set(
        manual_order=(list(manual_order) if manual_order is not None else None),
        choose_order=_choose_automatic_order,
        compile_order=_compile_for_executive,
        decisions=_latest_complete_move_decisions,
        top_alternatives=complete_move_cfg.report_top_k,
    )
    order = list(executive_result.order)
    transitions = list(executive_result.transitions)
    blocked_pairs = set(executive_result.blocked_pairs)
    replan_trace = list(executive_result.replan_trace)
    dj_executive_report = dict(executive_result.executive_report)
    if lazy_graph_authority:
        # Scoring diagnostics need only the adjacent baseline/final edges, not
        # the full N*(N-1) graph. Keep 1.7.1 sparse even for reporting.
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
    latest_complete_route = complete_move_runs[-1] if complete_move_runs else None
    complete_route_order = list(latest_complete_route.order) if latest_complete_route and latest_complete_route.completed else []
    route_divergence = [
        position
        for position, (actual, proposed) in enumerate(zip(order, complete_route_order))
        if actual != proposed
    ]
    generated_signatures = (
        [decision.winner.plan.execution_signature for decision in latest_complete_route.decisions]
        if latest_complete_route is not None
        else []
    )
    compiled_signatures = [edge.plan.execution_signature for edge in transitions]
    signature_matches = sum(
        1 for generated, compiled in zip(generated_signatures, compiled_signatures) if generated and generated == compiled
    )
    complete_move_report = {
        "version": COMPLETE_MOVE_VERSION,
        "mode": complete_move_cfg.mode,
        "active": bool(not manual and complete_move_cfg.mode != "legacy"),
        "legacy_planner_preserved": True,
        "audio_safety_compiler_authoritative": True,
        "route_authoritative": bool(not manual and complete_move_cfg.mode == "move_authoritative" and complete_route_order == order),
        "shadow_only": bool(not manual and complete_move_cfg.mode == "move_shadow"),
        "lookahead_moves": complete_move_cfg.lookahead_moves,
        "adaptive_thinking": adaptive_thinking_status(),
        "commit_moves": 1,
        "receding_horizon": True,
        "structural_deck_operations": True,
        "true_wait_compiled": True,
        "advanced_structural_operations_opt_in": True,
        "ai_headphones": ai_headphones_status(planner_mode=complete_move_cfg.mode),
        "ai_headphones_authoritative_decisions": (
            sum(
                1
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
                if isinstance(decision.headphones, dict) and bool(decision.headphones.get("authoritative"))
            )
        ),
        "ai_headphones_rendered_candidates": (
            sum(
                int(decision.headphones.get("rendered_candidates", 0))
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
                if isinstance(decision.headphones, dict)
            )
        ),
        "thinking_budget_levels": {
            level: sum(
                1
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
                if decision.thinking_budget is not None
                and decision.thinking_budget.brittleness.level == level
            )
            for level in ("easy", "normal", "brittle")
        },
        "thinking_budget_deep_search_escalations": sum(
            1
            for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
            if decision.thinking_budget is not None and decision.thinking_budget.lookahead_escalated
        ),
        "complete_move_generated_candidates": (
            sum(
                len(decision.candidates)
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
            )
        ),
        "crate_memory_informed_decisions": (
            sum(
                1
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
                if isinstance(decision.winner.crate_memory, dict)
                and abs(float(decision.winner.crate_memory.get("adjustment", 0.0) or 0.0)) > 1e-9
            )
        ),
        "crate_memory_total_adjustment": round(
            sum(
                float(decision.winner.crate_memory.get("adjustment", 0.0) or 0.0)
                for decision in (latest_complete_route.decisions if latest_complete_route is not None else ())
                if isinstance(decision.winner.crate_memory, dict)
            ),
            6,
        ),
        "lazy_candidate_retrieval": True,
        "lazy_candidate_width": complete_move_cfg.lazy_candidate_width,
        "lazy_expansion_width": complete_move_cfg.lazy_expansion_width,
        "search_branch_width": complete_move_cfg.search_branch_width,
        "opener_candidates": complete_move_cfg.opener_candidates,
        "sparse_graph_authority": lazy_graph_authority,
        "materialized_graph_pairs": len(graph),
        "possible_graph_pairs": len(analyses) * max(0, len(analyses) - 1),
        "run_count": len(complete_move_runs),
        "proposed_order": complete_route_order,
        "final_order": list(order),
        "order_matches": bool(complete_route_order and complete_route_order == list(order)),
        "divergence_positions": route_divergence,
        "generated_transition_signature_matches": signature_matches,
        "compiled_transition_count": len(compiled_signatures),
        "latest_route": (
            latest_complete_route.to_dict(top_k=complete_move_cfg.report_top_k)
            if latest_complete_route is not None
            else None
        ),
    }
    final_timing_state = build_timing_state(
        [analyses[index] for index in order],
        [edge.plan for edge in transitions],
        intent=timing_intent,
        projected_total_seconds=estimate_set_duration(order, analyses, transitions),
        fallback_progress=1.0,
    )
    plan = SetPlan(
        plan_id=plan_id,
        tracks=analyses,
        order=order,
        transitions=transitions,
        baseline_order=baseline_order,
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        estimated_duration_seconds=estimate_set_duration(order, analyses, transitions),
        source=source,
        algorithm=algorithm,
        manual=manual,
        warnings=list(dict.fromkeys(warnings)),
        automatic_delivery=automatic_delivery,
        listener=listener,
        metadata={
            "tempo_policy": tempo_policy,
            "preferred_style": preferred_style,
            "journey": journey,
            "timing": {
                "intent": (timing_intent or SetTimingIntent()).normalized().to_dict(),
                "state": final_timing_state.to_dict(),
                "time_aware_progress": timing_intent is not None and timing_intent.target_duration_seconds is not None,
            },
            "journey_arc": journey_metadata(order, analyses, journey=journey),
            "professional_story": set_story_report(order, analyses, journey=journey),
            "set_director_story": set_director_report(order, analyses, transitions, journey=journey),
            "dj_memory": build_dj_memory([analyses[index] for index in order], [edge.plan for edge in transitions]).to_dict(),
            "listener": listener,
            "listener_profile_confidence": (
                listener_profile.confidence if listener_profile is not None else 0.0
            ),
            "automatic_delivery": automatic_delivery,
            "complete_move_planner": complete_move_report,
            "dj_executive": dj_executive_report,
            "canonical_set_score": {
                "baseline": round(baseline_score, 6),
                "optimized": round(optimized_score, 6),
                "compiled_transition_objective_sum": round(compiled_transition_score, 6),
                "comparison_basis": "order_score",
            },
            "fallback_transition_count": sum(
                1 for edge in transitions if edge.plan.fallback_level > 0
            ),
            "minimum_body_seconds": minimum_body_seconds,
            "minimum_body_fraction": minimum_body_fraction,
            "directed_edge_count": len(graph),
            "performance": {"planning": session.to_dict()},
            "large_crate_planning": {
                "enabled": len(analyses) >= _large_set_threshold(),
                "track_count": len(analyses),
                "threshold": _large_set_threshold(),
                "graph_mode": session.graph_mode,
                "graph_pair_count": session.graph_pair_count,
                "graph_scout_cues": session.graph_scout_cues,
                "policy_start_limit": (
                    _env_int("DJAI_LARGE_SET_POLICY_STARTS", 4, minimum=2, maximum=8)
                    if len(analyses) >= _large_set_threshold()
                    else len(analyses)
                ),
                "local_improve_passes": (
                    _env_int("DJAI_LARGE_SET_LOCAL_PASSES", 2, minimum=1, maximum=4)
                    if len(analyses) >= _large_set_threshold()
                    else 8
                ),
                "maximum_replans": max_replans,
                "final_edges_restore_full_cues": True,
            },
            "quality_loop": {
                "enabled": bool(quality_loop is not None and quality_loop.enabled),
                "top_k": quality_loop.top_k if quality_loop is not None else 0,
                "sample_rate": quality_loop.sample_rate if quality_loop is not None else None,
                "checked_transitions": sum(1 for edge in transitions if edge.plan.quality_checked),
                "passed_transitions": sum(
                    1
                    for edge in transitions
                    if bool(
                        dict(edge.plan.metadata.get("quality_loop", {})).get(
                            "selection_gate_passed",
                            dict(edge.plan.metadata.get("quality_loop", {})).get("gate_passed", False),
                        )
                    )
                ),
                "measurement_version": MEASUREMENT_VERSION if quality_loop is not None else None,
            },
            "compiled_execution": True,
            "unified_transition_intelligence": {
                "enabled": True,
                "simple_pro_shared_brain": True,
                "review_needed_is_replan": True,
                "blocked_edges": [list(pair) for pair in sorted(blocked_pairs)],
                "replan_count": len(replan_trace),
                "replan_trace": replan_trace,
                "maximum_replans": max_replans,
            },
            "saund_ai_1": {
                "unified_planner_version": UNIFIED_PLANNER_VERSION,
                "full_set_benchmark_version": FULL_SET_BENCHMARK_VERSION,
                "learned_authority_policy": "bounded_and_safety_gated",
            },
        },
    )
    plan.metadata["full_set_benchmark"] = evaluate_full_set(plan).to_dict()
    if quality_loop is not None and quality_loop.enabled and quality_loop.log_path:
        finalize_measurement_run(
            quality_loop.log_path,
            start_offset=measurement_run_offset,
            accepted_signatures=[
                edge.plan.execution_signature
                for edge in transitions
                if edge.plan.measurement is not None
            ],
            accepted_plans=[
                edge.plan for edge in transitions if edge.plan.measurement is not None
            ],
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
