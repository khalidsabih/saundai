from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..models import TrackAnalysis
from ..intelligence.hybrid_next_track import score_hybrid_next_track
from .set_state import build_set_state, next_track_context_score
from .director import direct_set_intent, intent_track_score
from .dj_move import move_opportunity_score


@dataclass(frozen=True, slots=True)
class SetPolicyConfig:
    horizon: int = 3
    beam: int = 5
    future_discount: float = 0.78
    context_weight: float = 0.32
    hybrid_weight: float = 0.56
    move_weight: float = 0.30

    @classmethod
    def from_environment(cls) -> "SetPolicyConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default
        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return default
        return cls(
            horizon=max(1, min(6, _int("DJAI_SET_LOOKAHEAD_DEPTH", 3))),
            beam=max(2, min(12, _int("DJAI_SET_LOOKAHEAD_BEAM", 5))),
            future_discount=max(0.35, min(0.95, _float("DJAI_SET_FUTURE_DISCOUNT", 0.78))),
            context_weight=max(0.0, min(0.75, _float("DJAI_SET_CONTEXT_WEIGHT", 0.32))),
            hybrid_weight=max(0.0, min(0.85, _float("DJAI_HYBRID_NEXT_TRACK_WEIGHT", 0.56))),
            move_weight=max(0.0, min(0.65, _float("DJAI_MOVE_ORDER_WEIGHT", 0.30))),
        )


def _edge_score(edge: object) -> float:
    return float(getattr(edge, "objective_score", 0.0))


def _move_score(edge: object, track_a: TrackAnalysis, track_b: TrackAnalysis) -> tuple[float, dict[str, object]]:
    plan = getattr(edge, "plan", None)
    metadata = getattr(plan, "metadata", {}) if plan is not None else {}
    cached = metadata.get("dj_move_opportunity", {}) if isinstance(metadata, dict) else {}
    if isinstance(cached, dict) and cached.get("score") is not None:
        try:
            return float(cached["score"]), dict(cached)
        except (TypeError, ValueError):
            pass
    return move_opportunity_score(track_a, track_b)


def _simulate_value(
    prefix: list[int],
    remaining: set[int],
    graph: Mapping[tuple[int, int], object],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
    total_count: int,
    config: SetPolicyConfig,
    depth: int,
) -> float:
    if depth <= 0 or not remaining:
        return 0.0
    current = prefix[-1]
    progress = min(1.0, len(prefix) / max(1, total_count - 1))
    state = build_set_state([analyses[index] for index in prefix])
    scored: list[tuple[float, int]] = []
    for candidate in remaining:
        edge = graph.get((current, candidate))
        if edge is None:
            continue
        context, _ = next_track_context_score(
            state, analyses[candidate], journey=journey, progress=progress
        )
        intent = direct_set_intent(journey, progress, state)
        director, _ = intent_track_score(intent, state, analyses[candidate])
        contextual = 0.72 * context + 0.28 * director
        baseline = (1.0 - config.context_weight) * _edge_score(edge) + config.context_weight * contextual
        hybrid = score_hybrid_next_track(
            analyses[current],
            analyses[candidate],
            edge_score=_edge_score(edge),
            set_state=state,
            journey=journey,
            progress=progress,
            plan=getattr(edge, "plan", None),
        )
        hybrid_immediate = (1.0 - config.hybrid_weight) * baseline + config.hybrid_weight * hybrid.score
        move_score, _ = _move_score(edge, analyses[current], analyses[candidate])
        immediate = (1.0 - config.move_weight) * hybrid_immediate + config.move_weight * move_score
        scored.append((immediate, candidate))
    scored.sort(reverse=True)
    best = -1e9
    for immediate, candidate in scored[: config.beam]:
        future = _simulate_value(
            prefix + [candidate],
            remaining - {candidate},
            graph,
            analyses,
            journey=journey,
            total_count=total_count,
            config=config,
            depth=depth - 1,
        )
        best = max(best, immediate + config.future_discount * future)
    return 0.0 if best < -1e8 else best


def choose_next_track(
    prefix: Sequence[int],
    remaining: Sequence[int],
    graph: Mapping[tuple[int, int], object],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
    config: SetPolicyConfig | None = None,
) -> tuple[int, dict[str, float]]:
    """Choose the next track using set state plus a receding-horizon future search."""
    if not prefix or not remaining:
        raise ValueError("choose_next_track requires a current prefix and at least one remaining track")
    cfg = config or SetPolicyConfig.from_environment()
    current = prefix[-1]
    total_count = len(prefix) + len(remaining)
    progress = min(1.0, len(prefix) / max(1, total_count - 1))
    state = build_set_state([analyses[index] for index in prefix])
    values: list[tuple[float, int, float, float, float, float, float, float, dict[str, Any]]] = []
    remaining_set = set(remaining)
    for candidate in remaining:
        edge = graph.get((current, candidate))
        if edge is None:
            continue
        context, _ = next_track_context_score(state, analyses[candidate], journey=journey, progress=progress)
        intent = direct_set_intent(journey, progress, state)
        director, _ = intent_track_score(intent, state, analyses[candidate])
        contextual = 0.72 * context + 0.28 * director
        baseline = (1.0 - cfg.context_weight) * _edge_score(edge) + cfg.context_weight * contextual
        hybrid = score_hybrid_next_track(
            analyses[current],
            analyses[candidate],
            edge_score=_edge_score(edge),
            set_state=state,
            journey=journey,
            progress=progress,
            plan=getattr(edge, "plan", None),
        )
        hybrid_immediate = (1.0 - cfg.hybrid_weight) * baseline + cfg.hybrid_weight * hybrid.score
        move_score, move_meta = _move_score(edge, analyses[current], analyses[candidate])
        immediate = (1.0 - cfg.move_weight) * hybrid_immediate + cfg.move_weight * move_score
        future = _simulate_value(
            list(prefix) + [candidate],
            remaining_set - {candidate},
            graph,
            analyses,
            journey=journey,
            total_count=total_count,
            config=cfg,
            depth=cfg.horizon - 1,
        )
        total = immediate + cfg.future_discount * future
        values.append((total, candidate, immediate, future, hybrid.score, hybrid.confidence, hybrid.learned_authority, move_score, move_meta))
    if not values:
        raise RuntimeError("Set policy found no reachable next track")
    values.sort(reverse=True)
    total, candidate, immediate, future, hybrid_score, hybrid_confidence, learned_authority, move_score, move_meta = values[0]
    return candidate, {
        "policy_value": round(total, 6),
        "immediate_value": round(immediate, 6),
        "future_value": round(future, 6),
        "horizon": float(cfg.horizon),
        "beam": float(cfg.beam),
        "hybrid_weight": round(float(cfg.hybrid_weight), 6),
        "hybrid_score": round(float(hybrid_score), 6),
        "hybrid_confidence": round(float(hybrid_confidence), 6),
        "learned_authority": round(float(learned_authority), 6),
        "move_weight": round(float(cfg.move_weight), 6),
        "move_opportunity_score": round(float(move_score), 6),
        "move_best_exit_bar": float(move_meta.get("best_exit_bar", -1)) if move_meta.get("best_exit_bar") is not None else -1.0,
        "move_best_entry_bar": float(move_meta.get("best_entry_bar", -1)) if move_meta.get("best_entry_bar") is not None else -1.0,
    }


def build_receding_horizon_order(
    start: int,
    graph: Mapping[tuple[int, int], object],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
    end: int | None = None,
    config: SetPolicyConfig | None = None,
) -> tuple[list[int], list[dict[str, float]]]:
    cfg = config or SetPolicyConfig.from_environment()
    order = [start]
    unused = set(range(len(analyses))) - {start}
    if end is not None:
        unused.discard(end)
    trace: list[dict[str, float]] = []
    while unused:
        nxt, decision = choose_next_track(
            order, sorted(unused), graph, analyses, journey=journey, config=cfg
        )
        trace.append({"position": float(len(order)), "track_index": float(nxt), **decision})
        order.append(nxt)
        unused.remove(nxt)
    if end is not None and end not in order:
        order.append(end)
    return order, trace
