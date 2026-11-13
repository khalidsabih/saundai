from __future__ import annotations

"""Order-optimization orchestration for set planning.

This module assembles deterministic greedy seeds, receding-horizon policy
candidates, unified-planner candidates and local refinement. Musical route
scoring remains owned by set_planner and is injected as a callback.
"""

from collections.abc import Callable

from .brain.set_policy import SetPolicyConfig, build_receding_horizon_order
from .brain.unified_planner import build_unified_ai_order
from .journey import JourneyMode
from .models import TrackAnalysis
from .planner_large_crate import env_int, large_set_threshold
from .planner_route_search import greedy_orders, local_improve, valid_order
from .set_models import TransitionEdge


def optimize_order_candidates(
    analyses: list[TrackAnalysis],
    graph: dict[tuple[int, int], TransitionEdge],
    *,
    score_order: Callable[[list[int]], float],
    start_index: int | None = None,
    end_index: int | None = None,
    journey: JourneyMode = "balanced",
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

    if count == 2:
        exact = [
            order
            for order in ([0, 1], [1, 0])
            if valid_order(order, count, start_index, end_index)
            and not any((a, b) in blocked for a, b in zip(order, order[1:]))
        ]
        if not exact:
            raise RuntimeError("The optimizer could not construct a valid two-track order.")
        return max(exact, key=score_order)

    candidates = list(
        greedy_orders(
            graph,
            count,
            start=start_index,
            end=end_index,
            blocked_pairs=blocked,
        )
    )

    policy_config = SetPolicyConfig.from_environment()
    policy_starts = (
        [start_index]
        if start_index is not None
        else [index for index in range(count) if index != end_index]
    )
    large_set = count >= large_set_threshold()

    if large_set and start_index is None and candidates:
        start_limit = env_int(
            "DJAI_LARGE_SET_POLICY_STARTS",
            4,
            minimum=2,
            maximum=8,
        )
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
                policy_start,
                graph,
                analyses,
                journey=journey,
                end=end_index,
                config=policy_config,
            )
            if (
                valid_order(policy_order, count, start_index, end_index)
                and not any(
                    (a, b) in blocked
                    for a, b in zip(policy_order, policy_order[1:])
                )
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
        if valid_order(unified.order, count, start_index, end_index):
            candidates.append(unified.order)
    except (RuntimeError, ValueError):
        pass

    if not candidates:
        raise RuntimeError("The optimizer could not construct a valid set order.")

    best = max(candidates, key=score_order)
    local_passes = (
        env_int(
            "DJAI_LARGE_SET_LOCAL_PASSES",
            2,
            minimum=1,
            maximum=4,
        )
        if large_set
        else 8
    )
    return local_improve(
        best,
        score_order=score_order,
        start=start_index,
        end=end_index,
        max_passes=local_passes,
    )
