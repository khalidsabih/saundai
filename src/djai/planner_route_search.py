from __future__ import annotations

"""Generic route-search mechanics for set planning.

This module deliberately does not own musical scoring, model policy, or
Executive orchestration. Callers provide the route score function.
"""

from collections.abc import Callable, Iterable

from .set_models import TransitionEdge


def valid_order(
    order: list[int],
    count: int,
    start: int | None,
    end: int | None,
) -> bool:
    return (
        len(order) == count
        and len(set(order)) == count
        and set(order) == set(range(count))
        and (start is None or order[0] == start)
        and (end is None or order[-1] == end)
    )


def greedy_orders(
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
            allowed = [
                candidate
                for candidate in sorted(unused)
                if (current, candidate) not in blocked
            ]
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
        if valid_order(order, count, start, end):
            yield order


def local_improve(
    order: list[int],
    *,
    score_order: Callable[[list[int]], float],
    start: int | None,
    end: int | None,
    max_passes: int = 8,
) -> list[int]:
    best = list(order)
    best_score = score_order(best)
    count = len(best)
    first_mutable = 1 if start is not None else 0
    last_exclusive = count - 1 if end is not None else count

    for _ in range(max_passes):
        improved = False

        for left in range(first_mutable, last_exclusive):
            for right in range(left + 1, last_exclusive):
                candidate = list(best)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                score = score_order(candidate)
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True

        for source in range(first_mutable, last_exclusive):
            for target in range(first_mutable, last_exclusive):
                if source == target:
                    continue
                candidate = list(best)
                node = candidate.pop(source)
                candidate.insert(target, node)
                if not valid_order(candidate, count, start, end):
                    continue
                score = score_order(candidate)
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True

        for left in range(first_mutable, last_exclusive - 1):
            for right in range(left + 2, last_exclusive + 1):
                candidate = best[:left] + list(reversed(best[left:right])) + best[right:]
                if not valid_order(candidate, count, start, end):
                    continue
                score = score_order(candidate)
                if score > best_score + 1e-8:
                    best, best_score, improved = candidate, score, True

        if not improved:
            break

    return best
