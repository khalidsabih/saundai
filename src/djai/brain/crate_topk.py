from __future__ import annotations

"""Crate-size-aware search budgets. Search can widen; safety never changes."""

import os
from dataclasses import dataclass


def crate_aware_topk_enabled() -> bool:
    return os.environ.get("DJAI_CRATE_AWARE_TOPK", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


@dataclass(frozen=True, slots=True)
class CrateTopK:
    track_count: int
    route_initial: int
    route_expansions: tuple[int, ...]
    transition_pilot: int
    transition_max: int


def crate_topk(track_count: int) -> CrateTopK:
    """Select economical initial budgets without shrinking recovery ceilings.

    More remaining tracks means broader route uncertainty, but also more
    alternatives. The route search widens *before* the full graph fallback.
    Transition rehearsal uses the existing early-accept/expand mechanism.
    """
    count = max(2, int(track_count))
    if count <= 5:
        route, expansions, pilot, cap = 5, (8, 10), 5, 10
    elif count <= 15:
        route, expansions, pilot, cap = 4, (6, 10), 3, 10
    elif count <= 30:
        route, expansions, pilot, cap = 4, (6, 8), 3, 8
    elif count <= 60:
        route, expansions, pilot, cap = 3, (5, 8), 2, 8
    else:
        route, expansions, pilot, cap = 3, (5, 8), 2, 8
    available = count - 1
    initial = min(route, available)
    wider = tuple(dict.fromkeys(min(value, available) for value in expansions if min(value, available) > initial))
    return CrateTopK(
        track_count=count,
        route_initial=initial,
        route_expansions=wider,
        transition_pilot=pilot,
        transition_max=cap,
    )
