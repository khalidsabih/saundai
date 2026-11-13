from __future__ import annotations

"""Deterministic candidate-selection helpers for set planning.

This module owns only generic cue diversity and graph-scout candidate
selection. Musical scoring policy and sequence compilation remain in
set_planner.
"""

from .models import TransitionCandidate


def candidate_diverse_by_bar(
    candidates: list[tuple[float, TransitionCandidate]],
    *,
    limit: int,
    minimum_bar_distance: int = 2,
) -> list[TransitionCandidate]:
    selected: list[TransitionCandidate] = []
    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    for _, candidate in ranked:
        if all(
            abs(candidate.bar_index - existing.bar_index) >= minimum_bar_distance
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= limit:
            break
    if len(selected) < min(limit, len(candidates)):
        for _, candidate in ranked:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= limit:
                break
    return selected


def graph_scout_candidates(
    candidates: list[TransitionCandidate] | tuple[TransitionCandidate, ...],
    *,
    kind: str,
    limit: int,
) -> list[TransitionCandidate]:
    if limit <= 0 or len(candidates) <= limit:
        return list(candidates)

    scored: list[tuple[float, TransitionCandidate]] = []
    for candidate in candidates:
        low_vocal = 1.0 - max(
            0.0,
            min(1.0, float(candidate.vocal_probability)),
        )
        value = (
            0.44 * float(candidate.score)
            + 0.20 * float(candidate.boundary_strength)
            + 0.18 * float(candidate.phrase_strength)
            + 0.10 * float(candidate.beat_grid_confidence)
            + 0.08 * low_vocal
        )
        scored.append((value, candidate))

    selected = candidate_diverse_by_bar(
        scored,
        limit=limit,
        minimum_bar_distance=4,
    )
    if kind == "entry":
        earliest = min(
            candidates,
            key=lambda item: (
                float(item.time_seconds),
                -float(item.score),
            ),
        )
        if earliest.section_label == "intro" and earliest not in selected:
            if len(selected) >= limit:
                selected[-1] = earliest
            else:
                selected.append(earliest)

    return selected[:limit]
