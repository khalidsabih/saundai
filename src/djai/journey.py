from __future__ import annotations

import math
from typing import Literal, Sequence

from .models import TrackAnalysis

JourneyMode = Literal["balanced", "smooth", "build", "warmup", "peak", "sunset"]


def normalize_journey(value: str) -> JourneyMode:
    clean = value.strip().lower().replace("-", "_")
    aliases = {
        "warm_up": "warmup",
        "warm": "warmup",
        "peak_time": "peak",
        "peak-time": "peak",
        "ascending": "build",
    }
    clean = aliases.get(clean, clean)
    if clean not in {"balanced", "smooth", "build", "warmup", "peak", "sunset"}:
        raise ValueError(
            "journey must be one of: balanced, smooth, build, warmup, peak, sunset"
        )
    return clean  # type: ignore[return-value]


def target_energy_curve(journey: str, count: int) -> list[float]:
    """Return a deterministic target energy value for every set position."""
    mode = normalize_journey(journey)
    if count <= 0:
        return []
    if count == 1:
        return [0.62]

    curve: list[float] = []
    for index in range(count):
        x = index / (count - 1)
        if mode == "warmup":
            # Calm opening, steady development, enough headroom for another DJ/set.
            value = 0.32 + 0.43 * (x ** 0.82)
        elif mode == "build":
            value = 0.38 + 0.50 * (x ** 1.05)
        elif mode == "peak":
            # Start active, reach peak early, then preserve intensity.
            value = 0.66 + 0.23 * math.sin(min(1.0, x / 0.55) * math.pi / 2)
            value -= 0.05 * max(0.0, (x - 0.82) / 0.18)
        elif mode == "sunset":
            # Warm opening, emotional crest, gentle landing.
            value = 0.53 + 0.22 * math.sin(math.pi * x) - 0.17 * x
        elif mode == "smooth":
            value = 0.58 + 0.035 * math.sin(2.0 * math.pi * x)
        else:  # balanced
            value = 0.50 + 0.15 * math.sin(math.pi * x) + 0.035 * math.sin(3 * math.pi * x)
        curve.append(round(max(0.08, min(0.96, value)), 6))
    return curve


def curve_fit_error(
    order: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
) -> float:
    if not order:
        return 0.0
    target = target_energy_curve(journey, len(order))
    squared = [
        (float(analyses[track_index].energy) - target[position]) ** 2
        for position, track_index in enumerate(order)
    ]
    return float(math.sqrt(sum(squared) / max(1, len(squared))))


def curve_fit_score(
    order: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
) -> float:
    """Return 0..1 set-level agreement with the requested energy arc."""
    error = curve_fit_error(order, analyses, journey=journey)
    return float(max(0.0, min(1.0, math.exp(-((error / 0.24) ** 2)))))


def journey_metadata(
    order: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    *,
    journey: str,
) -> dict[str, object]:
    return {
        "mode": normalize_journey(journey),
        "target_energy": target_energy_curve(journey, len(order)),
        "actual_energy": [round(float(analyses[index].energy), 6) for index in order],
        "curve_fit_error": round(curve_fit_error(order, analyses, journey=journey), 6),
        "curve_fit_score": round(curve_fit_score(order, analyses, journey=journey), 6),
    }
