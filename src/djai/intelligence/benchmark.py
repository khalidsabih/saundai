from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .preferences import load_preferences
from .transition_ranker import load_transition_ranker


@dataclass(frozen=True, slots=True)
class BrainBenchmark:
    decisive_preferences: int
    unique_track_pairs: int
    ranker_loaded: bool
    cv_accuracy: float | None
    baseline_accuracy: float | None
    win_margin: float | None
    authority: float
    deployment_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_brain(
    preferences_path: str | Path = "ratings/transition-preferences.jsonl",
    model_path: str | Path = "config/transition-ranker.json",
    *,
    minimum_accuracy_lift: float = 0.05,
    minimum_comparisons: int = 100,
) -> BrainBenchmark:
    rows = [row for row in load_preferences(preferences_path) if row.get("choice") in {"a", "b"}]
    pair_keys = {
        f"{row.get('candidate_a', {}).get('track_a')}|{row.get('candidate_a', {}).get('track_b')}"
        for row in rows
    }
    model = load_transition_ranker(model_path)
    if not model:
        return BrainBenchmark(len(rows), len(pair_keys), False, None, None, None, 0.0, False)
    accuracy = float(model.get("cv_accuracy", 0.0))
    baseline = float(model.get("baseline_accuracy", 0.0))
    margin = accuracy - baseline
    ready = len(rows) >= minimum_comparisons and margin >= minimum_accuracy_lift and float(model.get("authority", 0.0)) > 0.0
    return BrainBenchmark(
        decisive_preferences=len(rows),
        unique_track_pairs=len(pair_keys),
        ranker_loaded=True,
        cv_accuracy=accuracy,
        baseline_accuracy=baseline,
        win_margin=margin,
        authority=float(model.get("authority", 0.0)),
        deployment_ready=ready,
    )
