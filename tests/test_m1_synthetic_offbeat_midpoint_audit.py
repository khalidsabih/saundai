"""Synthetic-only acoustic kick versus inferred half-beat midpoint experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-synthetic-offbeat-midpoint-audit.py"
    spec = importlib.util.spec_from_file_location("m1_offbeat_midpoint_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def events(count=20, period=.5):
    return [
        {"index": i, "status": "measured", "audio_attack_seconds": i * period}
        for i in range(count)
    ]


def test_half_beat_displacement_is_closer_to_inferred_midpoints():
    mod = load()
    grid = [i * .5 + .25 for i in range(2, 12)]
    original = list(grid)
    report = mod.evaluate(grid, events(), (("middle", None, None),))
    row = report["regions"][0]
    assert row["closer_to_inferred_midpoint_count"] == len(grid)
    assert row["within_inferred_midpoint_tolerance_count"] == len(grid)
    assert row["within_kick_tolerance_count"] == 0
    assert row["median_distance_to_inferred_midpoint_ms"] == 0.0
    assert report["midpoint_is_inferred_not_measured"] is True
    assert grid == original


def test_grid_aligned_to_kicks_remains_closer_to_kicks():
    mod = load()
    grid = [i * .5 + .02 for i in range(2, 12)]
    row = mod.evaluate(grid, events(), (("all", None, None),))["regions"][0]
    assert row["closer_to_kick_count"] == len(grid)
    assert row["within_kick_tolerance_count"] == len(grid)


def test_missing_adjacent_kick_never_creates_synthetic_midpoint():
    mod = load()
    broken = events(3)
    broken[1]["status"] = "insufficient_low_frequency_attack"
    row = mod.evaluate([.25, .75], broken, (("all", None, None),))["regions"][0]
    assert row["evaluable_beat_count"] == 0
    assert row["insufficient_acoustic_neighbor_count"] == 2
    assert row["median_distance_to_inferred_midpoint_ms"] is None


def test_invalid_grid_abstains():
    with pytest.raises(ValueError, match="grid"):
        load().evaluate([0.5, .5], events(), (("all", None, None),))
