"""Contracts for the read-only cue-local M1 beat clock audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-demo-local-clock-audit.py"
    spec = importlib.util.spec_from_file_location("m1_local_clock", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_clock_finds_stable_period_and_excludes_spurious_interval():
    mod = load()
    beats = [i * .5 for i in range(30)]
    beats[12] = beats[11] + .1
    result = mod.local_clock(beats, 3.0, 12.0)
    assert result["status"] == "measured"
    assert result["bpm"] == 120.0
    assert result["rejected_interval_count"] == 2


def test_local_clock_abstains_on_insufficient_or_unsorted_evidence():
    mod = load()
    assert mod.local_clock([0, .5, 1], 0, 1)["status"] == "insufficient_intervals"
    assert mod.local_clock([0, .5, .4, 1], 0, 2)["status"] == "invalid_grid"
    assert mod.local_clock([0, .5, 1], 2, 1)["status"] == "invalid_window"


def _tracks(bpm_a=122.0, bpm_b=124.0):
    return {
        "a.wav": {"beats_seconds": [i * 60.0 / bpm_a for i in range(160)], "bpm": 123.0},
        "b.wav": {"beats_seconds": [i * 60.0 / bpm_b for i in range(160)], "bpm": 123.0},
    }


def _row(rate_b=1.0):
    return {
        "track_a": "a.wav", "track_b": "b.wav",
        "exit_time_a_seconds": 15.0, "entry_time_b_seconds": 15.0,
        "transition_seconds": 10.0, "render_rate_a": 1.0,
        "render_rate_b": rate_b, "phase_nudge_seconds_b": 0.0,
        "points": [
            {"fraction": 0.0, "phase_error_ms": 0.0},
            {"fraction": 1.0, "phase_error_ms": -162.6},
        ],
    }


def test_equal_scalar_bpm_can_hide_local_grid_mismatch():
    mod = load()
    result = mod.compare([_row()], _tracks())
    assert result["evaluated_configs"] == 1
    assert result["median_predicted_local_clock_change_ms"] < -150
    assert result["matching_direction_count"] == 1
    assert result["counterfactual_only"] is True


def test_grid_consistent_rate_nearly_eliminates_predicted_drift():
    mod = load()
    result = mod.compare([_row(rate_b=122 / 124)], _tracks())
    assert result["evaluated_configs"] == 1
    assert abs(result["median_predicted_local_clock_change_ms"]) < .01


def test_missing_trace_evidence_is_not_mistaken_for_success():
    mod = load()
    row = _row()
    row["points"] = []
    result = mod.compare([row], _tracks())
    assert result["evaluated_configs"] == 0
    assert result["non_evaluable_count"] == 1
    assert result["median_absolute_unexplained_ms"] is None
