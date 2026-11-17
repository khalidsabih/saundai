"""Tests for the generator-grounded, read-only beat-index diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-demo-beat-index-audit.py"
    spec = importlib.util.spec_from_file_location("m1_demo_beat_index_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_generator_beats_have_no_index_jumps():
    mod = load()
    beats = [i * .5 for i in range(100)]
    summary = mod.compare_to_generator(beats, bpm=120, duration=50)
    assert summary["missing_event_index_jumps"] == 0
    assert summary["duplicate_event_index_jumps"] == 0
    assert summary["median_absolute_grid_error_ms"] == 0
    assert summary["largest_index_jump"] == 1


def test_missing_beat_and_duplicate_position_are_visible():
    mod = load()
    beats = [0, .5, 1.0, 2.0, 2.04, 2.5, 3.0]
    summary = mod.compare_to_generator(beats, bpm=120, duration=4)
    assert summary["missing_event_index_jumps"] == 1
    assert summary["duplicate_event_index_jumps"] == 1
    assert summary["longest_inferred_missing_run"] == 1


def test_invalid_beat_grid_fails_closed():
    mod = load()
    import pytest
    with pytest.raises(ValueError, match="increase"):
        mod.compare_to_generator([0.0, .5, .5], bpm=120, duration=3)


def test_phase_slope_separates_fixed_offset_from_accumulating_drift():
    mod = load()
    period = .5
    fixed = [i * period + .02 for i in range(100)]
    drifting = [i * (period + .002) + .02 for i in range(100)]
    fixed_report = mod.compare_to_generator(fixed, bpm=120, duration=51)
    drift_report = mod.compare_to_generator(drifting, bpm=120, duration=51)
    fixed_drift = fixed_report["sequential_index_drift"]
    drift = drift_report["sequential_index_drift"]
    assert abs(fixed_drift["slope_ms_per_beat"]) < .001
    assert abs(drift["slope_ms_per_beat"] - 2.0) < .001
    assert abs(drift["end_minus_start_signed_ms"] - 198) < .001


def test_nearest_grid_error_can_wrap_even_with_linear_drift():
    mod = load()
    period = .5
    beats = [i * (period + .002) for i in range(128)]
    report = mod.compare_to_generator(beats, bpm=120, duration=65)
    assert report["sequential_index_drift"]["slope_ms_per_beat"] == 2.0
    assert report["max_absolute_grid_error_ms"] <= 250.0
