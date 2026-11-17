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
