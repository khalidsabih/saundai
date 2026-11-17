"""Read-only integer/fractional beat-position replay contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-demo-phase-index-replay.py"
    spec = importlib.util.spec_from_file_location("m1_phase_index_replay", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_phase_position_components():
    m = load()
    beats = [i * 0.5 for i in range(100)]
    p = m.position_components(beats, 2.25, .5)
    assert p["index"] == 4
    assert p["fraction"] == pytest.approx(.5)
    assert p["position"] == pytest.approx(4.5)


def test_invalid_interval_abstains():
    m = load()
    assert m.position_components([0, .5, 1.5, 2.], 1.0, .5) is None
    assert m.position_components([0, .5, 1.0], 0, .5) is None


def _row():
    return {
        "track_a": "a.wav", "track_b": "b.wav",
        "exit_time_a_seconds": 4.25, "entry_time_b_seconds": 4.25,
        "transition_seconds": 5.0, "render_rate_a": 1.0,
        "render_rate_b": 1.0, "phase_nudge_seconds_b": 0.0,
        "points": [
            {"fraction": 0.0, "phase_error_ms": 0.0},
            {"fraction": .25, "phase_error_ms": 0.0},
            {"fraction": .5, "phase_error_ms": 0.0},
            {"fraction": .75, "phase_error_ms": 0.0},
            {"fraction": 1.0, "phase_error_ms": 0.0},
        ],
    }


def test_identical_beats_replay_zero_and_match_trace():
    m = load()
    beats = [i * .5 for i in range(100)]
    tracks = {
        "a.wav": {"beats": beats, "bpm": 120.0},
        "b.wav": {"beats": beats, "bpm": 120.0},
    }
    report = m.replay([_row()], tracks)
    assert report["evaluated"] == 1
    assert report["median_abs_integer_index_change_ms"] == 0
    assert report["median_abs_fraction_change_ms"] == 0
    assert report["median_abs_replay_trace_difference_ms"] == 0
    assert report["counterfactual_only"] is True


def test_offset_in_one_deck_is_decomposed_without_adjustment():
    m = load()
    beats = [i * .5 for i in range(100)]
    shifted = [v + (.1 if i >= 16 else 0) for i, v in enumerate(beats)]
    tracks = {
        "a.wav": {"beats": shifted, "bpm": 120.0},
        "b.wav": {"beats": beats, "bpm": 120.0},
    }
    report = m.replay([_row()], tracks)
    assert report["evaluated"] == 1
    example = report["examples"][0]
    assert example["replayed_phase_change_ms"] != 0
    assert abs(
        example["integer_index_change_ms"]
        + example["within_beat_fraction_change_ms"]
        - example["replayed_phase_change_ms"]
    ) < .002


def test_unknown_tracks_fail_closed_without_guessing():
    m = load()
    report = m.replay([_row()], {})
    assert report["evaluated"] == 0
    assert report["skipped"]["unknown_track"] == 1
