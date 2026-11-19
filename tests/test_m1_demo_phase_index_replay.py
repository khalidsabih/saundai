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
    # The first beat itself is a valid zero-phase sample under bisect_right.
    first = m.position_components([0, .5, 1.0], 0, .5)
    assert first is not None
    assert first["index"] == 0
    assert first["fraction"] == pytest.approx(0.0)
    assert first["position"] == pytest.approx(0.0)
    # Samples outside the supported grid must still abstain.
    assert m.position_components([0, .5, 1.0], -.001, .5) is None
    assert m.position_components([0, .5, 1.0], 1.0, .5) is None


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



def test_interval_budget_exposes_cumulative_short_intervals():
    m = load()
    beats = [i * .5 for i in range(30)]
    beats[10] = beats[9] + .10
    budget = m.interval_budget(beats, 3.0, 7.0, .5)
    assert budget["status"] == "measured"
    assert budget["outlier_interval_count"] == 2
    # A shortened beat bracket can be followed by a compensating long one,
    # so raw sum can be zero even with unsafe local variation.
    assert abs(budget["cumulative_interval_deviation_ms"]) < .01
    assert budget["max_abs_interval_deviation_ms"] >= 399.0


def test_phase_replay_reports_per_deck_excess_and_interval_budgets():
    m = load()
    beats = [i * .5 for i in range(100)]
    tracks = {
        "a.wav": {"beats": beats, "bpm": 120.0},
        "b.wav": {"beats": beats, "bpm": 120.0},
    }
    result = m.replay([_row()], tracks)
    assert result["median_abs_outgoing_clock_excess_ms"] == 0
    assert result["median_abs_incoming_clock_excess_ms"] == 0
    assert result["median_abs_clock_excess_difference_ms"] == 0
    example = result["examples"][0]
    assert example["outgoing_complete_interval_budget"]["status"] == "measured"
    assert example["incoming_complete_interval_budget"]["status"] == "measured"



def test_mean_interval_exposes_accumulated_bias_hidden_by_median():
    m = load()
    # Six median-500ms intervals, plus four 520ms intervals.
    periods = [.5] * 6 + [.52] * 4
    beats = [0.0]
    for period in periods:
        beats.append(beats[-1] + period)
    budget = m.interval_budget(beats, 0.0, beats[-1], .5)
    assert budget["status"] == "measured"
    assert budget["complete_interval_count"] == 10
    assert budget["median_interval_ms"] == 500.0
    assert budget["mean_interval_ms"] == 508.0
    assert budget["mean_minus_median_ms"] == 8.0
    assert budget["cumulative_interval_deviation_ms"] == 80.0
    assert budget["outlier_interval_count"] == 0


def test_regular_intervals_have_identical_mean_and_median():
    m = load()
    beats = [i * .5 for i in range(20)]
    budget = m.interval_budget(beats, 0.0, beats[-1], .5)
    assert budget["mean_minus_median_ms"] == 0.0
    assert budget["mean_implied_bpm"] == 120.0
    assert budget["median_implied_bpm"] == 120.0
