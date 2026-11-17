"""Generated 4/4 Beat This downbeat audit remains post-hoc and read-only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-beat-this-downbeat-timing-audit.py"
    spec = importlib.util.spec_from_file_location("m1_beat_this_downbeat_timing", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_generator_reference_has_128_beats_and_32_bar_starts():
    mod = load()
    beat, down = mod.synthetic_reference(122.)
    assert len(beat) == 128
    assert len(down) == 32
    assert beat[0] == down[0] == 0.
    assert down[1] == pytest.approx(4 * 60 / 122)
    assert down[-1] == pytest.approx(31 * 4 * 60 / 122)


def test_one_to_one_matches_and_strict_25ms_residuals_remain_descriptive():
    mod = load()
    report = mod.score_events([.01, .52, 1.1], [0., .5, 1.])
    assert report["matched_count"] == 2
    assert report["unmatched_prediction_count"] == 1
    assert report["missed_reference_count"] == 1
    assert report["matched_outside_25ms"] == 0
    assert report["reference_without_25ms_match"] == 1
    assert report["strict_residual_is_not_planner_gate"] is True
    assert report["maximum_absolute_error_ms"] == pytest.approx(20.)


def test_late_match_beyond_25ms_is_never_counted_strict():
    mod = load()
    report = mod.score_events([.04, .54], [0., .5])
    assert report["f1"] == 1.
    assert report["matched_outside_25ms"] == 2
    assert report["reference_without_25ms_match"] == 2
    assert report["p95_absolute_error_ms"] == pytest.approx(40.)


def test_zero_matched_events_not_falsely_promoted():
    mod = load()
    report = mod.score_events([.25, .75], [0., .5])
    assert report["matched_count"] == 0
    assert report["f1"] == 0.0
    assert report["maximum_absolute_error_ms"] is None


def test_predicted_downbeat_count_is_not_automatically_meter_consistency():
    mod = load()
    good = mod.bar_label_diagnostics(
        [0., .5, 1., 1.5, 2., 2.5, 3., 3.5],
        [1, 2, 3, 4, 1, 2, 3, 4], list(range(8)), [0., 2.]
    )
    assert good["non_four_beat_label_intervals"] == 0
    bad = mod.bar_label_diagnostics(
        [0., .5, 1., 1.5, 2., 2.5, 3., 3.5],
        [1, 2, 3, 1, 2, 1, 2, 3], list(range(8)),
        [0., 1.5, 2.5],
    )
    assert bad["non_four_beat_label_intervals"] == 2


def test_segment_windows_are_fixed_and_include_track_tail():
    mod = load()
    report = mod.segmented_scores([0., 7.99, 8.01, 15.9], [0., 8., 16.], 16.2)
    assert [(x["start_seconds"], x["end_seconds"]) for x in report] == [
        (0., 8.), (8., 16.), (16., 16.2),
    ]


def test_model_is_not_loaded_or_downloaded_by_benchmark():
    mod = load()
    assert mod.STRICT_RESIDUAL_MS == 25.
    assert mod.TOLERANCE_MS == 80.
    assert mod.BEATS_PER_BAR == 4
    assert mod.SYNTHETIC_BARS == 32
    assert len(mod.FIXTURES) == 2
