"""Read-only synthetic clock hypothesis test contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-demo-phase-clock-counterfactual.py"
    spec = importlib.util.spec_from_file_location("m1_demo_phase_clock", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _audit(a=122.0, b=124.0):
    return {"tracks": [
        {"filename": "a.wav", "estimated_bpm": 123.0,
         "beat_grid": {"generator_bpm": 122.0, "sequential_index_drift": {"implied_bpm_from_linear_fit": a}}},
        {"filename": "b.wav", "estimated_bpm": 123.0,
         "beat_grid": {"generator_bpm": 124.0, "sequential_index_drift": {"implied_bpm_from_linear_fit": b}}},
    ]}


def _trace(rate_b=1.0, observed=-325.203):
    return [{
        "track_a": "a.wav", "track_b": "b.wav",
        "exit_time_a_seconds": 30.0, "entry_time_b_seconds": 10.0,
        "transition_seconds": 20.0, "render_rate_a": 1.0, "render_rate_b": rate_b,
        "points": [
            {"fraction": 0.0, "phase_error_ms": 0.0},
            {"fraction": 1.0, "phase_error_ms": observed},
        ],
    }]


def test_equal_scalar_bpm_does_not_imply_equal_grid_clocks():
    mod = _module()
    report = mod.diagnose(_trace(), _audit())
    assert report["evaluated_rows"] == 1
    assert report["median_predicted_uncontrolled_phase_change_ms"] == -325.203
    assert report["matching_change_direction_count"] == 1
    assert report["counterfactual_only"] is True


def test_grid_consistent_rate_would_predict_no_clock_drift():
    mod = _module()
    report = mod.diagnose(_trace(rate_b=122 / 124, observed=0), _audit())
    assert report["median_predicted_uncontrolled_phase_change_ms"] == 0.0
    assert report["median_observed_phase_change_ms"] == 0.0
    assert report["median_abs_unexplained_change_ms"] == 0.0


def test_no_points_or_unknown_track_do_not_invent_phase():
    mod = _module()
    row = _trace()[0]
    row["points"] = []
    unknown = dict(row, track_a="missing.wav")
    report = mod.diagnose([row, unknown], _audit())
    assert report["evaluated_rows"] == 0
    assert report["no_phase_point_rows"] == 1
    assert report["skipped"]["unknown_track"] == 1


def test_invalid_audit_fails_closed():
    import pytest
    with pytest.raises(ValueError, match="synthetic beat-index"):
        _module().diagnose([], {})
