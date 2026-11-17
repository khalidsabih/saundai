"""Read-only phase-lock overlap correlation contract tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-correlate-phase-anomaly-overlaps.py"
    spec = importlib.util.spec_from_file_location("m1_correlate_phase_anomaly", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit():
    return {
        "tracks": [
            {
                "filename": "a.wav",
                "beat_grid": {"local_interval_diagnostics": {"outlier_clusters": [
                    {"start_seconds": 8.0, "end_seconds": 10.0},
                ]}},
            },
            {
                "filename": "b.wav",
                "beat_grid": {"local_interval_diagnostics": {"outlier_clusters": []}},
            },
        ],
    }


def _row(exit_a=5.0, residual=210.0):
    return {
        "track_a": "a.wav",
        "track_b": "b.wav",
        "exit_time_a_seconds": exit_a,
        "entry_time_b_seconds": 2.0,
        "transition_seconds": 4.0,
        "render_rate_a": 1.0,
        "render_rate_b": 1.0,
        "phase_nudge_seconds_b": 0.0,
        "reason": "required correction exceeds bounded phase-lock envelope",
        "max_residual_ms": residual,
    }


def test_correlates_overlap_without_changing_trace():
    mod = _module()
    touching = _row(exit_a=6.0, residual=140.0)
    outside = _row(exit_a=14.0, residual=280.0)
    rows = [touching, outside]
    result = mod.correlate(rows, _audit())
    assert result["groups"]["crosses_anomaly"]["rejected_configurations"] == 1
    assert result["groups"]["does_not_cross_anomaly"]["rejected_configurations"] == 1
    assert result["groups"]["crosses_anomaly"]["median_rejected_residual_ms"] == 140.0
    assert result["groups"]["does_not_cross_anomaly"]["median_rejected_residual_ms"] == 280.0
    assert result["causal_attribution"] is False
    assert rows == [touching, outside]


def test_outside_and_boundary_touch_do_not_count():
    mod = _module()
    result = mod.correlate([_row(exit_a=10.0), _row(exit_a=2.0)], _audit())
    assert result["groups"]["crosses_anomaly"]["rejected_configurations"] == 0
    assert result["groups"]["does_not_cross_anomaly"]["rejected_configurations"] == 2


def test_insufficient_residual_excluded_from_median():
    mod = _module()
    item = _row(exit_a=8.0, residual=999.0)
    item["reason"] = "insufficient reliable beat intervals"
    result = mod.correlate([item], _audit())
    assert result["groups"]["crosses_anomaly"]["measured_residual_count"] == 0
    assert result["groups"]["crosses_anomaly"]["median_rejected_residual_ms"] is None


def test_missing_track_skipped_and_invalid_audit_rejected():
    import pytest
    mod = _module()
    missing = _row()
    missing["track_a"] = "unknown.wav"
    result = mod.correlate([missing], _audit())
    assert result["skipped"]["trace_track_not_in_audit"] == 1
    with pytest.raises(ValueError, match="generated-demo"):
        mod.correlate([], {})
