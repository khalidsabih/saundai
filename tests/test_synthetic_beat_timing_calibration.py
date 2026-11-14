"""Synthetic drum calibration contracts: do not calibrate production."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.calibrate_synthetic_beat_timing import (
    align_nearest, evaluate_case, run, synthesize_click_kicks,
)


def test_exact_sample_onsets_and_frame_phase():
    sr, hop = 22050, 512
    wave, times, indices = synthesize_click_kicks(
        bpm=120, frame_phase=2/3, sample_rate=sr, hop_length=hop,
        beats=20,
    )
    assert indices[0] == sr + round(2/3*hop)
    assert indices[0]/sr == times[0]
    assert wave[indices[0]] > 0
    assert wave[indices[0]-1] == 0
    assert len(indices) == 20
    assert all(b>a for a, b in zip(indices, indices[1:]))


def test_timing_offset_sign_and_error_are_sample_precise():
    ref = [1.0+i*.5 for i in range(15)]
    observed = [x+.04 for x in ref]
    result = align_nearest(ref, observed)
    assert result["matched_count"] == 15
    assert result["median_signed_offset_ms"] == pytest.approx(40, abs=.001)
    assert result["matched_over_25ms"] == 15
    assert result["observed_late_count"] == 15
    assert align_nearest(ref, [x-.04 for x in ref])["observed_early_count"] == 15


def test_wrong_pulse_is_not_silently_marked_accurate():
    ref = [1.0+i*.5 for i in range(16)]
    double_time = [1.0+i*.25 for i in range(32)]
    result = align_nearest(ref, double_time)
    assert result["matched_count"] == 16
    assert result["matched_count"] < len(double_time)


def test_missing_events_explicitly_unknown():
    assert align_nearest([1.0], [])["status"] == "no_pairs"
    assert align_nearest([], [1.0])["status"] == "no_pairs"


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        synthesize_click_kicks(bpm=120, frame_phase=1, sample_rate=22050,
                               hop_length=512)


def test_no_production_changes_and_original_only(monkeypatch):
    import scripts.calibrate_synthetic_beat_timing as module
    monkeypatch.setattr(module, "evaluate_case", lambda **kw: {
        "status": "evaluated",
        "predictions_vs_true": {
            "status": "measured", "median_signed_offset_ms": 33.0,
        },
        "onset_peaks_vs_true": {
            "status": "measured", "median_signed_offset_ms": 30.0,
        },
    })
    report = run(bpms=(120,), phases=(0.0, .5))
    assert report["scenarios"] == 2
    assert report["evaluated_scenarios"] == 2
    assert report["phase_integrity_limit_changed"] is False
    assert report["global_offset_correction_authorized"] is False
    assert report["aggregate"]["median_scenario_prediction_offset_ms"] == 33.0


def test_actual_pipeline_quick_synthetic_case(tmp_path):
    # Small integration smoke: real analysis path on original audio.
    row = evaluate_case(bpm=120, frame_phase=0, workdir=tmp_path)
    assert row["status"] == "evaluated", row.get("reason")
    assert row["onset_count"] == 28
    assert row["first_onset_sample"] == 22050
    assert row["onset_peaks_vs_true"]["status"] == "measured"
    assert row["predictions_vs_true"]["status"] in ("measured", "no_pairs")
    assert not list(tmp_path.glob("synthetic_*.wav"))
