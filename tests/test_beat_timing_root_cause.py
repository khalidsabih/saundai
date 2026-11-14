import json

import numpy as np
import pytest

from scripts.audit_beat_timing_root_cause import (
    annotation_assisted_shift_sweep, audit, frame_grid_diagnosis,
    nearest_landmark_offsets, onset_landmarks,
)


def _beats(n=30):
    return [1.0 + .5*i for i in range(n)]


def test_frame_grid_lattice_is_a_measurement_not_a_causal_inference():
    sr, hop = 22050, 512
    pred = [i*hop/sr for i in (30, 50, 70)]
    m = frame_grid_diagnosis(pred, sr=sr, hop_length=hop)
    assert m["frame_period_ms"] == pytest.approx(23.22, abs=.01)
    assert m["share_within_1ms_of_lattice"] == 1.0
    assert m["frame_center_or_algorithmic_delay_not_established"]


def test_frame_grid_off_lattice_does_not_silently_round():
    sr, hop = 22050, 512
    pred = [30*hop/sr+.005, 50*hop/sr+.005]
    m = frame_grid_diagnosis(pred, sr=sr, hop_length=hop)
    assert m["median_residual_to_nearest_frame_ms"] == pytest.approx(5.0)


def test_nearest_audio_peaks_report_direction_and_limited_evidence():
    points = [1., 2., 3.]
    peaks = [.96, 1.96, 2.96]
    m = nearest_landmark_offsets(points, peaks)
    assert m["median_signed_timestamp_minus_peak_ms"] == pytest.approx(40.0)
    assert m["not_beat_ground_truth"]


def test_silence_no_peaks_is_not_success():
    assert nearest_landmark_offsets(_beats(), [])["status"] == "no_landmark_evidence"


def test_annotation_shift_sweep_is_reference_oracle_only():
    ref = _beats()
    predictions = [t+.04 for t in ref]
    res = annotation_assisted_shift_sweep(ref, predictions)
    assert res["status"] == "reference_assisted_oracle"
    assert res["oracle_uses_ground_truth"]
    assert res["not_a_production_offset"]
    assert res["best_shift_ms"] in range(-80, 81, 5)


def test_validation_selection_never_loads_test_paths(tmp_path, monkeypatch):
    import scripts.audit_beat_timing_root_cause as mod
    (tmp_path/"beats.txt").write_text("\n".join(map(str, _beats())))
    (tmp_path/"prediction.json").write_text(json.dumps({"beats_seconds": _beats()}))
    (tmp_path/"audio.wav").write_bytes(b"fixture")
    monkeypatch.setattr(mod, "onset_landmarks", lambda _: (_beats(), {
        "sample_rate": 22050, "hop_length": 512, "onset_peak_count": 30,
    }))
    manifest = {"dataset": "synthetic-fixture", "tracks": [
        {"id": "val", "split": "validation", "beats": "beats.txt",
         "prediction": "prediction.json", "audio": "audio.wav"},
        {"id": "unreadable-test", "split": "test",
         "beats": "nonexistent", "audio": "nonexistent"},
    ]}
    result = audit(manifest, tmp_path, precomputed_only=True)
    assert result["selected_count"] == 1
    assert result["evaluated_count"] == 1
    assert result["tracks"][0]["status"] == "evaluated"
    assert result["phase_integrity_limit_changed"] is False


def test_precomputed_only_missing_predictions_fail_closed(tmp_path):
    result = audit({"tracks": [{
        "id": "x", "split": "validation", "beats": "unavailable",
        "audio": "unavailable",
    }]}, tmp_path, precomputed_only=True)
    assert result["evaluated_count"] == 0


def test_silent_audio_yields_no_onset_landmarks(tmp_path):
    import soundfile as sf
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), np.zeros(22050*5, dtype=np.float32), 22050)
    peaks, meta = onset_landmarks(wav)
    assert peaks == []
    assert meta["onsets_not_verified_beats"]


def test_duplicate_track_ids_fail(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        audit({"tracks": [
            {"id": "x", "split": "validation"},
            {"id": "x", "split": "test"},
        ]}, tmp_path)
