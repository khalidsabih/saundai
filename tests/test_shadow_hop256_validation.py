"""Validation-only hop shadow comparison: strict split isolation and safety."""
import json

import numpy as np
import pytest

from scripts import benchmark_shadow_hop256_validation as mod


def _times(period=0.5, count=24):
    return [1.0+i*period for i in range(count)]


def test_rhythm_frontend_same_512_settings_as_production(monkeypatch):
    calls = []
    def fake_onset_strength(**kwargs):
        calls.append(("onset", kwargs["hop_length"]))
        return np.ones(100, dtype=float)
    def fake_beat_track(**kwargs):
        calls.append(("beat", kwargs["hop_length"]))
        return np.array([120.0]), np.arange(5, 55, 5)
    monkeypatch.setattr(mod.librosa.onset, "onset_strength", fake_onset_strength)
    monkeypatch.setattr(mod.librosa.beat, "beat_track", fake_beat_track)
    monkeypatch.setattr(mod, "_estimate_time_signature", lambda *a: (4, .5))
    monkeypatch.setattr(mod, "_estimate_downbeats", lambda beats, env, beats_per_bar: (beats[::4], .5))
    result = mod.deterministic_front_end(np.zeros(22050*5), 22050, 512)
    assert calls == [("onset", 512), ("beat", 512)]
    assert len(result["beats_seconds"]) == 10
    with pytest.raises(ValueError, match="unsupported hop"):
        mod.deterministic_front_end(np.zeros(100), 22050, 128)


def test_complete_metrics_for_identical_beats():
    reference = _times()
    pred = {"beats_seconds": reference, "downbeats_seconds": reference[::4],
            "bpm": 120.0, "time_signature": 4}
    row = mod.evaluate_prediction(reference, reference[::4], pred, runtime={
        "wall_seconds": .1, "python_traced_peak_mib": .1,
    })
    assert row["beat"]["f_measure"] == 1.0
    assert row["downbeat"]["f_measure"] == 1.0
    assert row["timing"]["signed_median_ms"] == 0.0
    assert row["pulse"]["class"] == "similar_pulse_period"


def test_validation_selection_never_touches_test_audio(monkeypatch, tmp_path):
    seen = []
    def fake_eval(item, root):
        seen.append(item["id"])
        exact = {"status": "evaluated", "beat": {"f_measure": 1.0, "matched": 24,
            "reference_count": 24, "predicted_count": 24, "matched_over_25ms": 0},
            "downbeat": {"f_measure": 1.0, "matched": 6,
                "reference_count": 6, "predicted_count": 6, "matched_over_25ms": 0},
            "timing": {"status": "measured", "signed_median_ms": 0.0},
            "pulse": {"class": "similar_pulse_period"}, "predicted_meter": 4,
            "runtime": {"wall_seconds": 1.0, "python_traced_peak_mib": 1.0}}
        return {"id": item["id"], "status": "evaluated",
                "production": exact, "deterministic_512": exact, "shadow_256": exact}
    monkeypatch.setattr(mod, "evaluate_track", fake_eval)
    output = mod.compare({"tracks": [
        {"id": "val", "split": "validation", "audio": "x"},
        {"id": "test", "split": "test", "audio": "DO_NOT_READ"},
    ]}, tmp_path)
    assert seen == ["val"]
    assert output["comparison_paired_count"] == 1
    assert output["selected_split"] == "validation"
    assert output["production_changes"] is False
    assert output["phase_integrity_limit_changed"] is False
    assert output["shadow_is_not_automatically_promotable"]


def test_missing_audio_is_not_reported_as_success(tmp_path):
    report = mod.compare({"tracks": [{
        "id": "missing", "split": "validation",
        "beats": "missing.labels", "audio": "missing.wav",
    }]}, tmp_path)
    assert report["comparison_paired_count"] == 0
    assert report["summary"]["shadow_256"]["evaluated_count"] == 0


def test_duplicate_ids_rejected_before_access(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        mod.compare({"tracks": [
            {"id": "x", "split": "validation"},
            {"id": "x", "split": "test"},
        ]}, tmp_path)
