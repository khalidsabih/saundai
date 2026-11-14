"""Contracts for validation-only meter/downbeat evidence investigations."""
import json

import numpy as np
import pytest

from scripts import diagnose_meter_downbeat_evidence as mod


def _beats(n=48):
    return [1.0+0.5*i for i in range(n)]


def test_accent_scores_detect_supported_three_beat_pattern():
    beat = _beats(48)
    env = np.ones(1200, dtype=float)
    indices = np.asarray(np.rint(np.array(beat)*22050/512), dtype=int)
    for i, idx in enumerate(indices):
        env[idx] = 9.0 if i % 3 == 0 else 1.0
    result = mod.accent_evidence(beat, env)
    assert result["status"] == "measured"
    assert result["meters"]["3"]["best_phase"] == 0
    assert result["meters"]["3"]["accent_score"] > result["meters"]["4"]["accent_score"]
    assert result["not_a_production_candidate"] is True


def test_equal_accents_abstain():
    beat = _beats()
    env = np.ones(1200, dtype=float)
    result = mod.accent_evidence(beat, env)
    assert result["abstained"]
    assert result["suggested_meter_or_abstain"] is None


def test_correct_meter_wrong_phase_classification_requires_oracle():
    kind = mod.classify_error(
        reference_meter_value=4,
        predicted_meter=4,
        pulse_class="similar_pulse_period",
        predicted_downbeat_f1=.1,
        reference_meter_phase_oracle={
            "status": "reference_assisted_oracle",
            "best_phase_f_measure": .9,
        },
    )
    assert kind == "possible_wrong_downbeat_phase"
    assert mod.classify_error(
        reference_meter_value=4, predicted_meter=6,
        pulse_class="similar_pulse_period",
        predicted_downbeat_f1=.1, reference_meter_phase_oracle={},
    ) == "meter_mismatch"
    assert mod.classify_error(
        reference_meter_value=4, predicted_meter=6,
        pulse_class="possible_half_time",
        predicted_downbeat_f1=.1, reference_meter_phase_oracle={},
    ) == "pulse_period_mismatch_or_uncertain"


def test_validation_only_never_access_test(monkeypatch, tmp_path):
    called = []
    def fake_eval(item, root, *, precomputed_only=False):
        called.append(item["id"])
        return {
            "id": item["id"], "status": "evaluated", "predicted_meter": 4,
            "reference_meter": {"meter": 4}, "pulse": {"class": "similar_pulse_period"},
            "error_class": "downbeat_matches",
            "accent_evidence": {"status": "measured", "suggested_meter_or_abstain": 4},
            "reference_meter_phase_oracle": {
                "status": "reference_assisted_oracle", "best_phase_f_measure": 1.0,
            },
            "downbeat": {"f_measure": 1.0},
        }
    monkeypatch.setattr(mod, "evaluate_track", fake_eval)
    report = mod.diagnose({"tracks": [
        {"id": "validation", "split": "validation"},
        {"id": "test", "split": "test", "audio": "should-never-be-read"},
    ]}, tmp_path)
    assert called == ["validation"]
    assert report["aggregate"]["meter_agreement_count"] == 1
    assert report["aggregate"]["reference_assisted_phase_oracle_macro_f1"] == 1.0
    assert report["phase_integrity_limit_changed"] is False
    assert report["beat_grid_edit_authorized"] is False
    assert report["oracle_used_ground_truth"] is True


def test_missing_prediction_is_explicit_failure(tmp_path):
    beats = _beats()
    (tmp_path/"beats.txt").write_text("\n".join(str(x) for x in beats))
    report = mod.diagnose({"tracks": [{
        "id": "v", "split": "validation",
        "beats": "beats.txt", "audio": "nonexistent",
    }]}, tmp_path, precomputed_only=True)
    assert report["evaluated_count"] == 0
    assert report["tracks"][0]["status"] == "missing_saved_prediction"


def test_saved_prediction_without_audio_has_no_accent_evidence(tmp_path):
    beats = _beats()
    (tmp_path/"beats.txt").write_text("\n".join(str(x) for x in beats))
    (tmp_path/"downbeats.txt").write_text("\n".join(str(x) for x in beats[::4]))
    (tmp_path/"prediction.json").write_text(json.dumps({
        "beats_seconds": beats,
        "downbeats_seconds": beats[::4],
        "time_signature": 4,
    }))
    report = mod.diagnose({"tracks": [{
        "id": "v", "split": "validation", "beats": "beats.txt",
        "downbeats": "downbeats.txt", "prediction": "prediction.json",
    }]}, tmp_path, precomputed_only=True)
    assert report["evaluated_count"] == 1
    assert report["tracks"][0]["accent_evidence"]["status"] == "unavailable_without_audio"
    assert report["tracks"][0]["error_class"] == "downbeat_matches"


def test_duplicate_track_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        mod.diagnose({"tracks": [
            {"id": "x", "split": "validation"},
            {"id": "x", "split": "test"},
        ]}, tmp_path)
