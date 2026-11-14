"""Diagnostic-only counterfactual beat/bar benchmark contracts."""
import json

import pytest

from scripts import benchmark_pulse_meter_phase_roots as mod


def test_counterfactual_grids_do_not_mutate_originals():
    src = [1., 2., 3., 4., 5.]
    assert mod.counterfactual_grid(src, "unchanged") == src
    assert mod.counterfactual_grid(src, "half_even") == [1., 3., 5.]
    assert mod.counterfactual_grid(src, "half_odd") == [2., 4.]
    assert mod.counterfactual_grid(src, "double_midpoint") == [
        1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    assert src == [1., 2., 3., 4., 5.]
    with pytest.raises(ValueError, match="unsupported"):
        mod.counterfactual_grid(src, "invalid")


def test_reference_assisted_phase_recovers_wrong_phase():
    beats = [1.+.5*i for i in range(64)]
    ref = beats[2::4]
    out = mod.phase_oracle(beats, ref, 4)
    assert out["status"] == "oracle"
    assert out["best_phase"] == 2
    assert out["f_measure"] == 1
    assert out["used_reference_labels"] is True
    assert out["not_deployable"] is True


def test_oracle_never_claims_missing_meter():
    assert mod.phase_oracle([.5*i for i in range(50)], [1., 2.], None)["status"] == "unsupported_or_missing"


def test_identity_replay_keeps_baseline_beat_measure():
    beats = [1.+.5*i for i in range(60)]
    bars = beats[::4]
    out = mod.evaluate_values(beats, bars, beats, bars, 4)
    assert out["beat"]["f_measure"] == 1
    assert out["downbeat"]["f_measure"] == 1
    assert out["oracles"]["unchanged"]["beat"] == out["beat"]
    assert out["opportunity"]["phase_only_f1_gain_at_production_meter"] == 0
    assert out["reference_labels_used_for_oracles"]
    assert out["not_causal_proof"]


def test_half_time_counterfactual_can_explain_rate_error():
    reference = [1.+.5*i for i in range(60)]
    predicted = [1.+.25*i for i in range(119)]
    out = mod.evaluate_values(reference, reference[::4], predicted,
                              predicted[::4], 4)
    assert out["best_counterfactual_beat_mode"] in ("half_even", "half_odd")
    assert out["opportunity"]["pulse_counterfactual_gain_ge_10pp"]
    assert out["oracles"]["half_even"]["beat"]["f_measure"] == 1


def test_test_split_is_never_evaluated(monkeypatch, tmp_path):
    seen = []
    def evaluate(item, root, *, precomputed_only=False):
        seen.append(item["id"])
        return {"id": item["id"], "status": "missing_saved_prediction"}
    monkeypatch.setattr(mod, "evaluate_track", evaluate)
    result = mod.compare({"tracks": [
        {"id": "validation", "split": "validation"},
        {"id": "test", "split": "test", "audio": "do-not-read"},
    ]}, tmp_path, precomputed_only=True)
    assert seen == ["validation"]
    assert result["aggregate"]["evaluated"] == 0
    assert result["phase_integrity_limit_changed"] is False
    assert result["beat_grid_edit_authorized"] is False
    assert result["auto_promotion_allowed"] is False


def test_saved_prediction_without_audio_support(tmp_path):
    beats = [1.+.5*i for i in range(48)]
    (tmp_path/"beats.txt").write_text("\n".join(map(str, beats)))
    (tmp_path/"downbeats.txt").write_text("\n".join(map(str, beats[::4])))
    (tmp_path/"prediction.json").write_text(json.dumps({
        "beats_seconds": beats,
        "downbeats_seconds": beats[::4],
        "time_signature": 4,
    }))
    report = mod.compare({"tracks": [{
        "id": "v", "split": "validation", "beats": "beats.txt",
        "downbeats": "downbeats.txt", "prediction": "prediction.json",
    }]}, tmp_path, precomputed_only=True)
    assert report["aggregate"]["evaluated"] == 1
    assert report["aggregate"]["baseline_downbeat_macro_f1"] == 1
    assert report["reference_labels_used_for_oracles"]


def test_duplicate_ids_rejected_before_access(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        mod.compare({"tracks": [
            {"id": "x", "split": "validation"},
            {"id": "x", "split": "test"},
        ]}, tmp_path)
