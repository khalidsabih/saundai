"""Shadow-only hierarchical meter candidate contracts."""
import json

import numpy as np
import pytest

from scripts import shadow_hierarchical_meter as mod


def _features(beats=60):
    x = np.zeros((beats, 14), dtype=float)
    x[:, 0] = 1.0
    x[:, 1] = 0.1
    x[:, 2] = 1.0
    return x


def test_missing_evidence_abstains():
    result = mod.infer_meter(np.zeros((8, 14)))
    assert result["abstained"]
    assert result["meter"] is None
    assert mod.choose_downbeats([i*.5 for i in range(8)], result) == []


def test_constant_accent_and_chroma_abstain():
    result = mod.infer_meter(_features())
    assert result["abstained"]
    assert result["meter"] is None
    assert result["score_is_not_probability"]


def test_three_pulse_accents_expose_hierarchical_scores():
    x = _features()
    x[:, 0] = .02
    x[::3, 0] = 6.0
    result = mod.infer_meter(x)
    assert result["scores"]["3"]["score"] > 0
    assert result["scores"]["6"]["six_group_asymmetry"] is not None
    assert result["scores"]["6"]["six_group_asymmetry"] < mod.MIN_SIX_GROUP_ASYMMETRY
    assert result["top_meter_before_abstention"] in (3, 4, 6)
    assert result["meter"] in (3, 4, 6, None)


def test_feature_validation_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        mod.infer_meter(np.full((30, 14), np.nan))


def test_choose_downbeats_uses_inferred_meter_and_phase_only():
    beats = [1.0+i*.5 for i in range(40)]
    inferred = {"abstained": False, "meter": 4, "phase": 2}
    assert mod.choose_downbeats(beats, inferred) == beats[2::4]
    with pytest.raises(ValueError):
        mod.choose_downbeats(beats, {"abstained": False, "meter": 0, "phase": 0})


def test_only_validation_split_is_accessed_and_no_production_changes(monkeypatch, tmp_path):
    seen = []
    def fake_eval(item, root):
        seen.append(item["id"])
        return {
            "id": item["id"], "status": "evaluated",
            "candidate": {"abstained": True, "meter": None},
            "production_meter_correct": True,
            "autonomous_meter_correct": None,
            "production_downbeat": {"f_measure": .25},
            "shadow_with_abstention_fallback": {"f_measure": .25},
        }
    monkeypatch.setattr(mod, "evaluate_one", fake_eval)
    result = mod.compare({"tracks": [
        {"id": "val", "split": "validation"},
        {"id": "never-read", "split": "test", "audio": "unavailable"},
    ]}, tmp_path)
    assert seen == ["val"]
    assert result["summary"]["abstained_count"] == 1
    assert result["summary"]["baseline_downbeat_macro_f1_all"] == .25
    assert result["summary"]["fallback_shadow_downbeat_macro_f1_all"] == .25
    assert result["production_changes"] is False
    assert result["phase_integrity_limit_changed"] is False
    assert result["reference_labels_used_for_candidate"] is False


def test_invalid_manifest_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        mod.compare({"tracks": [
            {"id": "duplicated", "split": "validation"},
            {"id": "duplicated", "split": "test"},
        ]}, tmp_path)
    result = mod.compare({"tracks": [
        {"id": "missing", "split": "validation",
         "beats": "missing.label", "audio": "missing.wav"},
    ]}, tmp_path)
    assert result["summary"]["evaluated"] == 0
