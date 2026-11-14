"""Safety and inference contracts for fixed-meter phase-only shadow."""
import numpy as np
import pytest

from scripts import shadow_downbeat_phase_only as mod


def _features(n=64, meter=4, phase=1):
    x = np.zeros((n, 14), dtype=float)
    x[:, 0] = .05
    x[:, 1] = .05
    x[phase::meter, 0] = 8.0
    x[phase::meter, 1] = 4.0
    x[:, 2] = 1.0
    return x


def test_fixed_four_phase_confident_and_stable():
    result = mod.infer_phase_fixed_meter(_features(), 4)
    assert result["meter"] == 4
    assert result["leading_phase"] == 1
    assert result["half_winners"] == [1, 1]
    assert result["half_agreement"]
    assert not result["abstained"]
    beats = [1+i*.5 for i in range(64)]
    assert mod.downbeats_for_phase(beats, result) == beats[1::4]


def test_fixed_three_phase_global_half_index():
    # Odd half length: local phase must map back into GLOBAL beat index.
    x = _features(n=61, meter=3, phase=2)
    result = mod.infer_phase_fixed_meter(x, 3)
    assert result["meter"] == 3
    assert result["leading_phase"] == 2
    assert result["half_winners"] == [2, 2]


def test_flat_and_unsupported_meter_abstain():
    flat = np.ones((64, 14))
    assert mod.infer_phase_fixed_meter(flat, 4)["abstained"]
    missing = mod.infer_phase_fixed_meter(_features(), 0)
    assert missing["abstained"]
    assert mod.downbeats_for_phase([.5*i for i in range(64)], missing) == []


def test_phase_change_between_halves_abstains():
    x = _features(n=64, meter=4, phase=1)
    x[32:, 0] = .05
    x[32:, 1] = .05
    x[34::4, 0] = 8
    x[34::4, 1] = 4
    result = mod.infer_phase_fixed_meter(x, 4)
    assert not result["half_agreement"]
    assert result["abstained"]


def test_invalid_inputs_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        mod.infer_phase_fixed_meter(np.full((64, 14), np.nan), 4)
    with pytest.raises(ValueError, match="invalid"):
        mod.downbeats_for_phase([.5*i for i in range(64)], {
            "abstained": False, "meter": 4, "phase": 4,
        })


def test_validation_selection_excludes_test_and_production_safe(monkeypatch, tmp_path):
    seen = []
    def fake_one(item, root):
        seen.append(item["id"])
        return {
            "id": item["id"], "status": "evaluated",
            "production_meter": 4, "reference_meter": 4,
            "production_meter_matches_reference": True,
            "fixed_phase": {"abstained": True},
            "hierarchical_meter_phase": {"abstained": True},
            "baseline": {"f_measure": .4},
            "fixed_phase_only_with_fallback": {"f_measure": .4},
            "hierarchical_with_fallback": {"f_measure": .4},
            "fixed_phase_f1_delta": 0,
            "hierarchical_f1_delta": 0,
        }
    monkeypatch.setattr(mod, "evaluate_one", fake_one)
    result = mod.compare({"tracks": [
        {"id": "validation", "split": "validation"},
        {"id": "test", "split": "test", "audio": "must-never-touch"},
    ]}, tmp_path)
    assert seen == ["validation"]
    assert result["summary"]["fixed_phase_only"]["abstained"] == 1
    assert result["summary"]["baseline_downbeat_macro_f1"] == .4
    assert result["production_changes"] is False
    assert result["phase_integrity_limit_changed"] is False
    assert result["reference_labels_used_for_inference"] is False


def test_duplicate_manifest_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        mod.compare({"tracks": [
            {"id": "same", "split": "validation"},
            {"id": "same", "split": "test"},
        ]}, tmp_path)


def test_summary_reports_regression_and_meter_groups():
    common = {
        "status": "evaluated", "production_meter": 4,
        "fixed_phase": {"abstained": False},
        "hierarchical_meter_phase": {"abstained": True},
        "baseline": {"f_measure": .6},
        "fixed_phase_only_accepted": {"f_measure": .2},
        "fixed_phase_only_with_fallback": {"f_measure": .2},
        "hierarchical_with_fallback": {"f_measure": .6},
        "fixed_phase_f1_delta": -.4,
        "hierarchical_f1_delta": 0,
    }
    out = mod._summary([{**common, "id": "regression", "reference_meter": 3,
                         "production_meter_matches_reference": False}])
    assert out["per_track_fixed_phase_regressed"] == 1
    assert out["regressions"][0]["id"] == "regression"
    assert out["by_reference_meter"]["3"]["tracks"] == 1
