from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-gate-challenger.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_gate_challenger", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def counts(match, total=4):
    return {"match": match, "reference": total, "predicted": total}


def fixture(confidence, available=True, mismatch=0, accepted=False):
    manifest = {
        "dataset_id": "ballroom",
        "split": "validation",
        "items": [{"id": "a", "genre": "Waltz", "bpm": 120}],
    }
    selected = counts(1)
    candidate = counts(4) if available else None
    row = {
        "id": "a",
        "grid_accepted": accepted,
        "neural_available": available,
        "confidence": confidence,
        "selected_beat_counts": selected,
        "selected_downbeat_counts": selected,
        "candidate_beat_counts": candidate,
        "candidate_downbeat_counts": candidate,
        "selected_bpm": 80,
        "neural_bpm": 120 if available else None,
        "neural_grid_bpm_difference": mismatch if available else None,
    }
    shadow = {
        "schema_version": "saund-m1-ballroom-grid-shadow-v1",
        "track_count": 1,
        "per_track": [row],
    }
    return manifest, shadow


def test_threshold_exactly_frozen_point_accepts_new_grid():
    m = load()
    result = m.evaluate(*fixture(0.50))
    assert result["coverage"]["newly_accepted"] == 1
    assert result["challenger_metrics"]["beat_f1"] == 1.0
    assert result["challenger_metrics"]["downbeat_f1"] == 1.0
    assert result["challenger_metrics"]["tempo_mae_bpm"] == 0.0
    assert result["selected_metrics"]["tempo_mae_bpm"] == 40.0


def test_reject_below_threshold_missing_neural_and_grid_mismatch():
    m = load()
    for manifest, shadow in [fixture(0.499), fixture(0.70, available=False), fixture(0.70, mismatch=2.1)]:
        result = m.evaluate(manifest, shadow)
        assert result["coverage"]["newly_accepted"] == 0
        assert result["challenger_metrics"]["beat_f1"] == 0.25


def test_keep_already_accepted_and_detect_regression():
    m = load()
    manifest, shadow = fixture(0.8, accepted=True)
    shadow["per_track"][0]["selected_beat_counts"] = counts(4)
    shadow["per_track"][0]["selected_downbeat_counts"] = counts(4)
    result = m.evaluate(manifest, shadow)
    assert result["coverage"]["newly_accepted"] == 0
    assert result["challenger_metrics"]["beat_f1"] == 1.0
    assert result["regressions"]["beat_f1_decreased"] == 0


def test_challenger_reports_severe_beat_regression():
    m = load()
    manifest, shadow = fixture(0.51)
    shadow["per_track"][0]["selected_beat_counts"] = counts(4)
    shadow["per_track"][0]["candidate_beat_counts"] = counts(1)
    result = m.evaluate(manifest, shadow)
    assert result["regressions"]["beat_f1_decreased"] == 1
    assert result["regressions"]["beat_f1_drop_gt_0_10_tracks"] == ["a"]


def test_reject_mismatched_track_ids():
    m = load()
    manifest, shadow = fixture(0.6)
    shadow["per_track"][0]["id"] = "unknown"
    try:
        m.evaluate(manifest, shadow)
    except ValueError as exc:
        assert "track IDs mismatch" in str(exc)
    else:
        assert False, "must reject mismatched shadow outputs"
