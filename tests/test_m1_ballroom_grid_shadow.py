from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-grid-shadow.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_grid_shadow", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_scores_rejected_grid_without_replacing_selected():
    mod = load()
    ref = {
        "dataset_id": "ballroom",
        "split": "development",
        "items": [
            {"id": "one", "bpm": None,
             "beats_seconds": [0.5, 1.0, 1.5, 2.0],
             "downbeats_seconds": [0.5, 2.0]},
        ],
    }
    pred = {"one": {
        "selected": {
            "bpm": 120.0,
            "beats_seconds": [0.25, 0.75, 1.25, 1.75],
            "downbeats_seconds": [0.25, 1.75],
        },
        "candidate": {
            "available": True,
            "used": False,
            "confidence": 0.5,
            "authority": 0.1,
            "reason": "gate",
            "bpm": 120.0,
            "beats_seconds": [0.5, 1.0, 1.5, 2.0],
            "downbeats_seconds": [0.5, 2.0],
        },
    }}
    result = mod.score(ref, pred)
    assert result["coverage"]["neural_grid_rejected_with_candidate"] == 1
    assert result["selected_metrics"]["beat_f1"] == 0.0
    assert result["selected_metrics"]["downbeat_f1"] == 0.0
    assert result["neural_candidate_metrics"]["beat_f1"] == 1.0
    assert result["neural_candidate_metrics"]["downbeat_f1"] == 1.0
    assert result["per_track"][0]["selected_bpm"] == 120.0


def test_unavailable_candidate_excluded_from_coverage():
    mod = load()
    ref = {
        "dataset_id": "ballroom", "split": "development",
        "items": [{"id": "one", "bpm": None,
                   "beats_seconds": [0.5, 1.0],
                   "downbeats_seconds": [0.5]}],
    }
    pred = {"one": {
        "selected": {"bpm": 120.0, "beats_seconds": [0.5, 1.0], "downbeats_seconds": [0.5]},
        "candidate": {"available": False, "used": False, "reason": "missing"},
    }}
    result = mod.score(ref, pred)
    assert result["coverage"]["neural_unavailable"] == 1
    assert result["neural_candidate_metrics"] is None
    assert result["selected_metrics"]["beat_f1"] == 1.0


def test_grid_bpm_uses_median_interval():
    mod = load()
    assert mod.grid_bpm([0.5, 1.0, 1.5, 2.0]) == 120.0
    assert mod.grid_bpm([]) is None
