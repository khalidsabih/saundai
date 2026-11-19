from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    script = Path(__file__).resolve().parents[1] / "scripts" / "m1-neural-tempo-shadow.py"
    spec = importlib.util.spec_from_file_location("m1_neural_tempo_shadow", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference():
    return {
        "dataset_id": "groove_midi",
        "split": "development",
        "items": [
            {"id": "a", "bpm": 80.0},
            {"id": "b", "bpm": 120.0},
            {"id": "c", "bpm": 95.0},
        ],
    }


def test_candidate_scored_independent_of_beat_grid_gate():
    m = load()
    predictions = {
        "a": {
            "selected_bpm": 160.0, "neural_bpm": 80.5,
            "evidence": {"available": True, "used": False, "confidence": 0.49, "reason": "gate"},
        },
        "b": {
            "selected_bpm": 120.0, "neural_bpm": 120.0,
            "evidence": {"available": True, "used": True, "confidence": 0.85},
        },
        "c": {
            "selected_bpm": 130.0, "neural_bpm": 94.0,
            "evidence": {"available": True, "used": False, "confidence": 0.68},
        },
    }
    result = m.score(reference(), predictions)
    assert result["coverage"]["neural_candidate"] == 3
    assert result["coverage"]["neural_grid_accepted"] == 1
    assert result["coverage"]["neural_grid_rejected_with_candidate"] == 2
    assert result["metrics"]["selected_tempo_mae_bpm"] == pytest.approx(38.333333)
    assert result["metrics"]["neural_candidate_tempo_mae_bpm"] == 0.5
    assert result["metrics"]["candidate_tempo_mae_rejected_subset_bpm"] == 0.75
    assert result["per_track"][0]["selected_bpm"] == 160.0
    assert result["per_track"][0]["neural_candidate_bpm"] == 80.5


def test_unavailable_neural_never_counted_as_candidate():
    m = load()
    manifest = {"dataset_id": "groove_midi", "split": "development",
                "items": [{"id": "a", "bpm": 100.0}]}
    result = m.score(manifest, {"a": {
        "selected_bpm": 105.0, "neural_bpm": 100.0,
        "evidence": {"available": False, "used": False, "reason": "not installed"},
    }})
    assert result["coverage"]["neural_unavailable"] == 1
    assert result["metrics"]["neural_candidate_tempo_mae_bpm"] is None
    assert result["per_track"][0]["neural_candidate_bpm"] is None


def test_missing_prediction_and_invalid_candidate_fail_closed():
    m = load()
    with pytest.raises(ValueError, match="missing prediction"):
        m.score(reference(), {})
    manifest = {"dataset_id": "groove_midi", "split": "development",
                "items": [{"id": "a", "bpm": 100.0}]}
    with pytest.raises(ValueError, match="invalid neural candidate"):
        m.score(manifest, {"a": {"selected_bpm": 100, "neural_bpm": float("nan"),
                                   "evidence": {"available": True}}})
