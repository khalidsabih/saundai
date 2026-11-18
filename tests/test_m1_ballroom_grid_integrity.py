from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-grid-integrity.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_grid_integrity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid(beats=None, downs=None, bpm=120):
    return {
        "bpm": bpm,
        "beats_seconds": beats if beats is not None else [0, .5, 1, 1.5, 2, 2.5, 3, 3.5],
        "downbeats_seconds": downs if downs is not None else [0, 2],
    }


def test_consistent_grid_passes():
    m = load()
    result = m.grid_health(**{"bpm": 120, "beats": grid()["beats_seconds"], "downbeats": grid()["downbeats_seconds"]})
    assert result["passed"]
    assert result["bpm_difference"] == 0


def test_grid_bpm_mismatch_detected():
    m = load()
    assert "bpm_grid_mismatch" in m.grid_health(90, grid()["beats_seconds"], grid()["downbeats_seconds"])["failures"]


def test_offbeat_downbeat_detected():
    m = load()
    assert "downbeats_not_on_beats" in m.grid_health(120, grid()["beats_seconds"], [.2, 2.2])["failures"]


def test_frozen_gate_accepts_qualifying_whole_neural_grid():
    m = load()
    selected = grid(bpm=90)
    candidate = {**grid(), "available": True, "used": False, "confidence": .5}
    actual, used = m.choose(selected, candidate)
    assert used and actual["bpm"] == 120 and actual["beats_seconds"] == candidate["beats_seconds"]


def test_frozen_gate_rejects_low_confidence():
    m = load()
    selected = grid(bpm=90)
    candidate = {**grid(), "available": True, "used": False, "confidence": .499}
    actual, used = m.choose(selected, candidate)
    assert not used and actual is selected


def test_invalid_neural_grid_does_not_mask_structural_failure():
    m = load()
    selected = grid()
    candidate = {**grid(bpm=80), "available": True, "used": False, "confidence": .9}
    actual, used = m.choose(selected, candidate)
    assert not used and actual is selected


def test_empty_grid_fails_closed():
    m = load()
    result = m.grid_health(120, [], [0, 2])
    assert not result["passed"]


def test_integrity_report_covers_both_selected_and_challenger():
    m = load()
    manifest = {"items": [{"id": "a", "genre": "Waltz"}]}
    predictions = {"a": {
        "selected": grid(bpm=90),
        "candidate": {**grid(), "available": True, "used": False, "confidence": .51},
    }}
    result = m.evaluate(manifest, predictions)
    assert result["coverage"]["newly_accepted"] == 1
    assert result["coverage"]["selected_integrity_failures"] == 1
    assert result["coverage"]["challenger_integrity_failures"] == 0
