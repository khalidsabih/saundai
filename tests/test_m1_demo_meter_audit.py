"""Fixture-contract tests for read-only synthetic 4/4 meter audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "m1_demo_meter_audit", ROOT / "scripts" / "m1-demo-meter-audit.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_track(meter: int, *, neural_used: bool = False):
    beats = [0.5 * n for n in range(48)]
    return SimpleNamespace(
        time_signature=meter,
        time_signature_confidence=0.78,
        bpm=120.0,
        beat_grid_confidence=0.88,
        downbeat_confidence=0.76,
        beats_seconds=beats,
        downbeats_seconds=beats[::meter],
        neural_evidence={"beat_downbeat": {
            "available": True, "used": neural_used,
            "confidence": 0.82, "reason": "test",
        }},
    )


def test_audit_recognizes_known_generated_four_four(tmp_path):
    m = load()
    for name in m.DEMO_TRACKS:
        (tmp_path / name).touch()
    result = m.audit(tmp_path, lambda _: fake_track(4))
    assert result["all_meters_match_generator"] is True
    assert len(result["results"]) == 2
    assert all(r["inferred_beats_per_downbeat_interval"] == 4 for r in result["results"])
    assert result["production_changed"] is False


def test_audit_flags_incoming_six_without_overriding(tmp_path):
    m = load()
    for name in m.DEMO_TRACKS:
        (tmp_path / name).touch()
    result = m.audit(
        tmp_path,
        lambda path: fake_track(6, neural_used=True)
        if path.name == m.DEMO_TRACKS[1] else fake_track(4),
    )
    assert result["all_meters_match_generator"] is False
    assert result["results"][0]["meter_matches_generator"] is True
    assert result["results"][1]["predicted_meter"] == 6
    assert result["results"][1]["expected_meter"] == 4
    assert result["results"][1]["neural_used"] is True
    assert result["rhythm_policy_changed"] is False


def test_audit_requires_expected_fixture_files(tmp_path):
    m = load()
    try:
        m.audit(tmp_path, lambda _: fake_track(4))
    except ValueError as exc:
        assert "Missing expected generated fixture" in str(exc)
    else:
        raise AssertionError("Missing demo fixtures should be rejected")
