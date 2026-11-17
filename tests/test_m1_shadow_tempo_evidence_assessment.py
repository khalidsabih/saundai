"""Contract checks for read-only shadow tempo evidence assessment."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-shadow-tempo-evidence-assessment.py"
    spec = importlib.util.spec_from_file_location("m1_shadow_tempo_assessment", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(name, scalar, grid, audio):
    return {
        "filename": name,
        "predicted_meter": 4,
        "bpm": scalar,
        "grid_span_bpm": grid,
        "tempo_consistency_evidence": {
            "decision": "retain",
            "reason": "no_material_global_disagreement",
            "shadow_audio_windows": [
                {"bpm": v, "strength": 0.95} for v in audio
            ],
        },
    }


def test_strong_onset_evidence_remains_shadow_only():
    module = load()
    source = {
        "results": [
            sample("opening.wav", 123.0469, 122.95467, [122.44625, 122.46173, 122.44389]),
            sample("groove.wav", 123.0469, 123.99157, [123.61149, 123.61791, 123.61261]),
        ],
    }
    assessment = module.assess(source)
    assert all(t["status"] == "independent_evidence_agrees_shadow_only" for t in assessment["tracks"])
    assert all(t["original_decision"] == "retain" for t in assessment["tracks"])
    assert assessment["comparison"]["hypothetical_incoming_source_rate_to_match_outgoing"] < 1
    assert assessment["shadow_only"] is True
    assert assessment["changes_to_reconciliation_threshold"] is False


def test_missing_or_conflicting_windows_fail_closed():
    module = load()
    missing = sample("opening.wav", 123, 122, [122, 122, 122])
    missing["tempo_consistency_evidence"]["shadow_audio_windows"] = []
    conflicted = sample("groove.wav", 123, 122, [120, 124, 128])
    assessment = module.assess({"results": [missing, conflicted]})
    assert assessment["tracks"][0]["status"] == "insufficient_shadow_evidence"
    assert assessment["tracks"][1]["status"] == "independent_evidence_conflict_or_weak"
    assert assessment["comparison"] is None


def test_invalid_audit_raises():
    import pytest

    with pytest.raises(ValueError, match="demo meter audit"):
        load().assess({})
