from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_world_model_utilization import (
    _phrase_at, analyze_plan, load_world_records,
)


def _record(path: str, *, role: str = "breakdown") -> dict:
    return {
        "cache_path": "cached.json",
        "source_path": path,
        "fingerprint": "fingerprint",
        "created_at_utc": "2026-11-13",
        "payload": {
            "path": path,
            "version": "6.0",
            "bpm": 124.0,
            "phrases": [{
                "start_seconds": 10.0,
                "end_seconds": 30.0,
                "role": role,
                "section": "breakdown",
                "transition_readiness": 0.82,
                "beat_confidence": 0.7,
                "bass_activity": 0.15,
                "vocal_probability": 0.5,
            }],
            "phrase_graph": [],
            "neural_evidence": {},
            "stem_activity": {"bass": 0.5},
        },
    }


def test_phrase_mapping_respects_bounds():
    phrases = _record("a.wav")["payload"]["phrases"]
    assert _phrase_at(phrases, 10.0)["role"] == "breakdown"
    assert _phrase_at(phrases, 29.99)["readiness"] == 0.82
    assert _phrase_at(phrases, 30.0) is None
    assert _phrase_at(phrases, None) is None


def test_trace_cue_context_without_claiming_causality():
    plan = {
        "tracks": [
            {"path": "/temporary/crate/a.wav", "analysis": {"bpm": 124.0}},
            {"path": "/temporary/crate/b.wav", "analysis": {"bpm": 124.0}},
        ],
        "transitions": [{
            "from_index": 0, "to_index": 1,
            "plan": {
                "exit_time_a": 15.0, "entry_time_b": 18.0,
                "technique": "echo_out_drop_in", "beatmatched": False,
                "metadata": {},
            },
        }],
    }
    result = analyze_plan(plan, {
        "a.wav": [_record("/different/original/a.wav")],
        "b.wav": [_record("/different/original/b.wav", role="drop")],
    })
    assert len(result["decisions"]) == 1
    decision = result["decisions"][0]
    assert decision["cached_exit_phrase"]["role"] == "breakdown"
    assert decision["cached_entry_phrase"]["role"] == "drop"
    assert decision["cache_pair_resolved"]
    assert decision["source_of_cue_choice_proven"] is False
    assert "fingerprint" in result["track_knowledge"][0]["warning"]


def test_ambiguous_cache_skips_cue_mapping():
    plan = {
        "tracks": [{"path": "a.wav"}],
        "transitions": [{
            "from_index": 0, "to_index": 0,
            "plan": {"exit_time_a": 10.0, "entry_time_b": 10.0},
        }],
    }
    result = analyze_plan(plan, {"a.wav": [_record("a.wav"), _record("elsewhere/a.wav")]})
    assert result["track_knowledge"][0]["join_status"] == "ambiguous"
    assert result["decisions"][0]["cached_exit_phrase"] is None
    assert not result["decisions"][0]["cache_pair_resolved"]


def test_cache_loader_handles_valid_and_malformed(tmp_path: Path):
    base = tmp_path / "hash" / "world_model"
    base.mkdir(parents=True)
    (base / "world-model__6.0.json").write_text(json.dumps({
        "payload": _record("/different/audio/test.wav")["payload"],
        "fingerprint": "abc",
    }), encoding="utf-8")
    records, errors = load_world_records(tmp_path)
    assert errors == 0
    assert len(records["test.wav"]) == 1
