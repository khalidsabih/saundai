"""Read-only shadow rate phase comparison contracts."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    file = Path(__file__).resolve().parents[1] / "scripts" / "m1-shadow-rate-phase-replay.py"
    spec = importlib.util.spec_from_file_location("m1_shadow_rate_phase", file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assessment():
    return {"shadow_only": True, "comparison": {
        "hypothetical_incoming_source_rate_to_match_outgoing": 0.99,
        "rate_is_research_counterfactual_only": True,
    }}


def row():
    return {
        "track_a": "a.wav", "track_b": "b.wav",
        "exit_time_a_seconds": 8.0, "entry_time_b_seconds": 8.0,
        "transition_seconds": 8.0,
        "render_rate_a": 1.0, "render_rate_b": 1.0,
        "phase_nudge_seconds_b": 0.0,
        "points": [{"fraction": x / 8} for x in range(9)],
    }


def fake_profile(beats_a, beats_b, **params):
    residual = 210.0 if params["rate_b"] == 1.0 else 18.0
    return {
        "status": "replayed",
        "max_bounded_residual_ms": residual,
        "numeric_residual_below_25ms": residual <= 25,
    }


def test_rate_comparison_does_not_rewrite_input():
    mod = load()
    tracks = {
        "a.wav": {"beats": list(range(100)), "bpm": 120},
        "b.wav": {"beats": list(range(100)), "bpm": 120},
    }
    trace = [row()]
    before = dict(trace[0])
    result = mod.compare(trace, assessment(), tracks, fake_profile)
    assert result["paired_numeric_profiles"] == 1
    assert result["median_original_numeric_residual_ms"] == 210.0
    assert result["median_hypothetical_numeric_residual_ms"] == 18.0
    assert result["hypothetical_numeric_below_25ms_count"] == 1
    assert result["new_plan_created"] is False
    assert result["production_changed"] is False
    assert trace[0] == before


def test_rejects_unmarked_or_invalid_hypothetical_rate():
    import pytest
    mod = load()
    for candidate in [
        {"comparison": {"hypothetical_incoming_source_rate_to_match_outgoing": .99}},
        {"shadow_only": True, "comparison": {"hypothetical_incoming_source_rate_to_match_outgoing": .99}},
        {"shadow_only": True, "comparison": {"rate_is_research_counterfactual_only": True,
                                              "hypothetical_incoming_source_rate_to_match_outgoing": 0}},
    ]:
        with pytest.raises(ValueError, match="research-only"):
            mod.compare([], candidate, {}, fake_profile)


def test_missing_track_skipped_not_marked_safe():
    mod = load()
    result = mod.compare([row()], assessment(), {}, fake_profile)
    assert result["paired_numeric_profiles"] == 0
    assert result["hypothetical_numeric_below_25ms_count"] == 0
    assert result["skipped"]["unknown_track"] == 1
