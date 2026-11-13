from types import SimpleNamespace

from djai.decision_trace import candidate_pool_trace, candidate_snapshot, rejected_phase_trace


def _plan(technique="equal_power_blend"):
    return SimpleNamespace(
        execution_signature="sig-1",
        technique=technique,
        style="safe",
        exit_time_a=32.0,
        entry_time_b=16.0,
        transition_bars=8,
        target_bpm=124.0,
        render_rate_a=1.0,
        render_rate_b=1.02,
        beatmatched=True,
        score=0.8,
        cue_confidence=0.7,
        fallback_level=0,
        metadata={"transition_router": {"family": "blend"}, "selection_utility": 0.81},
    )


def test_snapshot_is_read_only_and_structured():
    p = _plan()
    before = dict(p.metadata)
    result = candidate_snapshot(p)
    assert result["technique"] == "equal_power_blend"
    assert result["render_rate_b"] == 1.02
    assert result["transition_router"]["family"] == "blend"
    assert p.metadata == before


def test_bounded_candidate_snapshot_marks_truncation():
    p = _plan()
    result = candidate_pool_trace([p, p, p], limit=2)
    assert result["total_candidates"] == 3
    assert result["retained_candidates"] == 2
    assert result["truncated"]
    assert len(result["candidates"]) == 2


def test_phase_rejection_keeps_measured_points():
    p = _plan()
    profile = {"safe": False, "max_residual_ms": 130.0, "points": [{"fraction": 0.5, "residual_ms": 120.0}]}
    result = rejected_phase_trace(p, profile)
    assert result["phase_integrity"]["points"][0]["residual_ms"] == 120.0
    assert result["candidate"]["signature"] == "sig-1"
    assert profile["safe"] is False
