import numpy as np
import pytest

from scripts.diagnose_continuous_beat_phase import (
    evaluate_sequence, diagnose, _independent_support,
)


def _events(beats):
    return [{"seconds": float(t), "prominence_ratio": 9.0} for t in beats]


def test_steady_pulse_is_supported_without_circular_beat_matching():
    times = [i * 0.5 for i in range(20)]
    out = evaluate_sequence(times, _events(times), start=2.0,
                            end=7.0, expected_period=0.5)
    assert out["status"] == "evaluated"
    assert out["hypotheses"][0]["name"] == "original_event_grid"
    assert out["hypotheses"][0]["onset_support_score"] > 0.85
    assert not out["beat_correction_authorized"]


def test_missing_beat_hypothesis_is_examined_not_accepted():
    times = [i * 0.5 for i in range(20) if i != 10]
    events = _events([i * 0.5 for i in range(20)])
    out = evaluate_sequence(times, events, start=3.5, end=7.0,
                            expected_period=0.5)
    names = [h["name"] for h in out["hypotheses"]]
    assert "single_missing_beat" in names
    assert not out["beat_correction_authorized"]


def test_extra_beat_hypothesis_is_examined_not_accepted():
    times = sorted([i * 0.5 for i in range(20)] + [5.25])
    out = evaluate_sequence(times, _events([i*0.5 for i in range(20)]),
                            start=3.5, end=7.0, expected_period=0.5)
    assert "single_extra_beat" in [h["name"] for h in out["hypotheses"]]
    assert not out["beat_correction_authorized"]


def test_invalid_input_fails_closed():
    assert evaluate_sequence([0.0, 0.0, 1.0], [], start=0,
                             end=1, expected_period=0.5)["status"] == "invalid_input"


def test_plan_mismatch_fails_before_using_cached_tracks(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "a"}, {"plan_id": "b"}, tmp_path)


def test_no_rejected_candidate_has_explicit_status(tmp_path):
    result = diagnose(
        {"plan_id": "test", "transitions": [{}]},
        {"plan_id": "test", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
