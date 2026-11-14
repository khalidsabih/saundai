import pytest

from scripts.compare_local_beat_disagreements import (
    compare_junction, diagnose, monotone_align,
)


def _beats(n=80, period=0.5):
    return [i*period for i in range(n)]


def _junction(start=12, end=22):
    return {
        "junction": {"left_index": 30, "right_index": 32},
        "source_start": float(start),
        "source_end": float(end),
    }


def test_exact_sequences_align_monotonically_with_zero_cost():
    a = _beats(25)
    result = monotone_align(a, a, period=0.5)
    assert result["status"] == "aligned"
    assert result["matched"] == len(a)
    assert result["unmatched_cached"] == 0
    assert result["unmatched_independent"] == 0
    assert result["total_alignment_cost"] == 0


def test_monotone_alignment_detects_unmatched_event():
    a = _beats(30)
    b = [x for i,x in enumerate(a) if i != 15]
    result = monotone_align(a, b, period=0.5)
    assert result["matched"] == len(b)
    assert result["unmatched_cached"] == 1
    assert result["unmatched_independent"] == 0


def test_40ms_disagreement_was_within_70ms_but_exceeds_25ms_screen():
    cached = _beats()
    independent = [x + (0.04 if abs(x-15.0) < 0.001 else 0.0)
                   for x in cached if 12 <= x <= 22]
    report = compare_junction(cached, independent, _junction(), expected_period=0.5)
    assert report["status"] == "compared"
    assert report["matched_over_25ms_but_within_70ms"] == 1
    assert report["classification"] == "matched_event_timing_disagreement"
    assert not report["beat_grid_edit_authorized"]


def test_local_missing_independent_event_is_reported_not_repaired():
    cached = _beats()
    independent = [x for x in cached if 12 <= x <= 22 and x != 15.0]
    report = compare_junction(cached, independent, _junction(), expected_period=0.5)
    assert report["status"] == "compared"
    assert any(abs(x["cached_seconds"]-15.0) < 0.001
               for x in report["unmatched_cached_events"])
    assert report["index_step_disagreements"]


def test_bad_timing_fails_closed():
    assert monotone_align([0.0, 0.0], [0.5], period=0.5)["status"] == "invalid_sequences"
    assert compare_junction(_beats(), [], _junction(),
                            expected_period=0.5)["status"] == "missing_comparable_events"


def test_plan_id_mismatch_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "a"}, {"plan_id": "b"}, tmp_path)


def test_no_rejected_candidate_skipped(tmp_path):
    result = diagnose(
        {"plan_id": "a", "transitions": [{}]},
        {"plan_id": "a", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
