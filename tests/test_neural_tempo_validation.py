from __future__ import annotations

import pytest

from djai.intelligence.tempo_validation import (
    prospective_impact, validate_references, validation_summary,
)


def _row(reference, baseline, action, suggested=None, quality="independently_verified"):
    return {
        "reference_quality": quality,
        "reference_bpm": reference,
        "status": "ok",
        "comparison": {"deterministic": {"absolute_error_bpm": abs(baseline-reference)}},
        "arbitration": {"decision": action, "suggested_bpm": suggested,
                        "changes_dj_decisions": False},
    }


def test_reference_provenance_is_required_for_verified_data() -> None:
    reference = {"filename": "track.wav", "reference_bpm": 123}
    assert validate_references([reference])[0]["reference_quality"] == "approximate"
    with pytest.raises(ValueError, match="verification_method"):
        validate_references([{**reference, "reference_quality": "independently_verified"}])
    assert validate_references([{
        **reference, "reference_quality": "independently_verified",
        "verification_method": "manually annotated 64 kick intervals",
    }])[0]["reference_quality"] == "independently_verified"


def test_rejects_bad_duplicate_and_nonfinite_references() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_references([
            {"filename": "a.wav", "reference_bpm": 123},
            {"filename": "a.wav", "reference_bpm": 125},
        ])
    for bpm in (0, -5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="BPM"):
            validate_references([{"filename": "bad.wav", "reference_bpm": bpm}])
    with pytest.raises(ValueError, match="unknown reference_quality"):
        validate_references([{"filename": "track.wav", "reference_bpm": 123,
                              "reference_quality": "unknown"}])


def test_false_correction_measured_against_unfolded_reference() -> None:
    # Correct initial tempo should NOT be changed to the double-time pulse.
    impact = prospective_impact(_row(78, 78.0, "review_candidate", 156))
    assert impact["hypothetical_adoption"] is True
    assert impact["would_introduce_error_over_tolerance"] is True
    assert impact["candidate_error_bpm"] == 78.0
    assert impact["would_worsen"] is True


def test_retain_and_abstain_never_simulate_a_correction() -> None:
    for action in ("retain", "abstain"):
        impact = prospective_impact(_row(123, 123.05, action))
        assert impact["action_evaluated"] == action
        assert impact["hypothetical_adoption"] is False
        assert impact["candidate_error_bpm"] is None


def test_verified_metrics_exclude_approximate_reference() -> None:
    improved = _row(78, 152, "review_candidate", 78.9)
    worsened = _row(123, 123.05, "review_candidate", 125.5)
    approximate = _row(100, 100, "review_candidate", 160, "approximate")
    rows = []
    for row in (improved, worsened, approximate):
        row["prospective_impact"] = prospective_impact(row)
        rows.append(row)
    summary = validation_summary(rows)
    assert summary["verified_reference_count"] == 2
    assert summary["approximate_reference_count"] == 1
    assert summary["review_candidates_simulated"] == 2
    assert summary["candidate_improvements"] == 1
    assert summary["candidate_regressions"] == 1
    assert summary["false_corrections_over_2_bpm"] == 1
    assert summary["false_correction_rate"] == .5
    assert summary["ready_for_live_adoption"] is False
    assert summary["live_adoptions"] == 0


def test_no_verified_ground_truth_means_no_validated_rate() -> None:
    approx = _row(78, 152, "review_candidate", 79, "approximate")
    approx["prospective_impact"] = prospective_impact(approx)
    report = validation_summary([approx])
    assert report["verified_reference_count"] == 0
    assert report["false_correction_rate"] is None
    assert report["ready_for_live_adoption"] is False
