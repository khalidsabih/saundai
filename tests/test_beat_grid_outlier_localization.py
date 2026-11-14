import pytest

from scripts.localize_beat_grid_outliers import (
    inspect_grid, nearest_independent_onset, diagnose,
)


def _beats(n=60, period=0.5):
    return [i * period for i in range(n)]


def test_clean_grid_has_no_outliers():
    result = inspect_grid(
        _beats(), [], start=10.0, end=17.0, expected_period=0.5,
    )
    assert result["status"] == "audited"
    assert result["outlier_count"] == 0
    assert not result["index_change_authorized"]


def test_single_displaced_timestamp_reports_original_and_audio_support():
    beats = _beats()
    beats[24] += 0.12
    events = [
        {"seconds": 12.0, "prominence_ratio": 5.0},
        {"seconds": 12.12, "prominence_ratio": 8.0},
    ]
    result = inspect_grid(
        beats, events, start=10.0, end=15.0, expected_period=0.5,
    )
    findings = [x for x in result["outliers"] if x["global_beat_index"] == 24]
    assert findings
    item = findings[0]
    assert abs(item["reference_offset_ms"]) > 45
    assert item["independent_onset_near_cached"]["seconds"] == 12.12
    assert item["independent_onset_near_reference"]["seconds"] == 12.0
    assert not item["index_change_authorized"]


def test_persistent_offset_after_missing_beat_is_flagged_not_repaired():
    beats = [i * 0.5 for i in range(60) if i != 25]
    result = inspect_grid(
        beats, [], start=10.0, end=20.0, expected_period=0.5,
    )
    assert result["outlier_count"] > 0
    assert any(
        "potential_missing_beat" in x["indexing_hypotheses"]
        for x in result["outliers"]
    )
    assert not result["index_change_authorized"]


def test_invalid_grid_fails_closed():
    assert inspect_grid([0.0, 0.0], [], start=0, end=2,
                        expected_period=0.5)["status"] == "invalid_grid_or_window"


def test_audio_candidate_has_finite_radius():
    events = [{"seconds": 4.1, "prominence_ratio": 10.0}]
    assert nearest_independent_onset(4.0, events) is None


def test_plan_mismatch_prevents_comparison(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "plan1"}, {"plan_id": "plan2"}, tmp_path)


def test_transition_without_rejected_candidate_is_skipped(tmp_path):
    result = diagnose(
        {"plan_id": "same", "transitions": [{}]},
        {"plan_id": "same", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
