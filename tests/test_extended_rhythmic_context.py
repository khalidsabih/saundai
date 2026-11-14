import numpy as np
import pytest

from scripts.verify_extended_rhythmic_context import (
    compare_phases, diagnose, phase_support, verify_junction,
)


def _events(times):
    return [{"seconds": float(t), "prominence_ratio": 10.0} for t in times]


def test_one_to_one_onsets_do_not_double_count():
    result = phase_support([1.0, 1.02], _events([1.01]))
    assert result["matched"] == 1
    assert result["tested"] == 2


def test_original_phase_supported_by_low_band():
    beats = [i*0.5 for i in range(10)]
    result = compare_phases(
        beats,
        {"low": _events(beats), "high": [], "full": _events(beats)},
        period=0.5,
    )
    assert result["low"]["interpretation"] == "original_phase_onset_preferred"
    assert result["low"]["original"]["matched"] == 10
    assert result["low"]["half_beat_later"]["matched"] == 0


def test_halfbeat_onsets_are_reported_as_alternative_not_beat_proof():
    beats = [i*0.5 for i in range(10)]
    result = compare_phases(
        beats,
        {"low": _events([x+0.25 for x in beats]), "high": [], "full": []},
        period=0.5,
    )
    assert result["low"]["interpretation"] == "half_beat_onset_preferred"
    assert result["low"]["original"]["matched"] == 0


def test_both_phase_subdivisions_remain_ambiguous():
    beats = [i*0.5 for i in range(10)]
    result = compare_phases(
        beats,
        {"low": _events(beats+[x+0.25 for x in beats]), "high": [], "full": []},
        period=0.5,
    )
    assert result["low"]["interpretation"] == "ambiguous"


def test_invalid_junction_does_not_touch_audio(tmp_path):
    result = verify_junction(
        [i*0.5 for i in range(40)],
        tmp_path/"nonexistent.wav",
        {"left_run_final_index": 7, "right_run_first_index": 6},
        expected_period=0.5,
    )
    assert result["status"] == "invalid_junction_indices"


def test_missing_audio_reports_unavailable(tmp_path):
    result = verify_junction(
        [i*0.5 for i in range(80)],
        tmp_path/"nonexistent.wav",
        {"left_run_final_index": 30, "right_run_first_index": 33},
        expected_period=0.5,
    )
    assert result["status"] == "audio_missing_or_invalid_window"


def test_mismatched_plan_ids_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "a"}, {"plan_id": "b"}, tmp_path)


def test_no_rejected_candidate_is_skipped(tmp_path):
    report = diagnose(
        {"plan_id": "a", "transitions": [{}]},
        {"plan_id": "a", "transitions": []}, tmp_path,
    )
    assert report["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
