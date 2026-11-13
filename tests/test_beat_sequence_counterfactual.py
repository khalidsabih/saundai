import numpy as np
import pytest

from scripts.counterfactual_beat_sequence import diagnose, evaluate, refine_sequence


def _beats(period=0.5, count=160):
    return [round(i * period, 6) for i in range(count)]


def test_consistent_supported_shift_changes_only_matching_beat():
    beats = _beats()
    observations = [{
        "cached_beat_seconds": 10.0,
        "audio_onset_peak_seconds": 10.012,
        "local_peak_to_median_ratio": 4.0,
    }]
    result = refine_sequence(beats, observations, expected_interval_seconds=0.5)
    assert len(result["changes"]) == 1
    assert result["refined_beats"][20] == pytest.approx(10.012)
    assert result["refined_beats"][19] == beats[19]
    assert len(result["refined_beats"]) == len(beats)


def test_rejects_weak_large_and_interval_anomalies():
    beats = _beats()
    beats[25] = 12.42  # creates an abnormally short interval
    observations = [{
        "cached_beat_seconds": beats[25],
        "audio_onset_peak_seconds": beats[25] + 0.02,
        "local_peak_to_median_ratio": 6.0,
    }, {
        "cached_beat_seconds": beats[30],
        "audio_onset_peak_seconds": beats[30] + 0.04,
        "local_peak_to_median_ratio": 4.0,
    }, {
        "cached_beat_seconds": beats[40],
        "audio_onset_peak_seconds": beats[40] + 0.01,
        "local_peak_to_median_ratio": 1.0,
    }]
    result = refine_sequence(beats, observations, expected_interval_seconds=0.5)
    assert not result["changes"]
    assert result["rejection_reasons"]["interval_constraint"] == 1
    assert result["rejection_reasons"]["weak_or_large_shift"] == 2


def test_hypothetical_phase_is_separate_from_baseline():
    beats = _beats()
    shifted = [{
        "cached_beat_seconds": 10.0,
        "audio_onset_peak_seconds": 10.01,
        "local_peak_to_median_ratio": 3.5,
    }]
    report = evaluate(beats, beats, shifted, [],
                      exit_time_a=10.0, entry_time_b=10.0, duration=8.0,
                      bpm=120, rate_a=1, rate_b=1)
    assert report["status"] == "evaluated"
    assert report["changes_a"]
    assert report["baseline_uncorrected_max_drift_ms"] == 0


def test_different_plan_id_is_refused(tmp_path):
    with pytest.raises(ValueError, match="different plan IDs"):
        diagnose({"plan_id": "a", "transitions": []},
                 {"plan_id": "b", "transitions": []}, tmp_path)


def test_no_rejected_candidate_stays_unmodified(tmp_path):
    plan = {"plan_id": "a", "transitions": [{}]}
    audit = {"plan_id": "a", "transitions": [{"transition": 1, "status": "no_rejected_rhythmic_candidate"}]}
    result = diagnose(plan, audit, tmp_path)
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
