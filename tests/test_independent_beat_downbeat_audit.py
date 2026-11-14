import numpy as np
import pytest

from scripts.audit_independent_beat_downbeat import (
    analyze_junction, compare_pulse_phases, compare_times,
    diagnose, four_beat_accent_hypotheses, run_independent_tracker,
)


def test_match_once_no_reused_detection():
    result = compare_times([1.00, 1.02], [1.01])
    assert result["matches"] == 1
    assert result["coverage"] == 0.5


def test_detected_halfbeat_preference_not_authorization():
    original = [i*0.5 for i in range(20)]
    detected = [x+0.25 for x in original]
    result = compare_pulse_phases(original, detected, period=0.5)
    assert result["half_beat_later"]["coverage"] == 1.0
    assert result["original"]["coverage"] == 0
    assert not result["phase_choice_authorized"]


def test_four_beat_accents_are_hypotheses_not_verified_downbeats():
    positions = [i*0.5 for i in range(24)]
    envelope = np.zeros(1200)
    for i in range(24):
        envelope[int(positions[i]*80)] = 5 if i % 4 == 0 else 1
    result = four_beat_accent_hypotheses(
        positions, envelope, window_start=0, frames_per_second=80,
    )
    assert result["status"] == "accent_hypotheses_only"
    assert result["highest_accent_bin"] == 0
    assert not result["downbeat_verified"]


def test_insufficient_context_cannot_claim_meter():
    result = four_beat_accent_hypotheses(
        [0.0, 0.5], np.ones(100),
        window_start=0, frames_per_second=80,
    )
    assert result["status"] == "insufficient_beat_context"
    assert not result["downbeat_verified"]


def test_short_silence_not_fabricated_beats():
    result = run_independent_tracker(
        np.zeros(500), sr=22050, start=0, target_bpm=120,
    )
    assert result["status"] == "insufficient_audio"


def test_missing_audio_is_explicit(tmp_path):
    beats = [i*0.5 for i in range(120)]
    out = analyze_junction(
        tmp_path/"missing.wav", beats,
        {"left_run_final_index": 40, "right_run_first_index": 43},
        expected_period=0.5,
    )
    assert out["status"] == "source_audio_unavailable"


def test_plan_mismatch_prevents_analysis(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "a"}, {"plan_id": "b"}, tmp_path)


def test_no_rejected_candidate_is_skipped(tmp_path):
    result = diagnose(
        {"plan_id": "same", "transitions": [{}]},
        {"plan_id": "same", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
