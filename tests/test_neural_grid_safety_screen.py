from __future__ import annotations

import pytest

from djai.intelligence.neural_grid_safety_screen import screen_full_grid_candidate


def _screen(**overrides):
    values = dict(
        baseline_bpm=123.0469, neural_bpm=123.5,
        fallback_beat_count=600, fallback_downbeat_count=150,
        neural_beat_count=600, neural_downbeat_count=150,
        fallback_beat_agreement=.93,
        neural_available=True, existing_gate_would_pass=True,
    )
    values.update(overrides)
    return screen_full_grid_candidate(**values)


def test_toc_toc_blocks_same_level_bpm_disagreement_despite_high_beat_agreement():
    result = _screen(
        baseline_bpm=86.1328, neural_bpm=88.2353,
        fallback_beat_count=304, fallback_downbeat_count=51,
        neural_beat_count=307, neural_downbeat_count=78,
        fallback_beat_agreement=.9674,
    )
    assert result["would_pass_provisional_screen"] is False
    assert "same_level_bpm_disagreement_over_2_percent" in result["screen_reasons"]
    assert result["approved_for_live_adoption"] is False
    assert result["changes_dj_decisions"] is False


def test_cola_blocks_low_beat_agreement_and_bpm_disagreement():
    result = _screen(
        baseline_bpm=123.0469, neural_bpm=120.0,
        fallback_beat_count=834, fallback_downbeat_count=209,
        neural_beat_count=834, neural_downbeat_count=280,
        fallback_beat_agreement=.4784,
    )
    assert result["would_pass_provisional_screen"] is False
    assert "same_level_bpm_disagreement_over_2_percent" in result["screen_reasons"]
    assert "fallback_beat_agreement_below_85_percent" in result["screen_reasons"]


def test_provisional_pass_does_not_authorize_live_adoption():
    result = _screen()
    assert result["would_pass_provisional_screen"] is True
    assert result["screen_reasons"] == []
    assert result["approved_for_live_adoption"] is False
    assert result["independent_downbeat_accuracy_verified"] is False


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1.0, 1.2])
def test_missing_or_invalid_agreement_fails_closed(value):
    result = _screen(fallback_beat_agreement=value)
    assert result["would_pass_provisional_screen"] is False
    assert "fallback_beat_agreement_unavailable" in result["screen_reasons"]


@pytest.mark.parametrize("neural_bpm", [None, 0.0, float("nan"), float("inf"), 240.1])
def test_missing_or_invalid_tempo_fails_closed(neural_bpm):
    result = _screen(neural_bpm=neural_bpm)
    assert result["would_pass_provisional_screen"] is False
    assert "bpm_evidence_invalid_or_outside_range" in result["screen_reasons"]


def test_octave_only_agreement_does_not_pass():
    result = _screen(baseline_bpm=156.0, neural_bpm=78.0)
    assert result["would_pass_provisional_screen"] is False
    assert "same_level_bpm_disagreement_over_2_percent" in result["screen_reasons"]


def test_no_fallback_evidence_or_existing_gate_is_never_promoted():
    assert "insufficient_fallback_beats" in _screen(fallback_beat_count=0)["screen_reasons"]
    assert "insufficient_fallback_downbeats" in _screen(fallback_downbeat_count=0)["screen_reasons"]
    assert "existing_grid_gate_did_not_pass" in _screen(existing_gate_would_pass=False)["screen_reasons"]
    assert "neural_evidence_unavailable" in _screen(neural_available=False)["screen_reasons"]


def test_thresholds_do_not_loosen():
    assert _screen(neural_bpm=123.5, fallback_beat_agreement=.85)["would_pass_provisional_screen"] is True
    assert _screen(neural_bpm=123.5, fallback_beat_agreement=.849)["would_pass_provisional_screen"] is False
