import pytest

from scripts.counterfactual_crossdeck_phase import (
    diagnose, hypothetical_incoming, phase_profile,
)


def _beats(n=150):
    return [round(i * 0.5, 6) for i in range(n)]


def _hyp(score=0.8):
    return {
        "name": "steady_pulse_at_independent_tempo",
        "status": "scored",
        "onset_support_score": score,
    }


def test_identical_steady_decks_replay_zero_bounded_phase_residual():
    beats = _beats()
    result = phase_profile(
        beats, beats, exit_a=10.0, entry_b=10.0,
        duration=8.0, bpm=120.0, rate_a=1.0, rate_b=1.0,
    )
    assert result["status"] == "replayed"
    assert result["max_bounded_residual_ms"] == 0.0
    assert result["numeric_residual_below_25ms"]


def test_large_beat_offset_does_not_qualify_for_hypothetical_replacement():
    beats = _beats()
    beats[25] += 0.12
    result = hypothetical_incoming(
        beats, [], cue_start=10.0, cue_end=20.0,
        expected_period=0.5, hypothesis=_hyp(),
    )
    assert result["status"] == "movement_exceeds_limit"
    assert result["proposed_beats"] is None


def test_weak_onset_support_fails_closed():
    result = hypothetical_incoming(
        _beats(), [], cue_start=10, cue_end=20,
        expected_period=0.5, hypothesis=_hyp(score=0.4),
    )
    assert result["status"] == "insufficient_independent_onset_support"


def test_downbeat_disagreement_rejects_hypothetical_grid():
    beats = _beats()
    beats[25] += 0.029
    result = hypothetical_incoming(
        beats, [beats[25]], cue_start=10, cue_end=20,
        expected_period=0.5, hypothesis=_hyp(),
    )
    assert result["status"] in ("downbeat_anchor_disagreement", "movement_exceeds_limit")


def test_plan_id_mismatch_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "one"}, {"plan_id": "two"}, tmp_path)


def test_no_rejected_candidate_is_skipped(tmp_path):
    result = diagnose(
        {"plan_id": "same", "transitions": [{}]},
        {"plan_id": "same", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
