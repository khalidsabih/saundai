from __future__ import annotations

from djai.intelligence.tempo_arbitration import arbitrate_tempo, tempo_candidates


def test_candidate_search_includes_half_and_double_time() -> None:
    candidates = tempo_candidates(151.9991, 78.9474)
    assert 75.9996 in candidates
    assert 78.9474 in candidates
    assert 151.9991 in candidates
    assert 157.8948 in candidates


def test_abstains_without_corroboration() -> None:
    result = arbitrate_tempo(151.9991, 78.9474)
    assert result.decision == "abstain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_requires_three_strong_and_consistent_onset_windows() -> None:
    assert arbitrate_tempo(
        152, 78, ((78, .9), (78, .9))
    ).decision == "abstain"
    assert arbitrate_tempo(
        152, 78, ((78, .9), (78, .1), (78, .9))
    ).decision == "abstain"
    assert arbitrate_tempo(
        152, 78, ((78, .9), (90, .9), (78, .9))
    ).decision == "abstain"
    assert arbitrate_tempo(
        152, 78, ((152, .9), (152, .9), (152, .9))
    ).decision == "abstain"


def test_review_candidate_is_never_live_adoption() -> None:
    result = arbitrate_tempo(
        152, 78.9474, ((78.7, .8), (78.9, .7), (79.0, .8))
    )
    assert result.decision == "review_candidate"
    assert result.suggested_bpm == 78.9474
    assert result.changes_dj_decisions is False


def test_preserve_deterministic_when_supported() -> None:
    result = arbitrate_tempo(
        123.0469, 125, ((123.0, .8), (123.1, .8), (123.0, .8))
    )
    assert result.decision == "abstain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_retain_when_all_support_deterministic_tempo() -> None:
    result = arbitrate_tempo(
        123.0469, 123.15, ((123.1, .8), (123.0, .8), (123.2, .8))
    )
    assert result.decision == "retain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_missing_nan_and_invalid_inputs_fail_closed() -> None:
    for value in (None, 0.0, float("nan"), float("inf")):
        assert arbitrate_tempo(value, 123.0, ((123, 1),) * 3).decision == "abstain"
        assert arbitrate_tempo(123.0, value, ((123, 1),) * 3).decision == "abstain"
    assert arbitrate_tempo(123, 123, ((float("nan"), 1),) * 3).decision == "abstain"
