from __future__ import annotations

import pytest

from djai.intelligence.tempo_arbitration import arbitrate_tempo, tempo_candidates, octave_family_agreement, onset_window_diagnostics


def test_candidate_search_includes_half_and_double_time() -> None:
    candidates = tempo_candidates(151.9991, 78.9474)
    assert any(value == pytest.approx(151.9991 / 2, abs=0.0001) for value in candidates)
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


def test_ambiguous_octave_family_never_selects_a_metrical_level() -> None:
    result = arbitrate_tempo(
        152, 78.9474, ((78.7, .8), (78.9, .7), (158.0, .8))
    )
    assert result.octave_family_agreement is True
    assert result.decision == "abstain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_unambiguous_candidate_remains_review_only() -> None:
    result = arbitrate_tempo(
        130, 100, ((100.0, .8), (100.1, .7), (99.9, .8))
    )
    assert result.decision == "review_candidate"
    assert result.suggested_bpm == 100.0
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

def test_octave_family_agreement_does_not_imply_strength() -> None:
    assert octave_family_agreement((78.071, 78.2229, 157.8877))
    assert not octave_family_agreement((78.0, 88.0, 157.0))


def test_duration_failure_is_diagnostic_not_a_fabricated_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import librosa
    import numpy as np
    import djai.audio as audio
    monkeypatch.setattr(audio, "load_audio", lambda path, sample_rate, mono: (
        np.zeros((1, 22050 * 5), dtype=np.float32), 22050
    ))
    result = onset_window_diagnostics(__import__("pathlib").Path("short.wav"))
    assert result["status"] == "insufficient_duration"
    assert result["windows"] == []
    assert result["duration_seconds"] < result["minimum_seconds"]


def test_no_periodicity_peak_reports_specific_window_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import librosa
    import numpy as np
    import djai.audio as audio
    monkeypatch.setattr(audio, "load_audio", lambda path, sample_rate, mono: (
        np.zeros((1, 22050 * 90), dtype=np.float32), 22050
    ))
    monkeypatch.setattr(librosa.onset, "onset_strength", lambda **kwargs: np.zeros(4000))
    result = onset_window_diagnostics(__import__("pathlib").Path("silent.wav"))
    assert result["status"] == "incomplete_periodicity"
    assert len(result["windows"]) == 3
    assert all(w["status"] == "no_periodicity_peak" for w in result["windows"])
