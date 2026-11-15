from __future__ import annotations

import pytest

from djai.intelligence.tempo_arbitration import arbitrate_tempo, tempo_candidates, octave_family_agreement, onset_window_diagnostics, evidence_coverage


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

def _diagnostics(values: tuple[tuple[float | None, float | None], ...]) -> tuple[dict, ...]:
    return tuple(
        {"index": i, "status": "ok" if bpm is not None else "no_periodicity_peak",
         "bpm": bpm, "strength": strength}
        for i, (bpm, strength) in enumerate(values, 1)
    )


def test_coverage_separates_missing_weak_and_conflicting_evidence() -> None:
    windows = _diagnostics((
        (78, .8), (156, .9), (78.2, .2), (None, None),
        (90, .8), (78, .5), (157, .6),
    ))
    coverage = evidence_coverage(windows, 78)
    assert coverage["total_windows"] == 7
    assert coverage["strong_windows"] == 5
    assert coverage["weak_windows"] == 1
    assert coverage["missing_windows"] == 1
    assert coverage["neural_family_supported_windows"] == 4
    assert coverage["conflicting_strong_windows"] == 1
    assert coverage["dominant_family_supported_windows"] == 4
    assert coverage["strong_coverage_fraction"] == pytest.approx(5 / 7, abs=.0001)


def test_five_strong_sections_allow_observational_retention() -> None:
    windows = _diagnostics((
        (123, .8), (123.1, .7), (None, None),
        (123.1, .7), (123, .8), (123.2, .6), (123.0, .9),
    ))
    observed = tuple((w["bpm"], w["strength"]) for w in windows if w["status"] == "ok")
    result = arbitrate_tempo(123.05, 123.1, observed, window_diagnostics=windows)
    assert result.decision == "retain"
    assert result.coverage["strong_windows"] == 6
    assert result.coverage["missing_windows"] == 1
    assert result.changes_dj_decisions is False


def test_strong_conflict_blocks_arbitration_even_with_majority() -> None:
    windows = _diagnostics((
        (78, .8), (78, .8), (156, .9), (78, .9),
        (90, .7), (78, .8), (78, .8),
    ))
    observed = tuple((w["bpm"], w["strength"]) for w in windows)
    result = arbitrate_tempo(152, 78, observed, window_diagnostics=windows)
    assert result.decision == "abstain"
    assert result.coverage["conflicting_strong_windows"] == 1
    assert result.reason == "contradictory strong onset evidence"


def test_missing_coverage_does_not_count_as_support() -> None:
    windows = _diagnostics(((123, .8), (None, None), (None, None),
                            (123, .8), (None, None), (123, .8), (None, None)))
    observed = tuple((w["bpm"], w["strength"]) for w in windows if w["status"] == "ok")
    result = arbitrate_tempo(123, 123, observed, window_diagnostics=windows)
    assert result.decision == "abstain"
    assert result.coverage["strong_coverage_fraction"] < .60

def test_long_track_samples_seven_nonoverlapping_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import librosa
    import numpy as np
    import djai.audio as audio
    import djai.intelligence.tempo_arbitration as arb

    monkeypatch.setattr(audio, "load_audio", lambda path, sample_rate, mono: (
        np.zeros((1, 22050 * 5), dtype=np.float32), 22050
    ))
    monkeypatch.setattr(librosa.onset, "onset_strength",
                        lambda **kwargs: np.zeros(11000))
    monkeypatch.setattr(arb, "_acf_tempo", lambda *args, **kwargs: (123.0, .8))
    result = onset_window_diagnostics(__import__("pathlib").Path("long.wav"))
    assert result["status"] == "complete"
    assert len(result["windows"]) == 7
    starts = [w["start_seconds"] for w in result["windows"]]
    assert starts == sorted(set(starts))
    assert all(b - a >= 24.0 - 0.1 for a, b in zip(starts, starts[1:]))

def test_nova_audio_family_support_is_not_neural_conflict() -> None:
    # Actual seven-window evidence: 61.5168 is half of the ~123 BPM pulse.
    windows = _diagnostics((
        (None, None), (123.0073, .9943), (123.0158, .8116),
        (160.2976, .1612), (61.5168, .9417),
        (123.0074, .855), (123.0282, .7983),
    ))
    coverage = evidence_coverage(windows, 125.0, deterministic_bpm=123.046875)
    assert coverage["strong_windows"] == 5
    assert coverage["dominant_family_supported_windows"] == 5
    assert coverage["conflicting_strong_windows"] == 0
    assert coverage["family_coverage_fraction"] == pytest.approx(5 / 7, abs=.0001)
    assert coverage["deterministic_supports_dominant_family"] is True
    assert coverage["neural_supports_dominant_family"] is False
    assert coverage["detector_disagreement"] is True
    observed = tuple((w["bpm"], w["strength"]) for w in windows if w["status"] == "ok")
    result = arbitrate_tempo(123.046875, 125, observed, window_diagnostics=windows)
    assert result.decision == "retain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_dizzy_still_abstains_for_insufficient_strong_coverage() -> None:
    windows = _diagnostics((
        (78.0343, .6187), (78.4795, .2717), (153.5406, .3848),
        (156.7579, .1897), (77.971, .3448), (64.1051, .1606),
        (78.3973, .2824),
    ))
    observed = tuple((w["bpm"], w["strength"]) for w in windows)
    result = arbitrate_tempo(151.9991, 78.9474, observed, window_diagnostics=windows)
    assert result.decision == "abstain"
    assert result.reason == "insufficient strong evidence coverage across song"
    assert result.coverage["strong_windows"] == 2
    assert result.changes_dj_decisions is False


def test_two_to_three_subdivision_is_diagnostic_not_octave_corroboration() -> None:
    # Cola: four ~122 BPM windows and three ~81 BPM windows, all strong.
    windows = _diagnostics((
        (122.4053, .8175), (122.4306, .5696), (122.3542, .5938),
        (81.1261, .8377), (81.1236, .5431), (81.1039, .8371),
        (122.4584, .8759),
    ))
    coverage = evidence_coverage(windows, 120.0, deterministic_bpm=123.0469)
    assert coverage["conflicting_strong_windows"] == 3
    assert coverage["possible_two_to_three_subdivision_windows"] == 3
    assert coverage["possible_two_to_three_subdivision_indices"] == [4, 5, 6]
    assert coverage["subdivision_aliases_are_corroboration"] is False
    result = arbitrate_tempo(
        123.0469, 120.0, tuple((w["bpm"], w["strength"]) for w in windows),
        window_diagnostics=windows,
    )
    assert result.decision == "abstain"
    assert result.suggested_bpm is None


def test_seven_consistent_windows_explicitly_retain_existing_bpm() -> None:
    # Lane 8 Fingerprint: pulse near 123.6, reference reported as ~124.
    windows = _diagnostics((
        (123.6466, .945), (123.669, .8729), (123.5729, .9363),
        (123.664, .8233), (123.5805, .8576), (123.6473, .8331),
        (123.629, .7873),
    ))
    result = arbitrate_tempo(
        123.046875, 125.0,
        tuple((w["bpm"], w["strength"]) for w in windows),
        window_diagnostics=windows,
    )
    assert result.decision == "retain"
    assert result.suggested_bpm is None
    assert "existing deterministic BPM" in result.reason
    assert result.changes_dj_decisions is False


def test_octave_only_agreement_does_not_falsely_retain_wrong_metrical_level() -> None:
    windows = _diagnostics(((78.0, .9),) * 7)
    result = arbitrate_tempo(
        156.0, 78.0, ((78.0, .9),) * 7,
        window_diagnostics=windows,
    )
    assert result.decision != "retain"
    assert result.suggested_bpm is None
    assert result.changes_dj_decisions is False


def test_strong_subdivision_conflict_still_abstains_even_if_baseline_close() -> None:
    windows = _diagnostics(((123.6, .8), (82.4, .9), (123.6, .8),
                            (123.6, .8), (82.4, .9), (123.6, .8),
                            (123.6, .8)))
    result = arbitrate_tempo(
        123.05, 125.0,
        tuple((w["bpm"], w["strength"]) for w in windows),
        window_diagnostics=windows,
    )
    assert result.decision == "abstain"
    assert result.coverage["possible_two_to_three_subdivision_windows"] == 2
