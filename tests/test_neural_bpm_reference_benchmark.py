from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from djai.intelligence.tempo_benchmark import compare_bpm, compare_estimators, evaluate, summarize, deterministic_bpm


def test_reference_bpm_errors_are_not_octave_folded() -> None:
    dizzy = compare_bpm(78, 78.9474)
    nova = compare_bpm(123, 125)
    assert dizzy["absolute_error_bpm"] == 0.9474
    assert nova["absolute_error_bpm"] == 2.0
    assert nova["absolute_percentage_error"] == 1.626
    assert compare_bpm(78, 156)["absolute_error_bpm"] == 78.0
    with pytest.raises(ValueError):
        compare_bpm(0, 100)


def test_summary_distinguishes_missing_and_actual_predictions() -> None:
    rows = [
        {"status": "ok", **compare_bpm(78, 78.9474)},
        {"status": "ok", **compare_bpm(123, 125.0)},
        {"status": "missing", "filename": "missing.mp3"},
    ]
    report = summarize(rows)
    assert report["reference_count"] == 3
    assert report["evaluated_count"] == 2
    assert report["unavailable_count"] == 1
    assert report["mean_absolute_error_bpm"] == 1.4737
    assert report["within_1_bpm_count"] == 1
    assert report["within_2_bpm_count"] == 2
    assert report["calibration_proven"] is False


def test_benchmark_is_observation_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "Nova.wav"
    audio.write_bytes(b"synthetic placeholder - inference mocked")
    calls = []

    def fake_predict(path):
        calls.append(path)
        return SimpleNamespace(
            available=True, bpm=125.0, confidence=0.34,
            tempo_confidence=0.4792, used=False, reason="grid retained",
        )

    monkeypatch.setattr("djai.intelligence.neural_ears.predict_beats", fake_predict)
    monkeypatch.setattr("djai.intelligence.tempo_benchmark.deterministic_bpm", lambda path: 123.0)
    result = evaluate(tmp_path, [
        {"filename": "Nova.wav", "reference_bpm": 123},
        {"filename": "Absent.wav", "reference_bpm": 120},
    ])
    assert calls == [audio]
    assert result["mode"] == "observation_only"
    assert result["rows"][0]["absolute_error_bpm"] == 2.0
    assert result["rows"][0]["beat_alignment_measured"] is False
    assert result["rows"][0]["downbeat_accuracy_measured"] is False
    assert result["rows"][1]["status"] == "missing"
    assert result["summary"]["evaluated_count"] == 1
    assert result["summary"]["paired_count"] == 1
    assert result["summary"]["deterministic_wins"] == 1
    assert result["rows"][0]["comparison"]["deterministic"]["absolute_error_bpm"] == 0.0

def test_estimator_comparison_counts_true_error_not_model_agreement() -> None:
    outcome = compare_estimators(123, 123, 125)
    assert outcome["winner"] == "deterministic"
    assert outcome["deterministic"]["absolute_error_bpm"] == 0
    assert outcome["neural"]["absolute_error_bpm"] == 2
    assert compare_estimators(78, 79, 78)["winner"] == "neural"
    assert compare_estimators(123, 125, None)["winner"] == "unavailable"
    assert compare_estimators(123, 122.995, 123.0)["winner"] == "tie_within_0_01_bpm"


def test_paired_summary_excludes_incomplete_rows() -> None:
    det = {"status": "ok", **compare_bpm(123, 125),
           "comparison": compare_estimators(123, 122, 125)}
    partial = {"status": "partial",
               "comparison": compare_estimators(78, 78, None)}
    summary = summarize([det, partial])
    assert summary["paired_count"] == 1
    assert summary["paired_mean_absolute_error_deterministic_bpm"] == 1.0
    assert summary["paired_mean_absolute_error_neural_bpm"] == 2.0
    assert summary["neural_wins"] == 0
    assert summary["unavailable_count"] == 1


def test_deterministic_bpm_mirrors_saund_initial_beat_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import djai.audio as audio
    import librosa

    sample_rate = 22050
    y = np.zeros((1, sample_rate * 5), dtype=np.float32)
    received = {}

    def fake_strength(**kwargs):
        received["strength"] = kwargs
        return np.ones(128, dtype=float)

    def fake_beat_track(**kwargs):
        received["beat_track"] = kwargs
        return np.asarray([123.0]), np.array([1, 2, 3])

    monkeypatch.setattr(audio, "load_audio", lambda path, sample_rate, mono: (y, sample_rate))
    monkeypatch.setattr(librosa.onset, "onset_strength", fake_strength)
    monkeypatch.setattr(librosa.beat, "beat_track", fake_beat_track)

    assert deterministic_bpm(Path("dummy.wav")) == 123.0
    assert received["strength"]["aggregate"] is np.median
    assert received["beat_track"]["units"] == "frames"
    assert received["beat_track"]["sparse"] is True
