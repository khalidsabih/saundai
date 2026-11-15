from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from djai.intelligence.tempo_benchmark import compare_bpm, evaluate, summarize


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
