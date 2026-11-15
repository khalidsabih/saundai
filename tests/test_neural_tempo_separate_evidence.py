from __future__ import annotations

import numpy as np
import pytest

from djai.intelligence.neural_ears import (
    NeuralBeatResult,
    _downbeat_confidence,
    _tempo_confidence,
    tempo_only_adoption,
)


def _result(*, bpm: float = 78.95, stability: float = 0.95, available: bool = True) -> NeuralBeatResult:
    beats = tuple(float(x * 60 / bpm) for x in range(80))
    return NeuralBeatResult(
        backend="beat_this",
        model="test",
        available=available,
        beats_seconds=beats,
        downbeats_seconds=beats[::4],
        bpm=bpm,
        confidence=0.34,
        agreement=0.1,
        authority=0.0,
        used=False,
        reason="grid retained",
        tempo_confidence=stability,
        beat_alignment_confidence=0.1,
        downbeat_confidence=0.95,
    )


def test_neural_tempo_is_measured_separately_from_beat_alignment() -> None:
    beats = [index * 60 / 78.95 for index in range(80)]
    assert _tempo_confidence(beats) > 0.95
    assert _downbeat_confidence(beats, beats[::4]) > 0.9
    assert _tempo_confidence([0, 1, 2]) == 0.0
    assert _downbeat_confidence(beats, [0.13, 3.17]) < 0.5


def test_tempo_only_defaults_to_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DJAI_NEURAL_TEMPO_ONLY", raising=False)
    adopted, reason = tempo_only_adoption(_result(), 78.0)
    assert not adopted
    assert "disabled" in reason


def test_tempo_only_requires_stability_independent_agreement_and_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DJAI_NEURAL_TEMPO_ONLY", "1")
    monkeypatch.setattr(
        "djai.intelligence.neural_ears.load_model_registry",
        lambda: type("Registry", (), {"authority": lambda self, name, confidence: 0.2})(),
    )
    assert tempo_only_adoption(_result(), 78.0)[0] is True
    assert tempo_only_adoption(_result(stability=0.4), 78.0)[0] is False
    assert tempo_only_adoption(_result(), 89.0)[0] is False
    assert tempo_only_adoption(_result(available=False), 78.0)[0] is False
    assert tempo_only_adoption(_result(), float("nan"))[0] is False
    assert tempo_only_adoption(_result(), 0.0)[0] is False
    monkeypatch.setattr(
        "djai.intelligence.neural_ears.load_model_registry",
        lambda: type("Registry", (), {"authority": lambda self, name, confidence: 0.0})(),
    )
    assert tempo_only_adoption(_result(), 78.0)[0] is False


def test_legacy_result_serialization_keeps_existing_fields() -> None:
    result = _result()
    payload = result.to_dict()
    assert payload["used"] is False
    assert payload["bpm"] == 78.95
    assert payload["tempo_confidence"] == 0.95
    assert payload["beat_alignment_confidence"] == 0.1
    assert payload["downbeat_confidence"] == 0.95
