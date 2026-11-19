"""Fast-track M1: existing Beat This adapter flag, fallback and rollback contracts.

Tests stub external inference and registry authority. Never load checkpoints,
download models, analyze user music or change production model selection.
"""
from __future__ import annotations

import sys
import types

from djai.intelligence import neural_ears


BEATS = tuple(i * .5 for i in range(48))
DOWNS = BEATS[::4]


def setup_model(monkeypatch, tmp_path, *, outputs=(BEATS, DOWNS), failure=None,
                authority=.34):
    checkpoint = tmp_path / "final0.ckpt"
    checkpoint.write_bytes(b"test-only fake checkpoint, not pretrained weights")
    monkeypatch.setenv("DJAI_BEAT_THIS_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "auto")
    monkeypatch.setenv("DJAI_MODEL_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("DJAI_BEAT_THIS_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("DJAI_BEAT_THIS_DEVICE", "cpu")
    monkeypatch.setenv("DJAI_NEURAL_BEAT_USE_THRESHOLD", "0.70")
    monkeypatch.delenv("DJAI_NEURAL_DOWNBEAT_INTEGRITY", raising=False)

    events = []
    package = types.ModuleType("beat_this")
    package.__path__ = []
    inference = types.ModuleType("beat_this.inference")

    class FakeFile2Beats:
        def __init__(self, checkpoint_path, device, dbn):
            events.append(("constructed", checkpoint_path, device, dbn))

        def __call__(self, path):
            events.append(("called", path))
            if failure is not None:
                raise failure
            return outputs

    inference.File2Beats = FakeFile2Beats
    monkeypatch.setitem(sys.modules, "beat_this", package)
    monkeypatch.setitem(sys.modules, "beat_this.inference", inference)
    monkeypatch.setattr(
        neural_ears, "load_model_registry",
        lambda: types.SimpleNamespace(authority=lambda name, confidence: authority),
    )
    return events


def test_flag_off_prevents_even_constructing_model(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path)
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS)
    assert not result.available and not result.used
    assert "disabled" in result.reason
    assert events == []


def test_missing_explicit_checkpoint_fails_closed(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path)
    monkeypatch.setenv("DJAI_BEAT_THIS_CHECKPOINT", str(tmp_path / "missing.ckpt"))
    result = neural_ears.predict_beats(tmp_path / "a.wav")
    assert not result.used and not result.available
    assert "missing" in result.reason
    assert events == []


def test_checkpoint_load_or_model_runtime_failure_falls_back(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path, failure=RuntimeError("simulated failure"))
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS)
    assert not result.used and not result.available
    assert "inference unavailable" in result.reason
    assert len(events) == 2


def test_shadow_model_authority_cannot_replace_grid(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path, authority=0.0)
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS,
                                       fallback_downbeats=DOWNS)
    assert result.available and not result.used
    assert result.beats_seconds == BEATS
    assert result.authority == 0.0
    assert len(events) == 2


def test_advisory_can_propose_valid_beat_grid_under_opt_in_guard(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path)
    monkeypatch.setenv("DJAI_NEURAL_DOWNBEAT_INTEGRITY", "1")
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS,
                                       fallback_downbeats=DOWNS)
    assert result.available and result.used
    assert result.bpm == 120.0
    assert result.beats_seconds == BEATS
    assert len(events) == 2


def test_dense_downbeats_rejected_even_if_beat_grid_is_accurate(tmp_path, monkeypatch):
    events = setup_model(monkeypatch, tmp_path, outputs=(BEATS, BEATS[::2]))
    monkeypatch.setenv("DJAI_NEURAL_DOWNBEAT_INTEGRITY", "1")
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS)
    assert result.available and not result.used
    assert "too dense" in result.reason
    assert len(events) == 2


def test_no_authority_never_overridden_by_confidence(tmp_path, monkeypatch):
    setup_model(monkeypatch, tmp_path, authority=0.0)
    monkeypatch.setenv("DJAI_NEURAL_BEAT_USE_THRESHOLD", "0.50")
    result = neural_ears.predict_beats(tmp_path / "a.wav", fallback_beats=BEATS)
    assert not result.used and result.authority == 0.0
