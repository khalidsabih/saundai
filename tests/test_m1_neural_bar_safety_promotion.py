"""M1 integrated improvement: Beat This! pinned model + coherent bar selection.

No checkpoint downloads, network inference or production policy changes.
"""
from __future__ import annotations

import sys
import types

import pytest

from djai.intelligence import neural_ears


def beat_times(bpm=122.0, count=129):
    return tuple(round(i * 60.0 / bpm / .02) * .02 for i in range(count))


def fake_inference(monkeypatch, tmp_path, *, beats, downs, authority=.34):
    checkpoint = tmp_path / "final0.ckpt"
    checkpoint.write_bytes(b"synthetic fake model")
    monkeypatch.setenv("DJAI_BEAT_THIS_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "1")
    monkeypatch.setenv("DJAI_MODEL_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("DJAI_BEAT_THIS_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("DJAI_BEAT_THIS_DEVICE", "cpu")
    monkeypatch.setenv("DJAI_NEURAL_BEAT_USE_THRESHOLD", "0.70")
    monkeypatch.delenv("DJAI_NEURAL_DOWNBEAT_INTEGRITY", raising=False)
    monkeypatch.delenv("DJAI_BEAT_THIS_SHA256", raising=False)
    monkeypatch.delenv("DJAI_NEURAL_BAR_SAFETY", raising=False)

    calls = []
    package = types.ModuleType("beat_this")
    package.__path__ = []
    inference = types.ModuleType("beat_this.inference")

    class Fake:
        def __init__(self, checkpoint_path, device, dbn):
            calls.append(("model", checkpoint_path))
        def __call__(self, audio_path):
            calls.append(("inference", audio_path))
            return beats, downs

    inference.File2Beats = Fake
    monkeypatch.setitem(sys.modules, "beat_this", package)
    monkeypatch.setitem(sys.modules, "beat_this.inference", inference)
    monkeypatch.setattr(
        neural_ears,
        "load_model_registry",
        lambda: types.SimpleNamespace(authority=lambda model, confidence: authority),
    )
    return calls, checkpoint


def test_long_span_bpm_recovers_122_from_frame_quantized_beats():
    beats = beat_times()
    assert abs(neural_ears._estimate_bpm(beats) - 122.0) < .75
    assert neural_ears._estimate_bpm([0.0, 0.5, 0.4]) == 0


@pytest.mark.parametrize("meter", [3, 4, 6])
def test_regular_bar_spacing_is_eligible_not_claimed_correct(meter):
    beats = beat_times(count=145)
    ok, reason = neural_ears.downbeat_bar_integrity(beats, beats[::meter])
    assert ok, reason


def test_surplus_downbeat_labels_are_vetoed_even_if_median_spacing_looks_plausible():
    beats = beat_times()
    downs = [beats[i] for i in range(0, 129, 4)]
    downs.insert(5, beats[17])
    assert neural_ears.downbeat_spacing_integrity(beats, downs)[0]
    ok, reason = neural_ears.downbeat_bar_integrity(beats, downs)
    assert not ok and "irregular" in reason


def test_bar_safety_veto_prevents_used_grid_with_extra_downbeat(tmp_path, monkeypatch):
    beats = beat_times()
    downs = [beats[i] for i in range(0, 129, 4)]
    downs.insert(5, beats[17])
    calls, _ = fake_inference(monkeypatch, tmp_path, beats=beats, downs=downs)
    result = neural_ears.predict_beats(
        tmp_path / "demo.wav", fallback_beats=beats, fallback_downbeats=downs
    )
    assert result.available and not result.used
    assert len(result.beats_seconds) == len(beats)
    assert "bar-grid safety veto" in result.reason
    assert len(calls) == 2


def test_regular_downbeats_can_still_be_selected(tmp_path, monkeypatch):
    beats = beat_times()
    calls, _ = fake_inference(monkeypatch, tmp_path, beats=beats, downs=beats[::4])
    result = neural_ears.predict_beats(
        tmp_path / "demo.wav", fallback_beats=beats, fallback_downbeats=beats[::4]
    )
    assert result.used
    assert abs(result.bpm - 122.0) < .75
    assert len(calls) == 2


def test_bad_checkpoint_sha256_fails_before_model_inference(tmp_path, monkeypatch):
    beats = beat_times()
    calls, _ = fake_inference(monkeypatch, tmp_path, beats=beats, downs=beats[::4])
    monkeypatch.setenv("DJAI_BEAT_THIS_SHA256", "0" * 64)
    result = neural_ears.predict_beats(tmp_path / "demo.wav")
    assert not result.available and not result.used
    assert "SHA256 mismatch" in result.reason
    assert calls == []


def test_correct_checkpoint_sha256_allows_inference(tmp_path, monkeypatch):
    import hashlib
    beats = beat_times()
    calls, checkpoint = fake_inference(
        monkeypatch, tmp_path, beats=beats, downs=beats[::4]
    )
    monkeypatch.setenv(
        "DJAI_BEAT_THIS_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    result = neural_ears.predict_beats(
        tmp_path / "demo.wav", fallback_beats=beats
    )
    assert result.available and result.used
    assert len(calls) == 2


def test_bad_sha256_setting_fails_closed(tmp_path, monkeypatch):
    beats = beat_times()
    calls, _ = fake_inference(monkeypatch, tmp_path, beats=beats, downs=beats[::4])
    monkeypatch.setenv("DJAI_BEAT_THIS_SHA256", "bad")
    result = neural_ears.predict_beats(tmp_path / "demo.wav")
    assert not result.available and not result.used
    assert "invalid DJAI_BEAT_THIS_SHA256" in result.reason
    assert calls == []
