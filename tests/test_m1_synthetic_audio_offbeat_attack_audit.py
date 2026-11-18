"""Independent high-band offbeat onset evidence remains synthetic-only."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-synthetic-audio-offbeat-attack-audit.py"
    spec = importlib.util.spec_from_file_location("m1_audio_offbeat_attacks", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hat_audio(*, bpm=120, seconds=8, sample_rate=22050, missing=()):
    data = np.zeros(int(seconds * sample_rate), dtype=np.float64)
    length = int(sample_rate * .05)
    rng = np.random.default_rng(31)
    noise = rng.normal(size=length)
    t = np.arange(length, dtype=np.float64) / sample_rate
    attack = np.concatenate(([0.0], np.diff(noise))) * np.exp(-t * 52)
    period = 60.0 / bpm
    for i in range(int(seconds / period)):
        if i in missing:
            continue
        position = int(round((i + .5) * period * sample_rate))
        data[position:position + length] += attack[:max(0, min(length, len(data)-position))]
    return data, sample_rate


def test_measures_high_band_offbeat_attacks_in_audio():
    module = load()
    samples, rate = hat_audio()
    events = module.detect_offbeat_transients(samples, rate, 120)
    measured = [e for e in events if e["status"] == "measured"]
    assert len(measured) >= 12
    assert max(abs(e["offset_from_synthetic_halfbeat_ms"]) for e in measured) < 60


def test_missing_high_frequency_attack_abstains():
    module = load()
    samples, rate = hat_audio(missing=(5,))
    events = module.detect_offbeat_transients(samples, rate, 120)
    assert events[5]["status"] != "measured"


def test_midpoint_grid_compares_to_measured_attacks_without_modification():
    module = load()
    beats = [.25 + .5 * i for i in range(12)]
    events = [{"status": "measured", "attack_seconds": t} for t in beats]
    original = list(beats)
    result = module.regional_hat_alignment(
        beats, events, (("middle", None, None),)
    )[0]
    assert result["within_80ms_count"] == 12
    assert result["median_absolute_attack_error_ms"] == 0.0
    assert original == beats


def test_missing_audio_evidence_is_not_fabricated():
    module = load()
    result = module.regional_hat_alignment(
        [0.5, 1.0], [{"status": "insufficient_high_frequency_attack"}],
        (("all", None, None),)
    )[0]
    assert result["comparable_count"] == 0
    assert result["within_80ms_count"] == 0
    assert result["median_absolute_attack_error_ms"] is None


def test_invalid_grid_is_rejected():
    module = load()
    with pytest.raises(ValueError, match="grid"):
        module.regional_hat_alignment([0.5, 0.5], [], (("all", None, None),))
