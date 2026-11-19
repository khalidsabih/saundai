"""Synthetic, no-audio regression tests for source-isolated phase proxy."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m1-rendered-phase-acoustic-check.py"


def load():
    spec = importlib.util.spec_from_file_location("m1_acoustic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_equal_period_pulses_have_zero_relative_drift():
    module = load()
    a = np.arange(0.5, 14.9, 0.5)
    b = a + 0.065
    result = module.compare_pulse_tracks(a.tolist(), b.tolist(), 15.0)
    assert result["status"] == "acoustic_proxy_measured"
    assert result["proxy_within_25ms"] is True
    assert result["max_absolute_relative_pulse_drift_ms"] < 0.01
    assert result["initial_relative_offset_ms"] == pytest.approx(65.0)


def test_clear_linear_drift_is_reported_not_certified():
    module = load()
    a = np.arange(0.5, 14.9, 0.5)
    b = a + 0.02 + 0.004 * a
    result = module.compare_pulse_tracks(a.tolist(), b.tolist(), 15.0)
    assert result["status"] == "acoustic_proxy_measured"
    assert result["proxy_within_25ms"] is False
    assert result["max_absolute_relative_pulse_drift_ms"] > 45.0


def test_missing_pulses_are_inconclusive():
    module = load()
    result = module.compare_pulse_tracks(
        [0.5, 1.0, 1.5], [0.55, 1.05, 1.55], 15.0
    )
    assert result["status"] == "inconclusive"
    assert result["proxy_within_25ms"] is None


def test_nearly_silent_audio_has_no_pulses():
    module = load()
    audio = np.zeros((2, 44_100 * 3), dtype=np.float32)
    assert module.acoustic_pulses(audio, 44_100) == []


def test_nonfinite_or_nonmonotonic_pulses_raise():
    module = load()
    with pytest.raises(ValueError):
        module.compare_pulse_tracks([0.5, float("nan")], [0.5, 1.0], 10)
    with pytest.raises(ValueError):
        module.compare_pulse_tracks([1.0, 0.5], [0.5, 1.0], 10)
