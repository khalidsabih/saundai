"""Unconditioned acoustic peak extraction cannot use a beat grid as an oracle."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-unconditioned-transient-localization.py"
    spec = importlib.util.spec_from_file_location("m1_unconditioned_events", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_audio(*, seconds=10, sample_rate=16000, phase=.0):
    audio = np.zeros(seconds * sample_rate, dtype=float)
    duration = np.arange(round(.05 * sample_rate)) / sample_rate
    signal = np.sin(2 * np.pi * 95 * duration) * np.exp(-duration * 45)
    for index in range(1, 18):
        start = round((index * .5 + phase) * sample_rate)
        audio[start:start + len(signal)] += signal
    return audio, sample_rate


def test_global_low_band_peak_detection_requires_no_grid_input():
    module = load()
    audio, sr = synthetic_audio()
    found = module.locate_events(audio, sr, band="low")
    assert len(found) >= 10
    assert all("time_seconds" in event for event in found)
    assert all(0 <= event["time_seconds"] < len(audio) / sr for event in found)
    assert all(a["time_seconds"] < b["time_seconds"] for a, b in zip(found, found[1:]))


def test_silent_input_abstains():
    module = load()
    audio = np.zeros(16000 * 8, dtype=float)
    assert module.locate_events(audio, 16000, band="high") == []
    rows = module.compare_events(np.arange(20) * .5, [])
    assert rows[0]["insufficient_event_evidence"] is True
    assert rows[0]["median_absolute_beat_error_ms"] is None


def test_comparison_does_not_change_grid_or_event_timestamps():
    module = load()
    grid = list(np.arange(20) * .5)
    events = [{"time_seconds": float(t)} for t in np.arange(20) * .5]
    original_grid, original_events = grid[:], [dict(e) for e in events]
    report = module.compare_events(grid, events)
    row = report[0]
    assert row["beat_within_tolerance_count"] == 19
    assert row["midpoint_within_tolerance_count"] == 0
    assert row["shared_event_count"] == 0
    assert grid == original_grid
    assert events == original_events


def test_midpoint_grid_proximity_and_duplicate_event_accounting():
    module = load()
    grid = list(np.arange(20) * .5)
    events = [{"time_seconds": float(t)} for t in np.arange(19) * .5 + .25]
    row = module.compare_events(grid, events)[0]
    assert row["midpoint_within_tolerance_count"] == 19
    assert row["beat_within_tolerance_count"] == 0
    assert row["unique_midpoint_events_matched"] == 19


def test_invalid_inputs_fail_closed():
    module = load()
    audio, sr = synthetic_audio()
    with pytest.raises(ValueError, match="frequency band"):
        module.locate_events(audio, sr, band="other")
    with pytest.raises(ValueError, match="beat grid"):
        module.compare_events([0.5, 0.5] * 10, [])
    with pytest.raises(ValueError, match="nonfinite"):
        module.compare_events(np.arange(20) * .5, [{"time_seconds": float("nan")}])
