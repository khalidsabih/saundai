"""Generator-window-guided audio kick detection remains independent of beat tracker."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-synthetic-audio-kick-audit.py"
    spec = importlib.util.spec_from_file_location("m1_synthetic_kicks", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pulse_audio(*, bpm=120, seconds=8, sr=16000, missing=()):
    audio = np.zeros(int(seconds * sr), dtype=float)
    period = 60.0 / bpm
    t = np.arange(int(.09 * sr)) / sr
    pulse = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 26)
    for index in range(int(seconds / period)):
        if index in missing:
            continue
        begin = int(round(index * period * sr))
        audio[begin:begin + len(pulse)] += pulse
    return audio, sr


def test_measures_waveform_pulses_not_track_analysis():
    mod = load()
    audio, sr = pulse_audio()
    events = mod.extract_kick_attacks(audio, sr, 120)
    assert len(events) >= 15
    measured = [x for x in events if x["status"] == "measured"]
    assert len(measured) >= 13
    # The independent filter/attack envelope incurs a small physical latency.
    assert max(abs(x["attack_minus_generator_ms"]) for x in measured) < 85
    report = mod.align_grid_to_attacks([.5 * i for i in range(15)], events)
    assert report["status"] == "measured"
    assert report["median_absolute_nearest_attack_error_ms"] < 80


def test_missing_kick_is_not_filled_from_oracle_schedule():
    mod = load()
    audio, sr = pulse_audio(missing=(5,))
    events = mod.extract_kick_attacks(audio, sr, 120)
    assert events[5]["status"] != "measured"


def test_grid_far_from_audio_attack_is_not_labeled_matched():
    mod = load()
    attacks = [
        {"index": i, "status": "measured", "audio_attack_seconds": i * .5}
        for i in range(20)
    ]
    result = mod.align_grid_to_attacks([i * .5 + .2 for i in range(17)], attacks)
    assert result["beats_outside_audio_attack_tolerance"] == 17
    assert result["median_absolute_nearest_attack_error_ms"] > 80


def test_missing_audio_evidence_abstains():
    mod = load()
    events = [{"index": i, "status": "insufficient_low_frequency_attack"}
              for i in range(8)]
    result = mod.align_grid_to_attacks([0.5, 1.0], events)
    assert result["status"] == "insufficient_independent_audio_evidence"
    assert result["compared_beats"] == 0


def test_rejects_invalid_grid_and_short_audio():
    mod = load()
    audio, sr = pulse_audio()
    with pytest.raises(ValueError, match="grid"):
        mod.align_grid_to_attacks([0.5, 0.5], [])
    with pytest.raises(ValueError, match="mono audio"):
        mod.extract_kick_attacks(audio[:100], sr, 120)


def test_region_boundaries_and_mismatch_counts():
    mod = load()
    grid = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
    events = [
        {"index": i, "status": "measured", "audio_attack_seconds": stamp}
        for i, stamp in enumerate([0.5, 1.0, 2.2, 2.7, 3.2, 4.0])
    ]
    regions = (("early", None, 2.5), ("late", 2.5, None))
    rows = mod.regional_alignment(grid, events, regions, tolerance_ms=80)
    assert [r["beat_count"] for r in rows] == [3, 3]
    assert rows[0]["outside_tolerance_count"] == 1
    assert rows[1]["outside_tolerance_count"] == 2
    assert rows[0]["median_absolute_nearest_attack_error_ms"] == 0.0
    assert rows[1]["median_absolute_nearest_attack_error_ms"] == 200.0


def test_no_measured_kicks_is_not_inferred_from_generator_reference():
    mod = load()
    rows = mod.regional_alignment(
        [0.5, 1.0, 1.5],
        [{"index": 1, "status": "insufficient_low_frequency_attack"}],
        (("all", None, None),),
    )
    assert rows[0]["beat_count"] == 3
    assert rows[0]["measured_nearest_count"] == 0
    assert rows[0]["outside_tolerance_count"] == 3
    assert rows[0]["median_absolute_nearest_attack_error_ms"] is None
