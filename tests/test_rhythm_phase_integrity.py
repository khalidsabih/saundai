"""Fail-closed research contracts for generator-independent beat-phase integrity."""
from __future__ import annotations

import numpy as np
import pytest

from djai.music.rhythm_phase_integrity import (
    classify_phase_window, inspect_waveform, phase_evidence_features,
    summarize_phase_integrity,
)


def fake_features(phase="kick", length=64, period=.5):
    times = np.arange(length, dtype=float) * period + .5
    if phase == "kick":
        low_beat, low_half, high_beat, high_half = 1., .01, .01, 1.
    elif phase == "offbeat":
        low_beat, low_half, high_beat, high_half = .01, 1., 1., .01
    elif phase == "ambiguous":
        low_beat, low_half, high_beat, high_half = 1., 1., 1., 1.
    elif phase == "weak":
        low_beat = low_half = high_beat = high_half = 0.
    else:
        raise ValueError("unknown phase")
    return {
        "times": times,
        "low_at_beat": np.full(length, low_beat),
        "low_at_midpoint": np.full(length, low_half),
        "high_at_beat": np.full(length, high_beat),
        "high_at_midpoint": np.full(length, high_half),
    }


def test_sustained_offbeat_requires_multiple_windows():
    result = summarize_phase_integrity(fake_features("offbeat"))
    assert result["decision"] == "sustained_offbeat_phase_inconsistency"
    assert len(result["sustained_offbeat_runs"]) == 1
    assert result["generator_reference_used"] is False
    assert result["beat_grid_modified"] is False
    assert result["authority"] == "shadow_only"


def test_ordinary_kick_grid_with_offbeat_hats_is_not_flagged():
    result = summarize_phase_integrity(fake_features("kick"))
    assert result["decision"] == "kick_aligned_with_support"
    assert result["sustained_offbeat_runs"] == []


@pytest.mark.parametrize("phase", ["ambiguous", "weak"])
def test_weak_or_mixed_bands_abstain(phase):
    result = summarize_phase_integrity(fake_features(phase))
    assert result["decision"] == "insufficient_evidence"
    assert result["sustained_offbeat_runs"] == []


def test_isolated_six_beat_phase_flip_cannot_trigger_sustained_detection():
    features = fake_features("kick")
    for key, alt in [
        ("low_at_beat", .01), ("low_at_midpoint", 1.),
        ("high_at_beat", 1.), ("high_at_midpoint", .01)
    ]:
        features[key][24:30] = alt
    result = summarize_phase_integrity(features)
    assert result["sustained_offbeat_runs"] == []


def test_unknown_meter_three_and_six_pulse_grid_is_not_an_automatic_veto():
    for meter in (3, 6):
        features = fake_features("kick", length=60)
        features["times"] += (meter - 3) * .1
        result = summarize_phase_integrity(features)
        assert result["decision"] == "kick_aligned_with_support"


def test_invalid_or_out_of_bounds_grid_fails_closed():
    audio = np.zeros(16000 * 10, dtype=float)
    with pytest.raises(ValueError, match="beat intervals"):
        phase_evidence_features(audio, 16000, np.arange(20) * .1)
    with pytest.raises(ValueError, match="bounds"):
        phase_evidence_features(audio, 16000, np.arange(20) * .5 + 1)


def test_silent_waveform_never_reports_false_phase():
    audio = np.zeros(16000 * 14, dtype=float)
    grid = np.arange(24, dtype=float) * .5 + .5
    report = inspect_waveform(audio, 16000, grid)
    assert report["decision"] == "insufficient_evidence"


def test_audio_with_real_low_kicks_and_offbeat_hats_remains_kick_aligned():
    sr = 16000
    seconds = 24
    audio = np.zeros(sr * seconds, dtype=float)
    generator = np.random.default_rng(99)
    kick_t = np.arange(int(.09 * sr)) / sr
    kick = np.sin(2 * np.pi * 80 * kick_t) * np.exp(-kick_t * 30)
    hat_t = np.arange(int(.04 * sr)) / sr
    hat = np.diff(np.r_[0, generator.normal(size=len(hat_t))]) * np.exp(-hat_t * 75)
    for i in range(47):
        a = round((i * .5 + .5) * sr)
        b = round((i * .5 + .75) * sr)
        if a + len(kick) <= len(audio):
            audio[a:a + len(kick)] += kick
        if b + len(hat) <= len(audio):
            audio[b:b + len(hat)] += .3 * hat
    beat_grid = np.arange(42, dtype=float) * .5 + .5
    result = inspect_waveform(audio, sr, beat_grid)
    assert result["decision"] == "kick_aligned_with_support"
    assert result["sustained_offbeat_runs"] == []


def test_audio_with_halfbeat_shifted_grid_reports_sustained_mismatch():
    sr = 16000
    seconds = 24
    audio = np.zeros(sr * seconds, dtype=float)
    generator = np.random.default_rng(91)
    kick_t = np.arange(int(.09 * sr)) / sr
    kick = np.sin(2 * np.pi * 80 * kick_t) * np.exp(-kick_t * 30)
    hat_t = np.arange(int(.04 * sr)) / sr
    hat = np.diff(np.r_[0, generator.normal(size=len(hat_t))]) * np.exp(-hat_t * 75)
    for i in range(47):
        a = round((i * .5 + .5) * sr)
        b = round((i * .5 + .75) * sr)
        if a + len(kick) <= len(audio):
            audio[a:a + len(kick)] += kick
        if b + len(hat) <= len(audio):
            audio[b:b + len(hat)] += .3 * hat
    displaced = np.arange(42, dtype=float) * .5 + .75
    report = inspect_waveform(audio, sr, displaced)
    assert report["decision"] == "sustained_offbeat_phase_inconsistency"
    assert len(report["sustained_offbeat_runs"]) >= 1



def test_high_frequency_rise_is_diagnostic_not_classification_authority():
    features = fake_features("offbeat")
    baseline = classify_phase_window(features, start=0)
    features["high_rise_at_beat"] = np.full(64, 1.)
    features["high_rise_at_midpoint"] = np.full(64, .01)
    augmented = classify_phase_window(features, start=0)
    assert baseline["status"] == "suspected_offbeat_lock"
    assert augmented["status"] == baseline["status"]
    assert augmented["high_rise_beat_dominant_count"] == 16
    assert augmented["high_rise_diagnostic_only"] is True


def test_attack_rise_requires_a_new_positive_transient():
    from djai.music.rhythm_phase_integrity import _sample_attack_rise

    sr = 16000
    power = np.zeros(sr * 3, dtype=float)
    power[sr:sr + 200] = 1.
    power[sr + 200:sr + 2000] = .2
    onset = _sample_attack_rise(power, sr, [1.0])[0]
    silence = _sample_attack_rise(power, sr, [2.0])[0]
    assert onset > 0
    assert silence == 0
