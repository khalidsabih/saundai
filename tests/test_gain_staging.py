from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import analysis
from djai.execution import normalization_gain, professional_normalization_gain
from djai.loudness import integrated_lufs, loudness_normalization_gain, true_peak_dbtp
from djai.measurement import measure_transition_audio
from djai.planner import plan_transition
from djai.set_renderer import _track_gain


def _tone(sample_rate: int, seconds: float, amplitude: float, frequency: float = 440.0) -> np.ndarray:
    frames = max(1, int(round(sample_rate * seconds)))
    t = np.arange(frames, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * t)
    return np.stack([mono, mono]).astype(np.float32)


def test_lufs_measurement_tracks_level_change():
    sample_rate = 48_000
    quiet = _tone(sample_rate, 2.0, 0.05)
    loud = _tone(sample_rate, 2.0, 0.20)

    quiet_lufs = integrated_lufs(quiet, sample_rate)
    loud_lufs = integrated_lufs(loud, sample_rate)

    assert loud_lufs > quiet_lufs
    assert loud_lufs - quiet_lufs == pytest.approx(12.04, abs=0.8)


def test_true_peak_ceiling_overrides_lufs_boost_request():
    gain = loudness_normalization_gain(
        integrated_loudness_lufs=-24.0,
        true_peak_db=-0.2,
        target_lufs=-12.0,
        true_peak_ceiling_dbtp=-1.0,
    )
    gain_db = 20.0 * math.log10(gain)

    assert gain < 1.0
    assert -0.2 + gain_db <= -1.0 + 1e-6


def test_professional_normalization_uses_legacy_rms_when_lufs_is_missing():
    expected = normalization_gain(-18.0, -13.0)
    actual = professional_normalization_gain(
        integrated_lufs=-80.0,
        true_peak_dbtp=-80.0,
        legacy_loudness_dbfs=-18.0,
    )
    assert actual == pytest.approx(expected)


def test_set_track_gain_is_bounded_by_true_peak_headroom(tmp_path):
    path = tmp_path / "track.wav"
    path.write_bytes(b"track")
    track = analysis(path, bpm=124.0)
    track.integrated_lufs = -18.0
    track.true_peak_dbtp = -2.0

    gain = _track_gain(track)
    gain_db = 20.0 * math.log10(gain)

    # LUFS target asks for +6 dB, but -1 dBTP ceiling permits only +1 dB.
    assert gain_db == pytest.approx(1.0, abs=0.02)


def test_transition_window_loudness_jump_can_hard_fail_with_safe_sample_peak(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    sample_rate = 48_000
    pre = _tone(sample_rate, 1.5, 0.04)
    transition = _tone(sample_rate, 2.0, 0.40)
    post = _tone(sample_rate, 1.5, 0.04)

    result = measure_transition_audio(
        pre=pre,
        transition=transition,
        post=post,
        sample_rate=sample_rate,
        plan=plan,
    )

    assert result.peak < 0.5
    assert result.loudness_jump_lu > 8.0
    assert any("short-term loudness" in failure for failure in result.hard_failures)


def test_transition_measurement_reports_true_peak_and_headroom(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    sample_rate = 48_000
    audio = _tone(sample_rate, 2.0, 0.80, frequency=997.0)

    result = measure_transition_audio(
        pre=audio,
        transition=audio,
        post=audio,
        sample_rate=sample_rate,
        plan=plan,
    )

    expected_true_peak = true_peak_dbtp(audio)
    assert result.true_peak_dbtp == pytest.approx(expected_true_peak, abs=0.02)
    assert result.true_peak_headroom_db == pytest.approx(-1.0 - expected_true_peak, abs=0.03)
    assert result.transition_lufs > -80.0
