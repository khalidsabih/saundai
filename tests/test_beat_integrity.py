from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import analysis
from djai.performance.deck import _phase_integrity_profile
from djai.planner import plan_transition
from djai.renderer import _time_stretch_phase_warp
from djai.set_planner import make_set_plan


def _track(tmp_path: Path, name: str, bpm: float = 124.0):
    path = tmp_path / f"{name}.wav"
    path.write_bytes(name.encode("utf-8"))
    track = analysis(path, bpm=bpm, camelot="8A", energy=0.58)
    track.duration_seconds = 90.0
    for candidate in track.exit_candidates:
        candidate.time_seconds = 20.0
        candidate.available_bars = 16
    for candidate in track.entry_candidates:
        candidate.time_seconds = 4.0
        candidate.available_bars = 16
    return track


def _apply_drift(track, *, drift_seconds_at_30s: float) -> None:
    base = np.asarray(track.beats_seconds, dtype=float)
    drift = np.clip(base / 30.0, 0.0, 1.0) * float(drift_seconds_at_30s)
    track.beats_seconds = (base + drift).tolist()
    track.variable_tempo = True


def test_stable_grid_needs_effectively_no_continuous_phase_warp(tmp_path: Path):
    track_a = _track(tmp_path, "stable-a")
    track_b = _track(tmp_path, "stable-b")
    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=8,
        exit_time_a=20.0,
        entry_time_b=4.0,
    )
    profile = plan.metadata["deck_performance"]["phase_integrity"]
    assert profile["safe"] is True
    assert profile["enabled"] is True
    assert profile["max_residual_ms"] <= 25.0
    assert max(abs(point["source_offset_seconds"]) for point in profile["points"]) < 0.006


def test_mild_drift_gets_bounded_continuous_phase_correction(tmp_path: Path):
    track_a = _track(tmp_path, "drift-a")
    track_b = _track(tmp_path, "drift-b")
    _apply_drift(track_b, drift_seconds_at_30s=0.030)

    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=8,
        exit_time_a=20.0,
        entry_time_b=4.0,
    )

    profile = plan.metadata["deck_performance"]["phase_integrity"]
    assert plan.beatmatched is True
    assert profile["safe"] is True
    assert profile["enabled"] is True
    assert profile["max_phase_error_ms"] > 1.0
    assert profile["max_residual_ms"] <= profile["residual_limit_ms"]
    assert any(abs(point["source_offset_seconds"]) >= 0.002 for point in profile["points"])
    assert plan.performance_program["annotations"]["deck_performance"]["phase_integrity"]["safe"] is True


def test_large_drift_is_rejected_by_phase_integrity(tmp_path: Path):
    track_a = _track(tmp_path, "unsafe-a")
    track_b = _track(tmp_path, "unsafe-b")
    _apply_drift(track_b, drift_seconds_at_30s=0.140)

    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=8,
        exit_time_a=20.0,
        entry_time_b=4.0,
    )

    profile = plan.metadata["deck_performance"]["phase_integrity"]
    assert plan.beatmatched is True
    assert profile["safe"] is False
    assert plan.requires_confirmation is True
    assert "phase" in (plan.warning or "").lower()



def test_phase_safe_finalists_are_prioritized_without_relaxing_gate(tmp_path: Path):
    from djai.brain.unified_transition import _prefer_viable_phase_candidates

    track_a = _track(tmp_path, "rank-a")
    track_b = _track(tmp_path, "rank-b")
    unsafe = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=8,
        exit_time_a=20.0,
        entry_time_b=4.0,
    )
    safe = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=4,
        exit_time_a=20.0,
        entry_time_b=4.0,
    )
    unsafe.metadata["deck_performance"]["phase_integrity"]["safe"] = False
    safe.metadata["deck_performance"]["phase_integrity"]["safe"] = True
    assert _prefer_viable_phase_candidates([unsafe, safe]) == [safe]
    assert _prefer_viable_phase_candidates([unsafe]) == [unsafe]



def _tone(seconds: float, sample_rate: int = 16_000, frequency: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    mono = 0.25 * np.sin(2.0 * np.pi * frequency * t)
    return np.stack([mono, mono]).astype(np.float32)


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    mono = np.asarray(audio[0], dtype=np.float64)
    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window))
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def test_phase_warp_renderer_preserves_pitch_and_output_clock():
    sample_rate = 16_000
    target_length = 5 * sample_rate
    audio = _tone(5.2, sample_rate=sample_rate, frequency=440.0)
    rendered = _time_stretch_phase_warp(
        audio,
        sample_rate=sample_rate,
        base_rate=1.0,
        target_length=target_length,
        correction_points=[
            {"fraction": 0.0, "source_offset_seconds": 0.0},
            {"fraction": 0.5, "source_offset_seconds": 0.018},
            {"fraction": 1.0, "source_offset_seconds": 0.028},
        ],
    )
    assert rendered.shape == (2, target_length)
    assert np.isfinite(rendered).all()
    assert _dominant_frequency(rendered, sample_rate) == pytest.approx(440.0, abs=7.0)


def test_automatic_set_replans_when_phase_lock_is_outside_envelope(tmp_path: Path):
    track_a = _track(tmp_path, "set-a")
    track_b = _track(tmp_path, "set-b")
    _apply_drift(track_b, drift_seconds_at_30s=0.160)

    plan = make_set_plan(
        [track_a, track_b],
        source="phase-integrity-test",
        tempo_policy="safe",
        preferred_style="safe",
        journey="balanced",
        minimum_body_seconds=8.0,
        minimum_body_fraction=0.05,
        automatic_delivery=True,
    )

    chosen = plan.transitions[0].plan
    if chosen.beatmatched:
        profile = chosen.metadata.get("deck_performance", {}).get("phase_integrity", {})
        assert profile.get("safe", True) is True
    else:
        assert chosen.metadata.get("beat_integrity_replanned", False) or chosen.metadata.get("automatic_fallback", False)
