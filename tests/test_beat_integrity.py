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



def test_continuous_beat_position_has_no_nearest_beat_discontinuity():
    from djai.performance.deck import _continuous_beat_position

    beats = [index * 0.5 for index in range(120)]
    before = _continuous_beat_position(
        beats, 10.249, expected_source_beat_seconds=0.5
    )
    after = _continuous_beat_position(
        beats, 10.251, expected_source_beat_seconds=0.5
    )
    assert before == pytest.approx(20.498)
    assert after == pytest.approx(20.502)
    assert after - before == pytest.approx(0.004)


def test_continuous_beat_position_rejects_missing_beat_intervals():
    from djai.performance.deck import _continuous_beat_position

    beats = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    assert _continuous_beat_position(
        beats, 1.5, expected_source_beat_seconds=0.5
    ) is None
    assert _continuous_beat_position(
        beats, 2.25, expected_source_beat_seconds=0.5
    ) is not None


def test_phase_integrity_stable_grids_across_nearest_beat_switches(tmp_path: Path):
    from djai.performance.deck import _phase_integrity_profile

    a = _track(tmp_path, "continuous-a", bpm=120.0)
    b = _track(tmp_path, "continuous-b", bpm=120.0)
    a.beats_seconds = [index * 0.5 for index in range(200)]
    b.beats_seconds = [index * 0.5 + 0.018 for index in range(200)]
    plan = plan_transition(
        a, b, force=True, tempo_policy="safe", style="safe",
        requested_bars=8, exit_time_a=20.0, entry_time_b=4.0,
    )
    profile = _phase_integrity_profile(
        a, b, plan, phase_nudge_seconds_b=0.0
    )
    assert profile["safe"] is True
    assert profile["max_phase_error_ms"] < 1.0
    assert profile["max_residual_ms"] < 1.0


def test_phase_integrity_rejects_real_accumulated_tempo_drift(tmp_path: Path):
    from djai.performance.deck import _phase_integrity_profile

    a = _track(tmp_path, "real-drift-a", bpm=120.0)
    b = _track(tmp_path, "real-drift-b", bpm=120.0)
    a.beats_seconds = [index * 0.5 for index in range(200)]
    # A valid but 2%-slower incoming beat clock accumulates >300 ms drift.
    b.beats_seconds = [index * 0.51 for index in range(200)]
    plan = plan_transition(
        a, b, force=True, tempo_policy="safe", style="safe",
        requested_bars=8, exit_time_a=20.0, entry_time_b=4.0,
    )
    profile = _phase_integrity_profile(
        a, b, plan, phase_nudge_seconds_b=0.0
    )
    assert profile["safe"] is False
    assert profile["max_phase_error_ms"] > 100.0
    assert profile["max_residual_ms"] > 25.0



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




def test_phase_rejection_trace_is_opt_in_and_preserves_gate(tmp_path: Path, monkeypatch):
    import json

    from djai.performance.deck import _trace_rejected_phase_lock

    track_a = _track(tmp_path, "trace-a")
    track_b = _track(tmp_path, "trace-b")
    plan = plan_transition(
        track_a, track_b, force=True, tempo_policy="safe",
        style="safe", requested_bars=8, exit_time_a=20.0, entry_time_b=4.0,
    )
    path = tmp_path / "phase-rejections.jsonl"
    profile = {
        "safe": False,
        "reason": "required correction exceeds bounded phase-lock envelope",
        "max_phase_error_ms": 125.0,
        "max_residual_ms": 100.0,
        "residual_limit_ms": 25.0,
        "points": [{"fraction": 0.0, "residual_ms": 0.0}],
    }

    monkeypatch.delenv("DJAI_PHASE_LOCK_TRACE_PATH", raising=False)
    _trace_rejected_phase_lock(track_a, track_b, plan, profile)
    assert not path.exists()

    monkeypatch.setenv("DJAI_PHASE_LOCK_TRACE_PATH", str(path))
    _trace_rejected_phase_lock(track_a, track_b, plan, profile)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    evidence = json.loads(lines[0])
    assert evidence["reason"] == profile["reason"]
    assert evidence["exceeds_residual_limit"] is True
    assert evidence["max_residual_ms"] == 100.0
    assert evidence["unmatched_points"] == evidence["sample_count"] - 1
    assert evidence["track_a"] == "trace-a.wav"
    assert profile["safe"] is False

    _trace_rejected_phase_lock(track_a, track_b, plan, {**profile, "safe": True})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1



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


def test_missing_beat_gap_numbering_is_opt_in_and_does_not_interpolate(monkeypatch):
    from djai.performance.deck import (
        _continuous_beat_position,
        _overlap_crosses_unreliable_beat_interval,
    )

    beats = [0.0, 0.5, 1.0, 2.0, 2.5, 3.0]
    monkeypatch.setenv("DJAI_BEAT_GAP_AWARE", "0")
    original = _continuous_beat_position(
        beats, 2.25, expected_source_beat_seconds=0.5
    )
    assert original is not None and abs(original - 3.5) < 1e-6

    monkeypatch.setenv("DJAI_BEAT_GAP_AWARE", "1")
    corrected = _continuous_beat_position(
        beats, 2.25, expected_source_beat_seconds=0.5
    )
    assert corrected is not None and abs(corrected - 4.5) < 1e-6
    # The missing beat interval itself is still forbidden evidence.
    assert _continuous_beat_position(
        beats, 1.5, expected_source_beat_seconds=0.5
    ) is None
    # Nor can sparse sampling hide a gap traversed by a complete overlap.
    assert _overlap_crosses_unreliable_beat_interval(
        beats, start_seconds=0.75, end_seconds=2.25,
        expected_source_beat_seconds=0.5,
    )
    assert not _overlap_crosses_unreliable_beat_interval(
        beats, start_seconds=2.1, end_seconds=2.9,
        expected_source_beat_seconds=0.5,
    )
