"""Elapsed-time-normalized phase correction tests.

Synthetic beat positions isolate controller behavior. No copyrighted audio or
external model weights are required.
"""
from types import SimpleNamespace

import pytest

from djai.performance import deck


def _profile(monkeypatch, samples, *, duration=16.0, terminal_drift_ms=32.0):
    monkeypatch.setenv("DJAI_DECK_PHASE_WARP", "1")
    monkeypatch.setenv("DJAI_DECK_PHASE_WARP_SAMPLES", str(samples))
    monkeypatch.setenv("DJAI_DECK_PHASE_WARP_MAX_MS", "45")
    monkeypatch.setenv("DJAI_DECK_PHASE_WARP_STEP_MAX_MS", "18")
    monkeypatch.setenv("DJAI_DECK_PHASE_RESIDUAL_MAX_MS", "25")
    monkeypatch.setenv("DJAI_BEAT_GAP_AWARE", "0")

    a_beats = [i * 0.5 for i in range(240)]
    b_beats = [i * 0.5 for i in range(240)]
    a = SimpleNamespace(
        beats_seconds=a_beats, beat_grid_confidence=0.95, duration_seconds=120.0
    )
    b = SimpleNamespace(
        beats_seconds=b_beats, beat_grid_confidence=0.95, duration_seconds=120.0
    )
    plan = SimpleNamespace(
        beatmatched=True, target_bpm=120.0, render_rate_a=1.0,
        render_rate_b=1.0, exit_time_a=0.0, entry_time_b=0.0,
        transition_seconds=duration, source_duration_b=duration
    )

    # Simulate an incoming clock that drifts only during the final two
    # seconds, reaching a 32ms terminal phase error. No musical inference
    # assumptions enter this controlled fixture.
    def fake_position(beats, time, **kwargs):
        if beats is a_beats:
            return time / 0.5
        drift_s = max(0.0, (time - (duration - 2.0)) / 2.0)
        return (time + terminal_drift_ms / 1000.0 * drift_s) / 0.5

    monkeypatch.setattr(deck, "_continuous_beat_position", fake_position)
    return deck._phase_integrity_profile(a, b, plan, phase_nudge_seconds_b=0.0)


def _max_slew_ms_per_s(profile, duration=16.0):
    points = profile["points"]
    return max(
        abs(b["source_offset_seconds"] - a["source_offset_seconds"]) * 1000.0
        / ((b["fraction"] - a["fraction"]) * duration)
        for a, b in zip(points, points[1:])
    )


def test_nine_sample_reference_retains_original_18ms_per_2s(monkeypatch):
    p = _profile(monkeypatch, 9)
    assert p["safe"] is True
    assert p["max_residual_ms"] == pytest.approx(14.0, abs=0.001)
    assert p["points"][-1]["source_offset_seconds"] == pytest.approx(0.018, abs=1e-7)
    assert _max_slew_ms_per_s(p) <= 9.001


def test_seventeen_samples_do_not_increase_slew_rate(monkeypatch):
    p = _profile(monkeypatch, 17)
    assert p["safe"] is True
    assert p["max_residual_ms"] == pytest.approx(14.0, abs=0.001)
    assert p["points"][-1]["source_offset_seconds"] == pytest.approx(0.018, abs=1e-7)
    assert _max_slew_ms_per_s(p) <= 9.001


def test_unsafe_terminal_error_still_abstains_at_both_densities(monkeypatch):
    for samples in (9, 17):
        p = _profile(monkeypatch, samples, terminal_drift_ms=60.0)
        assert p["safe"] is False
        assert p["max_residual_ms"] >= 42.0
        assert p["residual_limit_ms"] == 25.0
        assert p["source_delta_seconds_b"] == 0.0


def test_absolute_correction_cap_unchanged(monkeypatch):
    for samples in (9, 17):
        p = _profile(monkeypatch, samples, terminal_drift_ms=100.0)
        corrections = [abs(v["source_offset_seconds"]) for v in p["points"]]
        assert max(corrections) <= 0.0450001
        assert p["safe"] is False
