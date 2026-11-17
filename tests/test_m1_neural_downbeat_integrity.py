"""M1 opt-in neural downbeat structural integrity checks.

These tests do not download models, invoke audio inference, or alter the
production default confidence policy.
"""
from __future__ import annotations

from djai.intelligence.neural_ears import downbeat_spacing_integrity


def test_rejects_nearly_every_beat_marked_downbeat():
    beats = [n * 0.5 for n in range(129)]
    downs = beats[:124]
    accepted, reason = downbeat_spacing_integrity(beats, downs)
    assert not accepted
    assert "too dense" in reason


def test_accepts_regular_three_four_and_six_pulse_downbeats():
    beats = [n * 0.5 for n in range(120)]
    for meter in (3, 4, 6):
        accepted, reason = downbeat_spacing_integrity(beats, beats[::meter])
        assert accepted, (meter, reason)


def test_rejects_unsorted_and_nonfinite_events():
    assert not downbeat_spacing_integrity([0, .5, .4, 1.5], [0, 1.5])[0]
    assert not downbeat_spacing_integrity([0, .5, 1, 1.5], [0, float("nan")])[0]


def test_rejects_downbeats_every_two_beats():
    beats = [n * 0.5 for n in range(48)]
    assert not downbeat_spacing_integrity(beats, beats[::2])[0]
