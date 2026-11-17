"""Opt-in M1 neural bar-phase resolution, with explicit abstention controls."""
from __future__ import annotations

import numpy as np
import pytest

from djai.intelligence.neural_bar_phase import resolve_bar_phase


def scenario(*, meter=4, phase=0, bpm=120, count=128):
    sr, hop = 22050, 512
    beats = np.arange(count, dtype=float) * (60 / bpm) + .03
    duration = float(beats[-1] + 1)
    envelope = np.zeros(int(duration * sr / hop) + 10)
    for i, t in enumerate(beats):
        f = int(round(t * sr / hop))
        envelope[f] = 20. if i % meter == phase else 1.
    downbeats = beats[phase::meter]
    return beats, downbeats, envelope, sr, hop, duration


def resolve(beats, downbeats, onset, sr, hop, duration, *, meter=4, conf=.95):
    return resolve_bar_phase(
        beats, downbeats, onset, sample_rate=sr, hop_length=hop,
        meter=meter, meter_confidence=conf, duration_seconds=duration,
    )


@pytest.mark.parametrize("meter", [3, 4, 6])
def test_independent_acoustic_phase_plus_bar_landmarks_can_resolve(meter):
    beats, anchors, onset, sr, hop, duration = scenario(meter=meter, count=144)
    out = resolve(beats, anchors, onset, sr, hop, duration, meter=meter)
    assert out.eligible, out.reason
    assert out.meter == meter
    assert out.phase == 0
    assert len(out.downbeats_seconds) == len(anchors)
    assert np.allclose(out.downbeats_seconds, anchors)
    assert out.bar_landmark_agreement >= .90
    assert abs(out.bpm - 120) < .1


def test_no_acoustic_accent_abstains_even_with_regular_bar_landmarks():
    beats, anchors, onset, sr, hop, duration = scenario()
    onset[:] = 1
    out = resolve(beats, anchors, onset, sr, hop, duration)
    assert not out.eligible
    assert "ambiguous" in out.reason
    assert not out.beats_seconds


def test_wrong_bar_phase_abstains_even_when_acoustic_phase_is_strong():
    beats, anchors, onset, sr, hop, duration = scenario()
    wrong_anchors = beats[1::4]
    out = resolve(beats, wrong_anchors, onset, sr, hop, duration)
    assert not out.eligible
    assert "landmarks disagree" in out.reason


def test_drifting_acoustic_phase_abstains():
    beats, anchors, onset, sr, hop, duration = scenario()
    onset[:] = 0
    split = len(beats) // 2
    for i, t in enumerate(beats):
        f = int(round(t * sr / hop))
        onset[f] = 20 if (i % 4 == (0 if i < split else 1)) else 1
    out = resolve(beats, anchors, onset, sr, hop, duration)
    assert not out.eligible
    assert "phase disagrees" in out.reason


def test_single_neural_gap_abstains_before_cues():
    beats, anchors, onset, sr, hop, duration = scenario()
    beats[80:] += 3
    out = resolve(beats, anchors, onset, sr, hop, duration + 3)
    assert not out.eligible
    assert "unstable" in out.reason


def test_low_meter_confidence_rejects_even_strong_onsets():
    beats, anchors, onset, sr, hop, duration = scenario()
    out = resolve(beats, anchors, onset, sr, hop, duration, conf=.5)
    assert not out.eligible
    assert "meter" in out.reason


def test_insufficient_coverage_abstains():
    beats, anchors, onset, sr, hop, duration = scenario()
    out = resolve(beats[:40], anchors, onset, sr, hop, duration)
    assert not out.eligible
    assert "coverage" in out.reason


def test_invalid_audio_evidence_abstains():
    beats, anchors, onset, sr, hop, duration = scenario()
    onset[20] = float("nan")
    out = resolve(beats, anchors, onset, sr, hop, duration)
    assert not out.eligible
    assert "invalid" in out.reason


def test_resolution_evidence_omits_raw_timestamp_vectors():
    beats, anchors, onset, sr, hop, duration = scenario()
    result = resolve(beats, anchors, onset, sr, hop, duration)
    info = result.evidence()
    assert info["eligible"]
    assert "beats_seconds" not in info
    assert "downbeats_seconds" not in info
