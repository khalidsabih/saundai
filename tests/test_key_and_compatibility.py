from __future__ import annotations

import numpy as np

from djai.compatibility import harmonic_compatibility, tempo_compatibility
from djai.key_detection import MINOR_PROFILE, estimate_key_from_chroma
from djai.models import KeyEstimate


def test_key_estimator_detects_a_minor_profile() -> None:
    chroma = np.tile(np.roll(MINOR_PROFILE, 9)[:, None], (1, 32))
    result = estimate_key_from_chroma(chroma)
    assert result.pitch_class == 9
    assert result.mode == "minor"
    assert result.camelot == "8A"


def test_tempo_compatibility_penalizes_large_gap() -> None:
    close, close_difference = tempo_compatibility(124.0, 126.0)
    far, far_difference = tempo_compatibility(124.0, 104.0)
    assert close > far
    assert close_difference < far_difference
    assert far < 0.20


def test_harmonic_compatibility_uses_camelot_relationship() -> None:
    a_minor = KeyEstimate("A", 9, "minor", "8A", 0.95)
    e_minor = KeyEstimate("E", 4, "minor", "9A", 0.95)
    fs_major = KeyEstimate("F#", 6, "major", "2B", 0.95)
    assert harmonic_compatibility(a_minor, e_minor) > harmonic_compatibility(a_minor, fs_major)
