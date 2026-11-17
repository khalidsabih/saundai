"""Tests for read-only tracing of raw Beat This! downbeats to TSV labels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load():
    source = Path(__file__).resolve().parents[1] / "scripts" / "m1-beat-this-downbeat-origin-trace.py"
    spec = importlib.util.spec_from_file_location("m1_bt_origin", source)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_minimal_logits_decode_beats_and_snapped_downbeats():
    m = load()
    x = np.full((2, 200), -3., dtype=np.float32)
    for idx in (0, 25, 50, 75, 100, 125, 150):
        x[0, idx] = 4.
    for idx in (1, 26, 101):
        x[1, idx] = 3.
    result = m.decode_minimal(x)
    assert result["beats"] == pytest.approx([0., .5, 1., 1.5, 2., 2.5, 3.])
    assert result["downbeats"] == pytest.approx([0., .5, 2.])


def test_adjacent_logit_peaks_are_deduplicated():
    m = load()
    x = np.full((2, 150), -4., dtype=np.float32)
    x[0, 50] = x[0, 51] = 2.
    x[0, 101] = 2.
    x[1, 50] = 3.
    result = m.decode_minimal(x)
    assert result["beats"] == pytest.approx([1.01, 2.02])
    assert result["downbeats"] == pytest.approx([1.01])


def test_export_pairing_catches_extra_downbeat_label():
    m = load()
    compared = m.pair_times([0., .5, 1.], [0., .5, .75, 1.])
    assert compared["matched_count"] == 3
    assert compared["unmatched_b_count"] == 1
    assert compared["exact_sequence_within_tolerance"] is False


def test_legacy_and_new_export_can_be_compared_without_mutation():
    m = load()
    a = [0., .5, 1., 1.5]
    b = [0., .5, 1., 1.5]
    result = m.pair_times(a, b)
    assert result["exact_sequence_within_tolerance"]
    assert a == b


def test_synthetic_reference_is_scoring_only():
    m = load()
    r = m.downbeat_classification([0., .5, 4 * 60/122], [0., 4 * 60/122])
    assert r["near_reference_count"] == 2
    assert r["extra_label_times_seconds"] == [.5]
    assert r["reference_used_only_posthoc"] is True


def test_bad_numpy_activation_shape_fails_closed():
    m = load()
    with pytest.raises(ValueError, match="2 x frames"):
        m.decode_minimal(np.zeros((3, 100)))
    with pytest.raises(ValueError, match="nonfinite"):
        x = np.zeros((2, 100))
        x[0, 10] = np.nan
        m.decode_minimal(x)


def test_no_beat_reference_used_in_decoding():
    m = load()
    assert m.FPS == 50
    assert m.EXPORT_TOLERANCE_SECONDS == .011
    assert len(m.FIXTURES) == 2
