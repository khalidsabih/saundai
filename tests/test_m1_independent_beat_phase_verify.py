"""Synthetic tests for research-only independent beat timing and fail-closed coverage."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m1-independent-beat-phase-verify.py"


def module():
    spec = importlib.util.spec_from_file_location("m1_independent_beats", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_consensus_agrees_with_small_differences():
    m = module()
    ref = np.arange(0.5, 15.0, 0.5)
    predicted = ref + 0.011
    result = m.consensus(ref, predicted, 0.0, 15.0)
    assert result["status"] == "reference_agreement"
    assert result["max_disagreement_ms"] == pytest.approx(11.0)
    assert result["matched"] == len(ref)


def test_consensus_does_not_shift_half_beat_grid_to_claim_agreement():
    m = module()
    ref = np.arange(0.5, 15.0, 0.5)
    predicted = ref + 0.25
    result = m.consensus(ref, predicted, 0.0, 15.0)
    assert result["status"] == "inconclusive"


def test_consensus_sparse_references_are_inconclusive():
    m = module()
    result = m.consensus([0.5, 1.0, 5.0], [0.51, 1.01, 5.01], 0.0, 15.0)
    assert result["status"] == "inconclusive"


def test_signed_warp_inverse_is_monotonic_and_accurate():
    m = module()
    points = [
        {"fraction": 0.0, "source_offset_seconds": 0.0},
        {"fraction": 1.0, "source_offset_seconds": 0.018},
    ]
    original = [1.0, 2.0, 3.0]
    mapped = m.map_source_beats_to_render(original, 0.0, 5.0, 1.04, offsets=points)
    for source, rendered in zip(original, mapped):
        assert source == pytest.approx(
            1.04 * rendered + 0.018 * rendered / 5.0, abs=0.0001
        )


def test_nonmonotonic_warp_rejected():
    m = module()
    with pytest.raises(ValueError):
        m.map_source_beats_to_render(
            [0.3, 0.5], 0.0, 1.0, 1.0,
            offsets=[
                {"fraction": 0.0, "source_offset_seconds": 0.0},
                {"fraction": 0.5, "source_offset_seconds": 1.0},
                {"fraction": 1.0, "source_offset_seconds": -1.0},
            ],
        )


def test_relative_phase_drift_detects_fault_without_permitting_certification():
    m = module()
    a = np.arange(0.5, 15.0, 0.5)
    b = a + 0.010 + 0.004 * a
    comparison = m.relative_phase_from_events(a, b, 15.0)
    assert comparison["status"] == "beat_reference_proxy_measured"
    assert comparison["max_absolute_relative_drift_ms"] > 45.0
    assert comparison["proxy_within_25ms"] is False


def test_relative_phase_constant_offset_is_removed():
    m = module()
    a = np.arange(0.5, 15.0, 0.5)
    b = a + 0.06
    comparison = m.relative_phase_from_events(a, b, 15.0)
    assert comparison["status"] == "beat_reference_proxy_measured"
    assert comparison["max_absolute_relative_drift_ms"] == pytest.approx(0.0, abs=0.001)


def test_relative_phase_sparse_pairs_do_not_pass():
    m = module()
    comparison = m.relative_phase_from_events(
        [0.5, 1.0, 1.5], [0.51, 1.01, 1.51], 15.0
    )
    assert comparison["status"] == "inconclusive"
    assert comparison["proxy_within_25ms"] is None


def test_invalid_reference_times_rejected():
    m = module()
    with pytest.raises(ValueError):
        m.map_source_beats_to_render([0.8, 0.7], 0.0, 3.0, 1.0)
    with pytest.raises(ValueError):
        m.consensus([0.0, float("nan")], [0.0, 1.0], 0.0, 3.0)


def test_consensus_rejects_invalid_events_even_outside_window():
    m = module()
    with pytest.raises(ValueError):
        m.consensus([0.5, 1.0, float("nan")], [0.5, 1.0], 0.0, 2.0)
    with pytest.raises(ValueError):
        m.consensus([0.5, 1.0, 0.8, 3.0], [0.5, 1.0], 0.0, 0.7)
    with pytest.raises(ValueError):
        m.consensus([0.5, 1.0], [0.5, 1.0], 2.0, 1.0)


def test_checksum_bound_external_essentia_predictions(tmp_path):
    import hashlib
    import json
    m = module()
    files = {}
    for deck in ("A", "B"):
        p = tmp_path / f"{deck}.mp3"
        p.write_bytes(f"synthetic-audio-{deck}".encode())
        files[deck] = p
    method = "essentia_rhythmextractor2013_multifeature"
    tracks = {}
    for deck, p in files.items():
        tracks[deck] = {
            "audio_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "method": method, "bpm": 120.0, "confidence": 1.0,
            "beats_seconds": [0.5, 1.0, 1.5, 2.0, 2.5],
        }
    sidecar = tmp_path / "predictions.json"
    sidecar.write_text(json.dumps({
        "schema_version": "saund-m1-essentia-sidecar-v1",
        "method": method, "tracks": tracks
    }))
    result = m.external_essentia_predictions(sidecar, files)
    assert result["A"]["available"] is True
    assert result["B"]["source"] == "numpy1_sidecar"
    files["B"].write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.external_essentia_predictions(sidecar, files)


def test_external_essentia_rejects_tampered_timestamps(tmp_path):
    import hashlib
    import json
    m = module()
    sources = {}
    tracks = {}
    for deck in ("A", "B"):
        p = tmp_path / f"{deck}.wav"
        p.write_bytes(deck.encode())
        sources[deck] = p
        tracks[deck] = {
            "audio_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "method": "essentia_rhythmextractor2013_multifeature",
            "bpm": 120.0, "confidence": 0.9,
            "beats_seconds": [0.5, 1.0, float("nan"), 2.0],
        }
    sidecar = tmp_path / "invalid.json"
    sidecar.write_text(json.dumps({
        "schema_version": "saund-m1-essentia-sidecar-v1",
        "method": "essentia_rhythmextractor2013_multifeature",
        "tracks": tracks
    }))
    with pytest.raises(ValueError, match="Essentia A beats"):
        m.external_essentia_predictions(sidecar, sources)
