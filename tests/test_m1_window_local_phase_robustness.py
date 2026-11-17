"""Shadow-only local-versus-global phase comparison, with no oracle leakage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-window-local-phase-robustness.py"
    spec = importlib.util.spec_from_file_location("m1_local_phase_robustness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(*, competing=False, period=.5):
    return {
        "audio_periodicity": {
            "status": "supported_audio_periodicity",
            "period_seconds": period,
        },
        "phase_evidence": {
            "status": (
                "competing_acoustic_phase_trains" if competing
                else "insufficient_evidence"
            ),
            "authority": "shadow_abstain_only",
        },
    }


def test_unchanged_detector_is_called_with_full_audio_and_overlapping_windows():
    mod = load()
    calls = []

    def assessor(audio, sr):
        calls.append((len(audio), sr))
        return evidence(competing=(len(calls) > 1))

    result = mod.measure_local_vs_global(
        np.zeros(12 * 16000), 16000, assessor=assessor
    )
    assert calls == [(192000, 16000), (128000, 16000), (128000, 16000)]
    assert [(x["start_seconds"], x["end_seconds"])
            for x in result["local_windows"]] == [(0., 8.), (4., 12.)]
    assert result["local_competing_window_count"] == 2
    assert result["local_status"] == "recurrent_competing_acoustic_evidence_shadow_only"
    assert result["windows_overlap"] is True
    assert result["correct_phase_identified"] is False


def test_single_positive_window_cannot_create_recurrent_status():
    mod = load()
    calls = []

    def assessor(audio, sr):
        calls.append(len(audio))
        return evidence(competing=(len(calls) == 2))

    result = mod.measure_local_vs_global(
        np.zeros(12 * 16000), 16000, assessor=assessor
    )
    assert result["local_competing_window_count"] == 1
    assert result["local_status"] == "insufficient_evidence"


def test_local_period_spread_is_descriptive_not_a_selected_tempo():
    mod = load()
    calls = []

    def assessor(audio, sr):
        calls.append(len(audio))
        periods = (.5, .5, .5075)
        return evidence(competing=False, period=periods[len(calls)-1])

    report = mod.measure_local_vs_global(
        np.zeros(12 * 16000), 16000, assessor=assessor
    )
    assert report["local_period_range_ms"] == 7.5
    assert report["supported_period_window_count"] == 2
    assert report["local_status"] == "insufficient_evidence"


def test_abstaining_local_acf_cannot_be_counted_as_competing_phase():
    mod = load()
    def assessor(audio, sr):
        return {
            "audio_periodicity": {"status": "insufficient_evidence"},
            "phase_evidence": {
                "status": "insufficient_evidence", "reason": "periodicity_abstained"
            },
        }
    report = mod.measure_local_vs_global(
        np.zeros(12 * 16000), 16000, assessor=assessor
    )
    assert report["supported_period_window_count"] == 0
    assert report["local_period_range_ms"] is None
    assert report["local_status"] == "insufficient_evidence"


def test_invalid_and_short_audio_is_rejected():
    mod = load()
    with pytest.raises(ValueError, match="too short"):
        mod.measure_local_vs_global(np.zeros(3 * 16000), 16000, assessor=lambda a, sr: {})
    bad = np.zeros(12 * 16000)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="invalid waveform"):
        mod.measure_local_vs_global(bad, 16000, assessor=lambda a, sr: {})


def test_full_original_matrix_is_reused_and_has_no_promotion():
    mod = load()
    def abstaining_assessor(audio, sr):
        return {
            "audio_periodicity": {"status": "insufficient_evidence"},
            "phase_evidence": {"status": "insufficient_evidence"},
        }
    report = mod.run(assessor=abstaining_assessor)
    assert len(report["cases"]) == 52
    assert report["fixed_source_matrix_case_count"] == 52
    assert sum(c["variant"] == "two_phase" for c in report["cases"]) == 32
    assert sum(c["variant"] == "pulse_only" for c in report["cases"]) == 16
    assert sum(c["variant"] == "noise_only" for c in report["cases"]) == 4
    assert report["window_seconds"] == 8.
    assert report["window_hop_seconds"] == 4.
    assert report["local_windows_overlap"] is True
    assert report["detector_thresholds_modified"] is False
    assert report["generator_reference_used_in_detector"] is False
    assert report["beat_grid_used"] is False
    assert report["correct_phase_identified"] is False
    assert report["production_changed"] is False
    assert report["beat_grid_modified"] is False
    assert report["selected_configuration"] is None
    assert report["transition_rendered"] is False
