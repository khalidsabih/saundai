"""M1 2x2 period/anchor attribution is waveform-only and never authoritative."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-period-anchor-attribution.py"
    spec = importlib.util.spec_from_file_location("m1_period_anchor_attribution", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Comparator:
    def __init__(self):
        self.calls = []

    def compare_phase_trains(self, events, period, duration, *,
                             anchor_phase_seconds=None):
        self.calls.append((period, duration, anchor_phase_seconds))
        if duration == 12.:
            return {"anchor_phase_seconds": .2}
        return {
            "status": "insufficient_evidence",
            "anchor_phase_seconds": (
                anchor_phase_seconds if anchor_phase_seconds is not None else .1
            ),
            "authority": "shadow_abstain_only",
        }


def test_four_arms_share_window_events_and_convert_held_anchor_to_local_time():
    mod = load()
    comparator = Comparator()
    local_events = [{"time_seconds": .25, "rise_energy": .5}]
    full_events = [{"time_seconds": .25, "rise_energy": .5}]
    result = mod.evaluate_window_events(
        local_events, 8., 4., .5, .495, full_events, 12.,
        comparator=comparator,
    )
    assert tuple(result) == mod.ARMS
    assert len(comparator.calls) == 6
    assert comparator.calls[0] == (.5, 8., None)
    assert comparator.calls[1] == (.5, 12., None)
    assert abs(comparator.calls[2][2] - .2) < 1e-9
    assert comparator.calls[3] == (.495, 8., None)
    assert comparator.calls[4] == (.495, 12., None)
    assert abs(comparator.calls[5][2] - ((.2 - 4.) % .495)) < 1e-9
    assert result["local_period_held_anchor"]["authority"] == "shadow_abstain_only"


def test_global_and_local_period_abstentions_do_not_create_phase_claims():
    mod = load()
    comparator = Comparator()
    result = mod.evaluate_window_events(
        [], 8., 0., None, .5, [], 12., comparator=comparator,
    )
    assert result["global_period_local_anchor"]["reason"] == "audio_periodicity_abstained"
    assert result["global_period_held_anchor"]["status"] == "insufficient_evidence"
    assert result["local_period_local_anchor"]["status"] == "insufficient_evidence"
    assert result["local_period_held_anchor"]["status"] == "insufficient_evidence"


def test_unsupported_global_acoustic_anchor_abstains_not_fabricated():
    mod = load()
    class NoAnchor:
        def compare_phase_trains(self, events, period, duration, *, anchor_phase_seconds=None):
            return {"status": "insufficient_evidence", "reason": "too_few_localized_events"}
    result = mod.evaluate_window_events(
        [], 8., 4., .5, .5, [], 12., comparator=NoAnchor()
    )
    assert result["global_period_held_anchor"]["reason"] == "global_acoustic_anchor_unavailable"
    assert result["local_period_held_anchor"]["reason"] == "global_acoustic_anchor_unavailable"


def test_full_and_local_periods_come_from_waveforms_not_known_grid():
    mod = load()
    class Adaptive:
        def estimate_pulse_period(self, audio, sr):
            return {"status": "supported_audio_periodicity",
                    "period_seconds": .5 if len(audio) == 12 * sr else .495}
    class Locator:
        def locate_events(self, audio, sr, **kwargs):
            assert kwargs == {"band": "low", "minimum_spacing_ms": 85.,
                              "threshold_fraction": .02}
            return [{"time_seconds": .5, "rise_energy": 1.}]
    result = mod.compare_waveform(
        np.zeros(12 * 16000), 16000,
        adaptive=Adaptive(), locator=Locator(), comparator=Comparator(),
    )
    assert len(result["windows"]) == 2
    assert [w["period_delta_local_minus_global_ms"] for w in result["windows"]] == [-5., -5.]
    assert all(tuple(w["arms"]) == mod.ARMS for w in result["windows"])
    assert result["correct_phase_identified"] is False


def test_invalid_inputs_fail_and_do_not_call_dependencies():
    mod = load()
    with pytest.raises(ValueError, match="invalid or short"):
        mod.compare_waveform(np.zeros(100), 16000,
                             adaptive=None, locator=None, comparator=None)
    invalid = np.zeros(12 * 16000)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="invalid or short"):
        mod.compare_waveform(invalid, 16000,
                             adaptive=None, locator=None, comparator=None)


def test_52_case_matrix_and_all_policy_flags_remain_frozen():
    mod = load()
    def abstaining_assessor(wave, sr):
        return {
            "global_audio_periodicity": {"status": "insufficient_evidence"},
            "windows": [{
                "arms": {arm: {"status": "insufficient_evidence"}
                         for arm in mod.ARMS}
            } for _ in range(2)],
        }
    report = mod.run(assessor=abstaining_assessor)
    assert len(report["cases"]) == 52
    assert report["source_matrix_case_count"] == 52
    assert report["summary_by_fixture_jitter"]["two_phase_jitter_0ms"]["cases"] == 16
    assert report["summary_by_fixture_jitter"]["two_phase_jitter_14ms"]["cases"] == 16
    assert report["detector_thresholds_modified"] is False
    assert report["selected_configuration"] is None
    assert report["generator_reference_used_in_detector"] is False
    assert report["beat_grid_used"] is False
    assert report["correct_phase_identified"] is False
    assert report["production_changed"] is False
    assert report["beat_grid_modified"] is False
    assert report["transition_rendered"] is False
