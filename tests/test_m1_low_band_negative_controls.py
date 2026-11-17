"""Synthetic negative-control contracts for event-spacing research."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-low-band-negative-controls.py"
    spec = importlib.util.spec_from_file_location("m1_low_band_negative_controls", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_fixtures_are_deterministic_and_distinct():
    mod = load()
    first = mod.generate_control("stable_120")
    second = mod.generate_control("stable_120")
    assert np.array_equal(first[0], second[0])
    assert first[2] == second[2]
    assert len(mod.generate_control("fast_200")[2]) > len(first[2])
    assert len(mod.generate_control("syncopated_bass")[3]) > 0
    assert mod.generate_control("silence")[2] == []


def test_posthoc_match_does_not_double_count_one_detected_peak():
    mod = load()
    measured = [{"time_seconds": 1.01}]
    result = mod.match_recovery(measured, [1.0, 1.02], tolerance_ms=40)
    assert result["matched_count"] == 1
    assert result["recall"] == .5
    assert result["unmatched_detected_event_count"] == 0


def test_missing_reference_yields_null_recall_not_fake_success():
    mod = load()
    report = mod.match_recovery([], [])
    assert report["recall"] is None
    assert report["matched_count"] == 0
    assert report["unmatched_detected_event_count"] == 0


def test_negative_control_sweep_runs_every_setting_without_selection():
    mod = load()
    result = mod.run_controls()
    assert result["selected_configuration"] is None
    assert result["generator_reference_used_in_detector"] is False
    assert result["reference_used_for_posthoc_scoring"] is True
    assert [r["name"] for r in result["cases"]] == list(mod.CASES)
    assert all(len(row["settings"]) == len(mod.SWEEP) for row in result["cases"])
    silence = next(r for r in result["cases"] if r["name"] == "silence")
    assert all(setting["detected_event_count"] == 0 for setting in silence["settings"])
    assert all(setting["primary"]["recall"] is None for setting in silence["settings"])


def test_350ms_min_spacing_cannot_recover_every_true_200bpm_pulse():
    mod = load()
    result = mod.run_controls()
    fast = next(r for r in result["cases"] if r["name"] == "fast_200")
    spaced = next(r for r in fast["settings"] if r["minimum_spacing_ms"] == 350.)
    assert spaced["primary"]["recall"] < 1.


def test_unsupported_control_fails_explicitly():
    with pytest.raises(ValueError, match="unknown control"):
        load().generate_control("unsupported")
