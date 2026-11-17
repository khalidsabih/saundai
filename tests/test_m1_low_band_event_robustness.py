"""Research-only robustness sweep must not tune on its evaluation track."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-low-band-event-robustness.py"
    spec = importlib.util.spec_from_file_location("m1_low_band_robustness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLocator:
    def __init__(self, events):
        self.events = events
        self.settings = []

    def locate_events(self, audio, sample_rate, *, band, minimum_spacing_ms,
                      threshold_fraction):
        self.settings.append((band, minimum_spacing_ms, threshold_fraction))
        return [dict(event) for event in self.events]

    def compare_events(self, grid, events, *, windows):
        from importlib.util import spec_from_file_location, module_from_spec
        path = Path(__file__).resolve().parents[1] / "scripts" / "m1-unconditioned-transient-localization.py"
        spec = spec_from_file_location("m1_events_for_robustness_test", path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.compare_events(grid, events, windows=windows)


def test_sweep_evaluates_every_setting_without_selecting_one():
    module = load()
    grid = list(np.arange(32, dtype=float) * .5)
    original = list(grid)
    peaks = [{"time_seconds": float(t)} for t in np.arange(32) * .5]
    locator = FakeLocator(peaks)
    report = module.sweep_track(
        np.zeros(16000 * 17), 16000, grid,
        (("early", None, 8.), ("late", 8., None)), locator=locator,
    )
    assert len(report) == len(module.SWEEP)
    assert len(locator.settings) == len(module.SWEEP)
    assert [(s[1], s[2]) for s in locator.settings] == list(module.SWEEP)
    assert all(s[0] == "low" for s in locator.settings)
    assert all(row["global_low_event_count"] == 32 for row in report)
    assert all(len(row["regions"]) == 2 for row in report)
    assert grid == original


def test_phase_displacement_is_reported_without_grid_mutation():
    module = load()
    grid = list(np.arange(32, dtype=float) * .5)
    locator = FakeLocator([
        {"time_seconds": float(t)} for t in np.arange(31) * .5 + .25
    ])
    result = module.sweep_track(
        np.zeros(16000 * 17), 16000, grid,
        (("all", None, None),), locator=locator,
    )
    for setting in result:
        row = setting["regions"][0]
        assert row["midpoint_matches"] == 31
        assert row["beat_matches"] == 0
        assert row["shared_event_count"] == 0


def test_no_acoustic_events_is_reported_as_insufficient():
    module = load()
    result = module.sweep_track(
        np.zeros(16000 * 17), 16000, list(np.arange(32) * .5),
        (("all", None, None),), locator=FakeLocator([]),
    )
    assert all(r["global_low_event_count"] == 0 for r in result)
    assert all(r["regions"][0]["insufficient_event_evidence"] for r in result)
    assert all(r["regions"][0]["median_absolute_beat_error_ms"] is None for r in result)
