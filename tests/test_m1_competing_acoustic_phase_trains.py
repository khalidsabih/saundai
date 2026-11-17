"""Shadow-only competing-phase evidence must never choose a 'true kick'."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-competing-acoustic-phase-trains.py"
    spec = importlib.util.spec_from_file_location("m1_competing_phase_trains", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event_train(*, period=.5, duration=18., second_strength=None):
    events = []
    for i in range(int(duration / period) - 2):
        base = .5 + i * period
        events.append({"time_seconds": base, "rise_energy": 1.})
        if second_strength is not None:
            events.append({
                "time_seconds": base + period / 2,
                "rise_energy": second_strength,
            })
    return events


def test_two_acoustic_phases_are_detected_without_labeling_true_beats():
    mod = load()
    report = mod.compare_phase_trains(
        event_train(second_strength=.03), .5, 18.
    )
    assert report["status"] == "competing_acoustic_phase_trains"
    assert report["anchor_event_coverage"] >= .6
    assert report["opposite_event_coverage"] >= .6
    assert report["cannot_identify_correct_beat_phase"] is True
    assert report["authority"] == "shadow_abstain_only"


def test_single_steady_train_never_certifies_correct_beat_phase():
    mod = load()
    report = mod.compare_phase_trains(event_train(), .5, 18.)
    assert report["status"] == "insufficient_evidence"
    assert report["reason"] in {
        "too_few_localized_events",
        "not_enough_independent_opposite_phase_support",
    }


def test_extremely_weak_or_infrequent_second_train_abstains():
    mod = load()
    weak = mod.compare_phase_trains(event_train(second_strength=.00001), .5, 18.)
    assert weak["status"] == "insufficient_evidence"
    sparse = event_train()
    sparse.extend([
        {"time_seconds": .75 + i * .5, "rise_energy": .4}
        for i in (0, 2, 4, 6)
    ])
    result = mod.compare_phase_trains(sparse, .5, 18.)
    assert result["status"] == "insufficient_evidence"


def test_unsupported_period_and_no_events_abstain():
    mod = load()
    for value in (None, .1, float("nan"), 1.5):
        assert mod.compare_phase_trains([], value, 18.)["status"] == "insufficient_evidence"
    assert mod.compare_phase_trains([], .5, 18.)["status"] == "insufficient_evidence"


def test_does_not_change_input_events_or_depend_on_grid():
    mod = load()
    peaks = event_train(second_strength=.04)
    original = [dict(p) for p in peaks]
    mod.compare_phase_trains(peaks, .5, 18.)
    assert peaks == original


def test_audio_periodicity_abstention_does_not_invoke_peak_locator():
    mod = load()

    class Adaptive:
        def estimate_pulse_period(self, audio, sr):
            return {"status": "insufficient_evidence", "reason": "conflicting_period"}

    class Locator:
        def locate_events(self, *args, **kwargs):
            raise AssertionError("locator should not be called on ambiguous tempo")

    report = mod.assess_audio(
        np.zeros(16000 * 8), 16000,
        adaptive=Adaptive(), locator=Locator(),
    )
    assert report["phase_evidence"]["status"] == "insufficient_evidence"
    assert report["phase_evidence"]["authority"] == "shadow_abstain_only"


def test_report_has_no_production_or_beat_phase_authority():
    mod = load()
    report = mod.run()
    assert report["generator_reference_used"] is False
    assert report["beat_grid_used"] is False
    assert report["true_beat_phase_identified"] is False
    assert report["selected_configuration"] is None
    assert report["production_changed"] is False
    assert report["beat_grid_modified"] is False
    assert report["transition_rendered"] is False
    assert [case["name"] for case in report["cases"]] == list(mod.CASES)
