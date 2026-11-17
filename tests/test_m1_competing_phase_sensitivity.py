"""Deterministic, research-only sensitivity validation of competing phases."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-competing-phase-sensitivity.py"
    spec = importlib.util.spec_from_file_location("m1_competing_sensitivity", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_sweep_and_controls_are_predeclared():
    module = load()
    assert module.KICK_AMPLITUDES == (.02, .06, .12, .25, .5)
    assert module.NOISE_STDS == (0., .001, .005)
    assert module.CONTROL_VARIANTS == (
        "stable_pulses", "strong_offbeat_only", "noise_only",
    )


def test_synthetic_waveform_is_deterministic_and_has_only_waveform_output():
    module = load()
    a = module.synth_waveform(kick_amplitude=.12, noise_std=.005)
    b = module.synth_waveform(kick_amplitude=.12, noise_std=.005)
    assert len(a) == 2
    assert a[1] == module.SAMPLE_RATE
    assert a[0].ndim == 1
    assert np.array_equal(a[0], b[0])


def test_clean_negative_controls_differ_from_two_phase_waveform():
    module = load()
    mixed, _ = module.synth_waveform(kick_amplitude=.12, noise_std=0.)
    only_pulse, _ = module.synth_waveform(
        kick_amplitude=.12, noise_std=0., variant="stable_pulses"
    )
    only_bass, _ = module.synth_waveform(
        kick_amplitude=.12, noise_std=0., variant="strong_offbeat_only"
    )
    noise, _ = module.synth_waveform(
        kick_amplitude=.12, noise_std=0., variant="noise_only"
    )
    assert np.max(np.abs(mixed - only_pulse - only_bass)) < 1e-10
    assert np.all(noise == 0.)


def test_invalid_fixture_settings_fail_explicitly():
    module = load()
    with pytest.raises(ValueError, match="unknown fixture"):
        module.synth_waveform(kick_amplitude=.12, noise_std=0., variant="wrong")
    with pytest.raises(ValueError, match="nonnegative"):
        module.synth_waveform(kick_amplitude=-.1, noise_std=0.)


def test_sensitivity_report_contains_every_setting_and_no_promotion(monkeypatch):
    module = load()
    class Stub:
        pass
    monkeypatch.setattr(module, "load_script", lambda filename, name: Stub())
    def mock_summarize(wave, sample_rate, *, adaptive, locator, phase):
        return {
            "audio_periodicity": {"status": "insufficient_evidence"},
            "phase_evidence": {"status": "insufficient_evidence"},
        }
    monkeypatch.setattr(module, "summarize_case", mock_summarize)
    result = module.run()
    assert len(result["cases"]) == 24
    assert [x["kick_amplitude"] for x in result["cases"][:3]] == [.02] * 3
    assert result["selected_configuration"] is None
    assert result["generator_reference_used_in_detector"] is False
    assert result["beat_grid_used"] is False
    assert result["correct_phase_identified"] is False
    assert result["production_changed"] is False
    assert result["transition_rendered"] is False


def test_single_train_of_23_events_is_not_rejected_before_cycle_analysis():
    module = load()
    phase = module.load_script(
        "m1-competing-acoustic-phase-trains.py", "m1_sens_test_phase"
    )
    events = [
        {"time_seconds": .5 + i * .5, "rise_energy": 1.}
        for i in range(23)
    ]
    result = phase.compare_phase_trains(events, .5, 12.)
    assert result["status"] == "insufficient_evidence"
    assert result.get("reason") != "too_few_localized_events"
    assert result.get("complete_cycles", 0) >= 12
