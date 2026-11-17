"""M1 robustness matrix: deterministic waveform-only shadow controls."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-competing-phase-robustness.py"
    spec = importlib.util.spec_from_file_location("m1_competing_robustness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predeclared_robustness_matrix_has_no_tuning():
    mod = load()
    assert mod.SEEDS == (101, 907)
    assert mod.NOISE_TYPES == ("white", "lowpass")
    assert mod.KICK_TIMBRES == ("decay_87hz", "chirp_65_to_125hz")
    assert mod.JITTER_MS == (0., 14.)
    assert mod.MIXED_AMPLITUDES == (.06, .12)
    assert mod.EXPECTED_CASES == 52


def test_deterministic_waveforms_are_audio_only():
    mod = load()
    kwargs = dict(
        seed=101, noise_type="lowpass", timbre="chirp_65_to_125hz",
        jitter_ms=14., variant="two_phase", kick_amplitude=.06,
    )
    a = mod.make_waveform(**kwargs)
    b = mod.make_waveform(**kwargs)
    assert len(a) == 2
    assert a[1] == mod.SAMPLE_RATE
    assert a[0].ndim == 1
    assert np.array_equal(a[0], b[0])


def test_noise_types_and_kick_timbres_materially_change_audio():
    mod = load()
    kwargs = dict(seed=907, noise_type="white", timbre="decay_87hz",
                  jitter_ms=0., variant="two_phase", kick_amplitude=.12)
    first, _ = mod.make_waveform(**kwargs)
    second, _ = mod.make_waveform(**{**kwargs, "noise_type": "lowpass"})
    third, _ = mod.make_waveform(**{**kwargs, "timbre": "chirp_65_to_125hz"})
    fourth, _ = mod.make_waveform(**{**kwargs, "jitter_ms": 14.})
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, third)
    assert not np.array_equal(first, fourth)


def test_invalid_control_options_fail_without_defaulting():
    mod = load()
    with pytest.raises(ValueError, match="noise type"):
        mod.normalized_noise(np.random.default_rng(1), 100, "purple")
    args = dict(seed=101, noise_type="white", timbre="decay_87hz",
                jitter_ms=0., variant="two_phase")
    with pytest.raises(ValueError, match="unknown variant"):
        mod.make_waveform(**{**args, "variant": "wrong"})
    with pytest.raises(ValueError, match="unknown timbre"):
        mod.make_waveform(**{**args, "timbre": "wrong"})
    with pytest.raises(ValueError, match="invalid fixture"):
        mod.make_waveform(**{**args, "jitter_ms": -1})


def test_matrix_reports_all_cases_with_no_production_authority():
    mod = load()
    def abstaining_assessor(audio, sr):
        return {
            "periodicity": {"status": "insufficient_evidence"},
            "phase_evidence": {"status": "insufficient_evidence"},
        }
    report = mod.run(assessor=abstaining_assessor)
    assert len(report["cases"]) == 52
    assert report["summary_by_fixture"]["two_phase"]["cases"] == 32
    assert report["summary_by_fixture"]["pulse_only"]["cases"] == 16
    assert report["summary_by_fixture"]["noise_only"]["cases"] == 4
    assert all(row["competing_phase_count"] == 0
               for row in report["summary_by_fixture"].values())
    assert report["selected_configuration"] is None
    assert report["detector_thresholds_modified"] is False
    assert report["generator_reference_used_in_detector"] is False
    assert report["beat_grid_used"] is False
    assert report["correct_phase_identified"] is False
    assert report["production_changed"] is False
    assert report["beat_grid_modified"] is False
    assert report["transition_rendered"] is False
