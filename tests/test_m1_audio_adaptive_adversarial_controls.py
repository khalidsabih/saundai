"""Adversarial ACF research: post-hoc alias audits must not become authority."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-audio-adaptive-adversarial-controls.py"
    spec = importlib.util.spec_from_file_location("m1_audio_adversarial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adversarial_fixtures_are_deterministic_and_have_distinct_tempos():
    mod = load()
    a = mod.generate_case("halftime_accents_120")
    b = mod.generate_case("halftime_accents_120")
    assert np.array_equal(a[0], b[0])
    assert a[2] == b[2]
    step = mod.generate_case("tempo_step_120_to_200")[2]
    differences = np.diff(step)
    assert min(differences) < .35
    assert max(differences) > .45
    ramp = np.diff(mod.generate_case("tempo_ramp_100_to_220")[2])
    assert ramp[0] > ramp[-1]
    assert len(mod.generate_case("doubletime_bass_120")[3]) > 0


def test_alias_flag_only_uses_reference_for_posthoc_scoring():
    mod = load()

    class Adaptive:
        def evaluate_case(self, audio, sr, true_beats, distractors, *, locator, scoring):
            assert audio.ndim == 1
            return {
                "audio_periodicity": {
                    "status": "supported_audio_periodicity",
                    "period_seconds": 1.0,
                },
                "strategies": [],
            }

    row = mod.assess_case(
        "halftime_accents_120",
        adaptive=Adaptive(), locator=object(), scorer=object()
    )
    assert row["potential_half_or_double_time_alias_posthoc"] is True
    assert row["hypothetical_period_to_reference_ratio_posthoc"] == 2.0


def test_unsupported_estimate_abstains_without_fabricated_alias():
    mod = load()

    class Adaptive:
        def evaluate_case(self, audio, sr, true_beats, distractors, *, locator, scoring):
            return {
                "audio_periodicity": {
                    "status": "insufficient_evidence", "reason": "ambiguous"
                },
                "strategies": [{"name": "audio_adaptive", "status": "abstain"}],
            }

    row = mod.assess_case(
        "tempo_step_120_to_200",
        adaptive=Adaptive(), locator=object(), scorer=object()
    )
    assert row["potential_half_or_double_time_alias_posthoc"] is False
    assert row["hypothetical_period_to_reference_ratio_posthoc"] is None
    assert row["audio_periodicity"]["status"] == "insufficient_evidence"


def test_synthetic_meters_do_not_imply_grid_or_musical_meter_detection():
    mod = load()
    three = mod.generate_case("three_beat_accents")
    six = mod.generate_case("six_beat_accents")
    assert len(three[2]) == len(six[2])
    assert not np.array_equal(three[0], six[0])


def test_unknown_fixture_is_rejected():
    with pytest.raises(ValueError, match="unsupported adversarial"):
        load().generate_case("unknown")


def test_report_preserves_research_only_authority():
    result = load().run()
    assert result["selected_configuration"] is None
    assert result["generator_reference_used_for_period"] is False
    assert result["beat_grid_used_for_period"] is False
    assert result["reference_used_only_for_posthoc_scoring"] is True
    assert result["production_changed"] is False
    assert result["beat_grid_modified"] is False
    assert result["transition_rendered"] is False
    assert [row["name"] for row in result["cases"]] == list(load().CASES)
    assert all(len(row["strategies"]) == 2 for row in result["cases"])
