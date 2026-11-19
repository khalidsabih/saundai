"""Guard research-only audio-derived event spacing against oracle leakage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-audio-adaptive-event-spacing.py"
    spec = importlib.util.spec_from_file_location("m1_audio_adaptive_spacing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_rule_never_uses_a_known_bpm_or_generator_reference():
    mod = load()
    assert mod.adaptive_spacing_ms({"status": "insufficient_evidence"}) is None
    assert mod.adaptive_spacing_ms({
        "status": "supported_audio_periodicity", "period_seconds": .3
    }) == 165.
    assert mod.adaptive_spacing_ms({
        "status": "supported_audio_periodicity", "period_seconds": .5
    }) == 275.
    assert mod.adaptive_spacing_ms({
        "status": "supported_audio_periodicity", "period_seconds": .8
    }) == 320.


def test_silence_abstains_before_any_adaptive_event_selection():
    mod = load()
    audio = np.zeros(16000 * 8)
    evidence = mod.estimate_pulse_period(audio, 16000)
    assert evidence["status"] == "insufficient_evidence"
    assert mod.adaptive_spacing_ms(evidence) is None


def test_invalid_and_short_waveforms_abstain():
    mod = load()
    assert mod.estimate_pulse_period(np.zeros(100), 16000)["status"] == "insufficient_evidence"
    bad = np.zeros(16000 * 5)
    bad[10] = np.nan
    assert mod.estimate_pulse_period(bad, 16000)["status"] == "insufficient_evidence"


def test_extra_fixtures_have_known_truth_but_are_not_used_as_tempo_input():
    mod = load()
    fast = mod.extra_control("very_fast_260")
    omitted = mod.extra_control("kick_omissions")
    sync = mod.extra_control("alternating_bass")
    assert len(fast[2]) > len(omitted[2])
    assert len(sync[3]) > 0
    assert np.array_equal(
        mod.extra_control("very_fast_260")[0], fast[0]
    )


def test_every_case_comparison_includes_fixed_baseline_even_if_adaptive_abstains():
    mod = load()
    class Locator:
        def __init__(self):
            self.spacings = []
        def locate_events(self, audio, sample_rate, *, band, minimum_spacing_ms, threshold_fraction):
            self.spacings.append(minimum_spacing_ms)
            return []
    class Scoring:
        def match_recovery(self, events, references):
            return {"expected_count": len(references), "matched_count": 0}
    locator = Locator()
    result = mod.evaluate_case(
        np.zeros(16000 * 8), 16000, [], [],
        locator=locator, scoring=Scoring(),
    )
    assert [strategy["name"] for strategy in result["strategies"]] == [
        "fixed_250", "audio_adaptive"
    ]
    assert locator.spacings == [250.]
    assert result["strategies"][1]["status"] == "abstain"
    assert result["strategies"][1]["primary"] is None


def test_research_report_never_selects_a_setting_or_changes_beats():
    mod = load()
    result = mod.run()
    assert result["selected_configuration"] is None
    assert result["generator_reference_used_for_period"] is False
    assert result["beat_grid_used_for_period"] is False
    assert result["reference_used_only_for_posthoc_scoring"] is True
    assert result["production_changed"] is False
    assert result["transition_rendered"] is False
    assert len(result["cases"]) == 10
    assert all(len(case["strategies"]) == 2 for case in result["cases"])
