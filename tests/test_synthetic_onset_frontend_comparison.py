"""Read-only onset/front-end candidate comparison contracts."""
import pytest
import numpy as np

from scripts.compare_synthetic_onset_frontends import (
    OnsetVariant, VARIANTS, describe_variant, evaluate_configuration,
    evaluate_scenario, run, summarize,
)
from scripts.calibrate_synthetic_beat_timing import synthesize_click_kicks


def test_baseline_matches_production_front_end_constants():
    baseline = VARIANTS[0]
    assert baseline.hop == 512
    assert baseline.n_fft == 2048
    assert baseline.center is True
    assert describe_variant(baseline)["frame_interval_ms"] == pytest.approx(23.22, abs=.01)


def test_center_flag_is_not_claimed_to_change_stft_window_centering():
    uncentered = VARIANTS[-1]
    assert uncentered.center is False
    assert "STFT still centered" in describe_variant(uncentered)["timestamps"]
    assert describe_variant(uncentered)["directly_comparable_to_centered_baseline"] is False


def test_experiment_rejects_duplicate_variant_names():
    with pytest.raises(ValueError, match="unique"):
        run(bpms=(100.0,), phases=(0.0,),
            variants=(VARIANTS[0], VARIANTS[0]))


def test_synthetic_reference_is_sample_precise_and_comparison_is_measured():
    y, labels, indices = synthesize_click_kicks(
        bpm=100.0, frame_phase=0.0,
        sample_rate=22050, hop_length=512, beats=28,
    )
    assert indices[0] == 22050
    row = evaluate_configuration(y, labels, VARIANTS[0], 100.0)
    assert row["status"] == "evaluated", row.get("reason")
    assert row["peak_timing"]["matched_count"] >= 20
    assert row["beat_timing"]["matched_count"] >= 20
    assert row["beat_recall_130ms"] > 0.7


def test_summarization_counts_missing_predictions_not_as_success():
    baseline = VARIANTS[0]
    scenarios = [{"true_beats": 28, "variants": [{
        **describe_variant(baseline),
        "status": "evaluated",
        "beat_timing": {"status": "no_pairs", "matched_count": 0},
        "peak_timing": {"status": "no_pairs", "matched_count": 0},
        "prediction_count": 0,
        "predicted_beat_interval_ratio": None,
    }]}]
    res = summarize(scenarios, (baseline,))[0]
    assert res["total_matched_beats"] == 0
    assert res["aggregate_beat_recall_130ms"] == 0
    assert res["aggregate_beat_precision_130ms"] is None
    assert res["median_scenario_beat_offset_ms"] is None


def test_offline_harness_does_not_authorize_production_changes(monkeypatch):
    import scripts.compare_synthetic_onset_frontends as mod
    def fake_case(*, bpm, frame_phase, variants):
        return {"bpm": bpm, "frame_phase_fraction": frame_phase,
                "true_beats": 28, "variants": [
                    {**describe_variant(v), "status": "analysis_failed",
                     "reason": "synthetic mock"} for v in variants]}
    monkeypatch.setattr(mod, "evaluate_scenario", fake_case)
    report = run(bpms=(124,), phases=(0.0, 1/3))
    assert report["total_scenarios"] == 2
    assert report["evaluated_variant_scenarios"] == 0
    assert report["production_changes"] is False
    assert report["phase_integrity_limit_changed"] is False
    assert report["global_offset_correction_authorized"] is False
