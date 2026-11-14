"""Frozen independent-only comparison safety and attribution contracts."""
import numpy as np
import pytest

from scripts import benchmark_independent_bar_start as mod


def _manifest():
    return {"dataset": "New EDM Holdout", "provenance": "newly annotated and disjoint",
            "tracks": [{
                "id": "fresh-01", "split": "independent",
                "audio": "audio/fresh.wav", "beats": "labels/fresh.beats",
                "downbeats": "labels/fresh.downbeats"
            }]}


def test_refuses_ballroom_by_name_path_and_split():
    with pytest.raises(ValueError, match="Ballroom"):
        mod.validate_manifest(_manifest(), mod.Path("data/ballroom/manifest.json"))
    bad = _manifest()
    bad["dataset"] = "Ballroom Rhythm Dataset"
    with pytest.raises(ValueError, match="Ballroom"):
        mod.validate_manifest(bad, mod.Path("fresh.json"))
    bad = _manifest()
    bad["tracks"][0]["split"] = "test"
    with pytest.raises(ValueError, match="ONLY split"):
        mod.validate_manifest(bad, mod.Path("fresh.json"))
    bad = _manifest()
    bad["tracks"][0]["audio"] = "original-ballroom-mix.wav"
    with pytest.raises(ValueError, match="Ballroom"):
        mod.validate_manifest(bad, mod.Path("fresh.json"))


def test_provenance_and_unique_ids_required():
    bad = _manifest()
    bad.pop("provenance")
    with pytest.raises(ValueError, match="provenance"):
        mod.validate_manifest(bad, mod.Path("fresh.json"))
    bad = _manifest()
    bad["tracks"] *= 2
    with pytest.raises(ValueError, match="unique"):
        mod.validate_manifest(bad, mod.Path("fresh.json"))


def test_rhythm_consensus_requires_both_stable_votes_same_phase():
    def candidate(bass_phase, repeat_phase, strong=True):
        return {"cue_votes": {
            "bass_onset": {"leading_phase": bass_phase, "strong": strong},
            "bar_repetition": {"leading_phase": repeat_phase, "strong": True},
            "harmonic_change": {"leading_phase": 2, "strong": True},
        }}
    assert mod.rhythm_consensus(candidate(1, 1)) == 1
    assert mod.rhythm_consensus(candidate(1, 0)) is None
    assert mod.rhythm_consensus(candidate(1, 1, strong=False)) is None
    assert mod.rhythm_consensus({"cue_votes": {}}) is None


def test_frozen_fusion_and_rhythm_consensus_are_separate(monkeypatch):
    from scripts import benchmark_independent_bar_start as subject
    monkeypatch.setattr(subject, "infer_bar_start", lambda x, meter: {
        "abstained": True, "phase": None, "cue_votes": {
            "bass_onset": {"leading_phase": 1, "strong": True},
            "bar_repetition": {"leading_phase": 1, "strong": True},
            "harmonic_change": {"leading_phase": 3, "strong": True},
        }
    })
    beats = [1+i*.5 for i in range(64)]
    baseline = beats[0::4]
    result, candidate = subject.predicted_alternatives(
        beats, baseline, 4, np.zeros((64, 3))
    )
    assert result["frozen_fusion"] is baseline
    assert result["rhythm_consensus"] == beats[1::4]
    assert candidate["rhythm_consensus_phase"] == 1


def test_benchmark_does_not_access_tracks_until_manifest_verified(monkeypatch, tmp_path):
    def never(*args):
        pytest.fail("must not evaluate unverified manifest")
    monkeypatch.setattr(mod, "evaluate_track", never)
    bad = _manifest()
    bad["tracks"][0]["split"] = "validation"
    with pytest.raises(ValueError):
        mod.benchmark(bad, tmp_path, tmp_path/"new.json")


def test_summary_preserves_negative_deltas_and_regression_ids():
    row = {"id": "x", "status": "evaluated", "production_meter": 4,
           "methods": {
               name: {"f1": .6 if name == "production" else .2,
                      "delta_vs_production": 0 if name == "production" else -.4,
                      "fallback_used": name == "production"}
               for name in mod.METHODS}}
    result = mod.summary([row])
    assert result["method_comparison"]["rhythm_consensus"]["regressed"] == 1
    assert result["method_comparison"]["rhythm_consensus"]["regressions"][0]["id"] == "x"
