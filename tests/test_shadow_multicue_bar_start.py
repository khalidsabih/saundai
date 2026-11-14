"""Multicue bar-start inference, abstention and split isolation tests."""
import numpy as np
import pytest

from scripts import shadow_multicue_bar_start as mod


def _features(n=64, meter=4, phase=1):
    x = np.full((n, 3), .02, dtype=float)
    x[phase::meter, 0] = 5.0
    x[phase::meter, 1] = 3.0
    x[phase::meter, 2] = 2.5
    return x


def test_synthetic_consistent_three_cues_accept():
    out = mod.infer_bar_start(_features(), 4)
    assert out["meter"] == 4
    assert out["phase"] == 1
    assert not out["abstained"]
    assert out["cue_votes"]["bass_onset"]["strong"]
    assert out["cue_votes"]["harmonic_change"]["strong"]
    beats = [1+i*.5 for i in range(64)]
    assert mod.select_downbeats(beats, out["phase"], 4) == beats[1::4]


def test_phase_is_global_across_odd_halves():
    out = mod.infer_bar_start(_features(n=65, meter=3, phase=2), 3)
    assert out["phase"] == 2
    assert out["cue_votes"]["bass_onset"]["half_phases"] == [2, 2]


def test_flat_signals_abstain():
    out = mod.infer_bar_start(np.ones((64, 3)), 4)
    assert out["abstained"]
    assert not any(c["strong"] for c in out["cue_votes"].values())


def test_disagreement_abstains():
    x = np.full((64, 3), .05)
    x[0::4, 0] = 5
    x[1::4, 1] = 5
    x[2::4, 2] = 5
    out = mod.infer_bar_start(x, 4)
    assert out["abstained"]


def test_half_track_phase_change_abstains():
    x = _features()
    x[32:] = .02
    x[34::4] = [5, 3, 2.5]
    out = mod.infer_bar_start(x, 4)
    assert out["abstained"]


def test_missing_or_invalid_evidence_fails_closed():
    assert mod.infer_bar_start(_features(n=8), 4)["abstained"]
    assert mod.infer_bar_start(_features(), 0)["abstained"]
    with pytest.raises(ValueError, match="finite"):
        mod.infer_bar_start(np.full((64, 3), np.nan), 4)
    with pytest.raises(ValueError, match="invalid"):
        mod.select_downbeats([i*.5 for i in range(64)], 4, 4)


def test_test_split_never_read(monkeypatch, tmp_path):
    seen = []
    def fake_evaluate(item, root):
        seen.append(item["id"])
        return {"id": item["id"], "status": "unavailable_or_invalid_data"}
    monkeypatch.setattr(mod, "evaluate_one", fake_evaluate)
    result = mod.compare({"tracks": [
        {"id": "val", "split": "validation"},
        {"id": "never", "split": "test", "audio": "not-allowed"},
    ]}, tmp_path)
    assert seen == ["val"]
    assert result["production_changes"] is False
    assert result["phase_integrity_limit_changed"] is False
    assert result["autonomous_inference_uses_reference"] is False


def test_reference_cohort_summaries_do_not_depend_on_inference_labels():
    fake = {
        "id": "example", "status": "evaluated",
        "reference_meter": 4, "cohort": "clean_phase_failure",
        "candidate": {"abstained": True, "cue_votes": {}},
        "baseline_downbeat": {"f_measure": 0.0},
        "fusion_with_fallback": {"f_measure": 0.0},
        "cue_with_fallback": {c: {"f_measure": 0.0} for c in mod.CUES},
        "cue_f1_deltas": {c: 0.0 for c in mod.CUES},
        "fusion_f1_delta": 0.0,
    }
    s = mod.summarize([fake])
    assert s["by_cohort"]["clean_phase_failure"]["count"] == 1
    assert s["fusion_accepted"] == 0
    assert s["fusion_regressed_tracks"] == 0


def test_audio_chroma_novelty_uses_adjacent_time_frames(monkeypatch):
    """Regression: pitch-bin slicing produced a 20-long vs 19-long mismatch."""
    import librosa
    n_frames = 20
    power = np.ones((1025, n_frames), dtype=float)
    chroma = np.zeros((12, n_frames), dtype=float)
    chroma[0, :10] = 1
    chroma[1, 10:] = 1
    monkeypatch.setattr(librosa, "stft", lambda *args, **kwargs: power)
    monkeypatch.setattr(librosa.feature, "chroma_stft",
                        lambda *args, **kwargs: chroma.copy())
    features = mod.audio_cues(
        np.zeros(4096), sr=22050, beat_times=[.10, .22, .35]
    )
    assert features.shape == (3, 3)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)
    assert features[1, 1] > 0  # window includes the pitch change at frame 10


def test_incomplete_validation_shows_failure_reason_examples():
    summary = mod.summarize([
        {"id": "fail-1", "status": "unavailable_or_invalid_data",
         "reason": "ValueError: diagnostic one"},
        {"id": "fail-2", "status": "unavailable_or_invalid_data",
         "reason": "ValueError: diagnostic one"},
    ])
    assert summary["evaluated"] == 0
    assert summary["failed"] == 2
    assert summary["failure_reasons"]["ValueError: diagnostic one"] == 2
    assert summary["failed_examples"][0]["id"] == "fail-1"
