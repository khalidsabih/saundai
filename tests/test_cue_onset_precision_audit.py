import numpy as np

from scripts.audit_cue_onset_precision import nearest_supported_peaks, diagnose


def test_synthetic_local_peak_detects_bounded_shift():
    fps = 22050 / 128
    env = np.ones(2000, dtype=float)
    beats = [1.0, 2.0, 3.0, 4.0]
    for time in beats:
        ix = round((time + 0.02) * fps)
        env[ix] = 8.0
        env[ix-1] = 4.0
        env[ix+1] = 4.0
    hits = nearest_supported_peaks(beats, env, window_start_seconds=0)
    assert len(hits) == len(beats)
    assert all(abs(obs["shift_ms"]-20) < 4 for obs in hits)


def test_silent_onset_envelope_yields_no_supported_peaks():
    assert nearest_supported_peaks([1.0, 2.0, 3.0], np.zeros(2000),
                                   window_start_seconds=0) == []


def test_candidate_missing_is_explicit(tmp_path):
    plan = {
        "tracks": [{"path": "/crate/a.wav"}],
        "transitions": [{"from_index": 0, "to_index": 0, "plan": {"metadata": {}}}],
    }
    item = diagnose(plan, tmp_path)["transitions"][0]
    assert item["status"] == "no_rejected_rhythmic_candidate"
