import json
from pathlib import Path

import numpy as np

from scripts.validate_overlap_audio_tempo import tempo_candidates_from_envelope, validate


def test_synthetic_120_bpm_periodicity_detected():
    fps = 200.0
    samples = np.zeros(200 * 40, dtype=np.float64)
    samples[::100] = 1.0
    found = tempo_candidates_from_envelope(samples, frames_per_second=fps)
    assert found["status"] == "estimated"
    assert abs(found["best_bpm"] - 120.0) < 0.2


def test_synthetic_117_bpm_periodicity_detected():
    fps = 200.0
    samples = np.zeros(200 * 50, dtype=np.float64)
    period = fps * 60.0 / 117.0
    for i in range(97):
        j = round(i * period)
        if j < len(samples):
            samples[j] = 1.0
    found = tempo_candidates_from_envelope(samples, frames_per_second=fps)
    assert found["status"] == "estimated"
    assert abs(found["best_bpm"] - 117.0) < 0.5


def test_silence_not_given_a_tempo():
    result = tempo_candidates_from_envelope(np.zeros(6000), frames_per_second=200)
    assert result["status"] == "silent_or_flat_envelope"


def test_missing_audio_is_reported_without_mutating_cache(tmp_path: Path):
    plan = {
        "plan_id": "abc",
        "tracks": [
            {"path": "/crate/a.wav", "bpm": 120},
            {"path": "/crate/b.wav", "bpm": 117},
        ],
        "transitions": [{
            "from_index": 0,
            "to_index": 1,
            "plan": {"metadata": {"decision_trace_rejected_phase": {
                "candidate": {
                    "exit_time_a": 12.0, "entry_time_b": 14.0,
                    "transition_bars": 8, "target_bpm": 120,
                    "render_rate_a": 1.0, "render_rate_b": 120 / 117,
                }
            }}},
        }],
    }
    report = validate(plan, cache_root=tmp_path)
    assert len(report["tracks"]) == 2
    assert report["tracks"][0]["regions"][0]["audio_periodicity"]["status"] == "audio_unavailable"
    assert report["tracks"][1]["regions"][0]["audio_periodicity"]["status"] == "audio_unavailable"
    assert list(tmp_path.iterdir()) == []
