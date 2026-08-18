from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from djai.intelligence.embeddings import embed_array, embedding_similarity
from djai.intelligence.preferences import append_preference
from djai.intelligence.transition_ranker import (
    MINIMUM_COMPARISONS,
    load_transition_ranker,
    train_transition_ranker,
)


def _candidate(signature: str, score: float, *, creative: bool = False) -> dict:
    return {
        "execution_signature": signature,
        "track_a": f"a-{signature}.wav",
        "track_b": f"b-{signature}.wav",
        "style": "creative" if creative else "safe",
        "intent": "energy_lift" if creative else "smooth_continuation",
        "technique": "filter_build" if creative else "eq_bass_swap",
        "transition_bars": 8,
        "transition_seconds": 16.0,
        "exit_section": "breakdown" if creative else "outro",
        "entry_section": "drop" if creative else "intro",
        "cue_confidence": score,
        "analytical_score": score,
        "measured_score": score,
        "measurement_passed": True,
        "intent_score": score,
        "technique_score": score,
        "tempo_difference_percent": 2.0,
        "energy_delta": 0.2 if creative else 0.0,
        "embedding_similarity": score,
        "compatibility": {
            "tempo": score,
            "harmonic": score,
            "energy": score,
            "texture": score,
            "beat_grid": score,
            "phrase": score,
            "sections": score,
            "genre": score,
            "vocal_overlap": score,
            "bass_conflict": score,
            "boundary": score,
            "section_fit": score,
        },
        "measurement": {"overall": score},
    }


def test_statistical_embedding_is_deterministic_and_normalized():
    sr = 22050
    t = np.arange(sr * 2) / sr
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    first = embed_array(audio, sr, backend="statistical")
    second = embed_array(audio, sr, backend="statistical")
    assert first.backend == "statistical"
    assert first.fingerprint == second.fingerprint
    assert len(first.vector) == 96
    assert abs(np.linalg.norm(np.asarray(first.vector)) - 1.0) < 1e-5
    assert embedding_similarity(first.vector, second.vector) > 0.999


def test_pairwise_ranker_trains_with_grouped_validation(tmp_path: Path):
    preferences = tmp_path / "preferences.jsonl"
    for index in range(MINIMUM_COMPARISONS + 16):
        high = _candidate(f"high-{index}", 0.86, creative=index % 2 == 0)
        low = _candidate(f"low-{index}", 0.38, creative=index % 2 != 0)
        # Reuse several directional groups so grouped CV is meaningful.
        pair = index % 8
        high["track_a"] = low["track_a"] = f"track-{pair}-a.wav"
        high["track_b"] = low["track_b"] = f"track-{pair}-b.wav"
        append_preference(
            candidate_a=high if index % 2 == 0 else low,
            candidate_b=low if index % 2 == 0 else high,
            choice="a" if index % 2 == 0 else "b",
            listener="expert",
            path=preferences,
        )
    model_path = train_transition_ranker(preferences, tmp_path / "ranker.json")
    payload = json.loads(model_path.read_text())
    assert payload["comparison_count"] >= MINIMUM_COMPARISONS
    assert payload["fold_count"] >= 2
    assert payload["cv_accuracy"] >= payload["baseline_accuracy"]
    assert payload["authority"] > 0.0
    load_transition_ranker.cache_clear()
    assert load_transition_ranker(model_path) is not None
