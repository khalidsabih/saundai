from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from djai.intelligence.preferences import append_preference
from djai.intelligence.transition_ranker import MINIMUM_COMPARISONS
from djai.training.lab_dataset import (
    create_training_snapshot,
    train_shadow_ranker,
    training_lab_status,
)
from djai.training.neural_heads import train_neural_head


def _candidate(signature: str, score: float, track_a: Path, track_b: Path, *, creative: bool = False) -> dict:
    return {
        "execution_signature": signature,
        "track_a": str(track_a),
        "track_b": str(track_b),
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
        "rendered_transition_embedding": [score] + [0.0] * 23,
        "rendered_transition_embedding_backend": "statistical",
        "set_context_score": 0.7,
        "rejection_penalty": 0.0,
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
        "measurement": {
            "overall": score,
            "loudness_continuity": score,
            "spectral_continuity": score,
            "bass_cleanliness": score,
            "phase_safety": score,
            "artifact_safety": score,
            "silence_safety": score,
            "rhythm_coherence": score,
            "transient_clarity": score,
            "groove_continuity": score,
            "harmonic_continuity": score,
            "energy_intent_fit": score,
            "drop_payoff": score,
            "musical_overall": score,
        },
    }


def _write_tone(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 8000
    t = np.arange(sr // 4, dtype=np.float32) / sr
    audio = 0.1 * np.sin(2.0 * np.pi * frequency * t)
    sf.write(path, audio, sr)


def _populate_preferences(workspace: Path, count: int = MINIMUM_COMPARISONS + 12) -> Path:
    ratings = workspace / "ratings" / "transition-preferences.jsonl"
    tracks: list[tuple[Path, Path]] = []
    for pair in range(6):
        a = workspace / "audio" / f"pair-{pair}-a.wav"
        b = workspace / "audio" / f"pair-{pair}-b.wav"
        _write_tone(a, 180 + pair * 10)
        _write_tone(b, 260 + pair * 10)
        tracks.append((a, b))
    for index in range(count):
        track_a, track_b = tracks[index % len(tracks)]
        high = _candidate(f"high-{index}", 0.88, track_a, track_b, creative=index % 2 == 0)
        low = _candidate(f"low-{index}", 0.32, track_a, track_b, creative=index % 2 != 0)
        append_preference(
            candidate_a=high if index % 2 == 0 else low,
            candidate_b=low if index % 2 == 0 else high,
            choice="a" if index % 2 == 0 else "b",
            listener="training-expert",
            context={"session_id": f"lab-test-{index // 2}", "candidate_labels": ["A", "B"], "reasons": ["flow"]},
            path=ratings,
        )
    return ratings


def test_training_snapshot_is_immutable_model_ready_and_audio_free(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "workspace"
    _populate_preferences(workspace)
    monkeypatch.setenv("SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED", "0")

    snapshot = create_training_snapshot(workspace, name="unit-test")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["immutable"] is True
    assert manifest["raw_audio_bundled"] is False
    assert manifest["counts"]["decisive_preferences"] >= MINIMUM_COMPARISONS
    assert manifest["counts"]["tracks"] >= 2
    assert manifest["production_training_ready"] is False
    assert (snapshot / "pairwise_matrix.npz").exists()
    assert (snapshot / "rendered_judge.jsonl").exists()
    assert manifest["counts"]["rendered_judge_examples"] >= 2 * MINIMUM_COMPARISONS
    assert manifest["counts"]["rendered_judge_embedding_examples"] == manifest["counts"]["rendered_judge_examples"]
    assert not list(snapshot.rglob("*.wav"))

    rendered_rows = [
        json.loads(line)
        for line in (snapshot / "rendered_judge.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rendered_rows
    assert "render_embedding_00" in rendered_rows[0]["features"]
    assert rendered_rows[0]["group"].startswith("pair-")

    matrix = np.load(snapshot / "pairwise_matrix.npz")
    assert matrix["x"].shape[0] >= MINIMUM_COMPARISONS
    assert matrix["x"].shape[1] == len(matrix["feature_names"])


def test_shadow_training_never_promotes_live_model(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _populate_preferences(workspace)
    snapshot = create_training_snapshot(workspace, name="shadow")

    result = train_shadow_ranker(workspace, snapshot_id=snapshot.name)
    model = workspace / result["model_path"]
    payload = json.loads(model.read_text(encoding="utf-8"))

    assert result["shadow_only"] is True
    assert result["comparison_count"] >= MINIMUM_COMPARISONS
    assert payload["training_dataset_id"] == snapshot.name
    assert payload["promoted_to_live"] is False
    assert not (workspace / "config" / "transition-ranker.json").exists()
    assert model.parent.name == "training-lab-models"


def test_training_lab_status_exposes_first_train_progress(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _populate_preferences(workspace, count=MINIMUM_COMPARISONS + 2)

    status = training_lab_status(workspace)
    assert status["preferences"]["decisive"] >= MINIMUM_COMPARISONS
    assert status["preferences"]["choice_a"] > 0
    assert status["preferences"]["choice_b"] > 0
    assert status["train_ready"] is True
    assert status["policy"]["raw_audio_bundled_in_snapshots"] is False
    assert status["policy"]["shadow_training_changes_live_model"] is False


def test_rendered_judge_snapshot_preserves_tie_and_neither_training_semantics(tmp_path: Path):
    workspace = tmp_path / "workspace"
    ratings = _populate_preferences(workspace)
    track_a = workspace / "audio" / "pair-0-a.wav"
    track_b = workspace / "audio" / "pair-0-b.wav"
    a = _candidate("tie-a", 0.72, track_a, track_b)
    b = _candidate("tie-b", 0.70, track_a, track_b, creative=True)
    append_preference(candidate_a=a, candidate_b=b, choice="tie", path=ratings)
    append_preference(candidate_a=a, candidate_b=b, choice="neither", path=ratings)

    snapshot = create_training_snapshot(workspace, name="rendered-labels")
    rows = [json.loads(line) for line in (snapshot / "rendered_judge.jsonl").read_text().splitlines() if line.strip()]

    ties = [row for row in rows if row["label_source"] == "pairwise_tie"]
    neither = [row for row in rows if row["label_source"] == "absolute_rejection"]
    assert len(ties) == 2 and all(row["target"] == 0.5 for row in ties)
    assert len(neither) == 2 and all(row["target"] == 0.0 for row in neither)
    assert min(row["weight"] for row in neither) > max(row["weight"] for row in ties)


def test_rendered_judge_snapshot_trains_governed_shadow_head(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _populate_preferences(workspace)
    snapshot = create_training_snapshot(workspace, name="rendered-head")
    output = workspace / "config" / "rendered-dj-judge-shadow.json"

    train_neural_head(
        "rendered_judge",
        snapshot / "rendered_judge.jsonl",
        output,
        require_rights_cleared=False,
        iterations=120,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["model_id"] == "rendered_dj_judge"
    assert "render_embedding_00" in payload["feature_names"]
    assert payload["metrics"]["validation_examples"] > 0
