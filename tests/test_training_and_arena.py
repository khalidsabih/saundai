from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from djai.evaluation.arena import ArenaVote, append_arena_vote, arena_report
from djai.training.provenance import audit_dataset_registry
from djai.training.synthetic_failures import corrupt_audio


def test_redacted_source_package_has_only_reviewed_production_datasets():
    registry = json.loads(Path("datasets/registry.json").read_text(encoding="utf-8"))
    assert registry["policy"]["production_default_deny"] is True

    report = audit_dataset_registry("datasets/registry.json", lane="production")

    assert report["ready"] is True
    assert report["blockers"] == []

    for dataset in report["datasets"]:
        assert dataset["rights_reviewed"] is True
        assert dataset["commercial_training_allowed"] is True
        assert dataset["derivative_models_allowed"] is True
        assert dataset["audio_bundled"] is False


def test_registry_blocks_uncleared_production_source(tmp_path: Path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"datasets": [{
        "id": "unknown", "name": "Unknown", "lane": "production",
        "rights_reviewed": False, "commercial_training_allowed": None,
        "derivative_models_allowed": None,
    }]}))
    report = audit_dataset_registry(path, lane="production")
    assert report["ready"] is False
    assert len(report["blockers"]) >= 2


def test_synthetic_failure_generator_changes_audio():
    sr = 22050
    audio = np.zeros((sr, 2), dtype=np.float32)
    audio[:, 0] = 0.35 * np.sin(2 * np.pi * 80 * np.arange(sr) / sr)
    audio[:, 1] = audio[:, 0]
    damaged = corrupt_audio(audio, sr, "phase_inversion", 0.8)
    assert damaged.shape == audio.shape
    assert not np.allclose(damaged, audio)
    assert np.allclose(damaged[:, 0], audio[:, 0])
    assert np.allclose(damaged[:, 1], -audio[:, 1])


def test_arena_requires_clear_held_out_win_and_no_regression(tmp_path: Path):
    path = tmp_path / "arena.jsonl"
    for index in range(70):
        choice = "challenger" if index < 55 else "champion"
        append_arena_vote(ArenaVote(
            champion="3.1.2", challenger="4.0.0", choice=choice,
            evaluator=f"expert-{index % 5}", context_id=f"pair-{index}",
        ), path)
    report = arena_report(path, champion="3.1.2", challenger="4.0.0", minimum_decisive=50)
    assert report["challenger_effective_win_rate"] > 0.60
    assert report["deployment_ready"] is True
    append_arena_vote(ArenaVote(
        champion="3.1.2", challenger="4.0.0", choice="challenger",
        evaluator="expert-x", context_id="regression", technical_regression=True,
    ), path)
    report = arena_report(path, champion="3.1.2", challenger="4.0.0", minimum_decisive=50)
    assert report["deployment_ready"] is False
