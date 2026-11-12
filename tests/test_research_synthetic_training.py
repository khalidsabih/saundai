from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai.training.research_synthetic import (
    create_research_synthetic_snapshot,
    validate_research_synthetic_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    root.mkdir()
    sources = [{"source_id": "paper-1", "title": "Paper", "year": 2026}]
    manifest = {
        "schema_version": "1.0",
        "generator_version": "test",
        "seed": 1,
        "total_examples": 5,
        "counts": {
            "transition_priors": 1,
            "switch_points": 1,
            "functional_structure": 1,
            "time_aware_set": 1,
            "hard_negatives": 1,
        },
        "policy": {
            "raw_audio_included": False,
            "human_preference": False,
            "production_authority": False,
        },
        "sources": sources,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    provenance = {
        "confidence_tier": "B",
        "human_preference": False,
        "production_authority": False,
        "label_source": "research_synthetic_weak_supervision",
        "source_ids": ["paper-1"],
    }
    _write_jsonl(root / "transition_priors.jsonl", [{
        "schema_version": "1.0",
        "example_id": "transition-1",
        "task": "transition_prior",
        "context": {"tempo_adjustment_ratio": 0.03},
        "candidates": [{"candidate_id": "t1", "transition_beats": 32}],
        "label": {"technical_prior": "preferred_prior", "confidence": 0.82},
        "provenance": provenance,
    }])
    _write_jsonl(root / "switch_points.jsonl", [{
        "schema_version": "1.0",
        "example_id": "switch-1",
        "task": "switch_point_detection",
        "context": {"downbeat": True, "period_start": True, "energy_novelty": 0.9},
        "candidates": [{"candidate_id": "s1"}],
        "label": {"is_switch_point": True, "confidence": 0.9},
        "provenance": {**provenance, "confidence_tier": "A"},
    }])
    _write_jsonl(root / "functional_structure.jsonl", [{
        "schema_version": "1.0",
        "example_id": "structure-1",
        "task": "functional_segment",
        "context": {"energy": 0.9, "tension": 0.7, "percussive": True},
        "candidates": [{"candidate_id": "drop"}],
        "label": {"segment": "drop", "confidence": 0.72},
        "provenance": provenance,
    }])
    _write_jsonl(root / "time_aware_set.jsonl", [{
        "schema_version": "1.0",
        "example_id": "time-1",
        "task": "time_aware_next_track",
        "context": {
            "elapsed_seconds": 1800.0,
            "target_duration_seconds": 3600.0,
            "remaining_seconds": 1800.0,
            "progress": 0.5,
            "planned_peak_seconds": 2400.0,
            "in_closing_window": False,
            "ending_energy_target": 0.4,
            "journey_energy_target": 0.75,
        },
        "candidates": [
            {"candidate_id": "a", "energy": 0.74, "duration_seconds": 300.0, "phrase_fit": 0.9},
            {"candidate_id": "b", "energy": 0.45, "duration_seconds": 280.0, "phrase_fit": 0.8},
        ],
        "label": {"choice": "a", "ranking": ["a", "b"], "confidence": 0.62},
        "provenance": {**provenance, "confidence_tier": "C"},
    }])
    _write_jsonl(root / "hard_negatives.jsonl", [{
        "schema_version": "1.0",
        "example_id": "negative-1",
        "task": "hard_negative",
        "context": {"failure_family": "tempo_warp_extreme"},
        "candidates": [{"candidate_id": "n1", "tempo_adjustment_ratio": 0.25}],
        "label": {"choice": "neither", "confidence": 0.93},
        "provenance": {**provenance, "confidence_tier": "A"},
    }])
    return root


def test_validate_research_synthetic_dataset(tmp_path: Path) -> None:
    source = _source(tmp_path)
    result = validate_research_synthetic_dataset(source)

    assert result.total_examples == 5
    assert result.counts["time_aware_next_track"] == 1
    assert result.source_ids == ("paper-1",)


def test_import_creates_grouped_model_ready_snapshot(tmp_path: Path) -> None:
    source = _source(tmp_path)
    workspace = tmp_path / "workspace"
    snapshot = create_research_synthetic_snapshot(workspace, source, name="test-v1")

    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["shadow_training_only"] is True
    assert manifest["policy"]["production_authority"] is False
    assert manifest["outputs"]["next_track"]["rows"] == 2

    rows = [
        json.loads(line)
        for line in (snapshot / "next-track.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["group"] for row in rows} == {"time-1"}
    assert len({row["split"] for row in rows}) == 1
    assert sorted(row["target"] for row in rows) == [0.0, 1.0]
    assert all(row["rights_cleared"] is False for row in rows)
    assert all(row["production_authority"] is False for row in rows)


def test_validator_rejects_production_authority_claim(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / "hard_negatives.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["provenance"]["production_authority"] = True
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="authority"):
        validate_research_synthetic_dataset(source)


def test_validator_rejects_local_audio_paths(tmp_path: Path) -> None:
    source = _source(tmp_path)
    path = source / "transition_priors.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["candidates"][0]["audio_path"] = "/tmp/private.wav"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="forbidden"):
        validate_research_synthetic_dataset(source)


def test_import_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    source = _source(tmp_path)
    workspace = tmp_path / "workspace"
    first = create_research_synthetic_snapshot(workspace, source, name="same")
    second = create_research_synthetic_snapshot(workspace, source, name="same")

    assert first == second
    assert (workspace / "datasets" / "research-synthetic" / "LATEST").read_text().strip() == first.name


def test_next_track_training_respects_explicit_splits_and_reports_ranking(tmp_path: Path) -> None:
    source = _source(tmp_path)
    # Expand the one time-aware scenario into enough grouped scenarios to exercise
    # train/validation/test without falling back to row-level splitting.
    rows = []
    provenance = {
        "confidence_tier": "C",
        "human_preference": False,
        "production_authority": False,
        "label_source": "research_synthetic_weak_supervision",
        "source_ids": ["paper-1"],
    }
    for index in range(30):
        rows.append({
            "schema_version": "1.0",
            "example_id": f"time-{index:03d}",
            "task": "time_aware_next_track",
            "context": {
                "elapsed_seconds": float(index * 60),
                "target_duration_seconds": 3600.0,
                "remaining_seconds": float(max(0, 3600 - index * 60)),
                "progress": index / 30.0,
                "planned_peak_seconds": 2400.0,
                "in_closing_window": index >= 25,
                "ending_energy_target": 0.4,
                "journey_energy_target": 0.4 + (index / 100.0),
            },
            "candidates": [
                {"candidate_id": "a", "energy": 0.8, "duration_seconds": 300.0, "phrase_fit": 0.95},
                {"candidate_id": "b", "energy": 0.6, "duration_seconds": 300.0, "phrase_fit": 0.85},
                {"candidate_id": "c", "energy": 0.4, "duration_seconds": 300.0, "phrase_fit": 0.75},
                {"candidate_id": "d", "energy": 0.2, "duration_seconds": 300.0, "phrase_fit": 0.65},
            ],
            "label": {
                "choice": "a",
                "ranking": ["a", "b", "c", "d"],
                "confidence": 0.7,
            },
            "provenance": provenance,
        })
    _write_jsonl(source / "time_aware_set.jsonl", rows)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["counts"]["time_aware_set"] = len(rows)
    manifest["total_examples"] = 4 + len(rows)
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    workspace = tmp_path / "workspace"
    snapshot = create_research_synthetic_snapshot(workspace, source, name="split-test")

    from djai.training.research_synthetic import train_research_synthetic_shadow

    result = train_research_synthetic_shadow(workspace, dataset_id=snapshot.name)
    model = json.loads((workspace / result["model_path"]).read_text(encoding="utf-8"))
    pairwise = json.loads((workspace / result["pairwise_model_path"]).read_text(encoding="utf-8"))
    baseline = json.loads((workspace / result["baseline_model_path"]).read_text(encoding="utf-8"))
    metrics = model["metrics"]

    assert model["trainer"] == "mlp_weighted_pairwise_ranknet"
    assert pairwise["trainer"] == "weighted_pairwise_logistic_ranknet"
    assert baseline["trainer"] == "scalar_logistic_baseline"
    assert model["shadow_only"] is True
    assert model["authority"] == 0.0
    assert "comparison" in result
    assert (
        workspace / "config" / "research-synthetic-models" / "NEXT_TRACK_RESEARCH_LATEST"
    ).read_text().strip() == Path(result["model_path"]).name
    assert (
        workspace / "config" / "research-synthetic-models" / "NEXT_TRACK_RESEARCH_PAIRWISE_LATEST"
    ).read_text().strip() == Path(result["pairwise_model_path"]).name
    assert (
        workspace / "config" / "research-synthetic-models" / "NEXT_TRACK_RESEARCH_BASELINE_LATEST"
    ).read_text().strip() == Path(result["baseline_model_path"]).name

    imported_rows = [
        json.loads(line)
        for line in (snapshot / "next-track.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_train = sum(1 for row in imported_rows if row["split"] == "train")
    expected_validation = sum(1 for row in imported_rows if row["split"] == "validation")
    expected_test = sum(1 for row in imported_rows if row["split"] == "test")

    assert model["training_examples"] == expected_train
    assert model["validation_examples"] == expected_validation
    assert model["test_examples"] == expected_test
    assert metrics["validation_ranking"]["groups"] == expected_validation // 4
    assert metrics["test_ranking"]["groups"] == expected_test // 4
    assert metrics["validation_ranking"]["top1_accuracy"] is not None
    assert metrics["test_ranking"]["ndcg"] is not None
    assert "pairwise_accuracy" in result["comparison"]["test"]
    assert "mlp_pairwise" in result["comparison"]["validation"]
    assert "mlp_pairwise" in result["comparison"]["test"]["top1_accuracy"]
    assert "mlp_vs_pairwise" in result["comparison"]["test"]["ndcg"]
