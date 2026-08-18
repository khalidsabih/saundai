from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from djai.training.dataset_builder import (
    create_bootstrap_corpus,
    dataset_builder_status,
    generate_weak_labels,
    import_edm_cue_normalized,
    import_fma_metadata,
    import_listenbrainz_sequences,
    import_unmixdb_ground_truth,
    train_bootstrap_shadow_models,
)
from djai.training.source_catalog import get_source, source_catalog


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]




def test_rc11_1_edm_cue_import_creates_track_grouped_held_out_test_split(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    rows = []
    for index in range(120):
        rows.append({
            "track_id": f"track-{index}",
            "cue_points_seconds": [16.0, 32.0],
            "bpm": 120.0,
            "grid_offset_seconds": 0.0,
            "initial_beat_number": 1,
            "split": "train" if index < 100 else "validation",
        })
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "records.jsonl"
    import_edm_cue_normalized(annotations, output)
    normalized = _rows(output)
    splits = {row["split"] for row in normalized}
    assert {"train", "validation", "test"} <= splits
    by_group: dict[str, set[str]] = {}
    for row in normalized:
        by_group.setdefault(row["group"], set()).add(row["split"])
        assert row["payload"]["source_split"] in {"train", "validation", "test"}
    assert all(len(values) == 1 for values in by_group.values())


def test_rc11_1_duration_labels_form_grouped_ranking_candidates(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    payload = {
        "record_id": "transition-1",
        "source_dataset": "unmixdb",
        "source_record_id": "1",
        "kind": "transition_annotation",
        "split": "test",
        "group": "mix-1",
        "label_source": "ground_truth_overlap",
        "label_confidence": 0.95,
        "rights_cleared": False,
        "payload": {"duration_seconds": 16.0},
        "schema_version": "1.0",
    }
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    weak = generate_weak_labels(records, tmp_path / "weak.jsonl")
    rows = _rows(weak)
    assert len(rows) == 5
    assert len({row["ranking_group"] for row in rows}) == 1
    assert sum(row["target"] >= 0.5 for row in rows) == 1
    assert all(row["feature_space"] == "external-transition-duration-v4-ranking" for row in rows)
    ratios = sorted(row["corruption"]["duration_ratio"] for row in rows if row.get("corruption"))
    assert ratios == [0.75, 0.875, 1.125, 1.25]

def test_source_catalog_is_default_deny_and_marks_audio_downloads() -> None:
    catalog = source_catalog()
    assert catalog["policy"]["production_default_deny"] is True
    ids = {item["id"] for item in catalog["sources"]}
    assert {"edm-cue", "unmixdb", "fma", "listenbrainz", "medley2k"} <= ids
    fma = get_source("fma")
    assert any(asset.kind == "audio" for asset in fma.assets)
    assert fma.lane == "research"


def test_fma_metadata_import_preserves_track_license_without_auto_clear(tmp_path: Path) -> None:
    archive = tmp_path / "fma_metadata.zip"
    csv_text = (
        ",track,track,artist,set,set\n"
        ",title,license,name,split,subset\n"
        "1,Alpha,https://creativecommons.org/licenses/by/4.0/,Artist A,training,small\n"
        "2,Beta,https://creativecommons.org/licenses/by-nc/4.0/,Artist B,validation,small\n"
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("fma_metadata/tracks.csv", csv_text)
    output = tmp_path / "records.jsonl"
    import_fma_metadata(archive, output)
    rows = _rows(output)
    assert len(rows) == 2
    assert rows[0]["payload"]["license"].startswith("https://creativecommons.org/")
    assert rows[0]["rights_cleared"] is False
    assert rows[1]["split"] == "validation"


def test_listenbrainz_import_hashes_user_and_creates_only_adjacent_edges(tmp_path: Path) -> None:
    source = tmp_path / "1.listens"
    listens = [
        {"user_name": "real-user", "listened_at": 100, "track_metadata": {"artist_name": "A", "track_name": "One", "additional_info": {"recording_mbid": "r1"}}},
        {"user_name": "real-user", "listened_at": 200, "track_metadata": {"artist_name": "B", "track_name": "Two", "additional_info": {"recording_mbid": "r2"}}},
        {"user_name": "real-user", "listened_at": 300, "track_metadata": {"artist_name": "C", "track_name": "Three", "additional_info": {"recording_mbid": "r3"}}},
    ]
    source.write_text("".join(json.dumps(item) + "\n" for item in listens), encoding="utf-8")
    output = tmp_path / "records.jsonl"
    import_listenbrainz_sequences(source, output)
    rows = _rows(output)
    assert len(rows) == 2
    text = output.read_text(encoding="utf-8")
    assert "real-user" not in text
    assert rows[0]["payload"]["from_track_id"] == "r1"
    assert rows[0]["payload"]["to_track_id"] == "r2"
    assert rows[0]["label_confidence"] < 0.5


def test_listenbrainz_import_accepts_current_incremental_timestamp_schema(tmp_path: Path) -> None:
    source = tmp_path / "listenbrainz-listens-dump" / "listens" / "2026" / "7.listens"
    source.parent.mkdir(parents=True)
    listens = [
        {
            "user_id": 140328,
            "user_name": "should-not-leak",
            "timestamp": 300,
            "track_metadata": {"artist_name": "C", "track_name": "Three", "additional_info": {}},
            "recording_msid": "r3",
        },
        {
            "user_id": 140328,
            "user_name": "should-not-leak",
            "timestamp": 100,
            "track_metadata": {"artist_name": "A", "track_name": "One", "additional_info": {}},
            "recording_msid": "r1",
        },
        {
            "user_id": 140328,
            "user_name": "should-not-leak",
            "timestamp": 200,
            "track_metadata": {"artist_name": "B", "track_name": "Two", "additional_info": {}},
            "recording_msid": "r2",
        },
    ]
    source.write_text("".join(json.dumps(item) + "\n" for item in listens), encoding="utf-8")
    noise = tmp_path / "listenbrainz-listens-dump" / "metadata" / "recordings_cache.jsonl"
    noise.parent.mkdir(parents=True)
    noise.write_text(json.dumps({"user_id": 999, "timestamp": 150, "track_metadata": {"artist_name": "Noise", "track_name": "Ignore"}}) + "\n", encoding="utf-8")
    output = tmp_path / "records.jsonl"
    import_listenbrainz_sequences(tmp_path / "listenbrainz-listens-dump", output)
    rows = _rows(output)

    assert len(rows) == 2
    assert [(row["payload"]["from_track_id"], row["payload"]["to_track_id"]) for row in rows] == [("r1", "r2"), ("r2", "r3")]
    text = output.read_text(encoding="utf-8")
    assert "should-not-leak" not in text
    assert "140328" not in text


def test_listenbrainz_import_deduplicates_overlapping_files_before_building_edges(tmp_path: Path) -> None:
    root = tmp_path / "listenbrainz" / "listens"
    root.mkdir(parents=True)
    first = root / "1.listens"
    second = root / "2.listens"
    listen_a = {
        "user_name": "same-user",
        "timestamp": 100,
        "recording_msid": "r1",
        "track_metadata": {"artist_name": "A", "track_name": "One"},
    }
    listen_b = {
        "user_name": "same-user",
        "timestamp": 200,
        "recording_msid": "r2",
        "track_metadata": {"artist_name": "B", "track_name": "Two"},
    }
    listen_c = {
        "user_name": "same-user",
        "timestamp": 300,
        "recording_msid": "r3",
        "track_metadata": {"artist_name": "C", "track_name": "Three"},
    }
    first.write_text(json.dumps(listen_b) + "\n" + json.dumps(listen_a) + "\n", encoding="utf-8")
    second.write_text(json.dumps(listen_b) + "\n" + json.dumps(listen_c) + "\n", encoding="utf-8")

    output = tmp_path / "records.jsonl"
    import_listenbrainz_sequences(root.parent, output)
    rows = _rows(output)

    assert [(row["payload"]["from_track_id"], row["payload"]["to_track_id"]) for row in rows] == [
        ("r1", "r2"),
        ("r2", "r3"),
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert any("retained 3 unique" in note for note in manifest["notes"])


def test_rc11_6c_cold_target_known_artist_gets_train_only_artist_backoff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    corpus = workspace / "datasets" / "bootstrap-corpora" / "corpus-cold-target"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text(json.dumps({"corpus_id": corpus.name}), encoding="utf-8")

    rows: list[dict] = []

    def add(index: int, split: str, source_track: str, target_track: str, source_artist: str, target_artist: str) -> None:
        rows.append({
            "record_id": f"edge-{index}",
            "source_dataset": "listenbrainz",
            "source_record_id": str(index),
            "kind": "sequence_edge",
            "split": split,
            "group": f"user-{index}",
            "label_source": "behavioral_sequence",
            "label_confidence": 0.35,
            "rights_cleared": False,
            "payload": {
                "from_track_id": source_track,
                "to_track_id": target_track,
                "from_artist": source_artist,
                "to_artist": target_artist,
            },
            "schema_version": "1.0",
        })

    index = 0
    for n in range(5):
        add(index, "train", f"source-{n}", f"dest-x-{n}", "Source Artist", "Destination X")
        index += 1
    add(index, "train", "source-y", "dest-y-source", "Source Artist", "Destination Y")
    index += 1
    for n in range(10):
        add(index, "train", f"other-{n}", f"dest-y-{n}", "Other Artist", "Destination Y")
        index += 1
    add(index, "validation", "heldout-source", "cold-track-x", "Source Artist", "Destination X")
    index += 1
    add(index, "validation", "heldout-source-2", "cold-track-z", "Source Artist", "Destination Z")

    (corpus / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = train_bootstrap_shadow_models(workspace, corpus_id=corpus.name)
    generalizer = next(
        item for item in report["models"]
        if item["model_id"] == "next_track_sequence_generalizer" and item["status"] == "trained"
    )
    artifact = json.loads((workspace / generalizer["path"]).read_text(encoding="utf-8"))
    cold = artifact["cold_target_intelligence"]["evaluation"]["subgroups"]["validation"]

    assert cold["cold_target_known_artist"]["examples"] == 1
    assert cold["cold_target_known_artist"]["model"]["hit_at_1"] == 1.0
    assert cold["cold_target_known_artist"]["global_target_artist_baseline"]["hit_at_1"] == 0.0
    assert cold["cold_target_unknown_artist"]["examples"] == 1
    assert cold["cold_target_unknown_artist"]["model"]["hit_at_1"] == 0.0

    runtime_index = workspace / generalizer["path"]
    runtime_index = runtime_index.parent / "sequence-runtime.sqlite"
    assert runtime_index.exists()
    connection = sqlite3.connect(runtime_index)
    try:
        artist_key = "artist-" + hashlib.sha256("source artist".encode("utf-8")).hexdigest()[:24]
        target_key = "artist-" + hashlib.sha256("destination x".encode("utf-8")).hexdigest()[:24]
        row = connection.execute(
            "SELECT count,outgoing_count FROM artist_pair WHERE source_artist_key=? AND target_artist_key=?",
            (artist_key, target_key),
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    assert row[0] == 5
    assert row[1] == 6


def test_unmixdb_import_and_weak_labels_keep_ground_truth_separate_from_synthetic(tmp_path: Path) -> None:
    labels = tmp_path / "mix.labels.txt"
    labels.write_text(
        "0.0\t20.0\t1 track-a\n"
        "16.0\t36.0\t2 track-b\n"
        "16.0\t20.0\t1 cue-out\n"
        "16.0\t20.0\t2 cue-in\n",
        encoding="utf-8",
    )
    records = tmp_path / "records.jsonl"
    import_unmixdb_ground_truth(labels, records)
    normalized = _rows(records)
    assert any(row["kind"] == "transition_annotation" for row in normalized)
    weak = generate_weak_labels(records, tmp_path / "weak.jsonl")
    weak_rows = _rows(weak)
    assert weak_rows
    assert {row["label_source"] for row in weak_rows} >= {"ground_truth_overlap", "synthetic_corruption"}
    assert all(row["feature_space"] == "external-transition-duration-v4-ranking" for row in weak_rows)
    assert all("duration_ratio_to_reference" not in row["features"] for row in weak_rows)


def test_bootstrap_corpus_is_immutable_audio_free_and_production_filters_uncleared(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = workspace / "datasets" / "external" / "unmixdb"
    external.mkdir(parents=True)
    records = external / "records.jsonl"
    payload = {
        "record_id": "transition-1",
        "source_dataset": "unmixdb",
        "source_record_id": "1",
        "kind": "transition_annotation",
        "split": "train",
        "group": "g1",
        "label_source": "ground_truth_overlap",
        "label_confidence": 0.95,
        "rights_cleared": False,
        "payload": {"duration_seconds": 8.0},
        "schema_version": "1.0",
    }
    records.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (external / "manifest.json").write_text(json.dumps({"rights_cleared": False, "counts": {"transition_annotation": 1}}), encoding="utf-8")

    corpus = create_bootstrap_corpus(workspace, name="research")
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["immutable"] is True
    assert manifest["raw_audio_bundled"] is False
    assert manifest["counts"]["records"] == 1
    assert manifest["counts"]["weak_examples"] > 0
    assert not list(corpus.rglob("*.wav"))
    assert dataset_builder_status(workspace)["counts"]["normalized_records"] == 1

    with pytest.raises(ValueError, match="No rights-cleared"):
        create_bootstrap_corpus(workspace, name="production", production=True)



def test_bootstrap_shadow_training_is_research_only_and_does_not_touch_live_models(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = workspace / "datasets" / "external" / "mixed"
    external.mkdir(parents=True)
    records = external / "records.jsonl"
    rows: list[dict] = []
    splits = ("train", "train", "validation", "test")
    for index in range(32):
        split = splits[index % len(splits)]
        rows.append({
            "record_id": f"cue-{index}",
            "source_dataset": "edm-cue",
            "source_record_id": str(index),
            "kind": "cue_annotation",
            "split": split,
            "group": f"cue-group-{index}",
            "label_source": "expert_annotation",
            "label_confidence": 1.0,
            "rights_cleared": False,
            "payload": {"cue_seconds": 16.0 + index * 2.0, "bpm": 120.0 + (index % 8)},
            "schema_version": "1.0",
        })
        rows.append({
            "record_id": f"transition-{index}",
            "source_dataset": "unmixdb",
            "source_record_id": str(index),
            "kind": "transition_annotation",
            "split": split,
            "group": f"transition-group-{index}",
            "label_source": "ground_truth_overlap",
            "label_confidence": 0.95,
            "rights_cleared": False,
            "payload": {"duration_seconds": 4.0 + float(index % 8)},
            "schema_version": "1.0",
        })
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (external / "manifest.json").write_text(
        json.dumps({"rights_cleared": False, "counts": {"cue_annotation": 32, "transition_annotation": 32}}),
        encoding="utf-8",
    )

    corpus = create_bootstrap_corpus(workspace, name="shadow-test")
    report = train_bootstrap_shadow_models(workspace, corpus_id=corpus.name)

    assert report["shadow_only"] is True
    assert report["live_models_changed"] is False
    trained = {item["model_id"]: item for item in report["models"] if item["status"] == "trained"}
    assert {"cue_point_prior", "transition_duration_structure"} <= trained.keys()
    assert trained["cue_point_prior"]["evaluation_kind"] == "grouped_pairwise_ranking"
    assert trained["cue_point_prior"]["primary_metric"] == "top1_accuracy"
    assert trained["cue_point_prior"]["metrics"]["test"]["groups"] > 0
    assert "top1_accuracy" in trained["cue_point_prior"]["metrics"]["test"]
    assert trained["transition_duration_structure"]["evaluation_kind"] == "grouped_pairwise_ranking"
    assert trained["transition_duration_structure"]["primary_metric"] == "top1_accuracy"
    assert trained["transition_duration_structure"]["metrics"]["test"]["groups"] > 0
    assert "top1_accuracy" in trained["transition_duration_structure"]["metrics"]["test"]
    model_path = workspace / trained["cue_point_prior"]["path"]
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["research_shadow"] is True
    assert model["promotable_to_live"] is False
    assert "baseline" in model
    assert "accuracy_improvement_vs_baseline" in model
    assert "primary_improvement_vs_baseline" in model
    assert "offset_from_expert_seconds" not in model["feature_names"]
    assert not (workspace / "config" / "transition-ranker.json").exists()



def test_rc11_cue_labels_are_grouped_and_grid_aligned_hard_negatives(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    normalized = [
        {
            "record_id": "cue-a",
            "source_dataset": "edm-cue",
            "source_record_id": "track:0",
            "kind": "cue_annotation",
            "split": "train",
            "group": "edm-cue:track",
            "label_source": "expert_annotation",
            "label_confidence": 1.0,
            "rights_cleared": False,
            "payload": {"cue_seconds": 32.0, "bpm": 120.0, "grid_offset_seconds": 0.0, "initial_beat_number": 1},
            "schema_version": "1.0",
        },
        {
            "record_id": "cue-b",
            "source_dataset": "edm-cue",
            "source_record_id": "track:1",
            "kind": "cue_annotation",
            "split": "train",
            "group": "edm-cue:track",
            "label_source": "expert_annotation",
            "label_confidence": 1.0,
            "rights_cleared": False,
            "payload": {"cue_seconds": 40.0, "bpm": 120.0, "grid_offset_seconds": 0.0, "initial_beat_number": 1},
            "schema_version": "1.0",
        },
    ]
    records.write_text("".join(json.dumps(row) + "\n" for row in normalized), encoding="utf-8")
    weak = generate_weak_labels(records, tmp_path / "weak.jsonl")
    rows = _rows(weak)
    cue_a = [row for row in rows if row["ranking_group"] == "cue-a"]
    assert sum(row["target"] >= 0.5 for row in cue_a) == 1
    assert any(row["candidate_role"] == "hard_grid_aligned_negative" for row in cue_a)
    assert all(row["feature_space"] == "external-cue-structural-v3-ranking" for row in cue_a)
    # 40s is another expert cue on the same track and must not be synthesized as a negative for 32s.
    assert all(abs(row["features"]["candidate_time_seconds"] - 40.0) > 0.01 or row["target"] >= 0.5 for row in cue_a)


def test_dataset_status_persists_latest_benchmark_details(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    external = workspace / "datasets" / "external" / "mixed"
    external.mkdir(parents=True)
    rows: list[dict] = []
    for index in range(48):
        split = "train" if index < 32 else ("validation" if index < 40 else "test")
        rows.append({
            "record_id": f"cue-{index}",
            "source_dataset": "edm-cue",
            "source_record_id": str(index),
            "kind": "cue_annotation",
            "split": split,
            "group": f"track-{index}",
            "label_source": "expert_annotation",
            "label_confidence": 1.0,
            "rights_cleared": False,
            "payload": {"cue_seconds": 32.0 + index, "bpm": 120.0},
            "schema_version": "1.0",
        })
    records = external / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (external / "manifest.json").write_text(json.dumps({"rights_cleared": False, "counts": {"cue_annotation": len(rows)}}), encoding="utf-8")
    corpus = create_bootstrap_corpus(workspace, name="persistent")
    train_bootstrap_shadow_models(workspace, corpus_id=corpus.name)
    status = dataset_builder_status(workspace)
    assert status["latest_benchmark"]["corpus_id"] == corpus.name
    cue = next(item for item in status["latest_benchmark"]["models"] if item["model_id"] == "cue_point_prior")
    assert cue["evaluation_kind"] == "grouped_pairwise_ranking"
    assert "top1_accuracy" in cue["metrics"]["test"]
