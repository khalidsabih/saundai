from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "dataset-benchmark-registry.py"
    spec = importlib.util.spec_from_file_location("dataset_benchmark_registry", path)
    assert spec and spec.loader
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


def seed(tmp_path):
    tool = module()
    manifest = tmp_path / "private-dataset-split.json"
    manifest.write_text(json.dumps({"split": "test", "recordings": ["a", "b"]}), encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({
        "schema_version": "1.0",
        "sources": [{"id": "local", "rights_status": "unreviewed"}],
        "benchmarks": [
            {"id":"beat", "metric":"beat_f1", "direction":"maximize", "requires_human":False},
            {"id":"taste", "metric":"blind_win_rate", "direction":"maximize", "requires_human":True},
        ],
    }), encoding="utf-8")
    db = tmp_path / "runs.sqlite3"
    return tool, catalog, manifest, db


def record(tool, catalog, manifest, db, value=0.8, implementation="old", bench="beat",
           split="test", human="none", rights=True, safety=True):
    args = ["--catalog", str(catalog), "--db", str(db), "record",
            "--benchmark", bench, "--dataset", "local", "--split", split,
            "--manifest", str(manifest), "--source-commit", "abcdef1234567",
            "--implementation", implementation, "--value", str(value)]
    if safety:
        args.append("--safety-passed")
    if rights:
        args.append("--rights-confirmed")
    args.extend(["--human-evidence", human])
    return tool.main(args)


def test_catalog_is_valid_and_registry_writes_only_ledger(tmp_path):
    tool, catalog, manifest, db = seed(tmp_path)
    assert len(tool.catalog()["sources"]) >= 4
    assert record(tool, catalog, manifest, db) == 0
    assert manifest.read_text().startswith('{"split": "test"')
    with tool.connection(db) as connection:
        row = tool.select(connection, 1)
        assert row["dataset_fingerprint"] == tool.sha256(manifest)
        assert row["rights_confirmed"] == 1
    assert record(tool, catalog, manifest, db) == 0
    with tool.connection(db) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 1


def test_frozen_cohort_comparisons_and_champion_promotion(tmp_path):
    tool, catalog, manifest, db = seed(tmp_path)
    assert record(tool, catalog, manifest, db, 0.8) == 0
    assert record(tool, catalog, manifest, db, 0.9, "new") == 0
    with tool.connection(db) as connection:
        old, new = tool.select(connection, 1), tool.select(connection, 2)
        assert tool.assess(old, new, tool.catalog(catalog)["benchmarks"][0])["promotion_eligible"]
        assert not tool.assess(new, old, tool.catalog(catalog)["benchmarks"][0])["promotion_eligible"]
    assert tool.main(["--catalog", str(catalog), "--db", str(db), "promote", "--run-id", "1", "--initial"]) == 0
    assert tool.main(["--catalog", str(catalog), "--db", str(db), "promote", "--run-id", "2"]) == 0
    with tool.connection(db) as connection:
        assert tool.champion(connection, tool.select(connection, 2))["id"] == 2


def test_manifest_fingerprint_blocks_cross_dataset_comparison(tmp_path):
    tool, catalog, manifest, db = seed(tmp_path)
    assert record(tool, catalog, manifest, db, 0.8) == 0
    manifest.write_text(json.dumps({"split":"test","recordings":["different"]}), encoding="utf-8")
    assert record(tool, catalog, manifest, db, 0.9, "new") == 0
    with tool.connection(db) as connection:
        with pytest.raises(ValueError, match="incomparable"):
            tool.assess(tool.select(connection, 1), tool.select(connection, 2), tool.catalog(catalog)["benchmarks"][0])


def test_refuse_missing_rights_and_human_evidence_for_taste(tmp_path):
    tool, catalog, manifest, db = seed(tmp_path)
    with pytest.raises(ValueError, match="rights"):
        record(tool, catalog, manifest, db, rights=False)
    assert record(tool, catalog, manifest, db, 0.8, "candidate", "taste") == 0
    with pytest.raises(ValueError, match="human"):
        tool.main(["--catalog", str(catalog), "--db", str(db), "promote", "--run-id", "1", "--initial"])
    assert record(tool, catalog, manifest, db, 0.9, "reviewed", "taste", human="arena-report-1") == 0
    assert tool.main(["--catalog", str(catalog), "--db", str(db), "promote", "--run-id", "2", "--initial"]) == 0


def test_dev_result_cannot_be_champion_and_safety_failure_rejected_on_test(tmp_path):
    tool, catalog, manifest, db = seed(tmp_path)
    assert record(tool, catalog, manifest, db, split="development") == 0
    with pytest.raises(ValueError, match="test split"):
        tool.main(["--catalog", str(catalog), "--db", str(db), "promote", "--run-id", "1", "--initial"])
    with pytest.raises(ValueError, match="safety"):
        record(tool, catalog, manifest, db, 0.99, "unsafe", safety=False)
