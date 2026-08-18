from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from djai.brain.move_generator import CompleteMoveConfig
from djai.release_candidate import release_candidate_status


ROOT = Path(__file__).resolve().parents[1]


def _write_plan(root: Path, name: str, schema: str) -> None:
    target = root / "output" / name / "set-plan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"schema_version": schema}), encoding="utf-8")


def test_rc_promotes_complete_move_planner_and_compose_passes_authority(monkeypatch) -> None:
    monkeypatch.delenv("DJAI_PLANNER_MODE", raising=False)
    assert CompleteMoveConfig.from_environment().mode == "move_authoritative"
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "DJAI_PLANNER_MODE: ${DJAI_PLANNER_MODE:-move_authoritative}" in compose
    assert "DJAI_HEADPHONES: ${DJAI_HEADPHONES:-1}" in compose
    assert "DJAI_CRATE_MEMORY: ${DJAI_CRATE_MEMORY:-1}" in compose
    assert "/api/ready" in compose


def test_operational_preflight_preserves_legacy_plans_but_blocks_future_schema(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_HEADPHONES", "1")
    monkeypatch.setenv("DJAI_CRATE_MEMORY", "1")
    monkeypatch.setenv("DJAI_STRUCTURAL_DECK_OPS", "1")
    for name in ("output", "ratings", "accounts", "benchmarks"):
        (tmp_path / name).mkdir()

    _write_plan(tmp_path, "legacy", "7.0.0")
    status = release_candidate_status(tmp_path)
    assert status["operational_ready"] is True
    plans = status["upgrade_compatibility"]["plans"]
    assert plans["legacy_replan_required"] == 1
    assert plans["future_schema_blocked"] == 0

    _write_plan(tmp_path, "future", "9.0.0")
    status = release_candidate_status(tmp_path)
    assert status["operational_ready"] is False
    assert status["upgrade_compatibility"]["plans"]["future_schema_blocked"] == 1


def test_operational_preflight_blocks_unknown_crate_memory_schema(tmp_path: Path, monkeypatch) -> None:
    for name in ("output", "ratings", "accounts", "benchmarks"):
        (tmp_path / name).mkdir()
    memory = tmp_path / "ratings" / "crate-memory.sqlite3"
    with sqlite3.connect(memory) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata(key, value) VALUES ('schema_version', '99')")
    monkeypatch.setenv("DJAI_CRATE_MEMORY_PATH", str(memory))
    status = release_candidate_status(tmp_path)
    assert status["operational_ready"] is False
    assert status["upgrade_compatibility"]["crate_memory"]["compatible"] is False


def test_strict_rc_status_requires_real_external_release_evidence(tmp_path: Path, monkeypatch) -> None:
    for name in ("output", "ratings", "accounts", "benchmarks"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("SAUND_HERO_INTRO_RIGHTS_CONFIRMED", "0")
    status = release_candidate_status(tmp_path, strict=True, benchmark_suite="standard")
    assert status["release_certifiable"] is False
    blockers = {item["name"] for item in status["checks"] if item["level"] == "blocker"}
    assert "benchmark_champion" in blockers
    assert "dataset_provenance" in blockers
    assert "dataset_source_catalog" in blockers
    assert "hero_media_rights" in blockers


def test_release_gate_makes_benchmark_and_certification_mandatory_for_rc() -> None:
    gate = (ROOT / "scripts" / "release-gate.sh").read_text(encoding="utf-8")
    package = (ROOT / "scripts" / "package-release.sh").read_text(encoding="utf-8")
    assert 'rc_build=1' in gate
    assert 'release-candidate.py certify' in gate
    assert 'dist/release-certification.json' in gate
    assert 'RELEASE_CERTIFICATION.json' in package
