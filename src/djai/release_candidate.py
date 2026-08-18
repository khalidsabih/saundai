from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .brain.crate_memory import CRATE_MEMORY_SCHEMA_VERSION
from .build_info import build_identity
from .evaluation.release_benchmark import (
    DEFAULT_BENCHMARK_DB,
    BenchmarkStore,
    compare_benchmarks,
)
from .feature_store import FEATURE_STORE_SCHEMA_VERSION
from .intelligence.governance import audit_model_governance
from .training.provenance import audit_dataset_registry
from .version import DJAI_VERSION, MUSICAL_TIMELINE_SCHEMA_VERSION, PLAN_SCHEMA_VERSION

RELEASE_CANDIDATE_SCHEMA_VERSION = "2.0.0"
RELEASE_CHANNEL = "release-candidate"
RELEASE_CANDIDATE_NUMBER = 1
CERTIFICATION_SCHEMA_VERSION = "1.0"

CheckLevel = Literal["pass", "warning", "blocker"]


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    name: str
    level: CheckLevel
    detail: str
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["evidence"] is None:
            payload.pop("evidence")
        return payload


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _workspace_root(workspace: str | Path = ".") -> Path:
    return Path(workspace).expanduser().resolve()


def _path_writable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _schema_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value).replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def _plan_compatibility(root: Path, *, limit: int = 250) -> dict[str, Any]:
    current = 0
    legacy = 0
    future = 0
    unreadable = 0
    examples: list[dict[str, str]] = []
    current_tuple = _schema_tuple(PLAN_SCHEMA_VERSION)
    paths = sorted((root / "output").glob("**/set-plan.json"))[:limit] if (root / "output").exists() else []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            schema = str(payload.get("schema_version", "missing"))
        except (OSError, json.JSONDecodeError, TypeError):
            unreadable += 1
            if len(examples) < 8:
                examples.append({"path": str(path.relative_to(root)), "status": "unreadable"})
            continue
        if schema == PLAN_SCHEMA_VERSION:
            current += 1
            continue
        if _schema_tuple(schema) > current_tuple:
            future += 1
            status = "future_schema_blocked"
        else:
            legacy += 1
            status = "legacy_replan_required"
        if len(examples) < 8:
            examples.append({"path": str(path.relative_to(root)), "schema_version": schema, "status": status})
    return {
        "current_schema": PLAN_SCHEMA_VERSION,
        "scanned": len(paths),
        "current": current,
        "legacy_replan_required": legacy,
        "future_schema_blocked": future,
        "unreadable": unreadable,
        "examples": examples,
        "policy": "execution plans are never silently downgraded or migrated across incompatible signed schemas",
    }


def _crate_memory_compatibility(root: Path) -> dict[str, Any]:
    configured = os.environ.get("DJAI_CRATE_MEMORY_PATH", "ratings/crate-memory.sqlite3").strip() or "ratings/crate-memory.sqlite3"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return {
            "available": False,
            "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
            "compatible": True,
            "detail": "store will be created on first use",
        }
    observed: str | None = None
    try:
        connection = sqlite3.connect(path, timeout=2.0)
        try:
            row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            observed = str(row[0]) if row else None
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return {
            "available": True,
            "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
            "compatible": False,
            "observed_schema_version": observed,
            "detail": "crate-memory database could not be inspected",
        }
    compatible = observed == str(CRATE_MEMORY_SCHEMA_VERSION)
    return {
        "available": True,
        "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
        "compatible": compatible,
        "observed_schema_version": observed,
        "detail": "compatible" if compatible else "unsupported crate-memory schema",
    }


def _benchmark_certification(root: Path, suite: str) -> tuple[list[ReleaseCheck], dict[str, Any]]:
    checks: list[ReleaseCheck] = []
    database = Path(os.environ.get("SAUND_RELEASE_BENCHMARK_DB", DEFAULT_BENCHMARK_DB)).expanduser()
    if not database.is_absolute():
        database = root / database
    evidence: dict[str, Any] = {"suite": suite, "database_available": database.exists()}
    if not database.exists():
        checks.append(ReleaseCheck("benchmark_champion", "blocker", f"benchmark database is missing for suite {suite!r}"))
        return checks, evidence
    store = BenchmarkStore(database)
    champion = store.champion(suite)
    latest = store.latest(suite)
    if champion is None:
        checks.append(ReleaseCheck("benchmark_champion", "blocker", f"suite {suite!r} has no manually validated champion"))
        return checks, evidence
    champion_id, champion_snapshot = champion
    evidence["champion_run_id"] = champion_id
    evidence["champion_version"] = champion_snapshot.saund_version
    if latest is None:
        checks.append(ReleaseCheck("benchmark_challenger", "blocker", f"suite {suite!r} has no challenger run"))
        return checks, evidence
    latest_id, latest_snapshot = latest
    evidence["challenger_run_id"] = latest_id
    evidence["challenger_version"] = latest_snapshot.saund_version
    if latest_snapshot.saund_version != DJAI_VERSION:
        checks.append(
            ReleaseCheck(
                "benchmark_current_version",
                "blocker",
                f"latest {suite} run is SAUND {latest_snapshot.saund_version}, expected {DJAI_VERSION}",
            )
        )
        return checks, evidence
    comparison = compare_benchmarks(champion_snapshot, latest_snapshot)
    evidence["comparison"] = comparison.to_dict()
    if comparison.passed:
        checks.append(ReleaseCheck("benchmark_champion", "pass", f"current {suite} challenger passes the named champion", evidence=evidence))
    else:
        checks.append(
            ReleaseCheck(
                "benchmark_champion",
                "blocker",
                f"current {suite} challenger failed the champion gate",
                evidence={"blockers": list(comparison.blockers), "warnings": list(comparison.warnings)},
            )
        )
    return checks, evidence


def release_candidate_status(
    workspace: str | Path = ".",
    *,
    strict: bool = False,
    benchmark_suite: str | None = None,
) -> dict[str, Any]:
    root = _workspace_root(workspace)
    suite = (benchmark_suite or os.environ.get("SAUND_RELEASE_BENCHMARK_SUITE", "standard")).strip() or "standard"
    checks: list[ReleaseCheck] = []

    build = build_identity()
    if build.get("version_consistent"):
        checks.append(ReleaseCheck("version_identity", "pass", f"all authoritative version surfaces report {DJAI_VERSION}"))
    else:
        checks.append(ReleaseCheck("version_identity", "blocker", "runtime/build/frontend version identity is inconsistent", evidence=dict(build.get("versions", {}))))

    if "rc" in DJAI_VERSION.lower():
        checks.append(ReleaseCheck("release_channel", "pass", f"{DJAI_VERSION} is explicitly marked as a release candidate"))
    else:
        level: CheckLevel = "blocker" if strict else "warning"
        checks.append(ReleaseCheck("release_channel", level, f"{DJAI_VERSION} is not explicitly marked as an RC build"))

    planner_mode = os.environ.get("DJAI_PLANNER_MODE", "move_authoritative").strip().lower()
    headphones = _bool_env("DJAI_HEADPHONES", True)
    crate_memory = _bool_env("DJAI_CRATE_MEMORY", True)
    structural_ops = _bool_env("DJAI_STRUCTURAL_DECK_OPS", True)
    if planner_mode == "move_authoritative":
        checks.append(ReleaseCheck("planner_authority", "pass", "Complete DJMove planner is authoritative"))
    else:
        checks.append(
            ReleaseCheck(
                "planner_authority",
                "blocker" if strict else "warning",
                f"effective planner mode is {planner_mode!r}; RC certification requires move_authoritative",
            )
        )
    for name, enabled, detail in (
        ("ai_headphones", headphones, "real-audio commit-boundary rehearsal"),
        ("crate_memory", crate_memory, "bounded record/pair memory"),
        ("structural_operations", structural_ops, "deterministic deck-operation compiler"),
    ):
        checks.append(
            ReleaseCheck(
                name,
                "pass" if enabled else ("blocker" if strict else "warning"),
                f"{detail} is {'enabled' if enabled else 'disabled'}",
            )
        )

    writable: dict[str, bool] = {}
    for relative in ("output", "ratings", "accounts", "benchmarks"):
        path = root / relative
        ok = _path_writable(path)
        writable[relative] = ok
        checks.append(
            ReleaseCheck(
                f"writable_{relative}",
                "pass" if ok else "blocker",
                f"{relative}/ is {'writable/creatable' if ok else 'not writable'}",
            )
        )

    plans = _plan_compatibility(root)
    if plans["future_schema_blocked"] or plans["unreadable"]:
        checks.append(ReleaseCheck("persisted_plan_compatibility", "blocker", "persisted plans include unreadable or future execution schemas", evidence=plans))
    elif plans["legacy_replan_required"]:
        checks.append(ReleaseCheck("persisted_plan_compatibility", "warning", "legacy plans are preserved but must be replanned before rendering", evidence=plans))
    else:
        checks.append(ReleaseCheck("persisted_plan_compatibility", "pass", "persisted plan schemas are compatible", evidence=plans))

    crate_store = _crate_memory_compatibility(root)
    checks.append(
        ReleaseCheck(
            "crate_memory_schema",
            "pass" if crate_store["compatible"] else "blocker",
            str(crate_store["detail"]),
            evidence=crate_store,
        )
    )

    benchmark_evidence: dict[str, Any] = {"suite": suite, "required": strict}
    if strict:
        benchmark_checks, benchmark_evidence = _benchmark_certification(root, suite)
        checks.extend(benchmark_checks)

        registry = root / "datasets" / "registry.json"
        catalog = root / "datasets" / "source-catalog.json"
        if not registry.exists():
            checks.append(ReleaseCheck("dataset_provenance", "blocker", "datasets/registry.json is missing"))
        else:
            try:
                audit = audit_dataset_registry(registry, lane="production")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                checks.append(ReleaseCheck("dataset_provenance", "blocker", f"production dataset provenance audit failed: {exc}"))
            else:
                checks.append(
                    ReleaseCheck(
                        "dataset_provenance",
                        "pass" if audit.get("ready") else "blocker",
                        "production dataset provenance is ready" if audit.get("ready") else "production dataset provenance has blockers",
                        evidence={"dataset_count": audit.get("dataset_count", 0), "blockers": audit.get("blockers", [])},
                    )
                )
        checks.append(
            ReleaseCheck(
                "dataset_source_catalog",
                "pass" if catalog.exists() else "blocker",
                "datasets/source-catalog.json is present" if catalog.exists() else "datasets/source-catalog.json is missing",
            )
        )

        model_registry = root / "config" / "model-registry.json"
        if not model_registry.exists():
            checks.append(ReleaseCheck("model_governance", "blocker", "config/model-registry.json is missing"))
        else:
            governance = audit_model_governance(model_registry)
            checks.append(
                ReleaseCheck(
                    "model_governance",
                    "pass" if governance.ready else "blocker",
                    "model governance is ready" if governance.ready else "model governance has blockers",
                    evidence=governance.to_dict(),
                )
            )
        rights_confirmed = _bool_env("SAUND_HERO_INTRO_RIGHTS_CONFIRMED", False)
        checks.append(
            ReleaseCheck(
                "hero_media_rights",
                "pass" if rights_confirmed else "blocker",
                "hero intro distribution rights explicitly confirmed" if rights_confirmed else "SAUND_HERO_INTRO_RIGHTS_CONFIRMED=1 is required for certification",
            )
        )
        debris = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".orig", ".rej"} and ".git" not in path.parts
        ][:20]
        checks.append(
            ReleaseCheck(
                "patch_debris",
                "pass" if not debris else "blocker",
                "no .orig/.rej patch debris found" if not debris else "patch debris must be removed",
                evidence={"files": debris} if debris else None,
            )
        )

    blockers = [item for item in checks if item.level == "blocker"]
    warnings = [item for item in checks if item.level == "warning"]
    operational_blockers = [
        item
        for item in blockers
        if item.name
        in {
            "version_identity",
            "writable_output",
            "writable_ratings",
            "writable_accounts",
            "writable_benchmarks",
            "persisted_plan_compatibility",
            "crate_memory_schema",
        }
    ]
    return {
        "schema_version": RELEASE_CANDIDATE_SCHEMA_VERSION,
        "product": "SAUND.AI",
        "version": DJAI_VERSION,
        "release_channel": RELEASE_CHANNEL,
        "candidate_number": RELEASE_CANDIDATE_NUMBER,
        "strict": bool(strict),
        "operational_ready": not operational_blockers,
        "release_certifiable": not blockers if strict else None,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "authority": {
            "planner_mode": planner_mode,
            "ai_headphones": headphones,
            "crate_memory": crate_memory,
            "structural_deck_operations": structural_ops,
            "hard_audio_safety_authoritative": True,
        },
        "schemas": {
            "plan": PLAN_SCHEMA_VERSION,
            "musical_timeline": MUSICAL_TIMELINE_SCHEMA_VERSION,
            "feature_store": FEATURE_STORE_SCHEMA_VERSION,
            "crate_memory": CRATE_MEMORY_SCHEMA_VERSION,
        },
        "upgrade_compatibility": {
            "plans": plans,
            "crate_memory": crate_store,
        },
        "benchmark": benchmark_evidence,
        "writable_paths": writable,
        "checks": [item.to_dict() for item in checks],
    }


def certification_manifest(
    workspace: str | Path = ".",
    *,
    benchmark_suite: str | None = None,
) -> dict[str, Any]:
    status = release_candidate_status(workspace, strict=True, benchmark_suite=benchmark_suite)
    manifest: dict[str, Any] = {
        "certification_schema_version": CERTIFICATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product": "SAUND.AI",
        "version": DJAI_VERSION,
        "release_channel": RELEASE_CHANNEL,
        "candidate_number": RELEASE_CANDIDATE_NUMBER,
        "certified": bool(status.get("release_certifiable")),
        "build": build_identity(),
        "readiness": status,
        "policy": {
            "benchmark_champion_required": True,
            "production_data_provenance_required": True,
            "model_governance_required": True,
            "hard_audio_safety_non_overridable": True,
            "legacy_signed_execution_plans_never_silently_migrated": True,
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    manifest["certification_id"] = hashlib.sha256(canonical).hexdigest()
    return manifest
