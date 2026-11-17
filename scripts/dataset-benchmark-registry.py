#!/usr/bin/env python3
"""M0 dataset / benchmark registry: metadata-only audit, immutable result ledger.

Separate from the existing production release-benchmark champion gate. No audio
downloads, no model or DJ authority modifications, and no implicit promotions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks" / "dataset-registry" / "catalog.json"
DEFAULT_DB = ROOT / "benchmarks" / "dataset-registry-results.sqlite3"
RIGHTS = {"approved_local_research", "approved_commercial", "unreviewed", "review_required", "review_required_per_recording", "noncommercial_research_review_required", "unknown_unverified"}
SPLITS = {"development", "validation", "test"}
SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    implementation_id TEXT NOT NULL,
    metric_value REAL NOT NULL,
    safety_passed INTEGER NOT NULL,
    rights_confirmed INTEGER NOT NULL,
    human_evidence TEXT NOT NULL,
    notes TEXT NOT NULL,
    UNIQUE(benchmark_id, dataset_id, split, dataset_fingerprint, source_commit, implementation_id, metric_value, safety_passed, rights_confirmed, human_evidence)
);
CREATE TABLE IF NOT EXISTS champions (
    benchmark_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    PRIMARY KEY (benchmark_id, dataset_id, split, dataset_fingerprint)
);
"""


def catalog(path: Path = CATALOG) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "1.0":
        raise ValueError("unsupported catalog schema")
    sources = data.get("sources", [])
    benches = data.get("benchmarks", [])
    if len({s["id"] for s in sources}) != len(sources):
        raise ValueError("duplicate dataset ID")
    if len({b["id"] for b in benches}) != len(benches):
        raise ValueError("duplicate benchmark ID")
    for b in benches:
        if b["direction"] not in ("maximize", "minimize"):
            raise ValueError("unknown benchmark direction")
        if not b.get("metric"):
            raise ValueError("missing benchmark metric")
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connection(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def as_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def select(db: sqlite3.Connection, run_id: int) -> dict:
    row = as_dict(db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone())
    if row is None:
        raise ValueError(f"run {run_id} does not exist")
    return row


def champion(db: sqlite3.Connection, run: dict) -> dict | None:
    return as_dict(db.execute(
        """SELECT r.* FROM champions c JOIN runs r ON c.run_id=r.id
           WHERE c.benchmark_id=? AND c.dataset_id=? AND c.split=? AND c.dataset_fingerprint=?""",
        (run["benchmark_id"], run["dataset_id"], run["split"], run["dataset_fingerprint"]),
    ).fetchone())


def assess(a: dict, b: dict, definition: dict) -> dict:
    # a = incumbent, b = challenger. No cross-cohort comparisons.
    fields = ("benchmark_id", "dataset_id", "split", "dataset_fingerprint")
    if any(a[x] != b[x] for x in fields):
        raise ValueError("incomparable benchmark/dataset/split/fingerprint; use the same frozen cohort")
    direction = definition["direction"]
    improvement = b["metric_value"] - a["metric_value"]
    better = improvement > 0 if direction == "maximize" else improvement < 0
    pass_basic = bool(b["safety_passed"]) and bool(b["rights_confirmed"])
    requires_human = bool(definition.get("requires_human"))
    human_ok = not requires_human or b["human_evidence"] not in ("", "none", "not_collected")
    passed = better and pass_basic and human_ok
    return {
        "comparable": True, "incumbent_run": a["id"], "challenger_run": b["id"],
        "metric": definition["metric"], "direction": direction,
        "incumbent_value": a["metric_value"], "challenger_value": b["metric_value"],
        "delta": improvement, "improved": better, "safety_passed": bool(b["safety_passed"]),
        "rights_confirmed": bool(b["rights_confirmed"]), "human_evidence_present": human_ok,
        "promotion_eligible": passed,
        "caveat": "Metric improvement and human-evidence reference are declarations, not independent scientific verification. Product model promotion and release gate remain separate.",
    }


def dataset_approval(source: dict, args: argparse.Namespace) -> None:
    # Explicit per-run rights review is required; catalog never grants usage.
    if not args.rights_confirmed:
        raise ValueError("rights not confirmed: refuse to record as promotion-ready")
    if source["rights_status"] in ("unknown_unverified", "unreviewed") and not args.rights_confirmed:
        raise ValueError("dataset provenance not confirmed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog", help="show proposed dataset and benchmark definitions")
    sub.add_parser("status", help="show local ledger runs and champions")
    record = sub.add_parser("record", help="record externally measured numeric result on frozen dataset manifest")
    record.add_argument("--benchmark", required=True)
    record.add_argument("--dataset", required=True)
    record.add_argument("--split", choices=sorted(SPLITS), required=True)
    record.add_argument("--manifest", type=Path, required=True, help="frozen, private manifest; stored as hash only")
    record.add_argument("--source-commit", required=True)
    record.add_argument("--implementation", required=True)
    record.add_argument("--value", type=float, required=True)
    record.add_argument("--safety-passed", action="store_true")
    record.add_argument("--rights-confirmed", action="store_true", help="explicit reviewer attestation, NOT automatically inferred")
    record.add_argument("--human-evidence", default="none", help="identifier to external blind Arena report, not a self-score")
    record.add_argument("--notes", default="")
    compare = sub.add_parser("compare", help="compare two runs using same frozen cohort")
    compare.add_argument("--incumbent", type=int, required=True)
    compare.add_argument("--challenger", type=int, required=True)
    promote = sub.add_parser("promote", help="manually advance ledger champion, NOT live authority")
    promote.add_argument("--run-id", type=int, required=True)
    promote.add_argument("--initial", action="store_true", help="explicitly establish first benchmark champion")

    args = parser.parse_args(argv)
    data = catalog(args.catalog)
    if args.command == "catalog":
        print(json.dumps(data, indent=2))
        return 0
    if args.command == "record":
        by_src = {x["id"]: x for x in data["sources"]}
        by_bench = {x["id"]: x for x in data["benchmarks"]}
        if args.dataset not in by_src or args.benchmark not in by_bench:
            raise ValueError("unknown dataset or benchmark")
        bench = by_bench[args.benchmark]
        if not math.isfinite(args.value):
            raise ValueError("metric value must be finite")
        if len(args.source_commit.strip()) < 8 or not args.implementation.strip():
            raise ValueError("pin source commit and implementation ID")
        if not args.manifest.is_file():
            raise ValueError("manifest missing; freeze a dataset before recording")
        if args.split == "test" and not args.safety_passed:
            raise ValueError("test result with failed safety cannot be recorded as a promotion candidate; use a development diagnostic")
        dataset_approval(by_src[args.dataset], args)
        fingerprint = sha256(args.manifest)
        with connection(args.db) as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO runs
                (created_at, benchmark_id, dataset_id, split, dataset_fingerprint, source_commit,
                 implementation_id, metric_value, safety_passed, rights_confirmed, human_evidence, notes)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now(timezone.utc).isoformat(), args.benchmark, args.dataset, args.split,
                 fingerprint, args.source_commit.strip(), args.implementation.strip(), args.value,
                 int(args.safety_passed), int(args.rights_confirmed), args.human_evidence, args.notes),
            )
            if cursor.lastrowid:
                run_id = cursor.lastrowid
            else:
                row = db.execute(
                    """SELECT id FROM runs WHERE benchmark_id=? AND dataset_id=? AND split=? AND
                       dataset_fingerprint=? AND source_commit=? AND implementation_id=? AND
                       metric_value=? AND safety_passed=? AND rights_confirmed=? AND human_evidence=?""",
                    (args.benchmark, args.dataset, args.split, fingerprint, args.source_commit.strip(),
                     args.implementation.strip(), args.value, int(args.safety_passed),
                     int(args.rights_confirmed), args.human_evidence),
                ).fetchone()
                run_id = row["id"]
        print(json.dumps({"run_id": run_id, "fingerprint": fingerprint, "benchmark": args.benchmark, "record_only": True}, indent=2))
        return 0
    with connection(args.db) as db:
        if args.command == "status":
            runs = [dict(x) for x in db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 30")]
            champions = [dict(x) for x in db.execute("SELECT * FROM champions ORDER BY benchmark_id, dataset_id")]
            print(json.dumps({"runs": runs, "champions": champions, "independent_release_gate": "scripts/release-benchmark.py"}, indent=2))
            return 0
        by_bench = {x["id"]: x for x in data["benchmarks"]}
        if args.command == "compare":
            a, b = select(db, args.incumbent), select(db, args.challenger)
            print(json.dumps(assess(a, b, by_bench[a["benchmark_id"]]), indent=2))
            return 0
        run = select(db, args.run_id)
        if run["benchmark_id"] not in by_bench:
            raise ValueError("unknown benchmark")
        if run["split"] != "test":
            raise ValueError("champions require untouched test split")
        if not run["rights_confirmed"] or not run["safety_passed"]:
            raise ValueError("champion requires rights and safety attestation")
        old = champion(db, run)
        if old is None:
            if not args.initial:
                raise ValueError("first champion needs --initial")
            if by_bench[run["benchmark_id"]].get("requires_human") and run["human_evidence"] in ("none", "not_collected", ""):
                raise ValueError("blind human evidence required")
        else:
            if args.initial:
                raise ValueError("--initial not allowed when champion exists")
            if not assess(old, run, by_bench[run["benchmark_id"]])["promotion_eligible"]:
                raise ValueError("challenger did not pass the improvement/safety/evidence gate")
        db.execute(
            """INSERT INTO champions (benchmark_id, dataset_id, split, dataset_fingerprint, run_id)
               VALUES (?,?,?,?,?) ON CONFLICT(benchmark_id,dataset_id,split,dataset_fingerprint)
               DO UPDATE SET run_id=excluded.run_id""",
            (run["benchmark_id"], run["dataset_id"], run["split"], run["dataset_fingerprint"], run["id"]),
        )
        print(json.dumps({"champion_run_id": run["id"], "registry_only": True, "product_model_authority_unchanged": True}))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError, KeyError) as exc:
        print(f"Dataset/benchmark registry: {exc}", file=sys.stderr)
        sys.exit(2)
