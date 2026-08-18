#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from djai.evaluation.release_benchmark import (
    BenchmarkStore,
    BenchmarkThresholds,
    benchmark_runtime_status,
    benchmark_suite_catalog,
    compare_benchmarks,
    extract_release_benchmark,
)
from djai.set_planner import load_set_plan


def _store(path: str) -> BenchmarkStore:
    return BenchmarkStore(Path(path))


def _resolve_run(store: BenchmarkStore, suite: str, value: str):
    if value == "latest":
        result = store.latest(suite)
        if result is None:
            raise SystemExit(f"No benchmark runs exist for suite {suite!r}.")
        return result
    try:
        run_id = int(value)
    except ValueError as exc:
        raise SystemExit("Benchmark run must be an integer id or 'latest'.") from exc
    snapshot = store.get(run_id)
    if snapshot is None:
        raise SystemExit(f"Benchmark run {run_id} does not exist.")
    return run_id, snapshot


def _thresholds(args: argparse.Namespace) -> BenchmarkThresholds:
    return BenchmarkThresholds(
        maximum_quality_regression=args.max_quality_regression,
        maximum_analysis_time_regression_fraction=args.max_analysis_regression,
        maximum_total_time_regression_fraction=args.max_total_regression,
        maximum_planning_time_regression_fraction=args.max_planning_regression,
        maximum_render_time_regression_fraction=args.max_render_regression,
        maximum_memory_regression_fraction=args.max_memory_regression,
        maximum_fallback_increase=args.max_fallback_increase,
        require_performance_telemetry=not args.allow_missing_performance,
        require_human_preference=args.require_human,
    )


def _comparison_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-quality-regression", type=float, default=0.015)
    parser.add_argument("--max-analysis-regression", type=float, default=0.20)
    parser.add_argument("--max-total-regression", type=float, default=0.15)
    parser.add_argument("--max-planning-regression", type=float, default=0.20)
    parser.add_argument("--max-render-regression", type=float, default=0.15)
    parser.add_argument("--max-memory-regression", type=float, default=0.20)
    parser.add_argument("--max-fallback-increase", type=int, default=0)
    parser.add_argument("--allow-missing-performance", action="store_true")
    parser.add_argument("--require-human", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record and compare SAUND.AI release benchmarks."
    )
    parser.add_argument(
        "--db",
        default="benchmarks/benchmark-results.sqlite3",
        help="SQLite benchmark history path",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("suites", help="Print canonical benchmark suite definitions")

    record = sub.add_parser("record", help="Record metrics from a completed set plan")
    record.add_argument("plan", help="set-plan.json with embedded analyses")
    record.add_argument("--suite", default="ad-hoc")
    record.add_argument(
        "--human-preference",
        choices=("champion", "challenger", "tie", "neither"),
    )
    record.add_argument("--notes")
    record.add_argument("--process-peak-rss-mb", type=float)

    compare = sub.add_parser("compare", help="Compare a run against the suite champion")
    compare.add_argument("--suite", required=True)
    compare.add_argument("--challenger", default="latest")
    compare.add_argument("--require-version")
    _comparison_args(compare)

    promote = sub.add_parser("promote", help="Promote a benchmark run to suite champion")
    promote.add_argument("--suite", required=True)
    promote.add_argument("--run-id", required=True, type=int)
    promote.add_argument("--force-first", action="store_true", help="Allow first champion without comparison")
    _comparison_args(promote)

    sub.add_parser("status", help="Show benchmark history/champion status")

    args = parser.parse_args()

    if args.command == "suites":
        print(json.dumps(benchmark_suite_catalog(), indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        print(json.dumps(benchmark_runtime_status(ROOT, database=args.db), indent=2))
        return 0

    store = _store(args.db)

    if args.command == "record":
        plan = load_set_plan(args.plan)
        catalog = benchmark_suite_catalog()
        expected = catalog.get(args.suite, {}).get("track_count")
        if expected is not None and len(plan.order) != int(expected):
            raise SystemExit(
                f"Suite {args.suite!r} requires {expected} tracks; plan contains {len(plan.order)}."
            )
        snapshot = extract_release_benchmark(
            plan,
            suite=args.suite,
            process_peak_rss_mb=args.process_peak_rss_mb,
            human_blind_preference=args.human_preference,
            notes=args.notes,
        )
        run_id = store.record(snapshot)
        print(json.dumps({"run_id": run_id, "snapshot": snapshot.to_dict()}, indent=2))
        return 0

    if args.command == "compare":
        champion = store.champion(args.suite)
        if champion is None:
            raise SystemExit(
                f"Suite {args.suite!r} has no champion. Promote an initial real benchmark first."
            )
        challenger_id, challenger = _resolve_run(store, args.suite, args.challenger)
        if args.require_version and challenger.saund_version != args.require_version:
            raise SystemExit(
                f"Latest challenger is SAUND {challenger.saund_version}, expected {args.require_version}. "
                "Run the current version on the benchmark crate and record it first."
            )
        champion_id, champion_snapshot = champion
        result = compare_benchmarks(
            champion_snapshot,
            challenger,
            thresholds=_thresholds(args),
        )
        print(
            json.dumps(
                {
                    "champion_run_id": champion_id,
                    "challenger_run_id": challenger_id,
                    "comparison": result.to_dict(),
                },
                indent=2,
            )
        )
        return 0 if result.passed else 2

    if args.command == "promote":
        challenger = store.get(args.run_id)
        if challenger is None:
            raise SystemExit(f"Benchmark run {args.run_id} does not exist.")
        if challenger.suite != args.suite:
            raise SystemExit(
                f"Benchmark run {args.run_id} belongs to {challenger.suite!r}, not {args.suite!r}."
            )
        champion = store.champion(args.suite)
        if champion is None:
            if not args.force_first:
                raise SystemExit(
                    "This suite has no champion. Re-run with --force-first after manually validating the first benchmark."
                )
            if not challenger.production_safe or challenger.hard_failure_count:
                raise SystemExit("Cannot promote an unsafe initial champion.")
        else:
            _, champion_snapshot = champion
            comparison = compare_benchmarks(
                champion_snapshot,
                challenger,
                thresholds=_thresholds(args),
            )
            if not comparison.passed:
                print(json.dumps(comparison.to_dict(), indent=2))
                return 2
        store.promote(args.suite, args.run_id)
        print(json.dumps({"promoted": True, "suite": args.suite, "run_id": args.run_id}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
