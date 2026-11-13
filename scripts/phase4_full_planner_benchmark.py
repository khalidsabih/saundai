#!/usr/bin/env python3
from __future__ import annotations

"""Controlled 13-track, end-to-end Phase 4 planner capture and comparison.

Invoke this same script from both worktrees with PYTHONPATH selecting the
implementation. Captures a failed plan as evidence rather than hiding a
safety/planning rejection. Does not render audio or alter safety decisions.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _peak_memory() -> dict[str, object]:
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    return {"peak_rss_bytes": int(raw * multiplier), "peak_rss_raw": raw}


def _capture(args: argparse.Namespace) -> int:
    from djai.cache import AnalysisBatchStats, analyze_tracks, discover_tracks
    from djai.candidate_renderer import QualityGateConfig
    from djai.decision import validate_execution_plan
    from djai.planning_context import PlanningSession, scoring_model_fingerprint
    from djai.set_planner import make_set_plan

    folder = Path(args.audio_dir).expanduser().resolve()
    paths = discover_tracks(folder, recursive=False)
    if len(paths) != 13:
        raise SystemExit(f"Expected 13 supported tracks in {folder}; found {len(paths)}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_dir).expanduser().resolve()
    # An explicit isolated cache makes cold graph vs warm graph semantics
    # reproducible without deleting any production application caches.
    os.environ["DJAI_CACHE_DIR"] = str(cache_root / "runtime")
    analysis_root = cache_root / "analysis"
    if args.cold_graph:
        import shutil
        shutil.rmtree(cache_root / "runtime" / "graph-pairs", ignore_errors=True)

    fingerprints = {path.name: _digest(path) for path in paths}
    start = time.perf_counter()
    stats = AnalysisBatchStats()
    report: dict[str, object] = {
        "schema": "phase4-full-planner-v1",
        "track_count": len(paths),
        "audio_fingerprints": fingerprints,
        "scoring_model_fingerprint": None,
        "config": {
            "rehearse": not args.no_rehearse,
            "automatic_delivery": not args.no_automatic_delivery,
            "graph_workers": args.graph_workers,
            "cold_graph": args.cold_graph,
            "cold_analysis": args.cold_analysis,
            "config_dir": os.environ.get("DJAI_CONFIG_DIR"),
            "runtime_cache": str(cache_root / "runtime"),
        },
        "status": "not_started",
        "error": None,
        "timings_seconds": {},
        "memory": {},
    }
    try:
        analysis_started = time.perf_counter()
        analyses = analyze_tracks(
            paths, workers=1, use_cache=not args.cold_analysis,
            cache_dir=analysis_root, stats=stats,
        )
        report["analysis_digests"] = [
            hashlib.sha256(json.dumps(
                analysis.to_dict(include_beats=True),
                sort_keys=True, separators=(",", ":"), allow_nan=True,
            ).encode("utf-8")).hexdigest()
            for analysis in analyses
        ]
        report["timings_seconds"]["analysis"] = round(time.perf_counter() - analysis_started, 4)
        report["analysis_stats"] = stats.to_dict()

        session = PlanningSession()
        report["scoring_model_fingerprint"] = scoring_model_fingerprint(session.scoring_model)
        plan_started = time.perf_counter()
        quality_loop = None if args.no_rehearse else QualityGateConfig.from_environment(
            enabled=True, top_k=args.top_k, log_path=None,
        )
        try:
            plan = make_set_plan(
                analyses, source="phase4-full-planner",
                tempo_policy="safe", preferred_style="auto", journey="balanced",
                minimum_body_seconds=30.0, minimum_body_fraction=0.35,
                graph_workers=args.graph_workers, planning_session=session,
                quality_loop=quality_loop,
                automatic_delivery=not args.no_automatic_delivery,
            )
        finally:
            report["timings_seconds"]["planning"] = round(time.perf_counter() - plan_started, 4)
            report["graph_stats"] = session.to_dict()

        for edge in plan.transitions:
            validate_execution_plan(edge.plan)
        report["status"] = "passed"
        report["order"] = list(plan.order)
        report["transitions"] = [{
            "from": edge.from_index,
            "to": edge.to_index,
            "signature": edge.plan.execution_signature,
            "technique": edge.plan.technique,
            "fallback_level": edge.plan.fallback_level,
            "quality_checked": edge.plan.quality_checked,
            "measurement_passed": edge.plan.measurement_passed,
        } for edge in plan.transitions]
        report["fallback_count"] = sum(edge.plan.fallback_level > 0 for edge in plan.transitions)
        report["emergency_count"] = sum(edge.plan.transition_mode == "emergency_cut" for edge in plan.transitions)
        report["planning_breakdown"] = dict(plan.metadata.get("performance", {})).get("planning", {})
        report["quality_summary"] = plan.metadata.get("quality_loop", {})
        report["large_crate_planning"] = plan.metadata.get("large_crate_planning", {})
        report["replan_summary"] = plan.metadata.get("unified_transition_intelligence", {})
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["timings_seconds"]["total"] = round(time.perf_counter() - start, 4)
        report["memory"] = _peak_memory()
        output.write_text(json.dumps(report, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Full-planner {report['status']}: {output}")
    print(f"  analysis={report['timings_seconds'].get('analysis')}s "
          f"planning={report['timings_seconds'].get('planning')}s "
          f"total={report['timings_seconds'].get('total')}s")
    print(f"  cache={report.get('graph_stats', {}).get('graph_cache_hits')} hits / "
          f"{report.get('graph_stats', {}).get('graph_cache_misses')} misses")
    if report["status"] != "passed":
        print(f"  REJECTED: {report['error']}", file=sys.stderr)
        return 2
    print(f"  fallback={report['fallback_count']} emergency={report['emergency_count']}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    right = json.loads(Path(args.current).read_text(encoding="utf-8"))
    print("Phase 4 full-planner comparison:")
    for field in ("schema", "track_count", "audio_fingerprints", "analysis_digests",
                  "scoring_model_fingerprint", "status", "order", "transitions",
                  "fallback_count", "emergency_count"):
        match = left.get(field) == right.get(field)
        print(f"  {field}: {'MATCH' if match else 'DIFFERENT'}")
    for name, report in (("baseline", left), ("current", right)):
        times = report.get("timings_seconds", {})
        graph = report.get("graph_stats", {})
        print(f"  {name}: status={report.get('status')}, "
              f"analysis={times.get('analysis')}s, planning={times.get('planning')}s, "
              f"graph={graph.get('graph_seconds')}s, "
              f"hits={graph.get('graph_cache_hits')} misses={graph.get('graph_cache_misses')}, "
              f"rss={report.get('memory', {}).get('peak_rss_bytes')} bytes")
        if report.get("error"):
            print(f"    error: {report['error']}")
        breakdown = report.get("planning_breakdown", {})
        for key in ("route_search_seconds", "compile_rehearsal_seconds",
                    "executive_seconds", "route_search_calls", "compile_rehearsal_calls"):
            if key in breakdown:
                print(f"    {key}: {breakdown[key]}")
    if left.get("audio_fingerprints") != right.get("audio_fingerprints") or (
        left.get("analysis_digests") != right.get("analysis_digests")
    ) or left.get("scoring_model_fingerprint") != right.get("scoring_model_fingerprint"):
        print("INVALID CONTROL: source, analysis or scoring-model inputs differ.")
        return 1
    if left.get("status") != "passed" or right.get("status") != "passed":
        print("SAFETY/PLANNING GATE OPEN: do not sign off a rejected plan.")
        return 2
    if left.get("order") != right.get("order") or left.get("transitions") != right.get("transitions"):
        print("WARNING: end-to-end plan differs; investigate before sign-off.")
        return 1
    print("Full-planner identity passed; performance values are observations, not speedup proof.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--audio-dir", required=True)
    capture.add_argument("--cache-dir", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--cold-graph", action="store_true")
    capture.add_argument("--cold-analysis", action="store_true")
    capture.add_argument("--no-rehearse", action="store_true")
    capture.add_argument("--no-automatic-delivery", action="store_true")
    capture.add_argument("--graph-workers", type=int, default=1)
    capture.add_argument("--top-k", type=int, default=4)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("current")
    args = parser.parse_args()
    return _capture(args) if args.command == "capture" else _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
