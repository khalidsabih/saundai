#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from djai.cache import AnalysisBatchStats, analyze_tracks, discover_tracks
from djai.planning_context import PlanningSession
from djai.set_planner import build_transition_graph
from djai.version import DJAI_VERSION


def graph_signature(graph: dict[tuple[int, int], Any]) -> dict[str, tuple[object, ...]]:
    return {
        f"{key[0]}->{key[1]}": (
            edge.selected_style,
            edge.objective_score,
            edge.quality,
            edge.plan.execution_signature,
            edge.plan.exit_time_a,
            edge.plan.entry_time_b,
            edge.plan.technique,
            edge.plan.intent,
            edge.plan.transition_bars,
        )
        for key, edge in graph.items()
    }


def analyze_once(
    tracks: list[Path],
    *,
    cache_dir: Path,
    workers: int,
    use_cache: bool,
) -> tuple[list[Any], AnalysisBatchStats]:
    stats = AnalysisBatchStats()
    analyses = analyze_tracks(
        tracks,
        cache_dir=cache_dir,
        use_cache=use_cache,
        workers=workers,
        stats=stats,
    )
    return analyses, stats


def graph_once(analyses: list[Any], *, workers: int) -> tuple[dict[Any, Any], PlanningSession]:
    session = PlanningSession()
    graph = build_transition_graph(
        analyses,
        workers=workers,
        session=session,
    )
    return graph, session


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SAUND.AI planning performance")
    parser.add_argument("folder", help="Folder containing supported audio tracks")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--analysis-workers", type=int, default=0)
    parser.add_argument("--graph-workers", type=int, default=0)
    parser.add_argument("--compare-serial", action="store_true")
    parser.add_argument("--warm-cache", action="store_true", help="Reuse existing benchmark caches")
    parser.add_argument("--output", default="output/benchmark-phase0.json")
    args = parser.parse_args()

    tracks = discover_tracks(args.folder, recursive=args.recursive)
    if len(tracks) < 2:
        raise SystemExit("Benchmark requires at least two supported audio files.")

    root = Path(".djai-cache/benchmark").resolve()
    if not args.warm_cache:
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    parallel_analyses, parallel_analysis_stats = analyze_once(
        tracks,
        cache_dir=root / "parallel-analysis",
        workers=args.analysis_workers,
        use_cache=args.warm_cache,
    )
    parallel_graph, parallel_session = graph_once(
        parallel_analyses,
        workers=args.graph_workers,
    )
    expected_compatibility_evaluations = sum(
        len(track_a.exit_candidates) * len(track_b.entry_candidates)
        for from_index, track_a in enumerate(parallel_analyses)
        for to_index, track_b in enumerate(parallel_analyses)
        if from_index != to_index
    )
    if parallel_session.compatibility_evaluations != expected_compatibility_evaluations:
        raise SystemExit(
            "Pair-evidence count mismatch: compatibility was not evaluated exactly once per unique cue pair."
        )

    report: dict[str, Any] = {
        "djai_version": DJAI_VERSION,
        "track_count": len(tracks),
        "directed_pair_count": len(tracks) * (len(tracks) - 1),
        "unique_cue_pair_count": expected_compatibility_evaluations,
        "one_compatibility_evaluation_per_cue_pair": True,
        "parallel": {
            "analysis": parallel_analysis_stats.to_dict(),
            "graph": parallel_session.to_dict(),
        },
    }

    if args.compare_serial:
        serial_analyses, serial_analysis_stats = analyze_once(
            tracks,
            cache_dir=root / "serial-analysis",
            workers=1,
            use_cache=args.warm_cache,
        )
        serial_graph, serial_session = graph_once(serial_analyses, workers=1)
        if serial_session.compatibility_evaluations != expected_compatibility_evaluations:
            raise SystemExit("Serial pair-evidence count differs from the unique cue-pair count.")
        serial_signature = graph_signature(serial_graph)
        parallel_signature = graph_signature(parallel_graph)
        report["serial"] = {
            "analysis": serial_analysis_stats.to_dict(),
            "graph": serial_session.to_dict(),
        }
        report["deterministic_match"] = serial_signature == parallel_signature
        report["speedup"] = {
            "analysis": (
                serial_analysis_stats.elapsed_seconds
                / max(parallel_analysis_stats.elapsed_seconds, 1e-9)
            ),
            "graph": serial_session.graph_seconds / max(parallel_session.graph_seconds, 1e-9),
        }
        if serial_signature != parallel_signature:
            raise SystemExit("Serial and parallel graph plans differ; benchmark failed.")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {output}")


if __name__ == "__main__":
    main()
