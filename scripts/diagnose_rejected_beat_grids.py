#!/usr/bin/env python3
"""Read-only diagnosis of beat intervals around rejected rhythmic overlaps.

Loads *full* beat arrays from pre-existing analysis cache artifacts. Never
re-analyzes audio or changes deck behavior/safety limits. Does not infer that
a detected beat is ground-truth correct.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def full_beat_cache(cache_root: Path, source_path: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Choose only unambiguous full-array analysis matching the exact source path."""
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(cache_root.rglob("*.json")):
        # Monolithic cache wraps TrackAnalysis under "analysis".
        # Feature projections and world models do not supply the complete grid.
        if "world_model" in path.parts:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            analysis = document.get("analysis")
            if not isinstance(analysis, dict):
                continue
            if str(analysis.get("path") or document.get("path") or "") != source_path:
                continue
            beats = analysis.get("beats_seconds") or []
            if len(beats) < 20:
                continue
            declared_count = analysis.get("beat_count")
            if declared_count is not None and len(beats) < int(declared_count):
                continue
            matches.append((path, analysis))
        except (OSError, ValueError, TypeError):
            continue
    if not matches:
        return None, []
    # Identical full arrays from separate cache copies are not ambiguous.
    grids = {tuple(item.get("beats_seconds", [])) for _, item in matches}
    if len(grids) != 1:
        return None, [str(path) for path, _ in matches]
    return matches[0][1], [str(path) for path, _ in matches]


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def interval_diagnostic(beats: list[float], *, start: float, end: float, expected: float) -> dict[str, Any]:
    if len(beats) < 3 or expected <= 0:
        return {"status": "insufficient_beats"}
    left = max(0, bisect.bisect_right(beats, start) - 2)
    right = min(len(beats) - 1, bisect.bisect_left(beats, end) + 1)
    intervals = []
    unusual = []
    for i in range(left, right):
        t0, t1 = float(beats[i]), float(beats[i + 1])
        if t1 <= start or t0 >= end:
            continue
        dt = t1 - t0
        intervals.append(dt)
        ratio = dt / expected
        if not (0.65 <= ratio <= 1.35):
            unusual.append({
                "beat_index": i,
                "from_seconds": round(t0, 6),
                "to_seconds": round(t1, 6),
                "interval_ms": round(dt * 1000, 3),
                "expected_ratio": round(ratio, 4),
                "integer_multiple": round(ratio),
            })
    return {
        "status": "ok" if intervals else "no_intervals_covering_overlap",
        "beat_intervals": len(intervals),
        "expected_interval_ms": round(expected * 1000, 3),
        "median_observed_interval_ms": round(median(intervals) * 1000, 3) if intervals else None,
        "min_interval_ms": round(min(intervals) * 1000, 3) if intervals else None,
        "max_interval_ms": round(max(intervals) * 1000, 3) if intervals else None,
        "unusual_interval_count": len(unusual),
        "unusual_intervals": unusual[:30],
        "unusual_intervals_truncated": len(unusual) > 30,
        "observed_span_seconds": round(sum(intervals), 6) if intervals else 0,
        "nominal_span_seconds": round(len(intervals) * expected, 6) if intervals else 0,
        "accumulated_interval_delta_ms": round((sum(intervals) - len(intervals)*expected) * 1000, 3) if intervals else None,
        "limits": "0.65-1.35 expected interval is the diagnostic reliability window, not an acceptance-threshold change.",
    }


def _segment(beats: list[float], source_start: float, rate: float, transition_seconds: float, bpm: float) -> dict[str, Any]:
    return interval_diagnostic(
        beats,
        start=source_start,
        end=source_start + transition_seconds * rate,
        expected=60.0 / bpm * rate,
    )


def diagnose(plan: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    cache = {}
    for i, row in enumerate(tracks):
        source = str(row.get("path") or "")
        cache[i] = full_beat_cache(cache_root, source)

    transitions = []
    for n, edge in enumerate(plan.get("transitions") or [], 1):
        p = edge.get("plan") or {}
        meta = p.get("metadata") or {}
        rejected = meta.get("decision_trace_rejected_phase") or {}
        failed = rejected.get("candidate") or {}
        profile = rejected.get("phase_integrity") or {}
        ai = edge.get("from_index")
        bi = edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or ai not in cache or bi not in cache:
            transitions.append({"transition": n, "status": "unresolved_indices"})
            continue
        a, files_a = cache[ai]
        b, files_b = cache[bi]
        item = {
            "transition": n,
            "from": tracks[ai].get("title"),
            "to": tracks[bi].get("title"),
            "rejected_candidate": failed,
            "reported_phase_reason": profile.get("reason"),
            "reported_max_residual_ms": profile.get("max_residual_ms"),
            "cache_paths_a": files_a,
            "cache_paths_b": files_b,
            "full_grid_available_a": a is not None,
            "full_grid_available_b": b is not None,
        }
        if not failed:
            item["status"] = "no_rejected_candidate_trace"
        elif a is None or b is None:
            item["status"] = "missing_or_ambiguous_full_beat_cache"
        else:
            t = _num(failed.get("transition_seconds"))
            # Candidate snapshots retain bars; convert only if explicit seconds missing.
            bpm = _num(failed.get("target_bpm"))
            bars = _num(failed.get("transition_bars"))
            if t is None and bpm and bars:
                t = bars * 4 * 60.0 / bpm
            ra, rb = _num(failed.get("render_rate_a")), _num(failed.get("render_rate_b"))
            xa, xb = _num(failed.get("exit_time_a")), _num(failed.get("entry_time_b"))
            if None in (t, bpm, ra, rb, xa, xb) or min(t, bpm, ra, rb) <= 0:
                item["status"] = "incomplete_rejected_candidate_timing"
            else:
                nudge = _num(failed.get("phase_nudge_seconds_b"))
                item["status"] = "interval_diagnostics_available"
                item["start_nudge_known"] = nudge is not None
                item["deck_a"] = _segment(a["beats_seconds"], xa, ra, t, bpm)
                item["deck_b"] = _segment(b["beats_seconds"], xb + (nudge or 0.0), rb, t, bpm)
                item["global_bpm_a"] = _num(a.get("bpm"))
                item["global_bpm_b"] = _num(b.get("bpm"))
                item["nominal_output_bpm_a"] = round(float(a.get("bpm", 0)) * ra, 5)
                item["nominal_output_bpm_b"] = round(float(b.get("bpm", 0)) * rb, 5)
                item["interpretation_limit"] = (
                    "Interval anomalies can explain apparent drift but cannot prove detector errors. "
                    "Without reference audio annotations, correct beat positions remain unknown. "
                    "Initial phase nudge is unknown unless explicitly recorded."
                )
        transitions.append(item)
    return {
        "schema": "saund-rejected-beat-grid-diagnostics-v1",
        "plan_id": plan.get("plan_id"),
        "transitions": transitions,
        "safety_behavior_changed": False,
        "requires_new_analysis": False,
        "limitations": [
            "The script reports local interval anomalies, not verified beat ground truth.",
            "A missing full-grid cache is not evidence that detected beats were good or bad.",
            "This script does not reproduce the full continuous phase-integrity algorithm.",
            "Cache matching by exact source path and identical full beats still does not verify source-file bytes.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-four-track/balanced/set-plan.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache"))
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-four-track/beat-grid-diagnostics.json"))
    args = parser.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")), args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transition_statuses": [r["status"] for r in report["transitions"]],
        "unusual_interval_counts": [{
            "a": r.get("deck_a", {}).get("unusual_interval_count"),
            "b": r.get("deck_b", {}).get("unusual_interval_count"),
        } for r in report["transitions"]],
    }, indent=2))


if __name__ == "__main__":
    main()
