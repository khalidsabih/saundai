#!/usr/bin/env python3
"""Read-only correlation of rejected phase-lock overlap windows with demo beat anomalies.

Uses an existing private phase JSONL trace and the separate synthetic-demo
beat-index audit JSON. Never runs planning, inference, or rendering and never
changes a beat grid or relaxes the phase-integrity envelope.

Caveat: the trace contains rejections only. Window intersection does not prove
that an anomalous beat caused the safety rejection.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def correlate(trace_rows: list[dict], audit: dict) -> dict:
    if not isinstance(audit, dict) or not isinstance(audit.get("tracks"), list):
        raise ValueError("expected a generated-demo beat-index audit")
    clusters_by_name = {}
    for track in audit["tracks"]:
        name = track.get("filename")
        if not isinstance(name, str):
            continue
        clusters = track.get("beat_grid", {}).get("local_interval_diagnostics", {}).get("outlier_clusters", [])
        clusters_by_name[name] = [
            {"start": float(c["start_seconds"]), "end": float(c["end_seconds"])}
            for c in clusters
            if _finite_number(c.get("start_seconds"))
            and _finite_number(c.get("end_seconds"))
            and float(c["end_seconds"]) > float(c["start_seconds"])
        ]
    groups = {"crosses_anomaly": [], "does_not_cross_anomaly": []}
    errors = Counter()
    for item in trace_rows:
        name_a, name_b = item.get("track_a"), item.get("track_b")
        if name_a not in clusters_by_name or name_b not in clusters_by_name:
            errors["trace_track_not_in_audit"] += 1
            continue
        fields = ("exit_time_a_seconds", "entry_time_b_seconds", "transition_seconds", "render_rate_a", "render_rate_b", "phase_nudge_seconds_b")
        if any(not _finite_number(item.get(k)) for k in fields):
            errors["missing_overlap_parameters"] += 1
            continue
        exit_a = float(item["exit_time_a_seconds"])
        entry_b = float(item["entry_time_b_seconds"]) + float(item["phase_nudge_seconds_b"])
        duration = float(item["transition_seconds"])
        rate_a, rate_b = float(item["render_rate_a"]), float(item["render_rate_b"])
        if duration <= 0 or rate_a <= 0 or rate_b <= 0:
            errors["invalid_overlap_parameters"] += 1
            continue
        intervals = (
            (name_a, exit_a, exit_a + duration * rate_a),
            (name_b, entry_b, entry_b + duration * rate_b),
        )
        crossed = []
        for name, start, end in intervals:
            for c in clusters_by_name[name]:
                if start < c["end"] and end > c["start"]:
                    crossed.append({"track": name, "cluster_start_seconds": c["start"], "cluster_end_seconds": c["end"]})
        residual = item.get("max_residual_ms")
        valid_residual = (
            _finite_number(residual)
            and float(residual) < 999.0
            and "insufficient" not in str(item.get("reason", "")).lower()
        )
        key = "crosses_anomaly" if crossed else "does_not_cross_anomaly"
        groups[key].append({
            "exit_time_a_seconds": exit_a,
            "entry_time_b_seconds": entry_b,
            "residual_ms": float(residual) if valid_residual else None,
            "crossed_clusters": crossed,
        })
    summarized = {}
    for name, rows in groups.items():
        measured = [row["residual_ms"] for row in rows if row["residual_ms"] is not None]
        summarized[name] = {
            "rejected_configurations": len(rows),
            "measured_residual_count": len(measured),
            "median_rejected_residual_ms": round(median(measured), 3) if measured else None,
            "example_rejected_cues": rows[:4],
        }
    return {
        "schema": "saund-m1-phase-anomaly-overlap-v1",
        "groups": summarized,
        "skipped": dict(errors),
        "trace_rows": len(trace_rows),
        "rejected_only": True,
        "causal_attribution": False,
        "notes": [
            "Correlation is not causation. Rejected traces lack accepted-candidate denominators.",
            "The audit's generator beat references are synthetic-only and must never influence production planning.",
            "Does not change beat indexes, phase-lock safety limits, renderer or rhythm policy.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--beat-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing report")
    audit = json.loads(args.beat_audit.read_text(encoding="utf-8"))
    rows = []
    for line in args.trace.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("trace row must be an object")
            rows.append(record)
    result = correlate(rows, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Phase overlap audit failed: {exc}") from exc
