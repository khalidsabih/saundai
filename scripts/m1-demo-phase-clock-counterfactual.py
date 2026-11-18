#!/usr/bin/env python3
"""Read-only synthetic demo tempo/grid consistency analysis of rejected phase traces.

Uses the generated-demo beat-index audit's inferred beat-clock BPM and a
private rejected phase-lock JSONL trace. This makes NO new plans, no beat
corrections and no authorization decisions. BPM from the synthetic generator
is a reference for this demo only; no oracle is permitted in production.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def _estimate(tracks):
    result = {}
    for track in tracks:
        name = track.get("filename")
        drift = track.get("beat_grid", {}).get("sequential_index_drift", {})
        bpm = drift.get("implied_bpm_from_linear_fit")
        if isinstance(name, str) and _number(bpm) and float(bpm) > 0:
            result[name] = {
                "grid_fit_bpm": float(bpm),
                "estimated_bpm": track.get("estimated_bpm"),
                "generator_bpm": track.get("beat_grid", {}).get("generator_bpm"),
            }
    return result


def _median(values):
    return round(median(values), 3) if values else None


def diagnose(rows, audit):
    if not isinstance(audit, dict) or not isinstance(audit.get("tracks"), list):
        raise ValueError("expected M1 synthetic beat-index audit")
    tracks = _estimate(audit["tracks"])
    groups = {"measured": [], "no_phase_points": []}
    skipped = {}
    for row in rows:
        a = tracks.get(row.get("track_a"))
        b = tracks.get(row.get("track_b"))
        fields = ("transition_seconds", "render_rate_a", "render_rate_b")
        if a is None or b is None:
            skipped["unknown_track"] = skipped.get("unknown_track", 0) + 1
            continue
        if any(not _number(row.get(k)) for k in fields):
            skipped["missing_rates_or_duration"] = skipped.get("missing_rates_or_duration", 0) + 1
            continue
        duration, rate_a, rate_b = (float(row[k]) for k in fields)
        if min(duration, rate_a, rate_b) <= 0:
            skipped["invalid_rates_or_duration"] = skipped.get("invalid_rates_or_duration", 0) + 1
            continue
        # Source time advances by render_rate * output time. Effective output
        # beat frequency is beat-clock BPM * source-time rate.
        bpm_a = a["grid_fit_bpm"] * rate_a
        bpm_b = b["grid_fit_bpm"] * rate_b
        scalar_a = a["estimated_bpm"]
        target = float(scalar_a) * rate_a if _number(scalar_a) else None
        predicted_ms = (
            duration * (bpm_a - bpm_b) / target * 1000.0
            if target is not None and target > 0 else None
        )
        points = row.get("points")
        points = points if isinstance(points, list) else []
        finite = [
            (float(p["fraction"]), float(p["phase_error_ms"]))
            for p in points if isinstance(p, dict)
            and _number(p.get("fraction")) and _number(p.get("phase_error_ms"))
        ]
        finite.sort(key=lambda p: p[0])
        observed_ms = finite[-1][1] - finite[0][1] if len(finite) >= 2 else None
        item = {
            "exit_time_a_seconds": row.get("exit_time_a_seconds"),
            "entry_time_b_seconds": row.get("entry_time_b_seconds"),
            "transition_seconds": duration,
            "planned_rate_a": rate_a,
            "planned_rate_b": rate_b,
            "grid_fit_output_bpm_a": round(bpm_a, 4),
            "grid_fit_output_bpm_b": round(bpm_b, 4),
            "output_clock_difference_bpm": round(bpm_a - bpm_b, 4),
            "uncontrolled_clock_drift_prediction_ms": round(predicted_ms, 3) if predicted_ms is not None else None,
            "observed_phase_change_ms": round(observed_ms, 3) if observed_ms is not None else None,
            "residual_after_clock_prediction_ms": (
                round(observed_ms - predicted_ms, 3)
                if observed_ms is not None and predicted_ms is not None else None
            ),
        }
        groups["measured" if observed_ms is not None else "no_phase_points"].append(item)
    valid = groups["measured"]
    predicted = [x["uncontrolled_clock_drift_prediction_ms"] for x in valid if x["uncontrolled_clock_drift_prediction_ms"] is not None]
    observed = [x["observed_phase_change_ms"] for x in valid]
    errors = [x["residual_after_clock_prediction_ms"] for x in valid if x["residual_after_clock_prediction_ms"] is not None]
    sign_match = [
        x for x in valid
        if x["uncontrolled_clock_drift_prediction_ms"] is not None
        and abs(x["uncontrolled_clock_drift_prediction_ms"]) > 1
        and abs(x["observed_phase_change_ms"]) > 1
        and (x["uncontrolled_clock_drift_prediction_ms"] > 0) == (x["observed_phase_change_ms"] > 0)
    ]
    return {
        "schema": "saund-m1-demo-phase-clock-counterfactual-v1",
        "trace_rows": len(rows),
        "evaluated_rows": len(valid),
        "no_phase_point_rows": len(groups["no_phase_points"]),
        "skipped": skipped,
        "track_clock_sources": tracks,
        "median_predicted_uncontrolled_phase_change_ms": _median(predicted),
        "median_observed_phase_change_ms": _median(observed),
        "median_abs_unexplained_change_ms": _median([abs(x) for x in errors]),
        "matching_change_direction_count": len(sign_match),
        "examples": valid[:6],
        "counterfactual_only": True,
        "notes": [
            "Synthetic generator-grounded grid-fit clocks are research diagnostics, not independent production meter evidence.",
            "This predicts uncorrected clock mismatch; it does not replay phase warps, jumps or phase-nudge effects.",
            "Global fitted BPM may be distorted by local beat compression, so predicted residual is not proof of causal explanation.",
            "Never substitute reference BPM into signed execution plans or relax the 25 ms phase residual limit.",
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
    rows = [json.loads(s) for s in args.trace.read_text(encoding="utf-8").splitlines() if s.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("trace must consist of JSON objects")
    result = diagnose(rows, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 phase clock audit failed: {exc}") from exc
