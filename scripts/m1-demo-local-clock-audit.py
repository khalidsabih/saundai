#!/usr/bin/env python3
"""Read-only local beat-clock counterfactual on the generated M1 demo pair.

Analyzes the two demo audio tracks once, then compares cue-local *observed*
beat intervals with rejected phase-lock trace trajectories.  It never changes
TrackAnalysis, plans, rendering, or safety.  In particular the script does NOT
apply derived rates to a plan or claim the synthetic fixture proves real-music
tempo estimation quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEMO_NAMES = {
    "01_opening_122bpm_Aminor.wav",
    "02_groove_124bpm_Eminor.wav",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def local_clock(beats, start, end, *, minimum_intervals=6):
    """Measure local BPM; report robust accepted intervals and rejected spans.

    Excludes intervals outside 85%-115% of the window median for estimating
    clock only, and always reports their count. This is *not* authorization to
    bridge over bad beats or correct them during a transition.
    """
    if not finite(start) or not finite(end) or end <= start:
        return {"status": "invalid_window"}
    values = [float(x) for x in beats]
    if any(not finite(x) for x in values) or any(b <= a for a, b in zip(values, values[1:])):
        return {"status": "invalid_grid"}
    intervals = [
        (a, b, b - a)
        for a, b in zip(values, values[1:])
        if a >= start and b <= end
    ]
    if len(intervals) < minimum_intervals:
        return {"status": "insufficient_intervals", "interval_count": len(intervals)}
    raw_period = median(x[2] for x in intervals)
    accepted = [x[2] for x in intervals if .85 * raw_period <= x[2] <= 1.15 * raw_period]
    discarded = [x for x in intervals if x[2] < .85 * raw_period or x[2] > 1.15 * raw_period]
    if len(accepted) < minimum_intervals:
        return {"status": "insufficient_clean_intervals",
                "interval_count": len(intervals),
                "rejected_interval_count": len(discarded)}
    return {
        "status": "measured",
        "bpm": round(60.0 / median(accepted), 5),
        "interval_count": len(intervals),
        "accepted_interval_count": len(accepted),
        "rejected_interval_count": len(discarded),
        "outlier_windows": [
            {"start_seconds": round(a, 6), "end_seconds": round(b, 6)}
            for a, b, _ in discarded[:8]
        ],
    }


def compare(rows, tracks):
    records = []
    skipped = {}
    for row in rows:
        a = tracks.get(row.get("track_a"))
        b = tracks.get(row.get("track_b"))
        if a is None or b is None:
            skipped["missing_demo_track"] = skipped.get("missing_demo_track", 0) + 1
            continue
        keys = ("exit_time_a_seconds", "entry_time_b_seconds", "transition_seconds",
                "render_rate_a", "render_rate_b", "phase_nudge_seconds_b")
        if any(not finite(row.get(k)) for k in keys):
            skipped["missing_trace_parameters"] = skipped.get("missing_trace_parameters", 0) + 1
            continue
        exit_a, entry_b, duration, rate_a, rate_b, nudge = (float(row[k]) for k in keys)
        if duration <= 0 or rate_a <= 0 or rate_b <= 0:
            skipped["invalid_trace_parameters"] = skipped.get("invalid_trace_parameters", 0) + 1
            continue
        estimate_a = local_clock(a["beats_seconds"], exit_a, exit_a + duration * rate_a)
        estimate_b = local_clock(b["beats_seconds"], entry_b + nudge,
                                 entry_b + nudge + duration * rate_b)
        points = row.get("points")
        points = points if isinstance(points, list) else []
        observed_points = sorted([
            (float(p["fraction"]), float(p["phase_error_ms"]))
            for p in points if isinstance(p, dict)
            and finite(p.get("fraction")) and finite(p.get("phase_error_ms"))
        ])
        observed = (observed_points[-1][1] - observed_points[0][1]
                    if len(observed_points) >= 2 else None)
        entry = {
            "exit_time_a_seconds": exit_a,
            "entry_time_b_seconds": entry_b,
            "duration_seconds": duration,
            "render_rate_a": rate_a,
            "render_rate_b": rate_b,
            "local_grid_a": estimate_a,
            "local_grid_b": estimate_b,
            "observed_phase_change_ms": round(observed, 3) if observed is not None else None,
            "predicted_local_clock_phase_change_ms": None,
            "unexplained_phase_change_ms": None,
        }
        if (estimate_a.get("status") == "measured"
                and estimate_b.get("status") == "measured"
                and finite(a.get("bpm")) and float(a["bpm"]) > 0):
            effective_a = estimate_a["bpm"] * rate_a
            effective_b = estimate_b["bpm"] * rate_b
            # Same output-beat coordinate convention as the phase controller:
            # (delta beat count)*target beat duration, measured in ms.
            # Plan target BPM is scalar A BPM for this standalone demo plan;
            # recorded explicit plan target, if present, takes precedence.
            target = row.get("target_bpm")
            target = float(target) if finite(target) and float(target) > 0 else float(a["bpm"]) * rate_a
            predicted = duration * (effective_a - effective_b) / target * 1000.0
            entry["predicted_local_clock_phase_change_ms"] = round(predicted, 3)
            if observed is not None:
                entry["unexplained_phase_change_ms"] = round(observed - predicted, 3)
        records.append(entry)
    evaluable = [r for r in records if r["unexplained_phase_change_ms"] is not None]
    signs = sum(
        1 for r in evaluable
        if abs(r["predicted_local_clock_phase_change_ms"]) > 1.0
        and abs(r["observed_phase_change_ms"]) > 1.0
        and (r["predicted_local_clock_phase_change_ms"] > 0) == (r["observed_phase_change_ms"] > 0)
    )
    summary = lambda values: round(median(values), 3) if values else None
    return {
        "schema": "saund-m1-demo-local-clock-audit-v1",
        "trace_rows": len(rows),
        "eligible_configs": len(records),
        "evaluated_configs": len(evaluable),
        "skipped": skipped,
        "median_absolute_unexplained_ms": summary([abs(r["unexplained_phase_change_ms"]) for r in evaluable]),
        "median_predicted_local_clock_change_ms": summary([r["predicted_local_clock_phase_change_ms"] for r in evaluable]),
        "median_observed_phase_change_ms": summary([r["observed_phase_change_ms"] for r in evaluable]),
        "matching_direction_count": signs,
        "non_evaluable_count": len(records) - len(evaluable),
        "examples": records[:8],
        "counterfactual_only": True,
        "notes": [
            "Observed local beat clock is not independent audio-onset or annotated reference evidence.",
            "Rejecting interval outliers for a BPM estimate never makes an unsafe overlap safe.",
            "End-to-end phase changes can include beat indexing discontinuities and warp effects not reproduced here.",
            "No production BPM, transition plan, correction envelope, or neural authority is modified.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track

    tracks = {}
    root = args.audio_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("audio root does not exist")
    for name in sorted(DEMO_NAMES):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing known generator fixture: {name}")
        analysis = analyze_track(path)
        tracks[name] = {"beats_seconds": analysis.beats_seconds, "bpm": analysis.bpm,
                        "neural_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used")}
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(r, dict) for r in rows):
        raise ValueError("trace must contain JSON objects")
    report = compare(rows, tracks)
    report["track_provenance"] = {
        name: {"scalar_bpm": track["bpm"], "neural_used": track["neural_used"]}
        for name, track in tracks.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 local clock audit failed: {exc}") from exc
