#!/usr/bin/env python3
"""Read-only beat-index audit for SAUND's generated, constant-tempo 4/4 demo pair.

Compares estimated beat timestamps against exact beat events from the demo
generator. This script is only valid for these known synthetic fixtures;
reference event times must never be fed into production analysis or planning.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEMO = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)


def compare_to_generator(beats, bpm, duration):
    period = 60.0 / bpm
    observed = [float(x) for x in beats]
    if len(observed) < 2 or any(not math.isfinite(x) for x in observed):
        raise ValueError("insufficient/nonfinite beat positions")
    if any(b <= a for a, b in zip(observed, observed[1:])):
        raise ValueError("beat positions must increase")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    # The generator places kicks at t=0, period, 2*period, ... with no swing.
    # Event-index gaps and duplicate nearest indices reveal dropped/doubled
    # beats even where a median BPM metric appears accurate.
    indices = [round(t / period) for t in observed]
    residuals_ms = [1000.0 * abs(t - i * period) for t, i in zip(observed, indices)]
    jumps = [b - a for a, b in zip(indices, indices[1:])]
    # A nearest-reference error can wrap near half a beat and look like a
    # missing/duplicate event. Independently track a sequential beat index to
    # expose accumulated tempo slope without modulo-beat wrapping.
    sequential_indices = [indices[0] + i for i in range(len(observed))]
    signed_errors = [
        1000.0 * (t - i * period)
        for t, i in zip(observed, sequential_indices)
    ]
    x = list(range(len(observed)))
    mid_x = statistics.mean(x)
    mid_error = statistics.mean(signed_errors)
    denominator = sum((n - mid_x) ** 2 for n in x)
    slope_ms_per_beat = (
        sum((n - mid_x) * (e - mid_error) for n, e in zip(x, signed_errors))
        / denominator
    )
    fitted_ms_per_minute = slope_ms_per_beat * bpm
    implied_period = period + slope_ms_per_beat / 1000.0
    first_third = signed_errors[: max(1, len(signed_errors) // 3)]
    last_third = signed_errors[-max(1, len(signed_errors) // 3):]
    # Inspect local intervals independently of the global fitted slope. A
    # single too-short/long interval can move the sequential beat index by one
    # pulse and make ordinary linear regression look like tempo drift.
    interval_events = []
    for i, (left, right) in enumerate(zip(observed, observed[1:])):
        interval = right - left
        deviation_ms = 1000.0 * (interval - period)
        if abs(deviation_ms) >= 0.15 * 1000.0 * period:
            interval_events.append({
                "between_observed_indices": [i, i + 1],
                "left_seconds": round(left, 6),
                "right_seconds": round(right, 6),
                "interval_ms": round(interval * 1000.0, 3),
                "deviation_from_generator_period_ms": round(deviation_ms, 3),
                "nearest_generator_index_step": jumps[i],
            })
    # Report the complete count and bounded details; do not silently assume
    # all phase residual is physical drift.
    quartiles = []
    for part in range(4):
        start = part * len(signed_errors) // 4
        end = (part + 1) * len(signed_errors) // 4
        quartiles.append(round(statistics.median(signed_errors[start:end]), 3))
    ideal_total = len(range(0, int((duration - 0.01) / period) + 1))
    return {
        "generator_bpm": bpm,
        "sequential_index_drift": {
            "reference_anchor_index": indices[0],
            "first_third_median_signed_ms": round(statistics.median(first_third), 3),
            "last_third_median_signed_ms": round(statistics.median(last_third), 3),
            "slope_ms_per_beat": round(slope_ms_per_beat, 4),
            "slope_ms_per_minute": round(fitted_ms_per_minute, 3),
            "implied_bpm_from_linear_fit": round(60.0 / implied_period, 4) if implied_period > 0 else None,
            "end_minus_start_signed_ms": round(signed_errors[-1] - signed_errors[0], 3),
            "index_assumption": "consecutive beats from first nearest-generator pulse; diagnose skipped/doubled beats separately",
        },
        "local_interval_diagnostics": {
            "expected_interval_ms": round(period * 1000.0, 3),
            "outlier_threshold_percent": 15,
            "outlier_interval_count": len(interval_events),
            "first_outlier_intervals": interval_events[:12],
            "signed_sequential_error_quartile_medians_ms": quartiles,
            "nearest_index_wraps": [
                {"between_observed_indices": [i, i + 1],
                 "event_index_step": step,
                 "left_seconds": round(observed[i], 6),
                 "right_seconds": round(observed[i+1], 6)}
                for i, step in enumerate(jumps) if step != 1
            ][:12],
        },
        "observed_count": len(observed),
        "expected_count_approx": ideal_total,
        "median_absolute_grid_error_ms": round(statistics.median(residuals_ms), 3),
        "max_absolute_grid_error_ms": round(max(residuals_ms), 3),
        "missing_event_index_jumps": sum(step > 1 for step in jumps),
        "duplicate_event_index_jumps": sum(step == 0 for step in jumps),
        "reverse_index_jumps": sum(step < 0 for step in jumps),
        "largest_index_jump": max(jumps) if jumps else 0,
        "longest_inferred_missing_run": max((step - 1 for step in jumps), default=0),
        "first_observed_seconds": round(observed[0], 6),
        "last_observed_seconds": round(observed[-1], 6),
        "reference_is_synthetic_only": True,
    }


def audit(audio_root, analyze):
    tracks = []
    for name, bpm in DEMO:
        path = (audio_root / name).resolve()
        if not path.is_file():
            raise ValueError(f"missing demo fixture: {path}")
        analysis = analyze(path)
        tracks.append({
            "filename": name,
            "estimated_bpm": analysis.bpm,
            "estimated_meter": analysis.time_signature,
            "neural_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
            "beat_grid": compare_to_generator(
                analysis.beats_seconds, bpm, analysis.duration_seconds
            ),
        })
    return {
        "schema": "saund-m1-demo-beat-index-audit-v1",
        "tracks": tracks,
        "production_changed": False,
        "rendered_transitions_tested": False,
        "warning": "Diagnostic synthetic reference only. No beat correction or permission to relax phase-lock safety.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track
    report = audit(args.audio_root.expanduser().resolve(), analyze_track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"M1 demo beat-index audit failed: {exc}") from exc
