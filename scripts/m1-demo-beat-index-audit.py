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
    ideal_total = len(range(0, int((duration - 0.01) / period) + 1))
    return {
        "generator_bpm": bpm,
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
