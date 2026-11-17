#!/usr/bin/env python3
"""Read-only meter audit for SAUND's explicitly generated 4/4 demo-set pair.

This is a *diagnostic*, not a meter override. Do not infer musical meter from
filenames or apply the known demo reference to arbitrary real music.
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

DEMO_TRACKS = (
    "01_opening_122bpm_Aminor.wav",
    "02_groove_124bpm_Eminor.wav",
)
REFERENCE_METER = 4


def measure(track) -> dict:
    beats = track.beats_seconds
    downs = track.downbeats_seconds
    intervals = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    down_intervals = [b - a for a, b in zip(downs, downs[1:]) if b > a]
    beat_period = statistics.median(intervals) if intervals else None
    down_period = statistics.median(down_intervals) if down_intervals else None
    observed = down_period / beat_period if beat_period and down_period else None
    neural = track.neural_evidence.get("beat_downbeat", {})
    if not isinstance(neural, dict):
        neural = {}
    return {
        "predicted_meter": track.time_signature,
        "meter_confidence": track.time_signature_confidence,
        "expected_meter": REFERENCE_METER,
        "meter_matches_generator": track.time_signature == REFERENCE_METER,
        "bpm": track.bpm,
        "beat_grid_confidence": track.beat_grid_confidence,
        "downbeat_confidence": track.downbeat_confidence,
        "beat_count": len(beats),
        "downbeat_count": len(downs),
        "inferred_beats_per_downbeat_interval": (
            round(observed, 4) if observed is not None and math.isfinite(observed) else None
        ),
        "neural_available": neural.get("available"),
        "neural_used": neural.get("used"),
        "neural_confidence": neural.get("confidence"),
        "neural_reason": neural.get("reason"),
    }


def audit(directory: Path, analyze) -> dict:
    rows = []
    for filename in DEMO_TRACKS:
        path = (directory / filename).resolve()
        if not path.is_file():
            raise ValueError(f"Missing expected generated fixture: {path}")
        row = {"filename": filename, **measure(analyze(path))}
        rows.append(row)
    return {
        "schema_version": "saund-m1-demo-meter-audit-v1",
        "source": "src/djai/demo.py generate_demo_set (four beats per bar)",
        "results": rows,
        "all_meters_match_generator": all(row["meter_matches_generator"] for row in rows),
        "production_changed": False,
        "rhythm_policy_changed": False,
        "rendered_transitions_tested": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Refusing to overwrite existing report")
    if not args.audio_root.is_dir():
        raise ValueError("Audio root must exist")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track

    report = audit(args.audio_root, analyze_track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_meters_match_generator"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Demo meter audit failed: {exc}") from exc
