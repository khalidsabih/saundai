#!/usr/bin/env python3
"""Read-only test of possible half-beat phase-lock on rights-free synthetic demos.

This compares detected beat timestamps against (1) waveform-measured kick
attacks and (2) midpoints between *consecutive measured kick attacks*. The
midpoints are only timing hypotheses, NOT independently measured offbeat
instrument attacks. The generator supplies pulse search windows only; never
use these references to modify a production beat grid or approve rendering.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location)

import importlib.util


def load_kick_extractor():
    path = ROOT / "scripts" / "m1-synthetic-audio-kick-audit.py"
    spec = importlib.util.spec_from_file_location("m1_synthetic_kick_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError("synthetic kick audit unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def median_ms(values):
    return round(float(statistics.median(values)), 3) if values else None


def evaluate(grid, kick_events, regions, *, tolerance_ms=80.0):
    """Compare exact beat positions to measured kicks and *inferred* midpoints.

    Missing adjacent kicks cannot produce a midpoint. The midpoint is NOT
    evidence of a hi-hat or other musical event without its own WAV detector.
    """
    if len(grid) < 2 or any(not math.isfinite(float(x)) for x in grid) or any(
        b <= a for a, b in zip(grid, grid[1:])
    ):
        raise ValueError("invalid detected beat grid")
    observed = sorted(
        (int(e["index"]), float(e["audio_attack_seconds"]))
        for e in kick_events if e.get("status") == "measured"
    )
    kicks = sorted((time, index) for index, time in observed)
    kick_times = [x[0] for x in kicks]
    midpoints = sorted(
        (0.5 * (left_time + right_time), left_index)
        for (left_index, left_time), (right_index, right_time)
        in zip(observed, observed[1:])
        if right_index == left_index + 1 and right_time > left_time
    )
    midpoint_times = [x[0] for x in midpoints]

    def closest(beat, timestamps):
        if not timestamps:
            return None
        pos = bisect.bisect_left(timestamps, beat)
        return min(
            (timestamps[k] for k in (pos - 1, pos) if 0 <= k < len(timestamps)),
            key=lambda t: abs(t - beat),
        )

    results = []
    for name, start, end in regions:
        subset = [
            float(t) for t in grid
            if (start is None or t >= start) and (end is None or t < end)
        ]
        kick_errors = []
        mid_errors = []
        both = []
        examples = []
        for beat in subset:
            kick = closest(beat, kick_times)
            midpoint = closest(beat, midpoint_times)
            if kick is None or midpoint is None:
                continue
            kick_ms = abs(beat - kick) * 1000.0
            midpoint_ms = abs(beat - midpoint) * 1000.0
            kick_errors.append(kick_ms)
            mid_errors.append(midpoint_ms)
            nearer = "midpoint" if midpoint_ms < kick_ms else "kick"
            both.append((kick_ms, midpoint_ms))
            if len(examples) < 5:
                examples.append({
                    "beat_seconds": round(beat, 6),
                    "distance_to_measured_kick_ms": round(kick_ms, 3),
                    "distance_to_inferred_midpoint_ms": round(midpoint_ms, 3),
                    "closer_to": nearer,
                })
        results.append({
            "region": name,
            "detected_beat_count": len(subset),
            "evaluable_beat_count": len(both),
            "insufficient_acoustic_neighbor_count": len(subset) - len(both),
            "closer_to_kick_count": sum(k <= m for k, m in both),
            "closer_to_inferred_midpoint_count": sum(m < k for k, m in both),
            "within_kick_tolerance_count": sum(k <= tolerance_ms for k, _ in both),
            "within_inferred_midpoint_tolerance_count": sum(m <= tolerance_ms for _, m in both),
            "median_distance_to_measured_kick_ms": median_ms(kick_errors),
            "median_distance_to_inferred_midpoint_ms": median_ms(mid_errors),
            "first_examples": examples,
        })
    return {
        "regions": results,
        "measured_kick_count": len(kicks),
        "inferred_midpoint_count": len(midpoints),
        "midpoint_is_inferred_not_measured": True,
        "no_beat_or_plan_modified": True,
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
    extractor = load_kick_extractor()
    tracks = []
    for name, bpm in extractor.FIXTURES:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing known generated fixture: {name}")
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        events = extractor.extract_kick_attacks(audio[:, 0], sample_rate, bpm)
        analysis = analyze_track(path)
        regions = (
            extractor.OUTGOING_REGIONS if name == extractor.FIXTURES[0][0]
            else (("entire_incoming", None, None),)
        )
        tracks.append({
            "filename": name,
            "estimated_scalar_bpm": analysis.bpm,
            "neural_grid_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
            **evaluate(analysis.beats_seconds, events, regions),
        })
    result = {
        "schema": "saund-m1-synthetic-offbeat-midpoint-audit-v1",
        "tracks": tracks,
        "production_changed": False,
        "transition_rendered": False,
        "no_beat_grid_corrected": True,
        "warning": (
            "Half-beat reference is the midpoint of adjacent independently measured kicks, "
            "not an independently detected hi-hat/onset. Synthetic-only hypothesis test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 offbeat midpoint audit failed: {exc}") from exc
