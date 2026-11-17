#!/usr/bin/env python3
"""Read-only exact phase-index decomposition for rejected synthetic M1 overlaps.

Replays the current deck's continuous beat-position logic at the recorded
sample fractions, then separates integer-index movement from within-beat
fractions. Never changes beats, TransitionPlans or the safety envelope.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEMO = ("01_opening_122bpm_Aminor.wav", "02_groove_124bpm_Eminor.wav")


def valid(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value))


def position_components(beats, seconds, expected_period):
    """Mirror deck's default sequential-index phase and expose its components."""
    if len(beats) < 2 or not valid(seconds) or expected_period <= 0:
        return None
    i = bisect.bisect_right(beats, seconds)
    if i == 0 or i >= len(beats):
        return None
    a, b = float(beats[i - 1]), float(beats[i])
    interval = b - a
    if not (.65 * expected_period <= interval <= 1.35 * expected_period):
        return None
    fraction = (seconds - a) / interval
    return {"index": i - 1, "fraction": fraction, "position": i - 1 + fraction,
            "interval_ms": interval * 1000.0}


def replay(trace, tracks):
    from djai.performance.deck import _continuous_beat_position

    evaluated = []
    skipped = {}
    for row in trace:
        a = tracks.get(row.get("track_a"))
        b = tracks.get(row.get("track_b"))
        if a is None or b is None:
            skipped["unknown_track"] = skipped.get("unknown_track", 0) + 1
            continue
        keys = ("exit_time_a_seconds", "entry_time_b_seconds", "transition_seconds",
                "render_rate_a", "render_rate_b", "phase_nudge_seconds_b")
        if any(not valid(row.get(k)) for k in keys):
            skipped["missing_trace_fields"] = skipped.get("missing_trace_fields", 0) + 1
            continue
        ea, eb, duration, ra, rb, nudge = (float(row[k]) for k in keys)
        if duration <= 0 or min(ra, rb) <= 0:
            skipped["invalid_trace_fields"] = skipped.get("invalid_trace_fields", 0) + 1
            continue
        # The original planner uses target BPM in the deck profile. The
        # generated pair's standalone planner target equals outgoing scalar BPM.
        target = float(a["bpm"]) * ra
        beat_seconds = 60.0 / target
        points = row.get("points") or []
        if not isinstance(points, list):
            points = []
        samples = []
        for point in points:
            if not isinstance(point, dict) or not valid(point.get("fraction")):
                continue
            fraction = float(point["fraction"])
            t_a = ea + duration * fraction * ra
            t_b = eb + nudge + duration * fraction * rb
            pa = position_components(a["beats"], t_a, beat_seconds * ra)
            pb = position_components(b["beats"], t_b, beat_seconds * rb)
            if pa is None or pb is None:
                continue
            native_a = _continuous_beat_position(
                a["beats"], t_a, expected_source_beat_seconds=beat_seconds * ra
            )
            native_b = _continuous_beat_position(
                b["beats"], t_b, expected_source_beat_seconds=beat_seconds * rb
            )
            if native_a is None or native_b is None or abs(native_a - pa["position"]) > 1e-8 or abs(native_b - pb["position"]) > 1e-8:
                raise ValueError("read-only phase decomposition diverged from deck implementation")
            samples.append({
                "fraction": fraction,
                "index_a": pa["index"], "index_b": pb["index"],
                "fraction_a": pa["fraction"], "fraction_b": pb["fraction"],
                "phase_a": pa["position"], "phase_b": pb["position"],
                "phase_difference_beats": pa["position"] - pb["position"],
                "interval_a_ms": pa["interval_ms"], "interval_b_ms": pb["interval_ms"],
            })
        if len(samples) < 2:
            skipped["insufficient_replayed_points"] = skipped.get("insufficient_replayed_points", 0) + 1
            continue
        first, last = samples[0], samples[-1]
        phase_change = ((last["phase_difference_beats"] - first["phase_difference_beats"]) * beat_seconds * 1000.0)
        integer_change = ((last["index_a"] - first["index_a"]) - (last["index_b"] - first["index_b"])) * beat_seconds * 1000.0
        fractional_change = (((last["fraction_a"] - first["fraction_a"]) - (last["fraction_b"] - first["fraction_b"])) * beat_seconds * 1000.0)
        observed = [float(p["phase_error_ms"]) for p in points if isinstance(p, dict) and valid(p.get("phase_error_ms"))]
        trace_change = observed[-1] - observed[0] if len(observed) >= 2 else None
        evaluated.append({
            "exit_time_a_seconds": ea, "entry_time_b_seconds": eb,
            "replayed_phase_change_ms": round(phase_change, 3),
            "integer_index_change_ms": round(integer_change, 3),
            "within_beat_fraction_change_ms": round(fractional_change, 3),
            "trace_phase_change_ms": round(trace_change, 3) if trace_change is not None else None,
            "replay_minus_trace_ms": round(phase_change - trace_change, 3) if trace_change is not None else None,
            "sample_count": len(samples),
            "first_sample": {k: round(v, 5) if isinstance(v, float) else v for k,v in first.items()},
            "last_sample": {k: round(v, 5) if isinstance(v, float) else v for k,v in last.items()},
        })
    measure = lambda key: round(median([abs(x[key]) for x in evaluated if x[key] is not None]), 3) if any(x[key] is not None for x in evaluated) else None
    return {
        "schema": "saund-m1-phase-index-replay-v1",
        "trace_rows": len(trace), "evaluated": len(evaluated),
        "skipped": skipped,
        "median_abs_integer_index_change_ms": measure("integer_index_change_ms"),
        "median_abs_fraction_change_ms": measure("within_beat_fraction_change_ms"),
        "median_abs_replay_trace_difference_ms": measure("replay_minus_trace_ms"),
        "examples": evaluated[:10], "counterfactual_only": True,
        "notes": [
            "Component decomposition uses the default sequential-index mode; rerun only with DJAI_BEAT_GAP_AWARE off.",
            "An integer beat-index difference is not automatically a missing beat: fractional movement can counterbalance it.",
            "The replay uses observed beat grids, not independently verified onset events.",
            "No correction, plan approval, rendering, or phase-limit changes are made.",
        ]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    if os.environ.get("DJAI_BEAT_GAP_AWARE", "0").strip().lower() not in {"0", "false", "off", "no", ""}:
        raise ValueError("default phase indexing required: disable DJAI_BEAT_GAP_AWARE for replay")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track

    tracks = {}
    for name in DEMO:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing synthetic demo fixture: {name}")
        analysis = analyze_track(path)
        tracks[name] = {"beats": analysis.beats_seconds, "bpm": analysis.bpm}
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(x, dict) for x in rows):
        raise ValueError("trace rows must be JSON objects")
    result = replay(rows, tracks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 phase replay failed: {exc}") from exc
