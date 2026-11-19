#!/usr/bin/env python3
"""Offline preflight: beat grids, downbeat-bar alignment and cue-window safety.

Only derives hypothetical bar-aligned cues. No DJ mixes are rendered; this is
not a replacement for listening tests, mix QC or cross-domain verification.
Requires explicitly rights-approved Ballroom validation data, never final test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_MIN = 0.50
BPM_GRID_MAX_ERROR = 2.0
BEAT_SNAP_MAX_SECONDS = 0.10
BAR_DURATION_RELATIVE_MAX_ERROR = 0.20
CUE_REF_MAX_SECONDS = 0.10
MIN_BARS_FOR_CUE = 5
CUE_BARS = 4


def load(script_name, module_name):
    source = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing module: {script_name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def meter_from_reference(item):
    """Count labeled beat events between successive annotated downbeats.

    Honors 3/4 waltzes and irregular reference bars; do not impose 4/4.
    """
    beats = item["beats_seconds"]
    downs = item["downbeats_seconds"]
    if len(downs) < 2:
        raise ValueError("at least two annotated downbeats required")
    counts = []
    for start, stop in zip(downs, downs[1:]):
        counts.append(sum(start - 0.035 <= b < stop - 0.035 for b in beats))
    counts = [c for c in counts if c > 0]
    if not counts:
        raise ValueError("no annotated beats per bar")
    return round(statistics.median(counts))


def grid_health(bpm, beats, downs, meter):
    failures = []
    if not finite(bpm) or bpm <= 0:
        failures.append("invalid_bpm")
    if not isinstance(beats, list) or len(beats) < 4 or any(not finite(x) or x < 0 for x in beats):
        failures.append("invalid_beats")
    elif any(b <= a for a, b in zip(beats, beats[1:])):
        failures.append("unordered_beats")
    if not isinstance(downs, list) or len(downs) < MIN_BARS_FOR_CUE or any(not finite(x) or x < 0 for x in downs):
        failures.append("insufficient_downbeats")
    elif any(b <= a for a, b in zip(downs, downs[1:])):
        failures.append("unordered_downbeats")
    if failures:
        return {"passed": False, "failures": failures}
    beat_interval = statistics.median(b - a for a, b in zip(beats, beats[1:]))
    bpm_grid = 60 / beat_interval
    if abs(bpm_grid - bpm) > BPM_GRID_MAX_ERROR:
        failures.append("bpm_grid_mismatch")
    offgrid = sum(min(abs(d - b) for b in beats) > BEAT_SNAP_MAX_SECONDS for d in downs)
    if offgrid:
        failures.append("offbeat_downbeats")
    bar_intervals = [b - a for a, b in zip(downs, downs[1:])]
    expected_bar = meter * beat_interval
    bad_bars = sum(abs(x - expected_bar) > expected_bar * BAR_DURATION_RELATIVE_MAX_ERROR for x in bar_intervals)
    if bad_bars:
        failures.append("irregular_bar_duration")
    # Hypothetical 4-bar transition boundary, anchored to actual predicted downbeats.
    cue_start = downs[-(CUE_BARS + 1)]
    cue_end = downs[-1]
    cue_span = cue_end - cue_start
    if abs(cue_span - CUE_BARS * expected_bar) > CUE_BARS * expected_bar * BAR_DURATION_RELATIVE_MAX_ERROR:
        failures.append("cue_window_duration_mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "grid_bpm": round(bpm_grid, 4),
        "meter_assumption_from_reference": meter,
        "offbeat_downbeats": offgrid,
        "irregular_bars": bad_bars,
        "cue_window_start": round(cue_start, 5),
        "cue_window_end": round(cue_end, 5),
        "cue_window_seconds": round(cue_span, 5),
    }


def choose(selected, candidate):
    """Mirror the already-frozen challenger rule, whole grid + BPM together."""
    if candidate.get("used") is True:
        return candidate, True
    confidence = candidate.get("confidence")
    bpm = candidate.get("bpm")
    beats = candidate.get("beats_seconds")
    if not (candidate.get("available") is True and finite(confidence) and confidence >= CONFIDENCE_MIN):
        return selected, False
    if not (finite(bpm) and isinstance(beats, list) and len(beats) >= 2):
        return selected, False
    if any(not finite(b) for b in beats) or any(b <= a for a, b in zip(beats, beats[1:])):
        return selected, False
    grid_bpm = 60 / statistics.median(b - a for a, b in zip(beats, beats[1:]))
    if abs(bpm - grid_bpm) > BPM_GRID_MAX_ERROR:
        return selected, False
    return candidate, True


def evaluate(manifest, outputs):
    rows = []
    for item in manifest["items"]:
        prediction = outputs[item["id"]]
        selected = prediction["selected"]
        candidate = prediction["candidate"]
        challenger, uses_neural = choose(selected, candidate)
        meter = meter_from_reference(item)
        s = grid_health(selected.get("bpm"), selected.get("beats_seconds"), selected.get("downbeats_seconds"), meter)
        c = grid_health(challenger.get("bpm"), challenger.get("beats_seconds"), challenger.get("downbeats_seconds"), meter)
        n = grid_health(candidate.get("bpm"), candidate.get("beats_seconds"), candidate.get("downbeats_seconds"), meter) if candidate.get("available") is True else None
        # The reference cue-phase diagnostic is descriptive: no ground truth
        # rendered transition was provided. Do not mark proximity as guaranteed.
        for g in (s, c, n):
            if g and "cue_window_start" in g:
                cue = g["cue_window_start"]
                g["cue_start_near_annotated_downbeat"] = min(abs(cue - t) for t in item["downbeats_seconds"]) <= CUE_REF_MAX_SECONDS
        rows.append({
            "id": item["id"],
            "genre": item.get("genre"),
            "reference_meter": meter,
            "previous_neural_accepted": candidate.get("used") is True,
            "challenger_uses_neural": uses_neural,
            "selected": s,
            "challenger": c,
            "raw_neural": n,
        })
    return {
        "schema_version": "saund-m1-transition-timing-preflight-v1",
        "policy": "frozen confidence>=0.50, BPM/median-grid BPM agreement<=2; previously accepted neural unchanged",
        "track_count": len(rows),
        "metrics": {
            "selected_grid_preflight_failures": sum(not r["selected"]["passed"] for r in rows),
            "challenger_grid_preflight_failures": sum(not r["challenger"]["passed"] for r in rows),
            "raw_neural_preflight_failures": sum(r["raw_neural"] is not None and not r["raw_neural"]["passed"] for r in rows),
            "challenger_cue_start_near_reference_downbeat": sum(r["challenger"].get("cue_start_near_annotated_downbeat") is True for r in rows),
            "challenger_cue_start_evaluable": sum("cue_start_near_annotated_downbeat" in r["challenger"] for r in rows),
            "challenger_neural_grids": sum(r["challenger_uses_neural"] for r in rows),
        },
        "per_track": rows,
        "production_changed": False,
        "rendered_transitions_tested": False,
        "cross_domain_tested": False,
        "warning": "This is a hypothetical 4-bar cue timing audit, not an audio render, DJ quality score, or production safety certification.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="optional raw inference fixture keyed by track ID")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("dataset_id") != "ballroom" or manifest.get("split") != "validation" or manifest.get("rights_confirmed") is not True:
        raise ValueError("requires approved Ballroom validation cohort; final test stays sealed")
    benchmark = load("m1-rhythm-benchmark.py", "m1_transition_benchmark")
    items = benchmark.validate_manifest(manifest, args.audio_root.expanduser().resolve())
    if any(i["beats_seconds"] is None or i["downbeats_seconds"] is None for i in items):
        raise ValueError("source beat and downbeat labels required")
    if args.predictions:
        outputs = json.loads(args.predictions.read_text())
    else:
        grid_shadow = load("m1-ballroom-grid-shadow.py", "m1_transition_grid_shadow")
        outputs = grid_shadow.run_inference(args.audio_root.expanduser().resolve(), items)
    if set(outputs) != {i["id"] for i in items}:
        raise ValueError("prediction IDs do not match manifest")
    result = evaluate(manifest, outputs)
    result["reference_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({
        "track_count": result["track_count"],
        "metrics": result["metrics"],
        "output": str(args.output),
        "rendered_transitions_tested": False,
        "production_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 transition timing preflight failed: {exc}") from exc
