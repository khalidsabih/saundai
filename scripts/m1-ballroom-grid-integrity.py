#!/usr/bin/env python3
"""Read-only structural safety audit for selected and frozen-challenger rhythm grids.

Uses local Ballroom audio and existing Beat This! evidence only. No production
gate edits, downloads, transition rendering, or audio copying. This is a grid
integrity check, NOT a rendered-mix or cross-domain safety certification.
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
THRESHOLD = 0.50
BPM_GRID_TOLERANCE = 2.0
DOWNBEAT_SNAP_TOLERANCE_SECONDS = 0.10


def benchmark():
    source = ROOT / "scripts" / "m1-rhythm-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_grid_integrity_benchmark", source)
    if not spec or not spec.loader:
        raise RuntimeError("benchmark unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def grid_health(bpm, beats, downbeats):
    errors = []
    if not valid_number(bpm) or bpm <= 0:
        errors.append("invalid_bpm")
    for name, events in (("beat", beats), ("downbeat", downbeats)):
        if not isinstance(events, list) or not events:
            errors.append(f"missing_{name}_events")
        elif any(not valid_number(v) or v < 0 for v in events):
            errors.append(f"invalid_{name}_timestamps")
        elif any(b <= a for a, b in zip(events, events[1:])):
            errors.append(f"nonmonotonic_{name}_timestamps")
    if errors:
        return {"passed": False, "failures": errors, "grid_bpm": None, "bpm_difference": None}
    if len(beats) < 4 or len(downbeats) < 2:
        errors.append("insufficient_grid_events")
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    median_interval = statistics.median(intervals) if intervals else None
    grid_bpm = 60 / median_interval if median_interval else None
    difference = abs(bpm - grid_bpm) if grid_bpm is not None else None
    if difference is None or difference > BPM_GRID_TOLERANCE:
        errors.append("bpm_grid_mismatch")
    # Predicted bar starts should align with predicted beats. This structural
    # constraint is independent from the Ballroom annotations and DJ rendering.
    j = 0
    far = 0
    for down in downbeats:
        while j + 1 < len(beats) and abs(beats[j + 1] - down) < abs(beats[j] - down):
            j += 1
        if not beats or abs(beats[j] - down) > DOWNBEAT_SNAP_TOLERANCE_SECONDS:
            far += 1
    if far:
        errors.append("downbeats_not_on_beats")
    return {
        "passed": not errors,
        "failures": errors,
        "grid_bpm": round(grid_bpm, 5) if grid_bpm else None,
        "bpm_difference": round(difference, 5) if difference is not None else None,
        "offgrid_downbeats": far,
    }


def choose(selected, neural):
    """Whole-grid selection only; rule is the pre-frozen challenger."""
    available = neural.get("available") is True
    used = neural.get("used") is True
    confidence = neural.get("confidence")
    neural_health = grid_health(neural.get("bpm"), neural.get("beats_seconds"), neural.get("downbeats_seconds")) if available else None
    consistency = (
        neural_health is not None
        and neural_health["bpm_difference"] is not None
        and neural_health["bpm_difference"] <= BPM_GRID_TOLERANCE
    )
    use_neural = used or (
        available and valid_number(confidence) and confidence >= THRESHOLD and consistency
    )
    return ({"bpm": neural.get("bpm"),
             "beats_seconds": neural.get("beats_seconds"),
             "downbeats_seconds": neural.get("downbeats_seconds")}
            if use_neural else selected), use_neural


def evaluate(manifest, predictions):
    rows = []
    for item in manifest["items"]:
        key = item["id"]
        p = predictions[key]
        selected = p["selected"]
        neural = p["candidate"]
        challenger, uses_neural = choose(selected, neural)
        s = grid_health(selected.get("bpm"), selected.get("beats_seconds"), selected.get("downbeats_seconds"))
        c = grid_health(challenger.get("bpm"), challenger.get("beats_seconds"), challenger.get("downbeats_seconds"))
        n = grid_health(neural.get("bpm"), neural.get("beats_seconds"), neural.get("downbeats_seconds")) if neural.get("available") is True else None
        rows.append({
            "id": key,
            "genre": item.get("genre"),
            "existing_gate_accepted": neural.get("used") is True,
            "challenger_uses_neural": uses_neural,
            "selected": s,
            "challenger": c,
            "raw_neural_candidate": n,
        })
    return {
        "schema_version": "saund-m1-grid-integrity-shadow-v1",
        "track_count": len(rows),
        "coverage": {
            "selected_integrity_failures": sum(not r["selected"]["passed"] for r in rows),
            "challenger_integrity_failures": sum(not r["challenger"]["passed"] for r in rows),
            "raw_candidate_integrity_failures": sum(r["raw_neural_candidate"] is not None and not r["raw_neural_candidate"]["passed"] for r in rows),
            "newly_accepted": sum(r["challenger_uses_neural"] and not r["existing_gate_accepted"] for r in rows),
        },
        "frozen_threshold": THRESHOLD,
        "per_track": rows,
        "production_changed": False,
        "warning": "Structural consistency only. Does NOT validate downbeat annotation accuracy, DJ bar-cue placement, transition quantization, render continuity, or cross-domain behavior.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--audio-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--predictions", type=Path, help="offline fixture keyed by ID; skips inference")
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("dataset_id") != "ballroom" or manifest.get("rights_confirmed") is not True:
        raise ValueError("requires a separately rights-approved Ballroom manifest")
    if manifest.get("split") != "validation":
        raise ValueError("audit existing validation split before final test reveal")
    items = benchmark().validate_manifest(manifest, args.audio_root.resolve())
    if args.predictions:
        predictions = json.loads(args.predictions.read_text())
    else:
        shadow_src = ROOT / "scripts" / "m1-ballroom-grid-shadow.py"
        spec = importlib.util.spec_from_file_location("m1_grid_integrity_inference", shadow_src)
        if not spec or not spec.loader:
            raise RuntimeError("shadow inference unavailable")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predictions = module.run_inference(args.audio_root.resolve(), items)
    if set(predictions) != {item["id"] for item in items}:
        raise ValueError("prediction track ID mismatch")
    result = evaluate(manifest, predictions)
    result["reference_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({
        "track_count": result["track_count"],
        "coverage": result["coverage"],
        "output": str(args.output),
        "production_changed": False,
        "rendered_transition_safety_tested": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 grid integrity failed: {exc}") from exc
