#!/usr/bin/env python3
"""Diagnose individual bar-interval outliers in approved Ballroom validation.

Target only records already flagged by the offline cue timing preflight.
Uses local, rights-reviewed audio and disabled model downloads. No promotion,
production changes or sealed test evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def import_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if not spec or not spec.loader:
        raise RuntimeError(f"missing {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def interval_diagnostic(beats, downbeats, meter, reference_downbeats=None):
    if not isinstance(beats, list) or len(beats) < 2 or not isinstance(downbeats, list) or len(downbeats) < 2:
        raise ValueError("insufficient beat/downbeat events")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in beats + downbeats):
        raise ValueError("nonfinite event timestamps")
    if any(b <= a for a, b in zip(beats, beats[1:])) or any(b <= a for a, b in zip(downbeats, downbeats[1:])):
        raise ValueError("events must strictly increase")
    period = statistics.median(b - a for a, b in zip(beats, beats[1:]))
    expected = meter * period
    intervals = []
    for index, (start, end) in enumerate(zip(downbeats, downbeats[1:])):
        duration = end - start
        ratio = duration / expected
        count_beats = sum(start - 0.035 <= b < end - 0.035 for b in beats)
        nearest = (min(abs(start - r) for r in reference_downbeats)
                   if reference_downbeats else None)
        intervals.append({
            "index": index,
            "start_seconds": round(start, 5),
            "end_seconds": round(end, 5),
            "duration_seconds": round(duration, 5),
            "expected_seconds": round(expected, 5),
            "bar_ratio": round(ratio, 4),
            "beat_events_between_downbeats": count_beats,
            "outlier_20pct": abs(ratio - 1) > .20,
            "closest_reference_downbeat_seconds": round(nearest, 5) if nearest is not None else None,
        })
    flagged = [i for i in intervals if i["outlier_20pct"]]
    return {
        "meter": meter,
        "predicted_bar_count": len(intervals),
        "outlier_count": len(flagged),
        "outliers_in_first_two": sum(i["index"] < 2 for i in flagged),
        "outliers_in_last_two": sum(i["index"] >= len(intervals) - 2 for i in flagged),
        "predicted_beat_period_seconds": round(period, 5),
        "intervals": intervals,
        "outliers": flagged,
    }


def inspect(manifest, preflight, predictions):
    if preflight.get("schema_version") != "saund-m1-transition-timing-preflight-v1":
        raise ValueError("unsupported timing preflight")
    manifest_by_id = {i["id"]: i for i in manifest["items"]}
    rows = [r for r in preflight["per_track"] if not r["challenger"]["passed"]
            or r["challenger"].get("cue_start_near_annotated_downbeat") is False]
    findings = []
    for r in rows:
        key = r["id"]
        if key not in manifest_by_id or key not in predictions:
            raise ValueError(f"unknown or missing track {key}")
        item = manifest_by_id[key]
        output = predictions[key]
        candidate = output["candidate"]
        # frozen gate, unchanged: previously accepted, otherwise confidence
        # >=0.50 with BPM/beat period <=2 BPM
        gate = import_script("m1-ballroom-transition-timing-shadow.py", "m1_bar_diagnostic_gate")
        selected, uses_neural = gate.choose(output["selected"], candidate)
        if uses_neural != r["challenger_uses_neural"]:
            raise ValueError(f"frozen choice changed for {key}")
        result = interval_diagnostic(
            selected["beats_seconds"], selected["downbeats_seconds"],
            r["reference_meter"], item["downbeats_seconds"],
        )
        result.update({
            "id": key, "genre": item.get("genre"),
            "challenger_uses_neural": uses_neural,
            "preflight_failures": r["challenger"]["failures"],
            "preflight_cue_phase_ok": r["challenger"].get("cue_start_near_annotated_downbeat"),
        })
        findings.append(result)
    return {
        "schema_version": "saund-m1-bar-interval-diagnostic-v1",
        "track_count": len(findings),
        "per_track": findings,
        "production_changed": False,
        "rendered_transitions_tested": False,
        "warning": "Outlier intervals are a diagnostic, not confirmation of incorrect musical bars; annotation proximity is descriptive.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    manifest = json.loads(args.manifest.read_text())
    preflight = json.loads(args.preflight.read_text())
    if manifest.get("dataset_id") != "ballroom" or manifest.get("split") != "validation" or manifest.get("rights_confirmed") is not True:
        raise ValueError("approved Ballroom validation manifest required")
    import hashlib
    if preflight.get("reference_manifest_sha256") != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
        raise ValueError("preflight and manifest hashes mismatch")
    benchmark = import_script("m1-rhythm-benchmark.py", "m1_bar_diagnostic_benchmark")
    items = benchmark.validate_manifest(manifest, args.audio_root.expanduser().resolve())
    flagged = {r["id"] for r in preflight["per_track"]
               if not r["challenger"]["passed"]
               or r["challenger"].get("cue_start_near_annotated_downbeat") is False}
    if not flagged or not flagged.issubset({i["id"] for i in items}):
        raise ValueError("no valid flagged tracks")
    subset = [i for i in items if i["id"] in flagged]
    grid_shadow = import_script("m1-ballroom-grid-shadow.py", "m1_bar_diagnostic_grid")
    predictions = grid_shadow.run_inference(args.audio_root.expanduser().resolve(), subset)
    result = inspect(manifest, preflight, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write("\n")
    for item in result["per_track"]:
        print(f"{item['id']}: neural={item['challenger_uses_neural']} "
              f"bar_outliers={item['outlier_count']} "
              f"start={item['outliers_in_first_two']} end={item['outliers_in_last_two']} "
              f"cue_phase_ok={item['preflight_cue_phase_ok']}")
        for i in item["outliers"][:8]:
            print(f"  bar {i['index']}: {i['start_seconds']:.2f}-{i['end_seconds']:.2f}s "
                  f"ratio={i['bar_ratio']:.2f} beats={i['beat_events_between_downbeats']} "
                  f"ref_offset={i['closest_reference_downbeat_seconds']}")
    print(f"Report: {args.output}")
    print("Rendered transitions tested: false; production changed: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 bar interval diagnostic failed: {exc}") from exc
