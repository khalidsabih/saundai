#!/usr/bin/env python3
"""Rights-gated, development-only Ballroom beat/downbeat and BPM/grid audit.

Runs SAUND's real analyzer twice per WAV (neural off/on); uses independently
annotated events from the manifest. No audio or predictions leave the machine.
Never changes production selection, beat grids, or model authority.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "m1_rhythm_benchmark_ballroom", ROOT / "scripts" / "m1-rhythm-benchmark.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("M1 benchmark implementation missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_bpm(beats):
    """Median positive beat interval, not an independent tempo ground truth."""
    if not isinstance(beats, list) or len(beats) < 2:
        return None
    diffs = [float(b) - float(a) for a, b in zip(beats, beats[1:])]
    if not diffs or any(not math.isfinite(x) or x <= 0 for x in diffs):
        return None
    median = statistics.median(diffs)
    return round(60 / median, 6) if median > 0 else None


def consistency(selected_bpm, beats):
    inferred = grid_bpm(beats)
    if inferred is None or not isinstance(selected_bpm, (int, float)) or selected_bpm <= 0:
        return {"grid_bpm": inferred, "bpm_grid_difference": None, "bpm_grid_ratio": None}
    return {
        "grid_bpm": inferred,
        "bpm_grid_difference": round(abs(float(selected_bpm) - inferred), 6),
        "bpm_grid_ratio": round(float(selected_bpm) / inferred, 6),
    }


def analyze(path, *, neural):
    code = (
        "import json,sys;"
        "from djai.analysis import analyze_track;"
        "a=analyze_track(sys.argv[1]);"
        "e=a.neural_evidence.get('beat_downbeat',{});"
        "print(json.dumps({'bpm':a.bpm,'beats_seconds':a.beats_seconds,"
        "'downbeats_seconds':a.downbeats_seconds,"
        "'neural':{k:e.get(k) for k in ('backend','available','used','bpm','confidence','reason')}}))"
    )
    env = os.environ.copy()
    env.update({
        "DJAI_NEURAL_BEATS": "1" if neural else "0",
        "DJAI_MODEL_ALLOW_DOWNLOAD": "0",
        "DJAI_BEAT_THIS_ALLOW_DOWNLOAD": "0",
        "DJAI_EMBEDDINGS_ENABLED": "0",
        "DJAI_CUE_MODEL_ENABLED": "0",
        "DJAI_TEMPO_CONSISTENCY": "0",
        "PYTHONPATH": str(ROOT / "src"),
    })
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{path.name}: analysis failed: {result.stderr[-1200:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path.name}: unreadable analysis output") from exc


def evaluate(manifest, deterministic, neural, benchmark):
    if set(deterministic) != set(neural):
        raise ValueError("analysis modes must cover identical track IDs")
    basic = {}
    per_track = []
    for label, values in (("deterministic", deterministic), ("neural_gated", neural)):
        predictions = {
            key: {k: row[k] for k in ("bpm", "beats_seconds", "downbeats_seconds")}
            for key, row in values.items()
        }
        basic[label] = benchmark.evaluate(manifest, predictions)
    for item in manifest["items"]:
        key = item["id"]
        d = deterministic[key]
        n = neural[key]
        de = consistency(d["bpm"], d["beats_seconds"])
        ne = consistency(n["bpm"], n["beats_seconds"])
        info = n.get("neural", {})
        per_track.append({
            "id": key,
            "reference_bpm": item.get("bpm"),
            "deterministic": {"bpm": d["bpm"], **de},
            "neural_gated": {"bpm": n["bpm"], **ne},
            "neural_candidate_bpm": info.get("bpm") if info.get("available") is True else None,
            "neural_available": info.get("available") is True,
            "neural_grid_accepted": info.get("used") is True,
            "neural_confidence": info.get("confidence"),
            "neural_reason": info.get("reason"),
        })
    return {
        "schema_version": "saund-m1-ballroom-grid-validation-v1",
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "track_count": len(manifest["items"]),
        "metrics": {
            mode: {
                "tempo_mae_bpm": report["metrics"]["tempo_mae_bpm"],
                "beat_f1": report["metrics"]["beat_f1"],
                "downbeat_f1": report["metrics"]["downbeat_f1"],
                "metric_coverage": report["metric_coverage"],
                "beat_details": report["beat_details"],
                "downbeat_details": report["downbeat_details"],
            }
            for mode, report in basic.items()
        },
        "per_track": per_track,
        "notes": (
            "The scalar neural candidate is diagnostic only. Grid BPM is inferred from the "
            "median selected beat interval; a BPM/grid mismatch can flag timing inconsistency "
            "but not certify correct phase, downbeat alignment or musical quality. "
            "No new model selection, threshold relaxation or production promotion."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite result")
    benchmark = benchmark_module()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "ballroom":
        raise ValueError("this validation lane requires independent Ballroom annotations")
    root = args.audio_root.expanduser().resolve()
    items = benchmark.validate_manifest(manifest, root)
    if not all(item.get("beats_seconds") is not None and item.get("downbeats_seconds") is not None for item in items):
        raise ValueError("Ballroom validation requires independent beat and downbeat labels for every item")
    deterministic = {}
    neural = {}
    for item in items:
        path = (root / item["audio"]).resolve()
        deterministic[item["id"]] = analyze(path, neural=False)
        neural[item["id"]] = analyze(path, neural=True)
    result = evaluate(manifest, deterministic, neural, benchmark)
    result["model_promoted"] = False
    result["production_changed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "track_count": result["track_count"],
        "metrics": {mode: result["metrics"][mode] for mode in ("deterministic", "neural_gated")},
        "neural_grid_accepted": sum(row["neural_grid_accepted"] for row in result["per_track"]),
        "model_promoted": False,
        "production_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"M1 Ballroom validation failed: {exc}", file=sys.stderr)
        sys.exit(2)
