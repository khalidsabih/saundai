#!/usr/bin/env python3
"""Offline M1 experiment: score Beat This! scalar tempo independently of its beat-grid gate.

Read-only to SAUND: never changes analysis BPM, beats, downbeats, registry or model authority.
Runs installed models only; never downloads weights. Results contain no raw audio.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_benchmark():
    path = ROOT / "scripts" / "m1-rhythm-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_rhythm_benchmark_shadow", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("M1 benchmark module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_bpm(value):
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def score(manifest, predictions):
    """Compare scalar tempo candidates only; never treat them as accepted beat grids."""
    rows = []
    for item in manifest["items"]:
        key = item["id"]
        prediction = predictions.get(key)
        if not isinstance(prediction, dict):
            raise ValueError(f"missing prediction for {key}")
        reference = item.get("bpm")
        if not valid_bpm(reference):
            raise ValueError(f"tempo reference required for {key}")
        selected = prediction.get("selected_bpm")
        if not valid_bpm(selected):
            raise ValueError(f"invalid selected tempo for {key}")
        candidate = prediction.get("neural_bpm")
        evidence = prediction.get("evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError(f"invalid neural evidence for {key}")
        available = evidence.get("available") is True
        if candidate is not None and not valid_bpm(candidate):
            raise ValueError(f"invalid neural candidate for {key}")
        if not available:
            candidate = None
        rows.append({
            "id": key,
            "reference_bpm": float(reference),
            "selected_bpm": round(float(selected), 6),
            "neural_candidate_bpm": round(float(candidate), 6) if candidate is not None else None,
            "selected_absolute_error_bpm": round(abs(float(selected) - float(reference)), 6),
            "candidate_absolute_error_bpm": round(abs(float(candidate) - float(reference)), 6) if candidate is not None else None,
            "neural_available": available,
            "grid_accepted": evidence.get("used") is True,
            "confidence": evidence.get("confidence"),
            "authority": evidence.get("authority"),
            "agreement": evidence.get("agreement"),
            "reason": str(evidence.get("reason", "")),
            "backend": str(evidence.get("backend", "")),
        })
    selected_errors = [r["selected_absolute_error_bpm"] for r in rows]
    candidates = [r for r in rows if r["candidate_absolute_error_bpm"] is not None]
    accepted = [r for r in candidates if r["grid_accepted"]]
    rejected = [r for r in candidates if not r["grid_accepted"]]
    def mean(values):
        return round(sum(values) / len(values), 6) if values else None
    return {
        "schema_version": "saund-m1-neural-tempo-shadow-v1",
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "track_count": len(rows),
        "metrics": {
            "selected_tempo_mae_bpm": mean(selected_errors),
            "neural_candidate_tempo_mae_bpm": mean([r["candidate_absolute_error_bpm"] for r in candidates]),
            "selected_tempo_mae_on_candidate_subset_bpm": mean([r["selected_absolute_error_bpm"] for r in candidates]),
            "candidate_tempo_mae_rejected_subset_bpm": mean([r["candidate_absolute_error_bpm"] for r in rejected]),
            "selected_tempo_mae_rejected_subset_bpm": mean([r["selected_absolute_error_bpm"] for r in rejected]),
        },
        "coverage": {
            "reference_tempo": len(rows),
            "neural_candidate": len(candidates),
            "neural_grid_accepted": len(accepted),
            "neural_grid_rejected_with_candidate": len(rejected),
            "neural_unavailable": len(rows) - len(candidates),
        },
        "per_track": rows,
        "warning": (
            "Counterfactual diagnostic only. Neural scalar BPM has NOT replaced selected BPM or "
            "beat/downbeat grids. Candidate BPM accuracy does not establish beat-grid quality. "
            "Do not treat this as a production-gate, held-out test, or model promotion."
        ),
    }


def infer(audio_root, items):
    script = (
        "import json,sys;"
        "from djai.analysis import analyze_track;"
        "a=analyze_track(sys.argv[1]);"
        "e=a.neural_evidence.get('beat_downbeat',{});"
        "print(json.dumps({'selected_bpm':a.bpm,'neural_bpm':e.get('bpm'),"
        "'evidence':{k:e.get(k) for k in "
        "('available','used','confidence','authority','agreement','reason','backend')}}))"
    )
    env = dict(os.environ)
    env.update({
        "DJAI_NEURAL_BEATS": "1",
        "DJAI_MODEL_ALLOW_DOWNLOAD": "0",
        "DJAI_BEAT_THIS_ALLOW_DOWNLOAD": "0",
        "DJAI_CUE_MODEL_ENABLED": "0",
        "DJAI_EMBEDDINGS_ENABLED": "0",
        "PYTHONPATH": str(ROOT / "src"),
    })
    result = {}
    for item in items:
        audio = (audio_root / item["audio"]).resolve()
        proc = subprocess.run(
            [sys.executable, "-c", script, str(audio)],
            cwd=str(ROOT), env=env, text=True, capture_output=True, check=False,
        )
        if proc.returncode:
            raise RuntimeError(f"analysis failed for {item['id']}: {proc.stderr[-1000:]}")
        try:
            result[item["id"]] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unreadable SAUND output for {item['id']}") from exc
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="offline fixture: precomputed shadow predictions")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    root = args.audio_root.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = load_benchmark().validate_manifest(manifest, root)
    if args.predictions:
        data = json.loads(args.predictions.read_text(encoding="utf-8"))
        if set(data) != {item["id"] for item in items}:
            raise ValueError("precomputed predictions must exactly match manifest IDs")
    else:
        data = infer(root, items)
    result = score(manifest, data)
    result["reference_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    result["inference_mode"] = "provided_predictions" if args.predictions else "installed_neural_shadow"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "track_count": result["track_count"],
        "coverage": result["coverage"],
        "metrics": result["metrics"],
        "model_promoted": False,
        "production_changed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"M1 tempo shadow failed: {exc}", file=sys.stderr)
        sys.exit(2)
