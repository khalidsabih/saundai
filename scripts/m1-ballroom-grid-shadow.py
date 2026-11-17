#!/usr/bin/env python3
"""Score selected and Beat This! candidate beat/downbeat grids against Ballroom labels.

Read-only research evaluation. Never changes SAUND runtime grids, authority,
model registry, or transition safety. Uses locally installed checkpoints only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def benchmark():
    src = ROOT / "scripts" / "m1-rhythm-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_benchmark", src)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_bpm(beats):
    if beats is None or len(beats) < 2:
        return None
    steps = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    return round(60 / statistics.median(steps), 4) if steps else None


def run_inference(root, items):
    snippet = (
        "import json,sys;"
        "from djai.analysis import analyze_track;"
        "a=analyze_track(sys.argv[1]);"
        "e=a.neural_evidence.get('beat_downbeat',{});"
        "print(json.dumps({'selected':{'bpm':a.bpm,'beats_seconds':a.beats_seconds,"
        "'downbeats_seconds':a.downbeats_seconds},'candidate':e}))"
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
    data = {}
    for item in items:
        wav = (root / item["audio"]).resolve()
        proc = subprocess.run([sys.executable, "-c", snippet, str(wav)],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"analysis failed for {item['id']}: {proc.stderr[-1000:]}")
        try:
            data[item["id"]] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unreadable analysis result for {item['id']}") from exc
    return data


def score(manifest, outputs):
    m = benchmark()
    selected, candidate, rows = {}, {}, []
    for item in manifest["items"]:
        key = item["id"]
        p = outputs.get(key)
        if not isinstance(p, dict) or not isinstance(p.get("selected"), dict):
            raise ValueError(f"missing selected output for {key}")
        e = p.get("candidate", {})
        if not isinstance(e, dict):
            raise ValueError(f"invalid neural evidence for {key}")
        s = p["selected"]
        selected[key] = s
        available = e.get("available") is True
        bv = e.get("beats_seconds") if available else None
        dv = e.get("downbeats_seconds") if available else None
        if available and (bv is None or dv is None):
            raise ValueError(f"available neural grid missing events for {key}")
        candidate[key] = {
            "bpm": e.get("bpm") if available else s.get("bpm"),
            "beats_seconds": bv if available else None,
            "downbeats_seconds": dv if available else None,
        }
        s_bpm = grid_bpm(s.get("beats_seconds"))
        n_bpm = grid_bpm(bv)
        rows.append({
            "id": key,
            "neural_available": available,
            "grid_accepted": e.get("used") is True,
            "confidence": e.get("confidence"),
            "authority": e.get("authority"),
            "reason": e.get("reason"),
            "selected_bpm": s.get("bpm"),
            "selected_grid_bpm": s_bpm,
            "selected_grid_bpm_difference": (
                round(abs(s["bpm"] - s_bpm), 4) if s_bpm is not None else None
            ),
            "neural_bpm": e.get("bpm") if available else None,
            "neural_grid_bpm": n_bpm,
            "neural_grid_bpm_difference": (
                round(abs(e["bpm"] - n_bpm), 4)
                if available and n_bpm is not None and isinstance(e.get("bpm"), (int, float)) else None
            ),
        })
    selected_score = m.evaluate(manifest, selected)
    # Unavailable neural grids have no candidate score; never score them as empty events.
    available_items = [i for i in manifest["items"] if outputs[i["id"]]["candidate"].get("available") is True]
    candidate_score = m.evaluate({**manifest, "items": available_items}, candidate) if available_items else None
    candidate_rows = {r["id"]: r for r in candidate_score["per_track"]} if candidate_score else {}
    selected_rows = {r["id"]: r for r in selected_score["per_track"]}
    for r in rows:
        key = r["id"]
        r["selected_beat_counts"] = selected_rows[key]["beat_counts"]
        r["selected_downbeat_counts"] = selected_rows[key]["downbeat_counts"]
        c = candidate_rows.get(key, {})
        r["candidate_beat_counts"] = c.get("beat_counts")
        r["candidate_downbeat_counts"] = c.get("downbeat_counts")
    return {
        "schema_version": "saund-m1-ballroom-grid-shadow-v1",
        "track_count": len(rows),
        "coverage": {
            "neural_candidate": len(available_items),
            "neural_grid_accepted": sum(r["grid_accepted"] for r in rows),
            "neural_grid_rejected_with_candidate": sum(r["neural_available"] and not r["grid_accepted"] for r in rows),
            "neural_unavailable": len(rows) - len(available_items),
        },
        "selected_metrics": selected_score["metrics"],
        "neural_candidate_metrics": candidate_score["metrics"] if candidate_score else None,
        "selected_event_details": {"beat": selected_score["beat_details"], "downbeat": selected_score["downbeat_details"]},
        "candidate_event_details": (
            {"beat": candidate_score["beat_details"], "downbeat": candidate_score["downbeat_details"]}
            if candidate_score else None
        ),
        "per_track": rows,
        "warnings": [
            "Candidate metrics use only available neural grids; compare coverage when availability <100%.",
            "Neither selected nor candidate grids were changed by this experiment.",
            "Ballroom is for permitted research evaluation only; do not train on it.",
            "Beat/downbeat F1 do not certify safe DJ transition rendering.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="fixture JSON keyed by manifest ID; no audio inference")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    root = args.audio_root.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "ballroom":
        raise ValueError("this evaluator only accepts Ballroom")
    items = benchmark().validate_manifest(manifest, root)
    if not all(i.get("beats_seconds") is not None and i.get("downbeats_seconds") is not None for i in items):
        raise ValueError("independent beat and downbeat timestamps required")
    if args.predictions:
        outputs = json.loads(args.predictions.read_text(encoding="utf-8"))
        if set(outputs) != {i["id"] for i in items}:
            raise ValueError("prediction IDs do not match manifest")
    else:
        outputs = run_inference(root, items)
    result = score(manifest, outputs)
    result["reference_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    result["inference_mode"] = "provided_predictions" if args.predictions else "installed_neural_shadow"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "track_count": result["track_count"],
        "coverage": result["coverage"],
        "selected_metrics": result["selected_metrics"],
        "neural_candidate_metrics": result["neural_candidate_metrics"],
        "production_changed": False,
        "model_promoted": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"M1 Ballroom grid shadow failed: {exc}", file=sys.stderr)
        sys.exit(2)
