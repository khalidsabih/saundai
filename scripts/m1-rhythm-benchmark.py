#!/usr/bin/env python3
"""M1 rhythm evaluator for independently annotated, local audio.

No dataset downloads. Neural inference is disabled by default to make the
frozen deterministic baseline repeatable. Annotations are never inferred from
SAUND predictions. Result reports are private and use only relative asset IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRIC_TO_BENCHMARK = {
    "tempo_mae_bpm": "m1_tempo_mae",
    "beat_f1": "m1_beat_f1",
    "downbeat_f1": "m1_downbeat_f1",
}


def valid_events(values, kind):
    if values is None:
        return None
    if not isinstance(values, list) or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError(f"{kind} must be nonnegative finite seconds or null")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError(f"{kind} must be sorted and strictly increasing")
    return [float(x) for x in values]


def count_matches(reference, predicted, tolerance):
    """One-to-one event matching; never count one detection multiple times."""
    matches = 0
    j = 0
    for x in reference:
        while j < len(predicted) and predicted[j] < x - tolerance:
            j += 1
        if j < len(predicted) and abs(predicted[j] - x) <= tolerance:
            matches += 1
            j += 1
    return matches


def event_counts(reference, predicted, tolerance):
    ref = valid_events(reference, "reference")
    pred = valid_events(predicted, "prediction")
    if ref is None:
        return None
    if pred is None:
        raise ValueError("prediction events unavailable")
    return {"match": count_matches(ref, pred, tolerance), "reference": len(ref), "predicted": len(pred)}


def score_events(rows):
    total = {"match": 0, "reference": 0, "predicted": 0}
    for row in rows:
        if row is None:
            continue
        for key in total:
            total[key] += row[key]
    if total["reference"] == 0:
        return None
    precision = total["match"] / total["predicted"] if total["predicted"] else 0.0
    recall = total["match"] / total["reference"]
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
        **total,
    }


def validate_manifest(manifest, audio_root):
    if manifest.get("schema_version") != "saund-m1-rhythm-v1":
        raise ValueError("unsupported reference manifest schema")
    split = manifest.get("split")
    if split not in {"development", "validation", "test"}:
        raise ValueError("split must be development, validation or test")
    dataset = manifest.get("dataset_id")
    if dataset not in {"giantsteps_tempo", "ballroom", "groove_midi", "local_34"}:
        raise ValueError("unknown M1 dataset ID")
    if manifest.get("rights_confirmed") is not True:
        raise ValueError("explicit local research rights review required")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("reference items must be nonempty")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("item must be object")
        track_id = item.get("id")
        if not isinstance(track_id, str) or not track_id or track_id in seen:
            raise ValueError("track IDs must be unique and nonempty")
        seen.add(track_id)
        rel = item.get("audio")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
            raise ValueError("audio paths must be relative")
        path = (audio_root / rel).resolve()
        if not path.is_relative_to(audio_root.resolve()):
            raise ValueError("audio path escapes root")
        if not path.is_file():
            raise ValueError(f"audio file missing for ID {track_id}")
        for label in ("beats_seconds", "downbeats_seconds"):
            valid_events(item.get(label), label)
        bpm = item.get("bpm")
        if bpm is not None and (isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or bpm <= 0):
            raise ValueError("reference BPM must be finite positive or null")
        if bpm is None and item.get("beats_seconds") is None and item.get("downbeats_seconds") is None:
            raise ValueError("each item must have independent reference BPM or event labels")
        if not isinstance(item.get("annotation_source"), str) or not item["annotation_source"].strip():
            raise ValueError("reference annotation source required for each item")
    return items


def predictions(audio_root, items, predictions_file, *, allow_neural):
    if predictions_file is not None:
        data = json.loads(predictions_file.read_text(encoding="utf-8"))
        if data.get("schema_version") != "saund-m1-predictions-v1" or not isinstance(data.get("predictions"), dict):
            raise ValueError("invalid prediction file")
        unknown = set(data["predictions"]) - {x["id"] for x in items}
        if unknown:
            raise ValueError("predictions include unknown track IDs")
        if set(data["predictions"]) != {x["id"] for x in items}:
            raise ValueError("predictions must cover every track exactly once")
        return data["predictions"]
    env = os.environ.copy()
    if not allow_neural:
        env["DJAI_NEURAL_BEATS"] = "0"
    env["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    env["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    env["DJAI_CUE_MODEL_ENABLED"] = "0"
    # Each source is independently analyzed, with existing SAUND analysis code.
    script = ("import json,sys;"
              "from djai.analysis import analyze_track;"
              "a=analyze_track(sys.argv[1]);"
              "print(json.dumps({'bpm':a.bpm,'beats_seconds':a.beats_seconds,"
              "'downbeats_seconds':a.downbeats_seconds}))")
    result = {}
    for item in items:
        path = (audio_root / item["audio"]).resolve()
        proc = subprocess.run([sys.executable, "-c", script, str(path)],
                              cwd=str(ROOT), env={**env, "PYTHONPATH": str(ROOT / "src")},
                              capture_output=True, text=True, check=False)
        if proc.returncode:
            raise RuntimeError(f"analysis failed for {item['id']}: {proc.stderr[-1000:]}")
        try:
            result[item["id"]] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unreadable SAUND analysis for {item['id']}") from exc
    return result


def evaluate(manifest, preds, *, beat_window=0.07, downbeat_window=0.10):
    rows = []
    for item in manifest["items"]:
        p = preds.get(item["id"])
        if not isinstance(p, dict):
            raise ValueError(f"missing prediction for {item['id']}")
        tempo_ref = item.get("bpm")
        tempo_pred = p.get("bpm")
        if tempo_ref is not None and (not isinstance(tempo_pred, (int, float)) or not math.isfinite(tempo_pred) or tempo_pred <= 0):
            raise ValueError(f"invalid predicted BPM for {item['id']}")
        beats = event_counts(item.get("beats_seconds"), p.get("beats_seconds"), beat_window)
        downbeats = event_counts(item.get("downbeats_seconds"), p.get("downbeats_seconds"), downbeat_window)
        rows.append({
            "id": item["id"],
            "tempo_absolute_error_bpm": abs(float(tempo_pred) - float(tempo_ref)) if tempo_ref is not None else None,
            "beat_counts": beats, "downbeat_counts": downbeats,
        })
    tempo_errors = [r["tempo_absolute_error_bpm"] for r in rows if r["tempo_absolute_error_bpm"] is not None]
    beats = score_events([r["beat_counts"] for r in rows])
    downs = score_events([r["downbeat_counts"] for r in rows])
    metrics = {
        "tempo_mae_bpm": round(sum(tempo_errors) / len(tempo_errors), 6) if tempo_errors else None,
        "beat_f1": beats["f1"] if beats else None,
        "downbeat_f1": downs["f1"] if downs else None,
    }
    return {
        "schema_version": "saund-m1-evaluation-v1",
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "track_count": len(rows),
        "metric_coverage": {"tempo": len(tempo_errors), "beat": sum(r["beat_counts"] is not None for r in rows),
                            "downbeat": sum(r["downbeat_counts"] is not None for r in rows)},
        "metrics": metrics, "beat_details": beats, "downbeat_details": downs,
        "per_track": rows,
        "metric_notes": "One-to-one greedy event match at 70ms beat / 100ms downbeat; absolute BPM MAE (no octave folding). Missing labels produce null rather than zero.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="private rights-reviewed reference manifest")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="optional precomputed SAUND outputs; useful for offline metric verification")
    parser.add_argument("--output", type=Path, required=True, help="new private result JSON path; no overwrite")
    parser.add_argument("--allow-neural", action="store_true", help="explicit opt-in: permit already installed neural model inference; never download weights")
    parser.add_argument("--record-ledger", action="store_true", help="record automatically computed, available metrics in local registry")
    parser.add_argument("--implementation", default="deterministic-baseline")
    parser.add_argument("--source-commit", help="required when recording results")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    root = args.audio_root.expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = validate_manifest(manifest, root)
    preds = predictions(root, items, args.predictions, allow_neural=args.allow_neural)
    result = evaluate(manifest, preds)
    result["reference_manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    result["inference_mode"] = "provided_predictions" if args.predictions else ("neural_opt_in" if args.allow_neural else "deterministic_only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    if args.record_ledger:
        if not args.source_commit or len(args.source_commit) < 8:
            raise ValueError("source commit required for ledger recording")
        registry = ROOT / "scripts" / "dataset-benchmark-registry.py"
        for metric_name, value in result["metrics"].items():
            if value is None:
                continue
            command = [sys.executable, str(registry), "record", "--benchmark", METRIC_TO_BENCHMARK[metric_name],
                       "--dataset", manifest["dataset_id"], "--split", manifest["split"],
                       "--manifest", str(args.manifest), "--source-commit", args.source_commit,
                       "--implementation", args.implementation, "--value", str(value),
                       "--rights-confirmed", "--safety-passed",
                       "--notes", "M1 measured; evidence="+str(args.output)]
            subprocess.run(command, cwd=str(ROOT), check=True)
    print(json.dumps({"result": str(args.output), "track_count": result["track_count"],
                      "metric_coverage": result["metric_coverage"], "metrics": result["metrics"],
                      "ledger_recorded": args.record_ledger}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"M1 rhythm benchmark failed: {exc}", file=sys.stderr)
        sys.exit(2)
