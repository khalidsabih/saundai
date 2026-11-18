#!/usr/bin/env python3
"""Read-only manifest-based beat-grid regression for independent rights-cleared audio.

The manifest holds explicit local sources + independently annotated beats.
No reference is supplied to the tracker: this script only evaluates saved
external predictions. A 'test' / 'sealed' split is NEVER evaluated. The report
is evidence only; no production grid, neural authority or render is changed.

Manifest shape:
{
  "tracks": [
    {"id":"track-a", "split":"validation", "audio":"audio/a.wav",
     "reference_beats":"labels/a.beats.txt",
     "reference_downbeats":"labels/a.downbeats.txt",
     "beat_this_predictions":"pred/a.beats",
     "saund_predictions":"pred/a-saund.beats"}
  ]
}
References contain one decimal timestamp per nonempty line (seconds).
Prediction TSV is Beat This's time TAB beat_number; SAUND prediction is
optional and uses the same TSV shape. No heuristic reference extraction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = {"validation"}
DEFAULT_TOLERANCE_MS = 70.0


def load_benchmark():
    p = ROOT / "scripts" / "m1-beat-this-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_independent_reference_matcher", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("M1 Beat This benchmark is missing")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_reference(path):
    if not path.is_file():
        raise ValueError(f"missing independent reference: {path}")
    values = []
    for row_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 1:
            raise ValueError(f"reference labels require one timestamp per line: {path}:{row_number}")
        try:
            val = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"nonnumeric reference timestamp: {path}:{row_number}") from exc
        if not math.isfinite(val) or val < 0 or values and val <= values[-1]:
            raise ValueError(f"nonfinite, negative or unordered reference: {path}:{row_number}")
        values.append(val)
    if len(values) < 2:
        raise ValueError(f"insufficient reference labels: {path}")
    return values


def evaluate_events(pred, ref, *, tolerance_ms=DEFAULT_TOLERANCE_MS):
    """One-to-one score; references influence scoring only, not model output."""
    if tolerance_ms <= 0 or tolerance_ms > 150:
        raise ValueError("invalid research matching tolerance")
    proposals = []
    for i, a in enumerate(pred):
        for j, b in enumerate(ref):
            error = 1000 * (a - b)
            if abs(error) <= tolerance_ms:
                proposals.append((abs(error), i, j, error))
    used_pred = set()
    used_ref = set()
    errors = []
    for _, i, j, error in sorted(proposals):
        if i not in used_pred and j not in used_ref:
            used_pred.add(i)
            used_ref.add(j)
            errors.append(error)
    precision = len(errors) / len(pred) if pred else 0.
    recall = len(errors) / len(ref) if ref else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {
        "reference_count": len(ref), "prediction_count": len(pred),
        "matched_count": len(errors),
        "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "maximum_matched_absolute_error_ms": (
            round(max(abs(x) for x in errors), 3) if errors else None
        ),
        "unmatched_predictions": len(pred) - len(errors),
        "missed_references": len(ref) - len(errors),
        "reference_events_not_matched_within_25ms": len(ref) - sum(
            abs(x) <= 25.0 for x in errors
        ),
        "research_tolerance_ms": tolerance_ms,
        "25ms_diagnostic_is_not_planner_gate": True,
    }


def _file(root, item, key, *, required=True):
    value = item.get(key)
    if not value:
        if required:
            raise ValueError(f"missing required manifest field: {key}")
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid manifest path for {key}")
    return root / value


def run(manifest, root):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tracks"), list):
        raise ValueError("manifest must provide tracks array")
    benchmark = load_benchmark()
    rows = []
    ids = set()
    for item in manifest["tracks"]:
        if not isinstance(item, dict):
            raise ValueError("track entries must be objects")
        name = item.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("track IDs must be unique and nonempty")
        ids.add(name)
        split = item.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError("only explicit validation split may be evaluated; sealed/test prohibited")
        audio = _file(root, item, "audio")
        if not audio.is_file():
            raise ValueError(f"missing music input: {audio}")
        information = sf.info(str(audio))
        if information.frames < information.samplerate * 4:
            raise ValueError(f"audio is shorter than 4 seconds: {audio}")
        duration = information.duration
        ref_beats = read_reference(_file(root, item, "reference_beats"))
        ref_downs_path = _file(root, item, "reference_downbeats", required=False)
        ref_downs = read_reference(ref_downs_path) if ref_downs_path else None
        predicted = benchmark.read_beat_tsv(_file(root, item, "beat_this_predictions"))
        baseline_path = _file(root, item, "saund_predictions", required=False)
        baseline = benchmark.read_beat_tsv(baseline_path) if baseline_path else None
        all_events = ref_beats + (ref_downs or []) + predicted["beats"] + (
            predicted["downbeats"]
        ) + (baseline["beats"] + baseline["downbeats"] if baseline else [])
        if any(x > duration + .1 for x in all_events):
            raise ValueError(f"labels or predictions extend beyond audio: {name}")
        row = {
            "id": name, "split": split, "duration_seconds": round(duration, 3),
            "beat_this_beats": evaluate_events(predicted["beats"], ref_beats),
            "beat_this_downbeats": (
                evaluate_events(predicted["downbeats"], ref_downs)
                if ref_downs is not None else None
            ),
            "saund_beats": (
                evaluate_events(baseline["beats"], ref_beats) if baseline else None
            ),
            "saund_downbeats": (
                evaluate_events(baseline["downbeats"], ref_downs)
                if baseline and ref_downs is not None else None
            ),
        }
        rows.append(row)
    if not rows:
        raise ValueError("manifest has no validation tracks")
    both = [r for r in rows if r["saund_beats"] is not None]
    regressions = [
        r["id"] for r in both
        if r["beat_this_beats"]["f1"] < r["saund_beats"]["f1"]
    ]
    return {
        "schema": "saund-m1-independent-audio-regression-v1",
        "selected_split": "validation", "track_count": len(rows),
        "paired_baseline_track_count": len(both),
        "beat_this_mean_beat_f1": round(
            sum(r["beat_this_beats"]["f1"] for r in rows) / len(rows), 4
        ),
        "paired_beat_f1_regression_track_ids": regressions,
        "tracks": rows,
        "rights_and_reference_provenance_must_be_checked_by_operator": True,
        "reference_used_for_inference": False, "production_changed": False,
        "model_authority_changed": False, "beat_grids_modified": False,
        "planner_gate_tested": False, "transition_rendered": False,
        "warning": "Validation only: independently annotated rights-cleared tracks required. "
                   "Do not infer production readiness from this report or change 25ms planner gate.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists() or a.output.is_symlink():
        raise ValueError("refusing to overwrite prior report")
    doc = json.loads(a.manifest.read_text(encoding="utf-8"))
    data = run(doc, a.root or a.manifest.parent)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(data, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 independent audio regression failed: {exc}") from exc
