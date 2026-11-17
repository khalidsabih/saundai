#!/usr/bin/env python3
"""Frozen M1 Ballroom confidence>=0.50 grid challenger, strictly offline.

Reads private shadow counts and independent BPM labels; no audio inference,
threshold optimization, model training, ledger writes, or production changes.
Policy frozen before evaluating disjoint validation recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN_THRESHOLD = 0.50
MAX_GRID_BPM_DISAGREEMENT = 2.0


def benchmark():
    path = ROOT / "scripts" / "m1-rhythm-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_gate_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark not found")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f1(counts):
    if not counts:
        return None
    denom = counts["reference"] + counts["predicted"]
    return 2 * counts["match"] / denom if denom else None


def evaluate(manifest, shadow):
    if shadow.get("schema_version") != "saund-m1-ballroom-grid-shadow-v1":
        raise ValueError("unsupported shadow format")
    if shadow.get("track_count") != len(manifest["items"]):
        raise ValueError("shadow track count mismatch")
    rows = shadow.get("per_track")
    if not isinstance(rows, list) or len(rows) != len(manifest["items"]):
        raise ValueError("shadow rows mismatch")
    indexed = {r["id"]: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {i["id"] for i in manifest["items"]}:
        raise ValueError("shadow track IDs mismatch")
    selected_beats, selected_downs = [], []
    challenged_beats, challenged_downs = [], []
    candidate_beats, candidate_downs = [], []
    tempo_sel, tempo_challenger, tempo_candidate = [], [], []
    details = []
    for item in manifest["items"]:
        r = indexed[item["id"]]
        if r.get("grid_accepted") and not r.get("neural_available"):
            raise ValueError("accepted grid not marked available")
        used = bool(r.get("grid_accepted"))
        available = r.get("neural_available") is True
        confidence = r.get("confidence")
        consistent = (
            isinstance(r.get("neural_grid_bpm_difference"), (float, int))
            and math.isfinite(r["neural_grid_bpm_difference"])
            and r["neural_grid_bpm_difference"] <= MAX_GRID_BPM_DISAGREEMENT
        )
        challenger_uses_neural = bool(
            used or (
                available
                and isinstance(confidence, (float, int))
                and not isinstance(confidence, bool)
                and math.isfinite(confidence)
                and confidence >= FROZEN_THRESHOLD
                and consistent
            )
        )
        sb, sd = r["selected_beat_counts"], r["selected_downbeat_counts"]
        nb, nd = r["candidate_beat_counts"], r["candidate_downbeat_counts"]
        if sb is None or sd is None:
            raise ValueError("missing selected event counts")
        if challenger_uses_neural and (nb is None or nd is None):
            raise ValueError("missing neural event counts")
        challenged_b = nb if challenger_uses_neural else sb
        challenged_d = nd if challenger_uses_neural else sd
        selected_beats.append(sb)
        selected_downs.append(sd)
        challenged_beats.append(challenged_b)
        challenged_downs.append(challenged_d)
        if available and nb is not None and nd is not None:
            candidate_beats.append(nb)
            candidate_downs.append(nd)
        reference_bpm = item.get("bpm")
        if reference_bpm is not None:
            s_bpm = r.get("selected_bpm")
            n_bpm = r.get("neural_bpm")
            if not isinstance(s_bpm, (float, int)) or not math.isfinite(s_bpm):
                raise ValueError("invalid selected BPM")
            if challenger_uses_neural and (not isinstance(n_bpm, (float, int)) or not math.isfinite(n_bpm)):
                raise ValueError("invalid challenger BPM")
            tempo_sel.append(abs(s_bpm - reference_bpm))
            tempo_challenger.append(abs((n_bpm if challenger_uses_neural else s_bpm) - reference_bpm))
            if available and isinstance(n_bpm, (float, int)) and math.isfinite(n_bpm):
                tempo_candidate.append(abs(n_bpm - reference_bpm))
        details.append({
            "id": item["id"],
            "genre": item.get("genre"),
            "confidence": confidence,
            "existing_gate_accepted": used,
            "challenger_uses_neural": challenger_uses_neural,
            "newly_accepted": challenger_uses_neural and not used,
            "selected_beat_f1": f1(sb),
            "challenger_beat_f1": f1(challenged_b),
            "selected_downbeat_f1": f1(sd),
            "challenger_downbeat_f1": f1(challenged_d),
            "beat_f1_change": (f1(challenged_b) - f1(sb)) if f1(challenged_b) is not None and f1(sb) is not None else None,
            "downbeat_f1_change": (f1(challenged_d) - f1(sd)) if f1(challenged_d) is not None and f1(sd) is not None else None,
        })
    scoring = benchmark().score_events
    def metrics(beats, downs, tempo):
        b = scoring(beats)
        d = scoring(downs)
        return {
            "beat_f1": b["f1"] if b else None,
            "downbeat_f1": d["f1"] if d else None,
            "tempo_mae_bpm": round(sum(tempo) / len(tempo), 6) if tempo else None,
        }
    newly = [r for r in details if r["newly_accepted"]]
    severe = [r["id"] for r in details if r["beat_f1_change"] is not None and r["beat_f1_change"] < -0.10]
    return {
        "schema_version": "saund-m1-ballroom-frozen-gate-challenger-v1",
        "frozen_policy": {
            "threshold": FROZEN_THRESHOLD,
            "max_neural_tempo_grid_disagreement_bpm": MAX_GRID_BPM_DISAGREEMENT,
            "rule": "keep current accepted grid; otherwise use neural candidate only if available, confidence>=0.50, and neural BPM/grid mismatch<=2; else retain selected",
        },
        "track_count": len(details),
        "coverage": {
            "newly_accepted": len(newly),
            "challenger_neural_total": sum(r["challenger_uses_neural"] for r in details),
            "existing_neural_total": sum(r["existing_gate_accepted"] for r in details),
        },
        "selected_metrics": metrics(selected_beats, selected_downs, tempo_sel),
        "challenger_metrics": metrics(challenged_beats, challenged_downs, tempo_challenger),
        "raw_candidate_metrics_available_subset": metrics(candidate_beats, candidate_downs, tempo_candidate),
        "regressions": {
            "beat_f1_decreased": sum(r["beat_f1_change"] is not None and r["beat_f1_change"] < -1e-9 for r in details),
            "downbeat_f1_decreased": sum(r["downbeat_f1_change"] is not None and r["downbeat_f1_change"] < -1e-9 for r in details),
            "beat_f1_drop_gt_0_10_tracks": severe,
        },
        "per_track": details,
        "production_changed": False,
        "model_promoted": False,
        "warning": "Shadow-only, development-frozen challenger; inspect disjoint validation before any production or holdout decision.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shadow", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    shadow = json.loads(args.shadow.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "ballroom" or manifest.get("rights_confirmed") is not True:
        raise ValueError("requires separately rights-approved Ballroom manifest")
    if manifest.get("split") != "validation":
        raise ValueError("frozen challenger confirmation requires validation split")
    if shadow.get("reference_manifest_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("shadow was generated with a different manifest")
    result = evaluate(manifest, shadow)
    result["reference_manifest_sha256"] = shadow["reference_manifest_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({k: result[k] for k in (
        "track_count", "coverage", "selected_metrics", "challenger_metrics",
        "raw_candidate_metrics_available_subset", "regressions", "production_changed",
        "model_promoted"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 frozen gate challenger failed: {exc}") from exc
