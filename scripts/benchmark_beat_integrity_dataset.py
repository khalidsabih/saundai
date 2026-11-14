#!/usr/bin/env python3
"""Read-only annotated-dataset Beat Integrity benchmark for SAUND.AI.

Manifest-based adapter for legally obtained audio and time-stamped beat /
downbeat ground truth. Never downloads copyrighted audio or presumes tempo-only
datasets contain beat labels. Always distinguish fixture tests from genuine
external-data validation. Runs the actual analyze_track pipeline unless a
precomputed analysis JSON is explicitly supplied.

Manifest JSON format:
{"dataset":"name","license_note":"...","tracks":[
 {"id":"001","audio":"audio/001.wav","beats":"annotations/001.beats",
  "downbeats":"annotations/001.downbeats","split":"test"}]}
Timestamp files: one beat time (seconds) per line, or whitespace/tab-delimited
time plus optional beat-in-bar index. This is a simple *normalized import*
format, not a claim to parse every native dataset annotation release.
A precomputed analysis JSON (optional) has beats_seconds/downbeats_seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np

BEAT_WINDOW_MS = 70.0
DOWNBEAT_WINDOW_MS = 100.0
JITTER_SCREEN_MS = 25.0
MIN_TRACKS_FOR_AGGREGATE = 1
ALLOWED_SPLITS = {"train", "validation", "test"}


def valid_times(values: list[float], *, allow_empty: bool = False) -> list[float]:
    a = [float(v) for v in values]
    if (not allow_empty and not a) or any(not math.isfinite(x) or x < 0 for x in a):
        raise ValueError("timestamps must be nonnegative, finite and nonempty")
    if any(y <= x for x, y in zip(a, a[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return a


def read_timestamp_labels(path: Path) -> tuple[list[float], list[int | None]]:
    """Support normalized times or times plus optional beat-in-bar positions.

    No dataset-specific indexing assumptions are made; annotated downbeats
    should be supplied explicitly rather than guessed from beat indices.
    """
    times = []
    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.replace(",", " ").split()
        if len(fields) > 2:
            raise ValueError(f"{path}: expected time and optional beat-in-bar, got {line!r}")
        times.append(float(fields[0]))
        positions.append(int(fields[1]) if len(fields) == 2 else None)
    return valid_times(times), positions


def match_events(
    reference: list[float], predicted: list[float], *, tolerance_ms: float,
) -> dict[str, Any]:
    """Monotone globally optimal maximum-cardinality, min-error matching.

    A single prediction cannot match two reference beats. This is more
    trustworthy than greedy nearest-neighbor pairing on irregular grids.
    """
    reference = valid_times(reference)
    predicted = valid_times(predicted, allow_empty=True)
    if tolerance_ms <= 0 or not math.isfinite(tolerance_ms):
        raise ValueError("invalid match tolerance")
    n, m = len(reference), len(predicted)
    count = [[0]*(m+1) for _ in range(n+1)]
    error = [[0.0]*(m+1) for _ in range(n+1)]
    decision = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            possibilities = [
                (count[i-1][j], -error[i-1][j], 1),
                (count[i][j-1], -error[i][j-1], 2),
            ]
            ms = abs(reference[i-1]-predicted[j-1])*1000
            if ms <= tolerance_ms:
                possibilities.append((count[i-1][j-1]+1, -(error[i-1][j-1]+ms), 3))
            chosen = max(possibilities)
            count[i][j], error[i][j], decision[i][j] = chosen[0], -chosen[1], chosen[2]
    i, j = n, m
    distances = []
    while i and j:
        move = decision[i][j]
        if move == 3:
            distances.append(abs(reference[i-1]-predicted[j-1])*1000)
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matched = len(distances)
    precision = matched/m if m else 0.0
    recall = matched/n
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "reference_count": n,
        "predicted_count": m,
        "matched": matched,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f_measure": round(f1, 5),
        "median_abs_error_ms": round(float(np.median(distances)), 3) if distances else None,
        "p95_abs_error_ms": round(float(np.percentile(distances, 95)), 3) if distances else None,
        "matched_over_25ms": sum(x > JITTER_SCREEN_MS for x in distances),
        "matching_tolerance_ms": tolerance_ms,
    }


def _aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    metrics = [r[label] for r in rows if r.get(label) is not None]
    if not metrics:
        return {"tracks_evaluated": 0, "status": "no_labels"}
    counts = [m["matched"] for m in metrics]
    references = [m["reference_count"] for m in metrics]
    predicted = [m["predicted_count"] for m in metrics]
    tp = sum(counts)
    fp = sum(predicted)-tp
    fn = sum(references)-tp
    micro_f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    return {
        "status": "measured",
        "tracks_evaluated": len(metrics),
        "macro_f_measure": round(float(np.mean([m["f_measure"] for m in metrics])), 5),
        "micro_f_measure": round(micro_f1, 5),
        "matched_over_25ms": sum(m["matched_over_25ms"] for m in metrics),
        "total_matched": tp,
        "total_reference": sum(references),
        "total_predicted": sum(predicted),
    }


def benchmark(
    manifest: dict[str, Any], root: Path, *,
    selected_split: str = "test", use_precomputed_only: bool = False,
) -> dict[str, Any]:
    if selected_split not in ALLOWED_SPLITS:
        raise ValueError("unknown split")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    seen = set()
    records = []
    for item in tracks:
        track_id = str(item.get("id") or "")
        if not track_id or track_id in seen:
            raise ValueError("track IDs must be unique and nonempty")
        seen.add(track_id)
        split = str(item.get("split") or "")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{track_id}: explicit train/validation/test split required")
        if split != selected_split:
            continue
        row = {"id": track_id, "split": split, "status": "pending"}
        try:
            annotation_path = root / str(item["beats"])
            reference, _ = read_timestamp_labels(annotation_path)
            ref_downbeats = None
            if item.get("downbeats"):
                ref_downbeats, _ = read_timestamp_labels(root / str(item["downbeats"]))
            prediction_source = "saund_analyze_track"
            if item.get("prediction"):
                obj = json.loads((root / str(item["prediction"])).read_text(encoding="utf-8"))
                if isinstance(obj.get("analysis"), dict):
                    obj = obj["analysis"]
                estimates = obj
                prediction_source = "precomputed_saund_analysis"
            elif use_precomputed_only:
                row["status"] = "missing_precomputed_analysis"
                records.append(row)
                continue
            else:
                from djai.analysis import analyze_track
                analysis = analyze_track(root / str(item["audio"]))
                estimates = {
                    "beats_seconds": analysis.beats_seconds,
                    "downbeats_seconds": analysis.downbeats_seconds,
                    "bpm": analysis.bpm,
                }
            estimated_beats = valid_times(
                estimates.get("beats_seconds") or [], allow_empty=True,
            )
            estimated_downbeats = valid_times(
                estimates.get("downbeats_seconds") or [], allow_empty=True,
            )
            row["beat"] = match_events(reference, estimated_beats, tolerance_ms=BEAT_WINDOW_MS)
            row["downbeat"] = (
                match_events(ref_downbeats, estimated_downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS)
                if ref_downbeats is not None else None
            )
            row["reference_has_downbeats"] = ref_downbeats is not None
            row["prediction_source"] = prediction_source
            row["predicted_bpm"] = estimates.get("bpm")
            row["status"] = "evaluated"
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            row["status"] = "unavailable_or_invalid_data"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        records.append(row)
    evaluated = [row for row in records if row["status"] == "evaluated"]
    return {
        "schema": "saund-beat-integrity-dataset-benchmark-v1",
        "dataset": str(manifest.get("dataset") or "unspecified"),
        "license_note": str(manifest.get("license_note") or "not specified; verify usage rights"),
        "selected_split": selected_split,
        "source_data_count": len(tracks),
        "selected_count": len(records),
        "evaluated_count": len(evaluated),
        "beat": _aggregate(evaluated, "beat"),
        "downbeat": _aggregate(evaluated, "downbeat"),
        "tracks": records,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "limitations": [
            "Reference beat/downbeat labels require separately licensed/accessible audio and verified annotation alignment.",
            "This measures beat and downbeat timestamp matches, NOT rendered-mix quality or false-safe DJ transition rates.",
            "No downbeat scores are computed when ground-truth downbeats are absent; tempo-only datasets cannot supply beat labels.",
            "70ms beat and 100ms downbeat tolerance are benchmark matching windows, NOT the 25ms production phase residual limit.",
            "The test split must not be used to choose hyperparameters or tune detector decisions.",
            "No external audio or reference annotations are bundled, downloaded, or redistributed.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path, help="Dataset root; default is manifest's directory")
    p.add_argument("--split", choices=sorted(ALLOWED_SPLITS), default="test")
    p.add_argument("--precomputed-only", action="store_true",
                   help="Only evaluate paths with saved SAUND analysis JSON; no audio processing")
    p.add_argument("--output", type=Path, default=Path("output/beat-integrity/dataset-report.json"))
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.root or args.manifest.parent
    report = benchmark(manifest, root, selected_split=args.split,
                       use_precomputed_only=args.precomputed_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "dataset": report["dataset"],
        "selected": report["selected_count"],
        "evaluated": report["evaluated_count"],
        "beat_macro_f_measure": report["beat"].get("macro_f_measure"),
        "downbeat_macro_f_measure": report["downbeat"].get("macro_f_measure"),
    }, indent=2))


if __name__ == "__main__":
    main()
