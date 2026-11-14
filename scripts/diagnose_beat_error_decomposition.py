#!/usr/bin/env python3
"""Read-only beat error decomposition on the Ballroom VALIDATION split.

Evaluate SAUND.AI on original annotated audio (or explicit saved predictions)
and distinguish signed timestamp offset, event-rate/octave ambiguity and
four- vs three-beat meter/downbeat-phase uncertainty. Reference-informed
phase searches are diagnostic oracles, NOT viable autonomous algorithms.

Never tune using the previously inspected test split; this command refuses
test by design. Neither write caches nor alter production beat/downbeat
predictions, phase-integrity policy, fallback decisions, or source audio.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
from scripts.benchmark_beat_integrity_dataset import (
    BEAT_WINDOW_MS, DOWNBEAT_WINDOW_MS, match_events,
    read_timestamp_labels, valid_times,
)

TIME_SCREEN_MS = 25.0
PULSE_RATIO_TOLERANCE = 0.18
SUPPORTED_METERS = (3, 4, 6)


def paired_offsets(reference: list[float], predicted: list[float],
                   *, tolerance_ms: float = BEAT_WINDOW_MS) -> dict[str, Any]:
    """One-to-one chronological event pairs, sign=prediction minus reference.

    The beat spacing in the source is generally much greater than tolerance.
    This greedy chronological pairer is for signed-offset analysis only;
    the benchmark's globally optimal match_events remains the authority for F1.
    """
    reference = valid_times(reference)
    predicted = valid_times(predicted, allow_empty=True)
    i = j = 0
    pairs = []
    tolerance = tolerance_ms/1000
    while i < len(reference) and j < len(predicted):
        dt = predicted[j]-reference[i]
        if abs(dt) <= tolerance:
            pairs.append((reference[i], dt*1000))
            i += 1
            j += 1
        elif dt < -tolerance:
            j += 1
        else:
            i += 1
    if not pairs:
        return {"status": "no_close_pairs", "paired_count": 0}
    times = np.asarray([p[0] for p in pairs], dtype=float)
    signed = np.asarray([p[1] for p in pairs], dtype=float)
    median = float(np.median(signed))
    centered = signed-median
    # Avoid asserting drift from a handful of points or very short recordings.
    slope = (float(np.polyfit(times-times[0], signed, 1)[0])
             if len(pairs) >= 12 and times[-1]-times[0] >= 10 else None)
    return {
        "status": "measured",
        "paired_count": len(pairs),
        "signed_median_ms": round(median, 3),
        "median_absolute_error_ms": round(float(np.median(abs(signed))), 3),
        "median_absolute_deviation_ms": round(float(np.median(abs(centered))), 3),
        "over_25ms_count": int(np.count_nonzero(abs(signed) > TIME_SCREEN_MS)),
        "early_count": int(np.count_nonzero(signed < -TIME_SCREEN_MS)),
        "late_count": int(np.count_nonzero(signed > TIME_SCREEN_MS)),
        "estimated_offset_slope_ms_per_s": round(slope, 4) if slope is not None else None,
        "slope_is_diagnostic_only": True,
    }


def event_rate_diagnosis(reference: list[float], predicted: list[float]) -> dict[str, Any]:
    reference = valid_times(reference)
    predicted = valid_times(predicted, allow_empty=True)
    if len(reference) < 12 or len(predicted) < 6:
        return {"status": "insufficient_events", "class": "unknown"}
    # Span-matched counts rather than full-track counts: track boundary offsets
    # can cause small count differences independent of tempo octave.
    left = max(reference[0], predicted[0])
    right = min(reference[-1], predicted[-1])
    if right-left < 6:
        return {"status": "insufficient_overlap", "class": "unknown"}
    ref_intervals = np.diff(reference)
    pred_intervals = np.diff(predicted)
    ref_period = float(np.median(ref_intervals))
    pred_period = float(np.median(pred_intervals))
    if min(ref_period, pred_period) <= 0:
        return {"status": "invalid_period", "class": "unknown"}
    ratio = pred_period/ref_period
    if abs(ratio-2) <= 2*PULSE_RATIO_TOLERANCE:
        kind = "possible_half_time"
    elif abs(ratio-0.5) <= 0.5*PULSE_RATIO_TOLERANCE:
        kind = "possible_double_time"
    elif abs(ratio-1) <= PULSE_RATIO_TOLERANCE:
        kind = "similar_pulse_period"
    else:
        kind = "other_pulse_period"
    return {
        "status": "measured",
        "class": kind,
        "reference_median_interval_ms": round(ref_period*1000, 3),
        "predicted_median_interval_ms": round(pred_period*1000, 3),
        "predicted_to_reference_period_ratio": round(ratio, 4),
        "not_confirmed_musical_tempo_error": True,
    }


def reference_meter(downbeats: list[float] | None, beats: list[float]) -> dict[str, Any]:
    if not downbeats or len(downbeats) < 4:
        return {"status": "missing_or_insufficient_downbeats", "meter": None}
    # Count reference beats between annotated bar boundaries; robust to
    # the start/end of clips not coinciding with a downbeat.
    ref = np.asarray(valid_times(beats), dtype=float)
    bars = np.asarray(valid_times(downbeats), dtype=float)
    counts = []
    for left, right in zip(bars, bars[1:]):
        n = int(np.count_nonzero((ref >= left-0.075) & (ref < right-0.075)))
        if n > 0:
            counts.append(n)
    if len(counts) < 3:
        return {"status": "insufficient_complete_bars", "meter": None}
    distribution = {str(k): counts.count(k) for k in sorted(set(counts))}
    winner = max(SUPPORTED_METERS, key=lambda m: counts.count(m))
    share = counts.count(winner)/len(counts)
    meter = winner if share >= 0.75 else None
    return {
        "status": "inferred_from_reference" if meter is not None else "ambiguous_reference_meter",
        "meter": meter,
        "complete_bar_interval_count": len(counts),
        "bar_length_distribution": distribution,
        "winner_share": round(share, 4),
        "not_an_autonomous_meter_prediction": True,
    }


def diagnostic_downbeat_phase_search(
    predicted_beats: list[float], reference_downbeats: list[float] | None,
    *, reference_meter_value: int | None,
) -> dict[str, Any]:
    """Ground-truth-assisted best possible periodic phase: an oracle ceiling.

    Uses the labels to pick the best phase and therefore cannot be deployed
    or interpreted as a detector improvement. No meter selection is learned.
    """
    if not reference_downbeats or reference_meter_value not in SUPPORTED_METERS:
        return {"status": "missing_reference_or_metre", "oracle_used_ground_truth": True}
    if len(predicted_beats) < 2*reference_meter_value:
        return {"status": "insufficient_predicted_beats", "oracle_used_ground_truth": True}
    scores = []
    for phase in range(reference_meter_value):
        hypothesized = predicted_beats[phase::reference_meter_value]
        result = match_events(
            reference_downbeats, hypothesized,
            tolerance_ms=DOWNBEAT_WINDOW_MS,
        )
        scores.append({"phase": phase, "f_measure": result["f_measure"]})
    winner = max(scores, key=lambda x: x["f_measure"])
    return {
        "status": "reference_assisted_oracle",
        "meter": reference_meter_value,
        "phase_scores": scores,
        "best_phase": winner["phase"],
        "best_phase_f_measure": winner["f_measure"],
        "oracle_used_ground_truth": True,
        "not_a_deployable_downbeat_method": True,
    }


def decompose_track(item: dict[str, Any], root: Path, *,
                    precomputed_only: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"id": str(item["id"]), "status": "pending"}
    try:
        reference, _ = read_timestamp_labels(root/str(item["beats"]))
        downbeats = None
        if item.get("downbeats"):
            downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        if item.get("prediction"):
            obj = json.loads((root/str(item["prediction"])).read_text(encoding="utf-8"))
            if isinstance(obj.get("analysis"), dict):
                obj = obj["analysis"]
            pred = obj
            row["prediction_source"] = "precomputed_saund_analysis"
        elif precomputed_only:
            return {**row, "status": "missing_precomputed_analysis"}
        else:
            from djai.analysis import analyze_track
            analysis = analyze_track(root/str(item["audio"]))
            pred = {
                "beats_seconds": analysis.beats_seconds,
                "downbeats_seconds": analysis.downbeats_seconds,
                "bpm": analysis.bpm,
                "time_signature": analysis.time_signature,
            }
            row["prediction_source"] = "saund_analyze_track"
        estimated = valid_times(pred.get("beats_seconds") or [], allow_empty=True)
        estimated_downbeats = valid_times(pred.get("downbeats_seconds") or [], allow_empty=True)
        row["beat"] = match_events(reference, estimated, tolerance_ms=BEAT_WINDOW_MS)
        row["downbeat"] = (match_events(downbeats, estimated_downbeats,
                                       tolerance_ms=DOWNBEAT_WINDOW_MS)
                           if downbeats else None)
        row["timing"] = paired_offsets(reference, estimated)
        row["pulse"] = event_rate_diagnosis(reference, estimated)
        row["reference_meter"] = reference_meter(downbeats, reference)
        row["downbeat_phase_oracle"] = diagnostic_downbeat_phase_search(
            estimated, downbeats, reference_meter_value=row["reference_meter"].get("meter"),
        )
        row["predicted_bpm"] = pred.get("bpm")
        row["predicted_meter"] = pred.get("time_signature")
        row["status"] = "evaluated"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def diagnose(manifest: dict[str, Any], root: Path, *,
             precomputed_only: bool = False) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    all_ids = [str(t.get("id") or "") for t in tracks]
    if any(not i for i in all_ids) or len(all_ids) != len(set(all_ids)):
        raise ValueError("manifest IDs must be unique and nonempty")
    if any(t.get("split") not in ("train", "validation", "test") for t in tracks):
        raise ValueError("all tracks require explicit split")
    selected = [t for t in tracks if t["split"] == "validation"]
    rows = [decompose_track(t, root, precomputed_only=precomputed_only) for t in selected]
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    metrics: dict[str, Any] = {}
    for kind in ("beat", "downbeat"):
        labeled = [r[kind]["f_measure"] for r in evaluated if r[kind] is not None]
        metrics[kind+"_macro_f_measure"] = (
            round(float(np.mean(labeled)), 5) if labeled else None
        )
    classes = {}
    for r in evaluated:
        c = r["pulse"]["class"]
        classes[c] = classes.get(c, 0)+1
    meters = {}
    for r in evaluated:
        met = r["reference_meter"]["meter"]
        key = str(met) if met is not None else "unknown"
        meters[key] = meters.get(key, 0)+1
    strong_beat_bad_downbeat = sum(
        r["beat"]["f_measure"] >= 0.90
        and r["downbeat"] is not None
        and r["downbeat"]["f_measure"] <= 0.10
        for r in evaluated
    )
    return {
        "schema": "saund-beat-error-decomposition-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(selected),
        "evaluated_count": len(evaluated),
        "aggregate": {
            **metrics,
            "pulse_classes": classes,
            "reference_meters": meters,
            "tracks_with_beat_f1_ge_90pct_downbeat_f1_le_10pct":
                strong_beat_bad_downbeat,
            "timing_paired_over_25ms": sum(
                r["timing"].get("over_25ms_count", 0) for r in evaluated
            ),
        },
        "tracks": rows,
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "limitations": [
            "Validation split only; do not tune to the previously inspected held-out test split.",
            "Signed timing pairs use a 70ms local greedy tolerance, while benchmark F1 uses global optimal matching.",
            "The 25ms local timing screen is not the production cross-deck phase-integrity residual.",
            "Reference-derived meter and best downbeat phase use ground-truth labels, so the oracle is NOT deployable.",
            "Median interbeat interval ratios only suggest possible half/double pulse tracking, never verify musical beat identity.",
            "Original beat and bar annotations may contain systematic timing offsets or meter assumptions.",
            "This dataset is Ballroom, not an EDM transition-safety benchmark.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/beat-benchmark/manifest.json"))
    parser.add_argument("--root", type=Path, help="Override relative path resolution root")
    parser.add_argument("--precomputed-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/beat-integrity/ballroom-validation-decomposition.json"))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Dataset manifest missing: {args.manifest}. Run the dataset preparation command first.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = diagnose(manifest, args.root or args.manifest.parent,
                      precomputed_only=args.precomputed_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "evaluated": report["evaluated_count"],
        "aggregate": report["aggregate"],
    }, indent=2))
    if report["evaluated_count"] == 0:
        parser.exit(2, "No validation tracks evaluated; this is not a benchmark result.\n")


if __name__ == "__main__":
    main()
