#!/usr/bin/env python3
"""Validation-only pulse/meter/phase bottleneck benchmark.

Uses UNCHANGED analyze_track or saved predictions, then scores reference-
assisted counterfactuals: predicted grid as-is, either parity half-rate,
and midpoint double-rate. At each grid, compare periodic downbeat phase
at the PRODUCTION meter versus the reference-derived meter. Each oracle
uses annotations and is NEVER an autonomous candidate. Comparisons isolate
diagnostic opportunity; they do not prove causal error attribution.

The previously inspected Ballroom test partition is never loaded. No
production grids, 25ms cross-deck gate, warp, fallbacks, or caches change.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from scripts.diagnose_beat_error_decomposition import (
    event_rate_diagnosis, paired_offsets, reference_meter, SUPPORTED_METERS,
)

MODES = ("unchanged", "half_even", "half_odd", "double_midpoint")
# Screen is for diagnostic grouping only, unrelated to production 25ms gate.
MIN_MEANINGFUL_F1_GAIN = 0.10


def counterfactual_grid(beats: list[float], mode: str) -> list[float]:
    times = valid_times(beats, allow_empty=True)
    if mode == "unchanged":
        return times.copy()
    if mode == "half_even":
        return times[::2]
    if mode == "half_odd":
        return times[1::2]
    if mode == "double_midpoint":
        if len(times) < 2:
            return times.copy()
        out = []
        for left, right in zip(times, times[1:]):
            out.extend((left, (left+right)/2.0))
        out.append(times[-1])
        return valid_times(out, allow_empty=True)
    raise ValueError("unsupported counterfactual mode")


def phase_oracle(
    predicted_beats: list[float],
    reference_downbeats: list[float] | None,
    meter: int | None,
) -> dict[str, Any]:
    if meter not in SUPPORTED_METERS or not reference_downbeats:
        return {"status": "unsupported_or_missing", "used_reference_labels": True}
    if len(predicted_beats) < 2*meter:
        return {"status": "insufficient_beats", "used_reference_labels": True}
    options = []
    for p in range(meter):
        metric = match_events(
            reference_downbeats, predicted_beats[p::meter],
            tolerance_ms=DOWNBEAT_WINDOW_MS,
        )
        options.append({"phase": p, "f_measure": metric["f_measure"]})
    best = max(options, key=lambda x: x["f_measure"])
    return {
        "status": "oracle", "best_phase": best["phase"],
        "f_measure": best["f_measure"], "scores": options,
        "used_reference_labels": True, "not_deployable": True,
    }


def evaluate_values(
    reference_beats: list[float],
    reference_downbeats: list[float] | None,
    predicted_beats: list[float],
    predicted_downbeats: list[float],
    predicted_meter: int | None,
) -> dict[str, Any]:
    ref_meter = reference_meter(reference_downbeats, reference_beats)
    meter = ref_meter.get("meter")
    pulse = event_rate_diagnosis(reference_beats, predicted_beats)
    beat = match_events(reference_beats, predicted_beats, tolerance_ms=BEAT_WINDOW_MS)
    downbeat = (match_events(reference_downbeats, predicted_downbeats,
                            tolerance_ms=DOWNBEAT_WINDOW_MS)
                if reference_downbeats else None)
    variants = {}
    for mode in MODES:
        grid = counterfactual_grid(predicted_beats, mode)
        variants[mode] = {
            "beat": match_events(reference_beats, grid, tolerance_ms=BEAT_WINDOW_MS),
            "phase_oracle_at_predicted_meter": phase_oracle(grid, reference_downbeats, predicted_meter),
            "phase_oracle_at_reference_meter": phase_oracle(grid, reference_downbeats, meter),
        }
    unchanged = variants["unchanged"]
    best_beat_mode = max(MODES, key=lambda m: variants[m]["beat"]["f_measure"])
    best_downbeat_mode = max(
        MODES,
        key=lambda m: variants[m]["phase_oracle_at_reference_meter"].get("f_measure", -1),
    )
    base_downbeat_f1 = downbeat["f_measure"] if downbeat else None
    original_phase = unchanged["phase_oracle_at_predicted_meter"].get("f_measure")
    reference_meter_phase = unchanged["phase_oracle_at_reference_meter"].get("f_measure")
    best_counterfactual_phase = variants[best_downbeat_mode]["phase_oracle_at_reference_meter"].get("f_measure")
    possible_pulse_gain = (
        variants[best_beat_mode]["beat"]["f_measure"]-beat["f_measure"]
    )
    meter_mismatch = meter in SUPPORTED_METERS and predicted_meter != meter
    phase_gap = (original_phase-base_downbeat_f1
                 if original_phase is not None and base_downbeat_f1 is not None else None)
    meter_gap = (reference_meter_phase-original_phase
                 if reference_meter_phase is not None and original_phase is not None else None)
    counterfactual_gap = (
        best_counterfactual_phase-reference_meter_phase
        if best_counterfactual_phase is not None and reference_meter_phase is not None else None
    )
    # Overlapping opportunity flags; never force a single causal explanation.
    return {
        "reference_meter": ref_meter,
        "predicted_meter": predicted_meter,
        "pulse": pulse,
        "beat": beat,
        "downbeat": downbeat,
        "timing": paired_offsets(reference_beats, predicted_beats),
        "meter_mismatch": meter_mismatch,
        "oracles": variants,
        "best_counterfactual_beat_mode": best_beat_mode,
        "best_counterfactual_downbeat_mode": best_downbeat_mode,
        "opportunity": {
            "counterfactual_pulse_beat_f1_gain": round(possible_pulse_gain, 5),
            "phase_only_f1_gain_at_production_meter": round(phase_gap, 5)
                if phase_gap is not None else None,
            "reference_meter_f1_change_with_optimized_phase": round(meter_gap, 5)
                if meter_gap is not None else None,
            "pulse_grid_f1_gain_after_reference_meter_phase": round(counterfactual_gap, 5)
                if counterfactual_gap is not None else None,
            "pulse_counterfactual_gain_ge_10pp": possible_pulse_gain >= MIN_MEANINGFUL_F1_GAIN,
            "phase_oracle_gain_ge_10pp": phase_gap is not None and phase_gap >= MIN_MEANINGFUL_F1_GAIN,
            "reference_meter_better_ge_10pp": meter_gap is not None and meter_gap >= MIN_MEANINGFUL_F1_GAIN,
            "pulse_grid_bar_oracle_gain_ge_10pp": (
                counterfactual_gap is not None and counterfactual_gap >= MIN_MEANINGFUL_F1_GAIN
            ),
        },
        "reference_labels_used_for_oracles": True,
        "not_causal_proof": True,
    }


def evaluate_track(item: dict[str, Any], root: Path, *,
                   precomputed_only: bool = False) -> dict[str, Any]:
    row = {"id": str(item["id"]), "status": "pending"}
    try:
        # Prediction is obtained without supplying reference data to analyze_track.
        if item.get("prediction"):
            pred = json.loads((root/str(item["prediction"])).read_text(encoding="utf-8"))
            if isinstance(pred.get("analysis"), dict):
                pred = pred["analysis"]
            row["prediction_source"] = "saved_analysis"
        elif precomputed_only:
            return {**row, "status": "missing_saved_prediction"}
        else:
            from djai.analysis import analyze_track
            result = analyze_track(root/str(item["audio"]))
            pred = {
                "beats_seconds": result.beats_seconds,
                "downbeats_seconds": result.downbeats_seconds,
                "time_signature": result.time_signature,
            }
            row["prediction_source"] = "unchanged_analyze_track"
        beats = valid_times(pred.get("beats_seconds") or [], allow_empty=True)
        downbeats = valid_times(pred.get("downbeats_seconds") or [], allow_empty=True)
        raw_meter = pred.get("time_signature")
        predicted_meter = int(raw_meter) if raw_meter is not None else None
        # Only here are ground-truth beat/downbeat labels opened.
        reference_beats, _ = read_timestamp_labels(root/str(item["beats"]))
        reference_downbeats = None
        if item.get("downbeats"):
            reference_downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        row.update(evaluate_values(
            reference_beats, reference_downbeats, beats, downbeats, predicted_meter
        ))
        row["status"] = "evaluated"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        row.update(status="unavailable_or_invalid_data", reason=f"{type(exc).__name__}: {exc}")
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "evaluated"]
    def average(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 5) if values else None
    def macro(path: str) -> float | None:
        return average([r[path]["f_measure"] for r in ok if r.get(path) is not None])
    opportunities = ("pulse_counterfactual_gain_ge_10pp", "phase_oracle_gain_ge_10pp",
                     "reference_meter_better_ge_10pp", "pulse_grid_bar_oracle_gain_ge_10pp")
    buckets: dict[str, Any] = {}
    for meter in (3, 4, 6, None):
        group = [r for r in ok if r["reference_meter"]["meter"] == meter]
        buckets[str(meter)] = {
            "count": len(group),
            "baseline_beat_f1": average([r["beat"]["f_measure"] for r in group]),
            "baseline_downbeat_f1": average([r["downbeat"]["f_measure"] for r in group if r["downbeat"]]),
            "meter_mismatch_count": sum(r["meter_mismatch"] for r in group),
        }
    return {
        "evaluated": len(ok),
        "baseline_beat_macro_f1": macro("beat"),
        "baseline_downbeat_macro_f1": macro("downbeat"),
        "reference_meter_groups": buckets,
        "pulse_classes": dict(sorted(Counter(r["pulse"]["class"] for r in ok).items())),
        "predicted_meter_counts": dict(sorted(Counter(str(r["predicted_meter"]) for r in ok).items())),
        "meter_mismatch_count": sum(r["meter_mismatch"] for r in ok),
        "opportunity_counts_overlapping": {key: sum(
            r["opportunity"][key] for r in ok) for key in opportunities},
        "oracle_macro_f1": {
            mode: {
                "beat": average([r["oracles"][mode]["beat"]["f_measure"] for r in ok]),
                "reference_meter_phase": average([
                    r["oracles"][mode]["phase_oracle_at_reference_meter"]["f_measure"]
                    for r in ok if r["oracles"][mode]["phase_oracle_at_reference_meter"].get("status") == "oracle"
                ]),
                "production_meter_phase": average([
                    r["oracles"][mode]["phase_oracle_at_predicted_meter"]["f_measure"]
                    for r in ok if r["oracles"][mode]["phase_oracle_at_predicted_meter"].get("status") == "oracle"
                ]),
            } for mode in MODES
        },
        "best_pulse_mode_counts": dict(sorted(Counter(
            r["best_counterfactual_beat_mode"] for r in ok).items())),
    }


def compare(manifest: dict[str, Any], root: Path, *, limit: int | None = None,
            precomputed_only: bool = False) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    items = manifest.get("tracks")
    if not isinstance(items, list):
        raise ValueError("manifest requires tracks list")
    ids = [str(t.get("id") or "") for t in items]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("unique nonempty IDs required")
    if any(t.get("split") not in ("train", "validation", "test") for t in items):
        raise ValueError("explicit split required")
    selected = [t for t in items if t["split"] == "validation"]
    if limit is not None:
        selected = selected[:limit]
    rows = [evaluate_track(item, root, precomputed_only=precomputed_only) for item in selected]
    return {
        "schema": "saund-pulse-meter-phase-root-cause-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(selected),
        "aggregate": aggregate(rows),
        "tracks": rows,
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "reference_labels_used_for_oracles": True,
        "auto_promotion_allowed": False,
        "limitations": [
            "Only Ballroom validation tracks are read; previously inspected test partition is excluded.",
            "Every counterfactual and best phase is selected using ground-truth labels and cannot be deployed.",
            "Counterfactual half- and double-rate grids are artificial diagnostic constructions, NOT autonomous beat detectors.",
            "Pulse-period ratios and overlapping opportunity flags are not proven causal diagnoses.",
            "Oracles are neither additive nor calibrated; no score here constitutes expected production performance.",
            "Reference meter is inferred from reference beat/downbeat annotations, not from audio.",
            "70ms beat and 100ms downbeat F1 tolerances do not alter the protected 25ms cross-deck residual gate.",
            "No changes to production meter, beats, downbeats, model authority, safety thresholds, warp or fallback.",
            "Ballroom music does not represent DJ/EDM transition safety.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/beat-benchmark/manifest.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--precomputed-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/ballroom-pulse-meter-phase-root-causes.json"
    ))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Missing manifest: {args.manifest}")
    report = compare(json.loads(args.manifest.read_text(encoding="utf-8")),
                     args.root or args.manifest.parent, limit=args.limit,
                     precomputed_only=args.precomputed_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "evaluated": report["aggregate"]["evaluated"],
        "aggregate": report["aggregate"],
    }, indent=2))
    if not report["selected_count"] or report["aggregate"]["evaluated"] != report["selected_count"]:
        parser.exit(2, "Incomplete validation. Inspect detailed report.\n")


if __name__ == "__main__":
    main()
