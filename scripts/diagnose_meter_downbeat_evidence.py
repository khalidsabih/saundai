#!/usr/bin/env python3
"""Validation-only meter and downbeat evidence diagnostics for SAUND.AI.

Separates wrong meter, wrong downbeat phase within a correct meter, and wrong
pulse/beat-grid. Logs autonomous periodic onset-accent evidence alongside
reference-assisted *oracle* scores (strictly diagnostic; NEVER deploy).
No model, production algorithm, cache, or phase safety limit is changed.
Previously inspected test split is never accessed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import librosa
import numpy as np

from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    BEAT_WINDOW_MS, DOWNBEAT_WINDOW_MS, match_events,
    read_timestamp_labels, valid_times,
)
from scripts.diagnose_beat_error_decomposition import (
    diagnostic_downbeat_phase_search, event_rate_diagnosis, reference_meter,
)

METERS = (3, 4, 6)


def accent_evidence(beat_times: list[float], envelope: np.ndarray,
                    *, sr: int = ANALYSIS_SAMPLE_RATE,
                    hop: int = HOP_LENGTH) -> dict[str, Any]:
    """Compare periodic onset accents across meters, without annotations.

    The inferred 'best' phase is from onset strength only. Not a genuine
    independent musical-structure classifier; ties and weak evidence abstain.
    """
    valid_times(beat_times, allow_empty=True)
    if sr < 1 or hop < 1:
        raise ValueError("sample rate and hop must be positive")
    e = np.asarray(envelope, dtype=float)
    if e.ndim != 1 or np.any(~np.isfinite(e)):
        raise ValueError("envelope must be one-dimensional and finite")
    if len(beat_times) < 12 or e.size == 0:
        return {"status": "insufficient_accent_evidence", "meters": {}}
    indices = librosa.time_to_frames(np.asarray(beat_times), sr=sr, hop_length=hop)
    indices = np.rint(indices).astype(int)
    sample = np.full(len(indices), np.nan)
    mask = (indices >= 0) & (indices < e.size)
    sample[mask] = e[indices[mask]]
    if np.count_nonzero(np.isfinite(sample)) < 12:
        return {"status": "insufficient_accent_evidence", "meters": {}}
    by_meter: dict[str, Any] = {}
    for meter in METERS:
        phase_mean = []
        phase_n = []
        for phase in range(meter):
            values = sample[phase::meter]
            values = values[np.isfinite(values)]
            phase_mean.append(float(np.mean(values)) if values.size else 0.0)
            phase_n.append(int(values.size))
        top = int(np.argmax(phase_mean))
        ranked = sorted(phase_mean, reverse=True)
        peak = ranked[0]
        average = float(np.mean(phase_mean))
        contrast = (peak-ranked[1])/max(abs(peak), 1e-8)
        score = max(0.0, (peak-average)/max(abs(peak)+abs(average), 1e-8))
        by_meter[str(meter)] = {
            "best_phase": top,
            "phase_mean_onset_strength": [round(x, 6) for x in phase_mean],
            "phase_counts": phase_n,
            "accent_score": round(score, 5),
            "best_second_contrast": round(max(0.0, contrast), 5),
        }
    ranked_meters = sorted(
        METERS, key=lambda m: by_meter[str(m)]["accent_score"], reverse=True,
    )
    first, second = (by_meter[str(m)]["accent_score"] for m in ranked_meters[:2])
    # Fixed conservative evidence threshold for *investigation* only.
    evidence_sufficient = first >= .08 and first-second >= .025
    return {
        "status": "measured",
        "meters": by_meter,
        "strongest_accent_meter": ranked_meters[0],
        "suggested_meter_or_abstain": ranked_meters[0] if evidence_sufficient else None,
        "abstained": not evidence_sufficient,
        "score_margin": round(first-second, 5),
        "not_a_production_candidate": True,
    }


def classify_error(*, reference_meter_value: int | None,
                   predicted_meter: int | None,
                   pulse_class: str, predicted_downbeat_f1: float | None,
                   reference_meter_phase_oracle: dict[str, Any]) -> str:
    """Mutually exclusive *diagnostic categories*, not causal proof."""
    if pulse_class != "similar_pulse_period":
        return "pulse_period_mismatch_or_uncertain"
    if reference_meter_value not in METERS:
        return "unknown_reference_meter"
    if predicted_meter != reference_meter_value:
        return "meter_mismatch"
    if predicted_downbeat_f1 is None:
        return "missing_downbeat_labels"
    if predicted_downbeat_f1 >= .80:
        return "downbeat_matches"
    if reference_meter_phase_oracle.get("status") == "reference_assisted_oracle":
        if (reference_meter_phase_oracle["best_phase_f_measure"] >= .80
                and reference_meter_phase_oracle["best_phase_f_measure"] >
                predicted_downbeat_f1 + .15):
            return "possible_wrong_downbeat_phase"
    return "downbeat_error_not_resolved_by_periodic_phase"


def evaluate_track(item: dict[str, Any], root: Path, *,
                   precomputed_only: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"id": str(item.get("id", "")), "status": "pending"}
    try:
        beats, _ = read_timestamp_labels(root/str(item["beats"]))
        downbeats = None
        if item.get("downbeats"):
            downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        if item.get("prediction"):
            obj = json.loads((root/str(item["prediction"])).read_text(encoding="utf-8"))
            if isinstance(obj.get("analysis"), dict):
                obj = obj["analysis"]
            estimates = obj
            row["prediction_source"] = "provided_saved_analysis"
        elif precomputed_only:
            return {**row, "status": "missing_saved_prediction"}
        else:
            from djai.analysis import analyze_track
            a = analyze_track(root/str(item["audio"]))
            estimates = {
                "beats_seconds": a.beats_seconds,
                "downbeats_seconds": a.downbeats_seconds,
                "time_signature": a.time_signature,
                "bpm": a.bpm,
            }
            row["prediction_source"] = "unchanged_analyze_track"
        pred_beats = valid_times(estimates.get("beats_seconds") or [], allow_empty=True)
        pred_downbeats = valid_times(estimates.get("downbeats_seconds") or [], allow_empty=True)
        raw_meter = estimates.get("time_signature")
        pred_meter = int(raw_meter) if raw_meter is not None else None
        row["predicted_meter"] = pred_meter
        row["reference_meter"] = reference_meter(downbeats, beats)
        row["pulse"] = event_rate_diagnosis(beats, pred_beats)
        row["beat"] = match_events(beats, pred_beats, tolerance_ms=BEAT_WINDOW_MS)
        row["downbeat"] = (
            match_events(downbeats, pred_downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS)
            if downbeats is not None else None
        )
        oracle = diagnostic_downbeat_phase_search(
            pred_beats, downbeats,
            reference_meter_value=row["reference_meter"].get("meter"),
        )
        row["reference_meter_phase_oracle"] = oracle
        row["error_class"] = classify_error(
            reference_meter_value=row["reference_meter"]["meter"],
            predicted_meter=pred_meter,
            pulse_class=row["pulse"]["class"],
            predicted_downbeat_f1=(
                row["downbeat"]["f_measure"] if row["downbeat"] else None
            ),
            reference_meter_phase_oracle=oracle,
        )
        if item.get("audio"):
            source, sr = load_audio(
                root/str(item["audio"]), sample_rate=ANALYSIS_SAMPLE_RATE,
                mono=False,
            )
            y = librosa.to_mono(source)
            envelope = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
            )
            row["accent_evidence"] = accent_evidence(
                pred_beats, envelope, sr=sr, hop=HOP_LENGTH,
            )
        else:
            row["accent_evidence"] = {
                "status": "unavailable_without_audio",
                "not_a_production_candidate": True,
            }
        row["status"] = "evaluated"
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def diagnose(manifest: dict[str, Any], root: Path, *,
             precomputed_only: bool = False,
             limit: int | None = None) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    seen = set()
    validation = []
    for item in tracks:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            raise ValueError("all IDs must be unique and nonempty")
        seen.add(tid)
        if item.get("split") not in ("train", "validation", "test"):
            raise ValueError("all tracks require explicit split")
        if item["split"] == "validation":
            validation.append(item)
    selected = validation[:limit] if limit is not None else validation
    rows = [evaluate_track(item, root, precomputed_only=precomputed_only)
            for item in selected]
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    meter_pairs = [(r["reference_meter"]["meter"], r["predicted_meter"])
                   for r in evaluated if r["reference_meter"]["meter"] in METERS]
    oracle_rows = [r for r in evaluated if
                   r["reference_meter_phase_oracle"].get("status")
                   == "reference_assisted_oracle" and r["downbeat"] is not None]
    measured_accents = [r for r in evaluated if
                        r["accent_evidence"]["status"] == "measured"]
    return {
        "schema": "saund-validation-meter-downbeat-evidence-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(selected),
        "evaluated_count": len(evaluated),
        "aggregate": {
            "error_classes": dict(sorted(Counter(r["error_class"] for r in evaluated).items())),
            "reference_meter_counts": dict(sorted(Counter(
                str(r["reference_meter"]["meter"]) for r in evaluated
            ).items())),
            "predicted_meter_counts": dict(sorted(Counter(
                str(r["predicted_meter"]) for r in evaluated
            ).items())),
            "meter_confusion": {str(ref): {str(pred): sum(
                a == ref and b == pred for a, b in meter_pairs
            ) for pred in (0, *METERS)} for ref in METERS},
            "reference_known_meter_count": len(meter_pairs),
            "meter_agreement_count": sum(a == b for a, b in meter_pairs),
            "accent_evidence_tracks": len(measured_accents),
            "accent_suggested_meters": dict(sorted(Counter(
                str(r["accent_evidence"]["suggested_meter_or_abstain"])
                for r in measured_accents
            ).items())),
            "oracle_evaluable_tracks": len(oracle_rows),
            "downbeat_baseline_macro_f1_on_oracle_subset": (
                round(float(np.mean([r["downbeat"]["f_measure"]
                                     for r in oracle_rows])), 5)
                if oracle_rows else None
            ),
            "reference_assisted_phase_oracle_macro_f1": (
                round(float(np.mean([
                    r["reference_meter_phase_oracle"]["best_phase_f_measure"]
                    for r in oracle_rows
                ])), 5) if oracle_rows else None
            ),
        },
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "beat_grid_edit_authorized": False,
        "oracle_used_ground_truth": True,
        "limitations": [
            "Validation split ONLY; never load the previously inspected test partition.",
            "Reference meter and reference-assisted phase use labels and cannot be deployed.",
            "Error classes are mutually exclusive triage labels, not proven causal diagnoses.",
            "Onset-accent meter evidence uses the same predicted beat grid; it is not an independent learned musical sensor.",
            "Precomputed predictions without source audio cannot yield onset evidence.",
            "70ms beat and 100ms downbeat F1 windows are NOT the 25ms production phase-integrity residual.",
            "No changes to production meter/downbeat logic, beat grids, cache, bounded warp or fallback.",
            "Ballroom annotations are not an EDM readiness or listening-quality benchmark.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path("data/beat-benchmark/manifest.json"))
    p.add_argument("--root", type=Path)
    p.add_argument("--precomputed-only", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path,
                   default=Path("output/beat-integrity/ballroom-meter-downbeat-evidence.json"))
    args = p.parse_args()
    if not args.manifest.is_file():
        p.error(f"Missing manifest: {args.manifest}")
    result = diagnose(json.loads(args.manifest.read_text(encoding="utf-8")),
                      args.root or args.manifest.parent,
                      precomputed_only=args.precomputed_only, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": result["selected_count"],
        "evaluated": result["evaluated_count"],
        "aggregate": result["aggregate"],
    }, indent=2))
    if result["selected_count"] == 0 or result["evaluated_count"] != result["selected_count"]:
        p.exit(2, "Some validation tracks were not evaluated; inspect detailed report.\n")


if __name__ == "__main__":
    main()
