#!/usr/bin/env python3
"""Read-only 512-vs-256 hop shadow comparison on Ballroom VALIDATION audio.

Three distinct outputs per track:
* production: untouched SAUND analyze_track (may use governed neural sensors)
* deterministic_512: replay the existing deterministic onset/beat/bar front-end
* shadow_256: same deterministic front-end at hop=256, offline only

Use the same original audio for the deterministic alternatives. Never modify
any model registry, beat cache, production analysis, cross-deck safety limit,
bounded warp, or fallback. Do not inspect/test the previously exposed test split.
The only persistent write is a requested JSON report.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import librosa
import numpy as np

from djai.analysis import (
    ANALYSIS_SAMPLE_RATE, HOP_LENGTH, _beat_grid_diagnostics,
    _estimate_downbeats, _estimate_time_signature, analyze_track,
)
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    BEAT_WINDOW_MS, DOWNBEAT_WINDOW_MS, match_events, read_timestamp_labels,
    valid_times,
)
from scripts.diagnose_beat_error_decomposition import (
    event_rate_diagnosis, paired_offsets, reference_meter,
)

BASELINE_HOP = HOP_LENGTH
SHADOW_HOP = 256
CONFIGS = ("production", "deterministic_512", "shadow_256")


def deterministic_front_end(y: np.ndarray, sr: int, hop: int) -> dict[str, Any]:
    """Mirror production deterministic front-end, differing ONLY by hop.

    This is not full-track analysis. Beat/downbeat evidence and meter are
    computed with production helper functions; no production metadata,
    neural authority, caching or decision pathways are changed.
    """
    if hop not in (BASELINE_HOP, SHADOW_HOP):
        raise ValueError("unsupported hop; controlled 512-vs-256 only")
    envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop, aggregate=np.median,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=envelope, sr=sr, hop_length=hop,
        units="frames", sparse=True,
    )
    frames = np.asarray(beat_frames, dtype=int)
    beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    confidence, _ = _beat_grid_diagnostics(beats, len(y)/sr)
    meter, meter_confidence = _estimate_time_signature(
        frames, envelope, confidence,
    )
    selected_meter = meter if meter in (3, 4, 6) else 4
    down_frames, down_confidence = _estimate_downbeats(
        frames, envelope, beats_per_bar=selected_meter,
    )
    downbeats = librosa.frames_to_time(
        down_frames, sr=sr, hop_length=hop,
    )
    estimated = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return {
        "beats_seconds": beats.tolist(),
        "downbeats_seconds": downbeats.tolist(),
        "bpm": estimated if estimated > 0 else 120.0,
        "time_signature": int(meter),
        "time_signature_confidence": float(meter_confidence),
        "downbeat_confidence": float(down_confidence),
        "beat_grid_confidence": float(confidence),
    }


def observe_work(callback) -> tuple[Any, dict[str, Any]]:
    """Wall time and Python-traced peak allocations; NOT total RSS.

    NumPy/FFT native allocations are not fully represented by tracemalloc.
    The reported peaks must not be described as full process memory.
    """
    tracemalloc.start()
    try:
        start = time.perf_counter()
        result = callback()
        elapsed = time.perf_counter()-start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {
        "wall_seconds": round(elapsed, 4),
        "python_traced_peak_mib": round(peak/(1024*1024), 3),
        "not_total_process_peak_rss": True,
    }


def evaluate_prediction(
    reference_beats: list[float],
    reference_downbeats: list[float] | None,
    prediction: dict[str, Any],
    *, runtime: dict[str, Any],
) -> dict[str, Any]:
    beats = valid_times(prediction.get("beats_seconds") or [], allow_empty=True)
    downbeats = valid_times(prediction.get("downbeats_seconds") or [], allow_empty=True)
    beat = match_events(reference_beats, beats, tolerance_ms=BEAT_WINDOW_MS)
    downbeat = (
        match_events(reference_downbeats, downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS)
        if reference_downbeats is not None else None
    )
    predicted_bpm = prediction.get("bpm")
    if predicted_bpm is not None and not math.isfinite(float(predicted_bpm)):
        raise ValueError("non-finite predicted BPM")
    return {
        "status": "evaluated",
        "beat": beat,
        "downbeat": downbeat,
        "timing": paired_offsets(reference_beats, beats),
        "pulse": event_rate_diagnosis(reference_beats, beats),
        "predicted_meter": prediction.get("time_signature"),
        "predicted_bpm": predicted_bpm,
        "prediction_count": len(beats),
        "downbeat_count": len(downbeats),
        "runtime": runtime,
    }


def evaluate_track(item: dict[str, Any], root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"id": str(item.get("id", "")), "status": "pending"}
    try:
        reference, _ = read_timestamp_labels(root/str(item["beats"]))
        ref_downbeats = None
        if item.get("downbeats"):
            ref_downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        row["reference_meter"] = reference_meter(ref_downbeats, reference)
        audio = root/str(item["audio"])
        # This is the exact production pipeline, including governed neural
        # models if they are in-use. It is observational, not shadow control.
        production, runtime = observe_work(lambda: analyze_track(audio))
        row["production"] = evaluate_prediction(
            reference, ref_downbeats, {
                "beats_seconds": production.beats_seconds,
                "downbeats_seconds": production.downbeats_seconds,
                "bpm": production.bpm,
                "time_signature": production.time_signature,
            }, runtime=runtime,
        )
        # Loading once for like-for-like deterministic 512/256 front ends.
        source, sr = load_audio(audio, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
        y = librosa.to_mono(source)
        for name, hop in (("deterministic_512", BASELINE_HOP),
                          ("shadow_256", SHADOW_HOP)):
            try:
                prediction, runtime = observe_work(
                    lambda h=hop: deterministic_front_end(y, sr, h)
                )
                row[name] = evaluate_prediction(
                    reference, ref_downbeats, prediction, runtime=runtime,
                )
            except Exception as exc:
                row[name] = {
                    "status": "analysis_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        row["status"] = "evaluated"
    except Exception as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def aggregate(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    measured = [r[config] for r in rows
                if r.get(config, {}).get("status") == "evaluated"]
    n = len(measured)
    beat = [r["beat"] for r in measured]
    downbeat = [r["downbeat"] for r in measured if r["downbeat"] is not None]
    signed = [r["timing"]["signed_median_ms"] for r in measured
              if r["timing"]["status"] == "measured"]
    times = [r["runtime"]["wall_seconds"] for r in measured]
    traced = [r["runtime"]["python_traced_peak_mib"] for r in measured]
    pulse = Counter(r["pulse"]["class"] for r in measured)
    meter = Counter(str(r["predicted_meter"]) for r in measured)
    def f1_summary(metric: list[dict[str, Any]]) -> dict[str, Any]:
        if not metric:
            return {"measured": 0, "macro_f_measure": None, "micro_f_measure": None}
        matched = sum(v["matched"] for v in metric)
        reference = sum(v["reference_count"] for v in metric)
        predicted = sum(v["predicted_count"] for v in metric)
        return {
            "measured": len(metric),
            "macro_f_measure": round(float(np.mean([v["f_measure"] for v in metric])), 5),
            "micro_f_measure": round(2*matched/(reference+predicted), 5)
                if reference+predicted else 0.0,
            "matched": matched,
            "reference": reference,
            "predicted": predicted,
            "matched_over_25ms": sum(v["matched_over_25ms"] for v in metric),
        }
    return {
        "evaluated_count": n,
        "beat": f1_summary(beat),
        "downbeat": f1_summary(downbeat),
        "tracks_with_measured_signed_offset": len(signed),
        "median_track_signed_offset_ms": round(float(np.median(signed)), 3)
            if signed else None,
        "pulse_period_classes": dict(sorted(pulse.items())),
        "predicted_meter_counts": dict(sorted(meter.items())),
        "median_runtime_seconds": round(float(np.median(times)), 4) if times else None,
        "median_python_traced_peak_mib": round(float(np.median(traced)), 3)
            if traced else None,
        "memory_metric_is_not_total_RSS": True,
    }


def compare(manifest: dict[str, Any], root: Path, *,
            limit: int | None = None) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest must contain a tracks list")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    seen = set()
    validation = []
    for item in tracks:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            raise ValueError("unique nonempty track IDs required")
        seen.add(tid)
        if item.get("split") not in ("train", "validation", "test"):
            raise ValueError("every track needs explicit train/validation/test split")
        if item["split"] == "validation":
            validation.append(item)
    selected = validation[:limit] if limit is not None else validation
    rows = [evaluate_track(item, root) for item in selected]
    summaries = {c: aggregate(rows, c) for c in CONFIGS}
    paired = [
        r for r in rows
        if r.get("deterministic_512", {}).get("status") == "evaluated"
        and r.get("shadow_256", {}).get("status") == "evaluated"
    ]
    beat_delta = [
        r["shadow_256"]["beat"]["f_measure"] -
        r["deterministic_512"]["beat"]["f_measure"] for r in paired
    ]
    down_delta = [
        r["shadow_256"]["downbeat"]["f_measure"] -
        r["deterministic_512"]["downbeat"]["f_measure"] for r in paired
        if r["shadow_256"]["downbeat"] is not None
        and r["deterministic_512"]["downbeat"] is not None
    ]
    return {
        "schema": "saund-shadow-hop256-ballroom-validation-v1",
        "dataset": manifest.get("dataset", "unspecified"),
        "selected_split": "validation",
        "validation_available": len(validation),
        "selected_count": len(selected),
        "track_evaluated_count": sum(r["status"] == "evaluated" for r in rows),
        "comparison_paired_count": len(paired),
        "summary": summaries,
        "paired_differences": {
            "median_beat_f1_delta_256_minus_512": (
                round(float(np.median(beat_delta)), 5) if beat_delta else None
            ),
            "median_downbeat_f1_delta_256_minus_512": (
                round(float(np.median(down_delta)), 5) if down_delta else None
            ),
            "paired_downbeat_count": len(down_delta),
        },
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "shadow_is_not_automatically_promotable": True,
        "limitations": [
            "Already-inspected Ballroom TEST tracks are never read.",
            "Production analyze_track may contain neural evidence while deterministic replay is non-neural. Use 512-vs-256 for matched comparison.",
            "The 256-hop shadow copies only rhythm and meter/downbeat front-end; it is not full 256-hop SAUND production analysis.",
            "70ms beat and 100ms downbeat benchmarks are not the protected 25ms cross-deck residual.",
            "Signed offsets are measured on close-pair subset; missing/out-of-window beats cannot contribute to that median.",
            "tracemalloc only estimates Python-traced allocations, NOT total RSS or native NumPy allocations.",
            "Sequential wall-clock time is environmentally variable and does not predict real-time safety.",
            "No automatic corrections, production algorithm switches, cache writes or safety-gate changes are performed.",
            "Ballroom is not a representative EDM transition or musical-quality release benchmark.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/beat-benchmark/manifest.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--limit", type=int, help="Only first N validation tracks; smoke test")
    parser.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/ballroom-shadow-hop256-validation.json",
    ))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    report = compare(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.root or args.manifest.parent, limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "track_evaluated": report["track_evaluated_count"],
        "paired": report["comparison_paired_count"],
        "summary": report["summary"],
        "paired_differences": report["paired_differences"],
    }, indent=2))
    if report["selected_count"] == 0 or (
        report["comparison_paired_count"] != report["selected_count"]
        or report["summary"]["production"]["evaluated_count"] != report["selected_count"]
    ):
        parser.exit(2, "Incomplete validation evaluation. See report; do not claim success.\n")


if __name__ == "__main__":
    main()
