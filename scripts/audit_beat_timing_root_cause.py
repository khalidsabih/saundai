#!/usr/bin/env python3
"""Read-only validation-only audit of timestamp delay vs frame grid and audio onsets.

Original annotation timestamps are reference labels. Onset peaks are noisy
audio landmarks, NOT verified beats. An annotation-assisted uniform time-shift
search is an oracle diagnostic, not permission to alter production beat grids.
The exposed Ballroom test split is never loaded. No file modifications other
than an explicitly requested output JSON.
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

import librosa
import numpy as np
from scipy.signal import find_peaks

from scripts.benchmark_beat_integrity_dataset import (
    BEAT_WINDOW_MS, match_events, read_timestamp_labels, valid_times,
)
from scripts.diagnose_beat_error_decomposition import paired_offsets
from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH
from djai.audio import load_audio

ONSET_RADIUS_MS = 110.0
SHIFT_CANDIDATES_MS = tuple(range(-80, 81, 5))


def nearest_landmark_offsets(
    timestamps: list[float], landmarks: list[float], *,
    radius_ms: float = ONSET_RADIUS_MS,
) -> dict[str, Any]:
    """Nearest onset landmark per timestamp, potentially reusing a peak.

    sign=timestamp minus landmark; diagnostics intentionally not a 1:1
    music-event score. Silence/no peaks returns insufficient evidence.
    """
    valid_times(timestamps, allow_empty=True)
    valid_times(landmarks, allow_empty=True)
    if not timestamps or not landmarks:
        return {"status": "no_landmark_evidence", "matched_count": 0}
    peaks = np.asarray(landmarks, dtype=float)
    differences = []
    for t in timestamps:
        at = int(np.searchsorted(peaks, t))
        candidates = [i for i in (at-1, at) if 0 <= i < len(peaks)]
        if candidates:
            i = min(candidates, key=lambda k: abs(t-peaks[k]))
            delta = (t-peaks[i])*1000
            if abs(delta) <= radius_ms:
                differences.append(delta)
    if not differences:
        return {"status": "no_landmark_evidence", "matched_count": 0}
    return {
        "status": "measured_landmark_proximity",
        "matched_count": len(differences),
        "median_signed_timestamp_minus_peak_ms": round(float(np.median(differences)), 3),
        "median_absolute_proximity_ms": round(float(np.median(np.abs(differences))), 3),
        "not_beat_ground_truth": True,
        "peaks_can_be_reused": True,
    }


def frame_grid_diagnosis(predictions: list[float], *, sr: int,
                         hop_length: int) -> dict[str, Any]:
    """Check actual prediction values against the librosa frame-time lattice.

    A lattice match does not tell which point inside an STFT window is
    perceptually the beat, and does not prove a fixed signal delay.
    """
    if sr <= 0 or hop_length <= 0:
        raise ValueError("sample rate and hop must be positive")
    period_ms = 1000*hop_length/sr
    if not predictions:
        return {"status": "no_predictions", "frame_period_ms": round(period_ms, 4)}
    ts = np.asarray(valid_times(predictions), dtype=float)
    frame_index = np.rint(ts*sr/hop_length)
    quantization_ms = np.abs(ts-frame_index*hop_length/sr)*1000
    return {
        "status": "measured",
        "sample_rate": sr,
        "hop_length": hop_length,
        "frame_period_ms": round(period_ms, 4),
        "half_frame_period_ms": round(period_ms/2, 4),
        "median_residual_to_nearest_frame_ms": round(float(np.median(quantization_ms)), 4),
        "maximum_residual_to_nearest_frame_ms": round(float(np.max(quantization_ms)), 4),
        "share_within_1ms_of_lattice": round(float(np.mean(quantization_ms <= 1)), 4),
        "frame_center_or_algorithmic_delay_not_established": True,
    }


def annotation_assisted_shift_sweep(
    reference: list[float], prediction: list[float],
) -> dict[str, Any]:
    """Retrospective reference-assisted offset sweep. NEVER deploy as a fix."""
    ref = valid_times(reference)
    pred = valid_times(prediction, allow_empty=True)
    raw = match_events(ref, pred, tolerance_ms=BEAT_WINDOW_MS)
    if not pred:
        return {"status": "no_predictions", "baseline_f_measure": raw["f_measure"]}
    candidates = []
    for shift in SHIFT_CANDIDATES_MS:
        adjusted = [t+shift/1000 for t in pred if t+shift/1000 >= 0]
        metric = match_events(ref, adjusted, tolerance_ms=BEAT_WINDOW_MS)
        candidates.append((shift, metric["f_measure"], metric["matched"]))
    best = max(candidates, key=lambda x: (x[1], x[2], -abs(x[0])))
    return {
        "status": "reference_assisted_oracle",
        "baseline_f_measure": raw["f_measure"],
        "best_shift_ms": best[0],
        "best_shift_f_measure": best[1],
        "f_measure_delta": round(best[1]-raw["f_measure"], 5),
        "candidate_range_ms": [-80, 80],
        "candidate_step_ms": 5,
        "oracle_uses_ground_truth": True,
        "not_a_production_offset": True,
        "small_f1_gain_does_not_rule_out_systematic_offset": True,
    }


def onset_landmarks(audio_path: Path) -> tuple[list[float], dict[str, Any]]:
    """Construct peak evidence using the SAME sample rate/hop as analysis.

    Recompute the onset envelope from the original waveform without
    changing any cache or saving resampled audio.
    """
    source, sr = load_audio(audio_path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    mono = librosa.to_mono(source)
    envelope = librosa.onset.onset_strength(
        y=mono, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    # A 3-frame local peak spacing avoids counting a single transient as many.
    # Prominence is adaptive to prevent silence/noise from dominating.
    if not len(envelope) or float(np.max(envelope)) <= 1e-8:
        peaks = np.array([], dtype=int)
    else:
        prominence = max(0.01, float(np.percentile(envelope, 75))*0.25)
        peaks, _ = find_peaks(envelope, prominence=prominence, distance=3)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=HOP_LENGTH).tolist()
    return times, {
        "sample_rate": sr,
        "hop_length": HOP_LENGTH,
        "onset_peak_count": len(times),
        "peak_definition": "local maxima in centered STFT-based librosa onset-strength envelope",
        "onsets_not_verified_beats": True,
    }


def analyze_one(item: dict[str, Any], root: Path, *,
                precomputed_only: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {"id": str(item.get("id", "")), "status": "pending"}
    try:
        reference, _ = read_timestamp_labels(root/str(item["beats"]))
        if item.get("prediction"):
            obj = json.loads((root/str(item["prediction"])).read_text(encoding="utf-8"))
            if isinstance(obj.get("analysis"), dict):
                obj = obj["analysis"]
            predicted = valid_times(obj.get("beats_seconds") or [], allow_empty=True)
            row["prediction_source"] = "provided_saved_analysis"
        elif precomputed_only:
            return {**row, "status": "missing_saved_analysis"}
        else:
            from djai.analysis import analyze_track
            predicted = valid_times(
                analyze_track(root/str(item["audio"])).beats_seconds, allow_empty=True,
            )
            row["prediction_source"] = "saund_analyze_track"
        audio_path = root/str(item["audio"])
        peaks, onset_info = onset_landmarks(audio_path)
        row["audio_evidence"] = onset_info
        row["reference_vs_prediction"] = paired_offsets(reference, predicted)
        row["frame_grid"] = frame_grid_diagnosis(
            predicted, sr=onset_info["sample_rate"], hop_length=HOP_LENGTH,
        )
        row["reference_vs_onset_peaks"] = nearest_landmark_offsets(reference, peaks)
        row["prediction_vs_onset_peaks"] = nearest_landmark_offsets(predicted, peaks)
        row["shift_oracle"] = annotation_assisted_shift_sweep(reference, predicted)
        row["status"] = "evaluated"
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def audit(manifest: dict[str, Any], root: Path, *,
          precomputed_only: bool = False) -> dict[str, Any]:
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest needs a tracks list")
    seen = set()
    selection = []
    for item in tracks:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            raise ValueError("unique, nonempty track IDs required")
        seen.add(tid)
        if item.get("split") not in ("train", "validation", "test"):
            raise ValueError("every track requires an explicit split")
        if item["split"] == "validation":
            selection.append(item)
    rows = [analyze_one(t, root, precomputed_only=precomputed_only) for t in selection]
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    signed = [
        r["reference_vs_prediction"]["signed_median_ms"] for r in evaluated
        if r["reference_vs_prediction"]["status"] == "measured"
    ]
    oracle = [
        r["shift_oracle"]["f_measure_delta"] for r in evaluated
        if r["shift_oracle"]["status"] == "reference_assisted_oracle"
    ]
    return {
        "schema": "saund-beat-timing-root-cause-audit-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(selection),
        "evaluated_count": len(evaluated),
        "aggregate": {
            "tracks_with_signed_offset": len(signed),
            "median_track_signed_offset_ms": round(float(np.median(signed)), 3) if signed else None,
            "tracks_with_positive_offset_over_25ms": sum(t > 25 for t in signed),
            "tracks_with_negative_offset_below_minus_25ms": sum(t < -25 for t in signed),
            "median_reference_assisted_shift_f1_gain": round(float(np.median(oracle)), 5) if oracle else None,
            "tracks_with_onset_peak_evidence": sum(
                r["audio_evidence"]["onset_peak_count"] > 0 for r in evaluated
            ),
        },
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "limitations": [
            "Validation only: the previously inspected Ballroom test set is not loaded.",
            "Ground-truth-assisted shift selection is an oracle, NOT a permissible autonomous timing correction.",
            "Onset-envelope peaks may be non-beat transients; proximity is not ground-truth beat accuracy.",
            "Prediction-onset and reference-onset peak searches are nearest-neighbor diagnostics, not causal proof.",
            "A timestamp on the 512/22050 frame lattice does not prove 23ms frame-centering delay.",
            "25ms annotation differences are not the production cross-deck phase-integrity residual.",
            "Ballroom audio and annotations may use conventions that differ from electronic dance music.",
            "No production beat grids, caches, fallbacks or cross-deck phase safety limits were changed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/beat-benchmark/manifest.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--precomputed-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("output/beat-integrity/ballroom-timing-root-cause.json"))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Missing dataset manifest: {args.manifest}")
    result = audit(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.root or args.manifest.parent,
        precomputed_only=args.precomputed_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": result["selected_count"],
        "evaluated": result["evaluated_count"],
        "aggregate": result["aggregate"],
    }, indent=2))
    if not result["evaluated_count"]:
        parser.exit(2, "No validation tracks were evaluated. No results can be claimed.\n")


if __name__ == "__main__":
    main()
