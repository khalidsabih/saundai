#!/usr/bin/env python3
"""Sample-accurate synthetic rhythm timing calibration: READ-ONLY research.

Generate original deterministic click-kicks in temporary WAV files with true
onset times defined at integer sample positions. Compare SAUND's unchanged
analysis timestamps and librosa onset-strength peaks with those true times.
Study frame-phase dependency without learning or applying a global correction.
No data is obtained from the already-exposed Ballroom test partition.

This is a controlled *timing diagnostic*, not proof of EDM beat recognition,
cross-deck residual safety, human mix quality, or production readiness.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH, analyze_track
from djai.audio import load_audio

DEFAULT_BPMS = (100.0, 124.0, 140.0)
DEFAULT_PHASE_FRACTIONS = (0.0, 1/3, 2/3)


def synthesize_click_kicks(*, bpm: float, frame_phase: float, sample_rate: int,
                           hop_length: int, beats: int = 28) -> tuple[np.ndarray, list[float], list[int]]:
    """Create known-time drum attacks; labels are exact integer sample locations.

    A click plus decaying low tone provides clearly defined instantaneous
    physical attacks and low-frequency content. Nothing is jittered randomly.
    """
    if not 50 <= bpm <= 220 or not 0 <= frame_phase < 1:
        raise ValueError("invalid tempo/frame phase")
    if sample_rate < 8000 or hop_length < 1 or beats < 12:
        raise ValueError("invalid sample rate/hop/beat count")
    start_sample = sample_rate + round(frame_phase*hop_length)
    sample_indices = [start_sample + round(i*60*sample_rate/bpm) for i in range(beats)]
    length = sample_indices[-1] + sample_rate
    wave = np.zeros(length, dtype=np.float32)
    tick_length = min(round(sample_rate*.11), sample_rate//2)
    t = np.arange(tick_length, dtype=np.float64)/sample_rate
    # Explicit nonzero sample at t=0; transient duration is much shorter
    # than the interbeat interval at supported BPMs.
    kick = (0.80*np.exp(-t*165.0) +
            0.25*np.exp(-t*34.0)*np.sin(2*np.pi*83*t)).astype(np.float32)
    for index in sample_indices:
        wave[index:index+tick_length] += kick
    return wave, [i/sample_rate for i in sample_indices], sample_indices


def align_nearest(reference: list[float], observed: list[float],
                  *, radius_ms: float = 130.0) -> dict[str, Any]:
    """Monotone one-to-one alignment within a local window.

    Sign convention: observed minus true physical onset. A match is a
    timing diagnostic; strict 130ms window is not the production 25ms gate.
    """
    if radius_ms <= 0:
        raise ValueError("radius must be positive")
    if not reference or not observed:
        return {"status": "no_pairs", "matched_count": 0}
    reference = [float(t) for t in reference]
    observed = [float(t) for t in observed]
    if any(not math.isfinite(t) or t < 0 for t in reference+observed):
        raise ValueError("times must be nonnegative finite")
    if any(b <= a for times in (reference, observed) for a, b in zip(times, times[1:])):
        raise ValueError("times must be increasing")
    pairs = []
    i = j = 0
    limit = radius_ms/1000
    while i < len(reference) and j < len(observed):
        delta = observed[j]-reference[i]
        if abs(delta) <= limit:
            pairs.append(delta*1000)
            i += 1
            j += 1
        elif delta < -limit:
            j += 1
        else:
            i += 1
    if not pairs:
        return {"status": "no_pairs", "matched_count": 0}
    values = np.asarray(pairs, dtype=float)
    return {
        "status": "measured",
        "matched_count": len(pairs),
        "median_signed_offset_ms": round(float(np.median(values)), 3),
        "median_absolute_error_ms": round(float(np.median(np.abs(values))), 3),
        "p95_absolute_error_ms": round(float(np.percentile(np.abs(values), 95)), 3),
        "matched_over_25ms": int(np.count_nonzero(np.abs(values) > 25.0)),
        "observed_late_count": int(np.count_nonzero(values > 25.0)),
        "observed_early_count": int(np.count_nonzero(values < -25.0)),
    }


def extract_onset_peak_times(path: Path) -> list[float]:
    source, sr = load_audio(path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    mono = librosa.to_mono(source)
    envelope = librosa.onset.onset_strength(
        y=mono, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    if not len(envelope) or float(np.max(envelope)) < 1e-8:
        return []
    prominence = max(0.01, float(np.percentile(envelope, 75))*.25)
    frames, _ = find_peaks(envelope, prominence=prominence, distance=3)
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP_LENGTH).tolist()


def evaluate_case(*, bpm: float, frame_phase: float, workdir: Path,
                  sample_rate: int = ANALYSIS_SAMPLE_RATE,
                  hop_length: int = HOP_LENGTH) -> dict[str, Any]:
    audio, actual_times, sample_indices = synthesize_click_kicks(
        bpm=bpm, frame_phase=frame_phase, sample_rate=sample_rate,
        hop_length=hop_length,
    )
    path = workdir / f"synthetic_{int(bpm)}_{round(frame_phase*1000)}.wav"
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    row: dict[str, Any] = {
        "bpm": bpm, "frame_phase_fraction": round(frame_phase, 6),
        "sample_rate": sample_rate, "onset_count": len(actual_times),
        "first_onset_sample": sample_indices[0],
        "first_onset_seconds": round(actual_times[0], 9),
        "physical_onset_definition": "first nonzero kick sample in original synthetic WAV",
        "status": "pending",
    }
    try:
        onsets = extract_onset_peak_times(path)
        row["onset_peak_count"] = len(onsets)
        row["onset_peaks_vs_true"] = align_nearest(actual_times, onsets)
        result = analyze_track(path)
        predicted = [float(t) for t in result.beats_seconds]
        row["prediction_count"] = len(predicted)
        row["predicted_bpm"] = round(float(result.bpm), 3)
        row["predictions_vs_true"] = align_nearest(actual_times, predicted)
        row["predicted_beat_interval_ratio"] = (
            round(float(np.median(np.diff(predicted)))/(60/bpm), 4)
            if len(predicted) >= 2 else None
        )
        row["status"] = "evaluated"
    except Exception as exc:
        # Preserve individual-case failures in report instead of pretending
        # that failed analysis is a successful calibration.
        row["status"] = "analysis_failed"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        path.unlink(missing_ok=True)
    return row


def run(*, bpms: tuple[float, ...] = DEFAULT_BPMS,
        phases: tuple[float, ...] = DEFAULT_PHASE_FRACTIONS) -> dict[str, Any]:
    if not bpms or not phases:
        raise ValueError("at least one BPM and frame phase required")
    rows = []
    with tempfile.TemporaryDirectory(prefix="saund-timing-cal-") as temp:
        directory = Path(temp)
        for bpm in bpms:
            for phase in phases:
                rows.append(evaluate_case(bpm=bpm, frame_phase=phase, workdir=directory))
    good = [row for row in rows if row["status"] == "evaluated"]
    result = [
        row["predictions_vs_true"]["median_signed_offset_ms"] for row in good
        if row["predictions_vs_true"]["status"] == "measured"
    ]
    onset = [
        row["onset_peaks_vs_true"]["median_signed_offset_ms"] for row in good
        if row["onset_peaks_vs_true"]["status"] == "measured"
    ]
    return {
        "schema": "saund-synthetic-sample-accurate-timing-v1",
        "dataset": "original synthetic click-kicks (no third-party audio)",
        "scenarios": len(rows), "evaluated_scenarios": len(good),
        "calibration_sample_rate": ANALYSIS_SAMPLE_RATE,
        "analysis_hop_length": HOP_LENGTH,
        "aggregate": {
            "scenarios_with_matched_predictions": len(result),
            "median_scenario_prediction_offset_ms": (
                round(float(np.median(result)), 3) if result else None
            ),
            "scenarios_with_matched_onset_peaks": len(onset),
            "median_scenario_onset_peak_offset_ms": (
                round(float(np.median(onset)), 3) if onset else None
            ),
        },
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "global_offset_correction_authorized": False,
        "limitations": [
            "Synthetic click-kicks are not a substitute for varied EDM or Ballroom tracks.",
            "An onset peak can be late relative to a waveform attack due to analysis-windowing.",
            "130ms temporal alignment is diagnostic, NOT the production 25ms cross-deck residual.",
            "Missing predicted beats or pulse octave disagreements are explicit failures/uncertainties.",
            "No production beat grid, model authority, cached beat grid, fallback or phase limit is modified.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("output/beat-integrity/synthetic-timing-calibration.json"))
    parser.add_argument("--quick", action="store_true",
                        help="Two synthetic scenarios for a fast smoke check")
    args = parser.parse_args()
    kwargs = {"bpms": (100.0,), "phases": (0.0, 2/3)} if args.quick else {}
    report = run(**kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "scenarios": report["scenarios"],
        "evaluated": report["evaluated_scenarios"],
        "aggregate": report["aggregate"],
    }, indent=2))
    if report["evaluated_scenarios"] != report["scenarios"]:
        parser.exit(2, "One or more synthetic scenarios failed; inspect the report.\n")


if __name__ == "__main__":
    main()
