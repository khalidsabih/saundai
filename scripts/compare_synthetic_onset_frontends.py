#!/usr/bin/env python3
"""Offline onset-front-end comparison on sample-accurate synthetic kicks.

Compare FFT window sizes, hop lengths and centering conventions on identical
original synthetic WAV samples. Scores are measured against *known physical
attack samples*. The default configuration calls the same librosa onset-strength
and beat-track routines as SAUND's deterministic front end, without modifying
SAUND analysis, its model registry, beat-grid caches, or safety thresholds.

CAUTION: center=False frames are timestamped by their WINDOW START, so their
apparent offset may improve artificially. Treat only center=True settings as
directly comparable timing alternatives. No global offset is authorized.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH
from scripts.calibrate_synthetic_beat_timing import (
    DEFAULT_BPMS, DEFAULT_PHASE_FRACTIONS, align_nearest, synthesize_click_kicks,
)


@dataclass(frozen=True)
class OnsetVariant:
    name: str
    n_fft: int
    hop: int
    center: bool = True


VARIANTS = (
    OnsetVariant("baseline_centered_fft2048_hop512", 2048, 512, True),
    OnsetVariant("centered_fft1024_hop512", 1024, 512, True),
    OnsetVariant("centered_fft4096_hop512", 4096, 512, True),
    OnsetVariant("centered_fft2048_hop256", 2048, 256, True),
    OnsetVariant("uncentered_fft2048_hop512_diagnostic", 2048, 512, False),
)


def describe_variant(variant: OnsetVariant) -> dict[str, Any]:
    if variant.n_fft < 256 or variant.hop < 1 or variant.n_fft % 2:
        raise ValueError("invalid transform dimensions")
    return {
        "name": variant.name,
        "n_fft": variant.n_fft,
        "hop_length": variant.hop,
        "center": variant.center,
        "frame_interval_ms": round(1000*variant.hop/ANALYSIS_SAMPLE_RATE, 4),
        "window_duration_ms": round(1000*variant.n_fft/ANALYSIS_SAMPLE_RATE, 4),
        "timestamps": ("librosa.frames_to_time frame grid; centered STFT input"
                       if variant.center else
                       "librosa.frames_to_time anchored to uncentered STFT window START"),
        "directly_comparable_to_centered_baseline": variant.center,
    }


def onset_and_beats(y: np.ndarray, sr: int, variant: OnsetVariant
                    ) -> tuple[list[float], list[float], float]:
    # Explicitly pass n_fft: uses the default mel/onset pipeline for the
    # baseline, and changes one transform parameter per experimental variant.
    env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=variant.hop, n_fft=variant.n_fft,
        center=variant.center, aggregate=np.median,
    )
    if len(env) == 0 or float(np.max(env)) <= 1e-8:
        return [], [], 0.0
    threshold = max(0.01, float(np.percentile(env, 75))*.25)
    peaks, _ = find_peaks(env, prominence=threshold, distance=3)
    peak_times = librosa.frames_to_time(
        peaks, sr=sr, hop_length=variant.hop,
    ).tolist()
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=variant.hop,
        units="frames", sparse=True,
    )
    beat_times = librosa.frames_to_time(
        frames, sr=sr, hop_length=variant.hop,
    ).tolist()
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return peak_times, beat_times, bpm


def evaluate_configuration(y: np.ndarray, reference: list[float],
                           variant: OnsetVariant, bpm: float) -> dict[str, Any]:
    row = describe_variant(variant)
    try:
        peaks, beats, predicted_bpm = onset_and_beats(
            y, ANALYSIS_SAMPLE_RATE, variant,
        )
        row.update({
            "status": "evaluated",
            "onset_peak_count": len(peaks),
            "prediction_count": len(beats),
            "predicted_bpm": round(predicted_bpm, 3),
            "peak_timing": align_nearest(reference, peaks),
            "beat_timing": align_nearest(reference, beats),
            "beat_recall_130ms": round(
                align_nearest(reference, beats)["matched_count"]/len(reference), 4
            ),
            "beat_precision_130ms": round(
                align_nearest(reference, beats)["matched_count"]/len(beats), 4
            ) if beats else 0.0,
            "predicted_beat_interval_ratio": (
                round(float(np.median(np.diff(beats)))/(60/bpm), 4)
                if len(beats) > 1 else None
            ),
        })
    except Exception as exc:
        row.update({"status": "analysis_failed",
                    "reason": f"{type(exc).__name__}: {exc}"})
    return row


def evaluate_scenario(*, bpm: float, frame_phase: float,
                      variants: tuple[OnsetVariant, ...] = VARIANTS
                      ) -> dict[str, Any]:
    y, true_times, samples = synthesize_click_kicks(
        bpm=bpm, frame_phase=frame_phase, sample_rate=ANALYSIS_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    rows = [evaluate_configuration(y, true_times, v, bpm) for v in variants]
    return {
        "bpm": bpm,
        "frame_phase_fraction": round(frame_phase, 6),
        "first_onset_sample": samples[0],
        "true_beats": len(true_times),
        "variants": rows,
    }


def summarize(scenarios: list[dict[str, Any]],
              variants: tuple[OnsetVariant, ...]) -> list[dict[str, Any]]:
    summary = []
    for variant in variants:
        rows = [
            row for sc in scenarios for row in sc["variants"]
            if row["name"] == variant.name and row["status"] == "evaluated"
        ]
        measured = [r for r in rows if r["beat_timing"]["status"] == "measured"]
        peak = [r for r in rows if r["peak_timing"]["status"] == "measured"]
        matched_beats = sum(r["beat_timing"]["matched_count"] for r in rows)
        true_beats = sum(s["true_beats"] for s in scenarios)
        predicted_beats = sum(r["prediction_count"] for r in rows)
        summary.append({
            **describe_variant(variant),
            "evaluated_scenarios": len(rows),
            "measured_beat_scenarios": len(measured),
            "measured_onset_scenarios": len(peak),
            "total_matched_beats": matched_beats,
            "total_true_beats": true_beats,
            "total_predicted_beats": predicted_beats,
            "aggregate_beat_recall_130ms": round(matched_beats/true_beats, 4)
                if true_beats else None,
            "aggregate_beat_precision_130ms": round(
                matched_beats/predicted_beats, 4
            ) if predicted_beats else None,
            "median_scenario_beat_offset_ms": round(float(np.median([
                r["beat_timing"]["median_signed_offset_ms"] for r in measured
            ])), 3) if measured else None,
            "median_scenario_beat_absolute_error_ms": round(float(np.median([
                r["beat_timing"]["median_absolute_error_ms"] for r in measured
            ])), 3) if measured else None,
            "median_scenario_peak_offset_ms": round(float(np.median([
                r["peak_timing"]["median_signed_offset_ms"] for r in peak
            ])), 3) if peak else None,
            "median_scenario_predicted_interval_ratio": round(float(np.median([
                r["predicted_beat_interval_ratio"] for r in rows
                if r["predicted_beat_interval_ratio"] is not None
            ])), 4) if any(
                r["predicted_beat_interval_ratio"] is not None for r in rows
            ) else None,
        })
    return summary


def run(*, bpms: tuple[float, ...] = DEFAULT_BPMS,
        phases: tuple[float, ...] = DEFAULT_PHASE_FRACTIONS,
        variants: tuple[OnsetVariant, ...] = VARIANTS) -> dict[str, Any]:
    if not bpms or not phases or not variants:
        raise ValueError("at least one BPM, phase and variant are required")
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    for v in variants:
        describe_variant(v)
    scenarios = [
        evaluate_scenario(bpm=bpm, frame_phase=phase, variants=variants)
        for bpm in bpms for phase in phases
    ]
    summaries = summarize(scenarios, variants)
    return {
        "schema": "saund-onset-front-end-comparison-v1",
        "dataset": "original deterministic synthetic clicks; exact integer sample onsets",
        "total_scenarios": len(scenarios),
        "variants_per_scenario": len(variants),
        "evaluated_variant_scenarios": sum(
            v["evaluated_scenarios"] for v in summaries
        ),
        "summary": summaries,
        "scenarios": scenarios,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "global_offset_correction_authorized": False,
        "limitations": [
            "center=False removes librosa onset-envelope frame shifting; it does NOT disable mel-STFT window centering.",
            "Best synthetic timing error alone cannot authorize a production algorithm change.",
            "130ms one-to-one matching window is diagnostic, NOT the production 25ms cross-deck limit.",
            "Synthetic click-kicks do not validate performance on real EDM or phrase/downbeat accuracy.",
            "This script calls deterministic librosa analysis, not neural model-authority routing.",
            "Production beat grids, caches, fallback, safety limits and live execution remain unchanged.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Compare two frame phases at 100 BPM")
    parser.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/onset-front-end-comparison.json",
    ))
    args = parser.parse_args()
    options = {"bpms": (100.0,), "phases": (0.0, 2/3)} if args.quick else {}
    report = run(**options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "scenarios": report["total_scenarios"],
        "variants": report["variants_per_scenario"],
        "evaluated_variant_scenarios": report["evaluated_variant_scenarios"],
        "summary": report["summary"],
    }, indent=2))
    expected = report["total_scenarios"]*report["variants_per_scenario"]
    if report["evaluated_variant_scenarios"] != expected:
        parser.exit(2, "Some configurations failed; see report for details.\n")


if __name__ == "__main__":
    main()
