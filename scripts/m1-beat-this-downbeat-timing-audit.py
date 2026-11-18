#!/usr/bin/env python3
"""M1 shadow external Beat This! bar/downbeat and timing-error validation.

Reads independently predicted Beat This! .beats files, then evaluates only
against the known four-beat-per-bar schedule of SAUND-generated demo WAVs.
The schedule enters POST-HOC evaluation, never model inference, beat choice,
phase correction, or transition planning. No model is installed/downloaded.

Synthetic bars cannot validate musical meter or downbeats on real music.
Report both 80ms research event score and STRICTER 25ms *descriptive*
residuals; the latter is NOT a passed planner phase-integrity gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_BARS = 32
BEATS_PER_BAR = 4
FIXTURES = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)
TOLERANCE_MS = 80.0
STRICT_RESIDUAL_MS = 25.0
# Predeclared source-time analysis windows; not boundaries optimized on data.
SEGMENT_SECONDS = 8.0
OUTGOING_REGION = ("between_compressions", 9.636281, 58.305306)


def module_from_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"required shadow benchmark script unavailable: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_reference(bpm, *, bars=SYNTHETIC_BARS, meter=BEATS_PER_BAR):
    """Source generator schedule from demo.py; scoring only, never inference."""
    if not np.isfinite(bpm) or bpm <= 0 or bars < 1 or meter < 1:
        raise ValueError("invalid known generator reference")
    period = 60.0 / float(bpm)
    beats = [i * period for i in range(bars * meter)]
    downbeats = [i * meter * period for i in range(bars)]
    return beats, downbeats


def unique_nearest_errors(predictions, references, *, tolerance_ms=TOLERANCE_MS):
    """One-to-one timestamp pairing; no phase shifting or beat-grid rewrite."""
    if not (0 < tolerance_ms <= 500):
        raise ValueError("invalid scoring tolerance")
    pred = [float(x) for x in predictions]
    ref = [float(x) for x in references]
    for arr in (pred, ref):
        if any(not math.isfinite(t) or t < 0 for t in arr):
            raise ValueError("invalid timestamp")
        if any(b <= a for a, b in zip(arr, arr[1:])):
            raise ValueError("events must be strictly increasing")
    candidates = []
    for i, p in enumerate(pred):
        for j, t in enumerate(ref):
            delta = 1000. * (p - t)
            if abs(delta) <= tolerance_ms:
                candidates.append((abs(delta), i, j, delta))
    matches = []
    used_pred = set()
    used_ref = set()
    for _, i, j, error in sorted(candidates):
        if i in used_pred or j in used_ref:
            continue
        used_pred.add(i)
        used_ref.add(j)
        matches.append((i, j, error))
    return sorted(matches)


def score_events(predictions, references, *, tolerance_ms=TOLERANCE_MS):
    matched = unique_nearest_errors(predictions, references, tolerance_ms=tolerance_ms)
    errors = [row[2] for row in matched]
    abs_errors = [abs(x) for x in errors]
    precision = len(matched) / len(predictions) if predictions else None
    recall = len(matched) / len(references) if references else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall
          else None)
    return {
        "predicted_count": len(predictions),
        "reference_count": len(references),
        "matched_count": len(matched),
        "missed_reference_count": len(references) - len(matched),
        "unmatched_prediction_count": len(predictions) - len(matched),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else 0.0 if predictions and references else None,
        "median_absolute_error_ms": round(float(np.median(abs_errors)), 3) if errors else None,
        "p95_absolute_error_ms": round(float(np.percentile(abs_errors, 95)), 3) if errors else None,
        "maximum_absolute_error_ms": round(max(abs_errors), 3) if errors else None,
        "matched_outside_25ms": sum(x > STRICT_RESIDUAL_MS for x in abs_errors),
        "reference_without_25ms_match": len(references) - sum(
            abs(x) <= STRICT_RESIDUAL_MS for x in abs_errors
        ),
        "research_tolerance_ms": tolerance_ms,
        "strict_residual_threshold_ms": STRICT_RESIDUAL_MS,
        "strict_residual_is_not_planner_gate": True,
        "first_unmatched_prediction_seconds": [
            round(float(predictions[i]), 6)
            for i in range(len(predictions)) if i not in {m[0] for m in matched}
        ][:10],
        "first_missed_reference_seconds": [
            round(float(references[j]), 6)
            for j in range(len(references)) if j not in {m[1] for m in matched}
        ][:10],
    }


def segmented_scores(predictions, references, duration):
    scores = []
    step_count = math.ceil(duration / SEGMENT_SECONDS)
    for index in range(step_count):
        start = index * SEGMENT_SECONDS
        end = min(duration, (index + 1) * SEGMENT_SECONDS)
        pred = [t for t in predictions if start <= t < end]
        ref = [t for t in references if start <= t < end]
        scores.append({
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            **score_events(pred, ref),
        })
    return scores


def bar_label_diagnostics(beats, numbers, reference_beats, predicted_downbeats):
    """Inspect 4/4 consistency without assuming 1 labels are correct."""
    if len(numbers) != len(beats):
        raise ValueError("beat-number labels inconsistent")
    increments = [int(b) for b in numbers]
    if any(n < 1 for n in increments):
        raise ValueError("invalid beat number label")
    downbeat_positions = [idx for idx, number in enumerate(numbers) if number == 1]
    intervals = [j - i for i, j in zip(downbeat_positions, downbeat_positions[1:])]
    interval_hist = {str(v): intervals.count(v) for v in sorted(set(intervals))}
    interval_s = [b - a for a, b in zip(predicted_downbeats, predicted_downbeats[1:])]
    return {
        "predicted_label_1_count": len(downbeat_positions),
        "reference_four_four_bar_count": len(reference_beats) // BEATS_PER_BAR,
        "predicted_beat_index_interval_between_label_1_histogram": interval_hist,
        "predicted_downbeat_seconds_interval_median": (
            round(statistics.median(interval_s), 5) if interval_s else None
        ),
        "non_four_beat_label_intervals": sum(value != BEATS_PER_BAR for value in intervals),
        "generator_four_four_reference_is_scoring_only": True,
    }


def evaluate_track(wav, tsv, *, parser):
    if wav.name not in dict(FIXTURES):
        raise ValueError("not a benchmark demo fixture")
    if not wav.is_file():
        raise ValueError(f"missing generated WAV: {wav}")
    info = sf.info(str(wav))
    duration = info.frames / info.samplerate
    if not np.isfinite(duration) or duration <= 1:
        raise ValueError("invalid WAV duration")
    output = parser.read_beat_tsv(tsv)
    beats = output["beats"]
    downbeats = output["downbeats"]
    if beats[-1] > duration + .1:
        raise ValueError("prediction outside WAV bounds")
    reference_beats, reference_downbeats = synthetic_reference(dict(FIXTURES)[wav.name])
    # Source generator generates 32 x 4 kick beats and one bar start per bar.
    if abs(duration - (reference_beats[-1] + 60. / dict(FIXTURES)[wav.name])) > .12:
        raise ValueError("demo waveform duration differs from known 32-bar generator")
    overall = score_events(beats, reference_beats)
    downbeat = score_events(downbeats, reference_downbeats)
    fail_region = None
    if wav.name == FIXTURES[0][0]:
        _, start, end = OUTGOING_REGION
        fail_region = {
            "beats": score_events(
                [x for x in beats if start <= x < end],
                [x for x in reference_beats if start <= x < end],
            ),
            "downbeats": score_events(
                [x for x in downbeats if start <= x < end],
                [x for x in reference_downbeats if start <= x < end],
            ),
        }
    return {
        "file": wav.name,
        "duration_seconds": round(duration, 4),
        "beat_timing_to_generated_pulse_schedule": overall,
        "downbeat_timing_to_generated_bar_starts": downbeat,
        "beat_timing_by_fixed_8second_regions": segmented_scores(
            beats, reference_beats, duration
        ),
        "label_consistency": bar_label_diagnostics(
            beats, output["beat_numbers"], reference_beats, downbeats
        ),
        "outgoing_failure_region": fail_region,
        "generated_reference_beat_count": len(reference_beats),
        "generated_reference_downbeat_count": len(reference_downbeats),
    }


def run(audio_root, predictions_root):
    parser = module_from_script("m1-beat-this-benchmark.py", "m1_downbeat_parser")
    tracks = []
    for filename, _ in FIXTURES:
        wav = audio_root.expanduser().resolve() / filename
        tsv = predictions_root.expanduser().resolve() / f"{Path(filename).stem}.beats"
        tracks.append(evaluate_track(wav, tsv, parser=parser))
    return {
        "schema": "saund-m1-external-beat-this-bar-and-timing-audit-v1",
        "tracks": tracks,
        "prediction_model": "Beat This! final0 external inference; TSV only",
        "reference_generator": "src/djai/demo.py:generate_demo_track",
        "generator_reference_used_only_in_posthoc_scoring": True,
        "beat_this_rerun": False,
        "model_download": False,
        "ground_truth_downbeat_labels_available_for_generated_demos_only": True,
        "real_music_validation_completed": False,
        "downbeat_phase_certified": False,
        "planner_25ms_gate_tested": False,
        "sealed_ballroom_test_touched": False,
        "beat_grid_modified": False,
        "production_changed": False,
        "transition_rendered": False,
        "selected_configuration": None,
        "warning": (
            "4/4 generator bar starts are known post-hoc scoring references. "
            "A 25ms timestamp-error count is NOT the planner's continuous "
            "25ms phase-integrity gate. No model promotion/render authorization."
        ),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-root", required=True, type=Path)
    p.add_argument("--predictions-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    result = run(args.audio_root, args.predictions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 Beat This bar/timing audit failed: {exc}") from exc
