#!/usr/bin/env python3
"""Read-only Beat This! 1.1.0 raw-logit -> minimal decoder -> TSV trace.

Requires model-inferred *.npy activations and companion *.beats from the
official "beat_this --activations" CLI, plus pre-existing original .beats.
Does not load the model, install dependencies, alter predictions, infer meter,
or authorize any transition.

Upstream v1.1.0 postprocessor.py minimal mode:
 - independent +/-3-frame maxima on beat and downbeat logits (50 fps)
 - logit > 0 and merge adjacent peaks <=1 frame apart
 - snap each downbeat peak to nearest beat, unique times
Upstream utils.py converts downbeats to beat-number-1 labels.

A post-hoc known 32-bar/4/4 synthetic schedule is used ONLY to categorize
exported extra downbeat labels. It never influences re-decoding or inference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)
FPS = 50
EXPORT_TOLERANCE_SECONDS = .011
EVAL_TOLERANCE_SECONDS = .08


def load_parser():
    path = ROOT / "scripts" / "m1-beat-this-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_bt_raw_trace_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("M1 TSV parser not available")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deduplicate_frame_peaks(indices):
    """Average consecutive peak frame indices, matching upstream v1.1.0."""
    if len(indices) == 0:
        return np.array([], dtype=np.float64)
    merged = []
    current = float(indices[0])
    count = 1
    for index in indices[1:]:
        index = float(index)
        if index - current <= 1.:
            count += 1
            current += (index - current) / count
        else:
            merged.append(current)
            current = index
            count = 1
    merged.append(current)
    return np.asarray(merged, dtype=np.float64)


def minimal_peaks(logits):
    """Recreate upstream strict equality to seven-frame maxpool and logit>0.

    Uses -infinity padding, matching torch maxpool's boundary handling.
    All-frame equal maxima can become multiple peaks, subsequently deduped.
    """
    signal = np.asarray(logits, dtype=np.float64)
    if signal.ndim != 1 or not len(signal) or not np.all(np.isfinite(signal)):
        raise ValueError("invalid raw logits")
    padded = np.pad(signal, (3, 3), mode="constant", constant_values=-np.inf)
    neighborhood = np.maximum.reduce([padded[i:i + len(signal)] for i in range(7)])
    frames = np.flatnonzero((signal == neighborhood) & (signal > 0.))
    return _deduplicate_frame_peaks(frames) / FPS


def decode_minimal(activations):
    """No source BPM, downbeat reference, or annotation enters this decoder."""
    x = np.asarray(activations)
    if x.ndim != 2 or x.shape[0] != 2 or x.shape[1] < FPS:
        raise ValueError("activation array must be 2 x frames, >=1 second")
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("nonfinite or nonnumeric activations")
    beats = minimal_peaks(x[0])
    original_downbeat_peaks = minimal_peaks(x[1])
    if len(beats):
        snapped = [
            float(beats[int(np.argmin(np.abs(beats - time)))])
            for time in original_downbeat_peaks
        ]
        downs = np.unique(np.asarray(snapped, dtype=np.float64))
    else:
        downs = np.asarray(original_downbeat_peaks, dtype=np.float64)
    return {
        "beats": beats.tolist(),
        "raw_downbeat_peak_seconds": original_downbeat_peaks.tolist(),
        "downbeats": downs.tolist(),
    }


def pair_times(a, b, *, tolerance=EXPORT_TOLERANCE_SECONDS):
    """One-to-one approximate comparison, without rewriting either sequence."""
    if tolerance <= 0:
        raise ValueError("invalid pairing tolerance")
    candidates = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if abs(float(x) - float(y)) <= tolerance:
                candidates.append((abs(float(x) - float(y)), i, j))
    used_a, used_b, matched = set(), set(), []
    for error, i, j in sorted(candidates):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matched.append(error)
    return {
        "a_count": len(a),
        "b_count": len(b),
        "matched_count": len(matched),
        "unmatched_a_count": len(a) - len(matched),
        "unmatched_b_count": len(b) - len(matched),
        "max_matched_difference_ms": round(max(matched) * 1000., 3) if matched else None,
        "exact_sequence_within_tolerance": (
            len(a) == len(b) == len(matched)
        ),
    }


def downbeat_classification(predicted, expected):
    """Post-hoc only: checks generated bar starts, not musical meter inference."""
    candidates = []
    for t in predicted:
        nearest = min(expected, key=lambda e: abs(e - t))
        candidates.append((t, nearest, abs(t - nearest)))
    return {
        "predicted_count": len(predicted),
        "reference_bar_count": len(expected),
        "near_reference_count": sum(error <= EVAL_TOLERANCE_SECONDS
                                    for _, _, error in candidates),
        "extra_label_times_seconds": [
            round(t, 6) for t, _, error in candidates
            if error > EVAL_TOLERANCE_SECONDS
        ],
        "reference_used_only_posthoc": True,
    }


def evaluate_track(name, original_root, activation_root, *, tsv_parser):
    orig = tsv_parser.read_beat_tsv(original_root / f"{Path(name).stem}.beats")
    new = tsv_parser.read_beat_tsv(activation_root / f"{Path(name).stem}.beats")
    npy = activation_root / f"{Path(name).stem}.npy"
    if not npy.is_file():
        raise ValueError(f"missing raw logit activations: {npy}")
    activations = np.load(npy, allow_pickle=False)
    decoded = decode_minimal(activations)
    generated_bpm = dict(FIXTURES)[name]
    expected_bars = [
        4 * i * 60. / generated_bpm for i in range(32)
    ]
    new_beats = pair_times(decoded["beats"], new["beats"])
    new_downs = pair_times(decoded["downbeats"], new["downbeats"])
    previous_beats = pair_times(orig["beats"], new["beats"])
    previous_downs = pair_times(orig["downbeats"], new["downbeats"])
    return {
        "file": name,
        "raw_frame_count": int(activations.shape[1]),
        "raw_frame_rate_fps": FPS,
        "raw_downbeat_logit_positive_frame_count": int(np.count_nonzero(activations[1] > 0)),
        "minimal_decoder_beat_count": len(decoded["beats"]),
        "minimal_decoder_downbeat_peak_before_snapping_count": len(decoded["raw_downbeat_peak_seconds"]),
        "minimal_decoder_downbeat_count": len(decoded["downbeats"]),
        "export_from_same_raw_activations": {
            "beats": new_beats,
            "downbeats": new_downs,
            "all_events_agree": (new_beats["exact_sequence_within_tolerance"]
                                  and new_downs["exact_sequence_within_tolerance"]),
        },
        "original_export_vs_activation_rerun": {
            "beats": previous_beats,
            "downbeats": previous_downs,
            "all_events_agree": (
                previous_beats["exact_sequence_within_tolerance"]
                and previous_downs["exact_sequence_within_tolerance"]
            ),
        },
        "posthoc_generator_reference": {
            "original_export": downbeat_classification(orig["downbeats"], expected_bars),
            "activation_rerun_export": downbeat_classification(new["downbeats"], expected_bars),
            "independently_redecoded_logits": downbeat_classification(decoded["downbeats"], expected_bars),
        },
        "classification_guidance": (
            "If redecoded logits and activation-run export agree, extra downbeats "
            "are already present in the independent minimal postprocessor's "
            "downbeat peak extraction, not created by the beat-number TSV writer. "
            "This does not establish whether the underlying raw logits or "
            "the postprocessor threshold caused unwanted musical labels."
        ),
    }


def run(original_root, activation_root):
    parser = load_parser()
    rows = [
        evaluate_track(name, original_root, activation_root, tsv_parser=parser)
        for name, _ in FIXTURES
    ]
    return {
        "schema": "saund-m1-beat-this-downbeat-origin-trace-v1",
        "tracks": rows,
        "upstream_release_inspected": "CPJKU/beat_this v1.1.0",
        "original_tsv_preserved": True,
        "activation_rerun_tsv_preserved": True,
        "reference_used_in_decoder": False,
        "raw_logits_generated_outside_this_script": True,
        "production_changed": False,
        "beat_grid_modified": False,
        "model_promoted": False,
        "transition_rendered": False,
        "selected_configuration": None,
        "limitations": [
            "Only two known 4/4 generated demos, not real-music validation.",
            "Independent NumPy recreation of minimal decoding should be checked against upstream output.",
            "No model inference or parameter tuning in this script.",
            "Raw downbeat score can be elevated on non-bar beats: identifying the causal reason requires further independent audio evaluation.",
            "No results authorize 4/4 forcing, production grids, or rendering."
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-predictions-root", type=Path, required=True)
    p.add_argument("--activation-predictions-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite report")
    report = run(args.original_predictions_root, args.activation_predictions_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"M1 downbeat origin trace failed: {exc}") from exc
