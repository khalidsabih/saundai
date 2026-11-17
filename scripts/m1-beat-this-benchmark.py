#!/usr/bin/env python3
"""M1 offline Beat This! external prediction benchmark (read-only, fail-closed).

Install/run the third-party model separately using its official CLI. This
script ONLY reads its .beats TSV files, the two known generated demo WAVs,
and optionally SAUND's *existing* beat analysis. Synthetic known BPM is used
solely to locate audible kick attacks AFTER predictions have been produced.
No neural promotion, model download, production grid edit, plan or render.

Usage: python scripts/m1-beat-this-benchmark.py
  --audio-root "$M1_AUDIO_ROOT" --predictions-root /tmp/saund-m1-beat-this-01
  --output /tmp/saund-m1-beat-this-comparison-01.json [--compare-saund]

Upstream .beats format: TIME_SECONDS TAB BEAT_NUMBER; downbeat number = 1.
The benchmark never treats matching demo kicks as proof of correct musical
downbeats or as general performance on commercially released songs.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)
MIDDLE_REGION = ("between_compressions", 9.636281, 58.305306)
EVAL_TOLERANCE_MS = 80.0


def load_kick_audit():
    path = ROOT / "scripts" / "m1-synthetic-audio-kick-audit.py"
    spec = importlib.util.spec_from_file_location("m1_beat_this_kick_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("existing synthetic kick audit unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_beat_tsv(path: Path):
    """Parse Beat This! standard TSV with strict monotonicity and numbers."""
    if not path.is_file():
        raise ValueError(f"missing model prediction: {path}")
    beats = []
    downbeats = []
    beat_numbers = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid beat TSV at line {line_number}: two columns required")
        try:
            timestamp = float(fields[0])
            number = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"invalid beat TSV at line {line_number}") from exc
        if not math.isfinite(timestamp) or timestamp < 0 or number < 1:
            raise ValueError(f"invalid beat time or number at line {line_number}")
        if beats and timestamp <= beats[-1]:
            raise ValueError(f"non-increasing beat timestamps at line {line_number}")
        beats.append(timestamp)
        beat_numbers.append(number)
        if number == 1:
            downbeats.append(timestamp)
    if len(beats) < 2:
        raise ValueError(f"insufficient model beats in {path}")
    return {"beats": beats, "downbeats": downbeats, "beat_numbers": beat_numbers}


def match_grid(grid, attack_events, *, start=None, end=None,
               tolerance_ms=EVAL_TOLERANCE_MS):
    """One-to-one nearest matching against independently measured WAV attacks.

    Generator BPM and pulse index are only used by extract_kick_attacks to
    establish reference attack measurements. They are never used to infer
    Beat This! predictions or to rewrite/shift a grid.
    """
    if tolerance_ms <= 0 or not math.isfinite(tolerance_ms):
        raise ValueError("invalid evaluation tolerance")
    detected = [float(x) for x in grid
                if (start is None or x >= start) and (end is None or x < end)]
    if any(not math.isfinite(x) for x in detected):
        raise ValueError("nonfinite grid time")
    reference = sorted(
        float(row["audio_attack_seconds"])
        for row in attack_events if row.get("status") == "measured"
        and (start is None or row["audio_attack_seconds"] >= start)
        and (end is None or row["audio_attack_seconds"] < end)
    )
    candidates = []
    for i, beat in enumerate(detected):
        position = bisect.bisect_left(reference, beat)
        for j in (position - 1, position):
            if 0 <= j < len(reference):
                error_ms = (beat - reference[j]) * 1000.
                if abs(error_ms) <= tolerance_ms:
                    candidates.append((abs(error_ms), i, j))
    seen_detected = set()
    seen_reference = set()
    errors = []
    for _, i, j in sorted(candidates):
        if i in seen_detected or j in seen_reference:
            continue
        seen_detected.add(i)
        seen_reference.add(j)
        errors.append((detected[i] - reference[j]) * 1000.)
    precision = len(errors) / len(detected) if detected else None
    recall = len(errors) / len(reference) if reference else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and precision + recall else None)
    return {
        "detected_beat_count": len(detected),
        "measured_reference_attack_count": len(reference),
        "one_to_one_matched_count": len(errors),
        "precision_at_80ms": round(precision, 4) if precision is not None else None,
        "recall_at_80ms": round(recall, 4) if recall is not None else None,
        "f1_at_80ms": round(f1, 4) if f1 is not None else None,
        "median_absolute_matched_error_ms": round(
            float(statistics.median(abs(x) for x in errors)), 3
        ) if errors else None,
        "missed_measured_attacks": len(reference) - len(errors),
        "unmatched_predictions": len(detected) - len(errors),
        "tolerance_ms": tolerance_ms,
    }


def evaluate_track(wav: Path, prediction_path: Path, audit, *,
                   analyze=None):
    if not wav.is_file():
        raise ValueError(f"missing rights-free WAV: {wav}")
    expected = dict(FIXTURES)
    if wav.name not in expected:
        raise ValueError("benchmark only permits the two known generated demo WAVs")
    audio, rate = sf.read(wav, dtype="float32", always_2d=True)
    if rate < 8000 or not len(audio):
        raise ValueError("invalid demo WAV")
    mono = np.asarray(audio[:, 0], dtype=np.float64)
    attacks = audit.extract_kick_attacks(mono, rate, expected[wav.name])
    predictions = read_beat_tsv(prediction_path)
    duration = len(mono) / rate
    if predictions["beats"][-1] > duration + .1:
        raise ValueError("predicted beat timestamps exceed WAV duration")
    is_outgoing = wav.name == FIXTURES[0][0]
    model = {
        "beat_count": len(predictions["beats"]),
        "downbeat_count": len(predictions["downbeats"]),
        "first_beat_seconds": [round(v, 6) for v in predictions["beats"][:6]],
        "first_downbeat_seconds": [round(v, 6) for v in predictions["downbeats"][:6]],
        "whole_track": match_grid(predictions["beats"], attacks),
        "outgoing_failure_region": (
            match_grid(predictions["beats"], attacks,
                       start=MIDDLE_REGION[1], end=MIDDLE_REGION[2])
            if is_outgoing else None
        ),
    }
    result = {
        "file": wav.name,
        "duration_seconds": round(duration, 3),
        "measured_audio_kick_count": sum(
            x.get("status") == "measured" for x in attacks
        ),
        "unmeasured_synthetic_kick_windows": sum(
            x.get("status") != "measured" for x in attacks
        ),
        "beat_this": model,
    }
    if analyze is not None:
        sa = analyze(wav)
        grid = [float(x) for x in sa.beats_seconds]
        result["saund_current"] = {
            "scalar_bpm": float(sa.bpm),
            "beat_count": len(grid),
            "whole_track": match_grid(grid, attacks),
            "outgoing_failure_region": (
                match_grid(grid, attacks,
                           start=MIDDLE_REGION[1], end=MIDDLE_REGION[2])
                if is_outgoing else None
            ),
        }
    return result


def run(audio_root: Path, predictions_root: Path, *,
        analyze=None):
    audit = load_kick_audit()
    tracks = []
    for filename, _ in FIXTURES:
        wav = audio_root.expanduser().resolve() / filename
        prediction = predictions_root.expanduser().resolve() / (Path(filename).stem + ".beats")
        tracks.append(evaluate_track(wav, prediction, audit, analyze=analyze))
    return {
        "schema": "saund-m1-beat-this-external-benchmark-v1",
        "tracks": tracks,
        "model_source": "CPJKU/beat_this",
        "predictions_from_external_offline_cli": True,
        "beat_this_model_and_checkpoint_must_be_recorded_separately": True,
        "model_weights_downloaded_by_this_script": False,
        "reference_is_measured_synthetic_kick_attack_only": True,
        "ground_truth_downbeat_accuracy_measured": False,
        "metric_scope": "known generated WAVs, 80 ms one-to-one match to measurable kick attacks",
        "outgoing_failure_region": list(MIDDLE_REGION),
        "correct_phase_certified": False,
        "sealed_ballroom_test_touched": False,
        "production_changed": False,
        "beat_grid_modified": False,
        "neural_policy_promoted": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": (
            "This is a bounded external pretrained-model benchmark, not a "
            "new beat-grid authority. Model TSV is untrusted prediction input; "
            "synthetic kick attacks are post-hoc test references only. "
            "80ms evaluation tolerance does NOT relax the planner's 25ms hard gate."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-saund", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    analyze = None
    if args.compare_saund:
        # No model/network downloads from the existing SAUND analysis path.
        os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
        os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
        os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
        os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
        sys.path.insert(0, str(ROOT / "src"))
        from djai.analysis import analyze_track
        analyze = analyze_track
    result = run(args.audio_root, args.predictions_root, analyze=analyze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 Beat This! benchmark failed: {exc}") from exc
