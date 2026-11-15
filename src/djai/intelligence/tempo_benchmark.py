"""Reference BPM evaluation for neural rhythm evidence.

Observational only: no changes to analysis, model weights, thresholds, or DJ decisions.
References are user-reported approximate tempos, not beat/downbeat annotations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

DEFAULT_REFERENCES = Path(__file__).resolve().parents[3] / "config" / "neural" / "bpm-reference-tracks.json"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".aif"}


def compare_bpm(reference_bpm: float, predicted_bpm: float) -> dict[str, float]:
    """Absolute error and relative error; no octave folding of wrong tempos."""
    if not math.isfinite(reference_bpm) or reference_bpm <= 0:
        raise ValueError("Reference BPM must be positive and finite")
    if not math.isfinite(predicted_bpm) or predicted_bpm <= 0:
        raise ValueError("Predicted BPM must be positive and finite")
    error = predicted_bpm - reference_bpm
    return {
        "reference_bpm": round(reference_bpm, 4),
        "predicted_bpm": round(predicted_bpm, 4),
        "signed_error_bpm": round(error, 4),
        "absolute_error_bpm": round(abs(error), 4),
        "absolute_percentage_error": round(100 * abs(error) / reference_bpm, 4),
    }


def deterministic_bpm(path: Path) -> float:
    """Measure SAUND's original librosa tempo estimate, before neural fusion.

    Mirrors analyze_track's audio loading, onset extraction, and beat_track
    parameters. This intentionally avoids a full structural analysis.
    """
    import librosa
    import numpy as np
    from ..analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH
    from ..audio import load_audio

    source_audio, sr = load_audio(path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    y = librosa.to_mono(source_audio)
    if librosa.get_duration(y=y, sr=sr) < 4.0:
        raise ValueError("Track is too short for SAUND analysis")
    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sr, hop_length=HOP_LENGTH,
        units="frames", sparse=True,
    )
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    # Original analysis defaults to 120 BPM when librosa returns no tempo.
    return bpm if bpm > 0.0 else 120.0


def compare_estimators(reference: float, deterministic: float | None,
                       neural: float | None) -> dict:
    """Compare both estimators to the same reference; never infer a winner from missing data."""
    output: dict = {"deterministic": None, "neural": None, "winner": "unavailable"}
    if deterministic is not None:
        output["deterministic"] = compare_bpm(reference, deterministic)
    if neural is not None:
        output["neural"] = compare_bpm(reference, neural)
    if output["deterministic"] is None or output["neural"] is None:
        return output
    det_error = output["deterministic"]["absolute_error_bpm"]
    neural_error = output["neural"]["absolute_error_bpm"]
    if abs(det_error - neural_error) <= 0.01:
        output["winner"] = "tie_within_0_01_bpm"
    else:
        output["winner"] = "deterministic" if det_error < neural_error else "neural"
    return output


def summarize(rows: list[dict]) -> dict:
    """Summarize available predictions while reporting missing data honestly."""
    observed = [r for r in rows if r.get("status") == "ok"]
    errors = [r["absolute_error_bpm"] for r in observed]
    percentages = [r["absolute_percentage_error"] for r in observed]
    paired = [r for r in rows if r.get("comparison", {}).get("deterministic")
              and r.get("comparison", {}).get("neural")]
    det_errors = [r["comparison"]["deterministic"]["absolute_error_bpm"] for r in paired]
    neural_errors = [r["comparison"]["neural"]["absolute_error_bpm"] for r in paired]
    return {
        "reference_count": len(rows),
        "evaluated_count": len(observed),
        "unavailable_count": len(rows) - len(observed),
        "mean_absolute_error_bpm": round(mean(errors), 4) if errors else None,
        "mean_absolute_percentage_error": round(mean(percentages), 4) if percentages else None,
        "within_1_bpm_count": sum(e <= 1.0 for e in errors),
        "within_2_bpm_count": sum(e <= 2.0 for e in errors),
        "paired_count": len(paired),
        "paired_mean_absolute_error_deterministic_bpm": round(mean(det_errors), 4) if det_errors else None,
        "paired_mean_absolute_error_neural_bpm": round(mean(neural_errors), 4) if neural_errors else None,
        "deterministic_wins": sum(r["comparison"]["winner"] == "deterministic" for r in paired),
        "neural_wins": sum(r["comparison"]["winner"] == "neural" for r in paired),
        "ties": sum(r["comparison"]["winner"] == "tie_within_0_01_bpm" for r in paired),
        "arbitration_review_candidates": sum(
            r.get("arbitration", {}).get("decision") == "review_candidate" for r in rows
        ),
        "arbitration_retain": sum(
            r.get("arbitration", {}).get("decision") == "retain" for r in rows
        ),
        "arbitration_abstentions": sum(
            r.get("arbitration", {}).get("decision") == "abstain" for r in rows
        ),
        "calibration_proven": False,
    }


def evaluate(audio_root: Path, references: list[dict]) -> dict:
    from .neural_ears import predict_beats
    from .tempo_arbitration import arbitrate_tempo, onset_window_diagnostics

    if not audio_root.is_dir():
        raise ValueError(f"Audio directory not found: {audio_root}")
    available = {
        p.name: p for p in audio_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    rows: list[dict] = []
    for item in references:
        filename = str(item["filename"])
        reference = float(item["reference_bpm"])
        row = {
            "filename": filename,
            "reference_source": item.get("source", "unspecified"),
            "status": "missing",
        }
        path = available.get(filename)
        if path is None:
            row["reason"] = "reference audio file not found"
        else:
            # Run both estimators independently, without altering the DJ Brain.
            deterministic = None
            neural = None
            failures = {}
            try:
                deterministic = deterministic_bpm(path)
            except Exception as exc:
                failures["deterministic"] = f"{type(exc).__name__}: {exc}"
            try:
                result = predict_beats(path)
                if result.available and result.bpm > 0:
                    neural = result.bpm
                    row.update(compare_bpm(reference, neural))
                    row.update({
                        "tempo_stability": result.tempo_confidence,
                        "grid_confidence": result.confidence,
                        "beat_alignment_measured": False,
                        "downbeat_accuracy_measured": False,
                        "full_grid_adopted": result.used,
                    })
                else:
                    failures["neural"] = result.reason
            except Exception as exc:
                failures["neural"] = f"{type(exc).__name__}: {exc}"
            row["comparison"] = compare_estimators(reference, deterministic, neural)
            try:
                diagnostics = (
                    onset_window_diagnostics(path)
                    if deterministic is not None and neural is not None
                    else {"status": "missing_estimator", "windows": []}
                )
                windows = tuple(
                    (float(w["bpm"]), float(w["strength"]))
                    for w in diagnostics["windows"]
                    if w["status"] == "ok"
                )
                arbitration = arbitrate_tempo(
                    deterministic, neural, windows,
                    window_diagnostics=tuple(diagnostics["windows"]),
                ).to_dict()
                arbitration["window_analysis_status"] = diagnostics["status"]
                if "duration_seconds" in diagnostics:
                    arbitration["analyzed_duration_seconds"] = diagnostics["duration_seconds"]
                if "minimum_seconds" in diagnostics:
                    arbitration["minimum_duration_seconds"] = diagnostics["minimum_seconds"]
                if arbitration["suggested_bpm"] is not None:
                    arbitration["reference_error_if_adopted_bpm"] = compare_bpm(
                        reference, arbitration["suggested_bpm"]
                    )["absolute_error_bpm"]
                row["arbitration"] = arbitration
            except Exception as exc:
                row["arbitration"] = {
                    "decision": "abstain",
                    "suggested_bpm": None,
                    "reason": f"arbitration evidence unavailable: {type(exc).__name__}",
                    "changes_dj_decisions": False,
                }
            row["status"] = ("ok" if neural is not None else
                             "partial" if deterministic is not None else "unavailable")
            if failures:
                row["estimator_failures"] = failures
        rows.append(row)
    return {
        "benchmark": "SAUND paired deterministic/neural BPM reference",
        "mode": "observation_only",
        "rows": rows,
        "summary": summarize(rows),
        "limitations": [
            "Two user-reported reference BPMs cannot establish model calibration",
            "Comparison uses the initial librosa BPM before optional consistency reconciliation or fusion",
            "Paired winners are descriptive, not statistically significant",
            "Tempo stability is not verified BPM accuracy",
            "Beat phase and musical downbeat accuracy are not measured",
            "Arbitration recommendations are for human review, never automatic tempo adoption",
            "Onset ACF window evidence can share signal-processing biases with the deterministic estimator",
            "No threshold or DJ Brain decision is modified",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observational neural BPM reference benchmark")
    parser.add_argument("--audio-root", type=Path, required=True, help="Directory containing reference audio")
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, help="Optional local JSON report file")
    args = parser.parse_args()
    references = json.loads(args.references.read_text(encoding="utf-8"))
    if not isinstance(references, list):
        parser.error("References must be a JSON array")
    report = evaluate(args.audio_root, references)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
