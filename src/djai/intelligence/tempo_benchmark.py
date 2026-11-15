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


def summarize(rows: list[dict]) -> dict:
    """Summarize available predictions while reporting missing data honestly."""
    observed = [r for r in rows if r.get("status") == "ok"]
    errors = [r["absolute_error_bpm"] for r in observed]
    percentages = [r["absolute_percentage_error"] for r in observed]
    return {
        "reference_count": len(rows),
        "evaluated_count": len(observed),
        "unavailable_count": len(rows) - len(observed),
        "mean_absolute_error_bpm": round(mean(errors), 4) if errors else None,
        "mean_absolute_percentage_error": round(mean(percentages), 4) if percentages else None,
        "within_1_bpm_count": sum(e <= 1.0 for e in errors),
        "within_2_bpm_count": sum(e <= 2.0 for e in errors),
        "calibration_proven": False,
    }


def evaluate(audio_root: Path, references: list[dict]) -> dict:
    from .neural_ears import predict_beats

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
            try:
                result = predict_beats(path)
                if result.available and result.bpm > 0:
                    row.update(compare_bpm(reference, result.bpm))
                    row.update({
                        "status": "ok",
                        "tempo_stability": result.tempo_confidence,
                        "grid_confidence": result.confidence,
                        "beat_alignment_measured": False,
                        "downbeat_accuracy_measured": False,
                        "full_grid_adopted": result.used,
                    })
                else:
                    row["status"] = "unavailable"
                    row["reason"] = result.reason
            except Exception as exc:
                row["status"] = "error"
                row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return {
        "benchmark": "SAUND real-track neural BPM reference",
        "mode": "observation_only",
        "rows": rows,
        "summary": summarize(rows),
        "limitations": [
            "Two user-reported reference BPMs cannot establish model calibration",
            "Tempo stability is not verified BPM accuracy",
            "Beat phase and musical downbeat accuracy are not measured",
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
