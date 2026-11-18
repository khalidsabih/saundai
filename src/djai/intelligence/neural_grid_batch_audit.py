"""Batch, read-only evaluation of neural grid gates against annotated filenames.

Runs the same fallback-aware audit for each unique reference track and
summarizes how an *experimental* screen would affect existing gate approvals.
No track analysis, DJ decision or model authority is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .neural_grid_gate_audit import audit_grid_gate
from .tempo_validation import validate_references


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif"}


def summarize_batch(rows: list[dict]) -> dict:
    completed = [r for r in rows if r["status"] == "ok"]
    existing = [r for r in completed if r["audit"]["neural_grid_gate_would_pass"]]
    rejected = [
        r for r in existing
        if not r["audit"]["counterfactual_safety_screen"]["would_pass_provisional_screen"]
    ]
    provisional = [
        r for r in existing
        if r["audit"]["counterfactual_safety_screen"]["would_pass_provisional_screen"]
    ]
    reasons = Counter(
        reason for row in rejected
        for reason in row["audit"]["counterfactual_safety_screen"]["screen_reasons"]
    )
    return {
        "reference_count": len(rows),
        "evaluated_count": len(completed),
        "unavailable_count": len(rows) - len(completed),
        "approximate_reference_count": sum(r["reference_quality"] == "approximate" for r in rows),
        "independently_verified_reference_count": sum(
            r["reference_quality"] == "independently_verified" for r in rows
        ),
        "existing_gate_approvals": len(existing),
        "existing_gate_rejections": len(completed) - len(existing),
        "provisional_screen_rejections_of_existing_approvals": len(rejected),
        "provisional_screen_passes_of_existing_approvals": len(provisional),
        "rejection_reason_counts": dict(sorted(reasons.items())),
        # Lack of independent beat/downbeat labels prevents estimating these.
        "false_acceptance_rate": None,
        "false_rejection_rate": None,
        "beat_alignment_accuracy": None,
        "downbeat_accuracy": None,
        "live_adoptions": 0,
        "ready_for_live_adoption": False,
    }


def evaluate_batch(audio_root: Path, references: list[dict]) -> dict:
    references = validate_references(references)
    if not audio_root.is_dir():
        raise ValueError(f"Audio directory not found: {audio_root}")
    available: dict[str, list[Path]] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            available.setdefault(path.name, []).append(path)
    rows: list[dict] = []
    for reference in references:
        name = reference["filename"]
        row = {
            "filename": name,
            "reference_bpm": reference["reference_bpm"],
            "reference_quality": reference["reference_quality"],
            "verification_method": reference.get("verification_method"),
            "status": "unavailable",
        }
        matches = available.get(name, [])
        if len(matches) != 1:
            row["reason"] = (
                "audio file missing" if not matches else
                "ambiguous audio basename (multiple files)"
            )
        else:
            try:
                audit = audit_grid_gate(matches[0])
                row["audit"] = audit
                row["status"] = "ok"
                # Merely contextual agreement with annotated BPM. It does NOT
                # validate beat or downbeat alignment and must not influence gate.
                neural_bpm = audit["neural_bpm"]
                if (
                    isinstance(neural_bpm, (int, float))
                    and math.isfinite(neural_bpm)
                ):
                    row["neural_reference_absolute_error_bpm"] = round(
                        abs(neural_bpm - float(reference["reference_bpm"])), 4
                    )
                else:
                    row["neural_reference_absolute_error_bpm"] = None
            except Exception as exc:
                row["reason"] = f"auditing failed: {type(exc).__name__}: {exc}"
        rows.append(row)
    return {
        "benchmark": "SAUND full-grid gate counterfactual batch (not adoption)",
        "mode": "observation_only",
        "rows": rows,
        "summary": summarize_batch(rows),
        "limitations": [
            "BPM annotations do not independently verify beat positions or downbeat phase.",
            "Approximate BPM labels are not independently verified ground truth.",
            "Two algorithms can share the same tempo, meter and phase errors.",
            "Screen cutoffs are research hypotheses; false acceptance and rejection rates are unknown.",
            "This calls the live-signature neural gate but does not execute analyze_track or a running DJ app.",
            "All experimental passes have approved_for_live_adoption=false; no DJ decision is changed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only full-grid gate audit of all reference tracks"
    )
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New JSON file; never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite output: {args.output}")
    try:
        references = json.loads(args.references.read_text(encoding="utf-8"))
        report = evaluate_batch(args.audio_root, references)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    # Avoid a partial success report that silently hides omitted tracks.
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Full per-track report: {args.output}")


if __name__ == "__main__":
    main()
