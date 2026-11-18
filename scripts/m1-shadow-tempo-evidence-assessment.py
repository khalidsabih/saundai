#!/usr/bin/env python3
"""Summarize opt-in tempo shadow evidence from synthetic M1 demo audit.

Read-only evidence comparison. Never reanalyzes audio, changes scalar BPM,
modifies an approved plan, weakens reconciliation policy, or renders output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def _finite_positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def assess(report: dict) -> dict:
    if not isinstance(report, dict) or not isinstance(report.get("results"), list):
        raise ValueError("expected M1 demo meter audit results")
    rows = []
    for track in report["results"]:
        evidence = track.get("tempo_consistency_evidence") or {}
        windows = evidence.get("shadow_audio_windows")
        scalar = track.get("bpm")
        grid = track.get("grid_span_bpm")
        if not _finite_positive(scalar) or not _finite_positive(grid):
            raise ValueError("missing positive scalar/grid BPM")
        status = "insufficient_shadow_evidence"
        onset_median = None
        max_window_disagreement = None
        grid_onset_disagreement = None
        hypothetical_blend = None
        strengths = []
        if isinstance(windows, list) and len(windows) == 3 and all(
            isinstance(w, dict)
            and _finite_positive(w.get("bpm"))
            and _finite_positive(w.get("strength"))
            for w in windows
        ):
            speeds = [float(w["bpm"]) for w in windows]
            strengths = [float(w["strength"]) for w in windows]
            onset_median = median(speeds)
            max_window_disagreement = max(abs(b / onset_median - 1) for b in speeds)
            grid_onset_disagreement = abs(onset_median / float(grid) - 1)
            # Exactly the existing reconcile_bpm independent evidence checks.
            if (
                min(strengths) >= .5
                and max_window_disagreement <= .006
                and grid_onset_disagreement <= .006
            ):
                status = "independent_evidence_agrees_shadow_only"
                hypothetical_blend = median([onset_median, float(grid)])
            else:
                status = "independent_evidence_conflict_or_weak"
        rows.append({
            "filename": track.get("filename"),
            "meter": track.get("predicted_meter"),
            "scalar_bpm": float(scalar),
            "grid_span_bpm": float(grid),
            "scalar_vs_grid_pct": round(100 * (float(grid) / float(scalar) - 1), 4),
            "original_decision": evidence.get("decision"),
            "original_reason": evidence.get("reason"),
            "status": status,
            "onset_median_bpm": round(onset_median, 5) if onset_median is not None else None,
            "minimum_onset_strength": round(min(strengths), 4) if strengths else None,
            "max_window_disagreement_pct": (
                round(100 * max_window_disagreement, 4)
                if max_window_disagreement is not None else None
            ),
            "grid_onset_disagreement_pct": (
                round(100 * grid_onset_disagreement, 4)
                if grid_onset_disagreement is not None else None
            ),
            "hypothetical_reconciled_bpm_not_authorized": (
                round(hypothetical_blend, 5)
                if hypothetical_blend is not None else None
            ),
        })
    comparison = None
    if len(rows) == 2 and all(
        r["hypothetical_reconciled_bpm_not_authorized"] is not None
        for r in rows
    ):
        a, b = (r["hypothetical_reconciled_bpm_not_authorized"] for r in rows)
        comparison = {
            "hypothetical_tempo_difference_bpm": round(b - a, 5),
            "hypothetical_incoming_source_rate_to_match_outgoing": round(a / b, 6),
            "existing_scalar_incoming_source_rate": round(
                rows[0]["scalar_bpm"] / rows[1]["scalar_bpm"], 6
            ),
            "rate_is_research_counterfactual_only": True,
        }
    return {
        "schema": "saund-m1-shadow-tempo-evidence-assessment-v1",
        "tracks": rows,
        "comparison": comparison,
        "shadow_only": True,
        "changes_to_reconciliation_threshold": False,
        "transition_rendered": False,
        "notes": [
            "Only source audio and beat-grid evidence from the operator's existing synthetic audit are compared.",
            "A hypothetical reconciled BPM does not override the existing 0.8% materiality gate.",
            "Agreement does not guarantee accurate beats/downbeat phase or a safe phase-locked overlap.",
            "Do not apply the hypothetical rate to a signed plan or treat it as an accepted transition.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing report")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    assessment = assess(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(assessment, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(assessment, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 shadow tempo assessment failed: {exc}") from exc
