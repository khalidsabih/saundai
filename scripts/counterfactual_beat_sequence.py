#!/usr/bin/env python3
"""Read-only, conservative beat-sequence counterfactual for rejected transitions.

Consumes the independently generated cue-onset audit and original full beat
caches. Proposed onset refinements must meet strict per-beat and per-interval
constraints; ambiguous or unsupported beats remain unchanged. Reports both
baseline and hypothetical phase trajectories using the same planner formula.
This is not permission to alter audio, beat grids, downbeats or phase limits.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from scripts.reconstruct_phase_drift import reconstruct

MAX_SHIFT_MS = 30.0
MIN_SUPPORT_RATIO = 2.0
MIN_INTERVAL_FRACTION = 0.90
MAX_INTERVAL_FRACTION = 1.10
MIN_OBSERVATION_COVERAGE = 0.90


def refine_sequence(
    beats: list[float], observations: list[dict[str, Any]], *,
    expected_interval_seconds: float,
) -> dict[str, Any]:
    """Return copy of beat sequence; always preserve order and beat count.

    This deliberately performs no insertion/deletion and no interpolation.
    If an overlap contains abnormal intervals, it cannot claim that a
    refinement removed the anomaly merely by shifting nearby onset peaks.
    """
    original = np.asarray(beats, dtype=float)
    result = original.copy()
    if len(original) < 4 or expected_interval_seconds <= 0 or not np.all(np.isfinite(original)):
        return {"status": "invalid_grid", "refined_beats": beats, "changes": []}
    if np.any(np.diff(original) <= 0):
        return {"status": "invalid_grid", "refined_beats": beats, "changes": []}
    changes: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for obs in observations:
        # Resolve timestamp to exact cached beat even when the audit reports
        # local (rather than global) beat indices.
        try:
            t = float(obs["cached_beat_seconds"])
            peak = float(obs["audio_onset_peak_seconds"])
            support = float(obs["local_peak_to_median_ratio"])
        except (KeyError, ValueError, TypeError):
            reasons["malformed_observation"] = reasons.get("malformed_observation", 0) + 1
            continue
        if not all(map(math.isfinite, (t, peak, support))):
            reasons["invalid_observation"] = reasons.get("invalid_observation", 0) + 1
            continue
        i = int(np.searchsorted(original, t))
        candidates = [j for j in (i-1, i) if 0 <= j < len(original)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(original[k]-t))
        if abs(original[j]-t) > 0.000002:
            reasons["timestamp_not_in_grid"] = reasons.get("timestamp_not_in_grid", 0) + 1
            continue
        delta_ms = (peak-t)*1000
        if support < MIN_SUPPORT_RATIO or abs(delta_ms) > MAX_SHIFT_MS:
            reasons["weak_or_large_shift"] = reasons.get("weak_or_large_shift", 0) + 1
            continue
        # Require BOTH neighboring intervals to remain plausible. This
        # knowingly leaves gross beat-indexing anomalies unresolved.
        left = peak - result[j-1] if j else expected_interval_seconds
        right = result[j+1] - peak if j+1 < len(result) else expected_interval_seconds
        if any(not MIN_INTERVAL_FRACTION <= dt/expected_interval_seconds <= MAX_INTERVAL_FRACTION
               for dt in (left, right)):
            reasons["interval_constraint"] = reasons.get("interval_constraint", 0) + 1
            continue
        result[j] = peak
        changes.append({
            "global_beat_index": j,
            "original_seconds": round(t, 6),
            "proposed_seconds": round(peak, 6),
            "shift_ms": round(delta_ms, 3),
        })
    return {
        "status": "refined_for_diagnostic",
        "refined_beats": result.tolist(),
        "changes": changes,
        "rejection_reasons": reasons,
    }


def evaluate(
    beats_a: list[float], beats_b: list[float],
    obs_a: list[dict[str, Any]], obs_b: list[dict[str, Any]],
    *, exit_time_a: float, entry_time_b: float, duration: float,
    bpm: float, rate_a: float, rate_b: float, nudge_b: float = 0.0,
    samples: int = 9,
) -> dict[str, Any]:
    if min(duration, bpm, rate_a, rate_b) <= 0:
        return {"status": "invalid_timing"}
    expected_a = rate_a * 60 / bpm
    expected_b = rate_b * 60 / bpm
    left = refine_sequence(beats_a, obs_a, expected_interval_seconds=expected_a)
    right = refine_sequence(beats_b, obs_b, expected_interval_seconds=expected_b)
    baseline = reconstruct(beats_a, beats_b, exit_time_a=exit_time_a,
        entry_time_b=entry_time_b, duration=duration, bpm=bpm,
        rate_a=rate_a, rate_b=rate_b, nudge_b=nudge_b, samples=samples)
    hypothesis = reconstruct(left["refined_beats"], right["refined_beats"],
        exit_time_a=exit_time_a, entry_time_b=entry_time_b, duration=duration,
        bpm=bpm, rate_a=rate_a, rate_b=rate_b, nudge_b=nudge_b, samples=samples)
    return {
        "status": "evaluated",
        "changes_a": left["changes"],
        "changes_b": right["changes"],
        "rejections_a": left.get("rejection_reasons", {}),
        "rejections_b": right.get("rejection_reasons", {}),
        "baseline_uncorrected_max_drift_ms": baseline["max_abs_detected_grid_drift_ms"],
        "hypothetical_uncorrected_max_drift_ms": hypothesis["max_abs_detected_grid_drift_ms"],
        "change_in_max_uncorrected_drift_ms": round(
            hypothesis["max_abs_detected_grid_drift_ms"]-baseline["max_abs_detected_grid_drift_ms"], 3),
        "baseline_trajectory": [p.get("detected_grid_phase_drift_ms") for p in baseline["phase_points"]],
        "hypothetical_trajectory": [p.get("detected_grid_phase_drift_ms") for p in hypothesis["phase_points"]],
        "note": "Uncorrected phase only; this does NOT replay bounded corrections or evaluate rendered audio and cannot establish phase-integrity passage.",
    }


def diagnose(plan: dict[str, Any], audit: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    audit_by_number = {int(row["transition"]): row for row in audit.get("transitions") or []}
    memo: dict[int, Any] = {}
    results: list[dict[str, Any]] = []
    if audit.get("plan_id") != plan.get("plan_id"):
        raise ValueError("Audit and set plan belong to different plan IDs")
    for number, transition in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": number}
        observed = audit_by_number.get(number) or {}
        if observed.get("status") != "audited":
            row["status"] = observed.get("status", "missing_audit")
            results.append(row)
            continue
        ai, bi = transition.get("from_index"), transition.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            row["status"] = "invalid_track_indices"
            results.append(row)
            continue
        rejected = ((transition.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            row["status"] = "missing_rejected_candidate"
            results.append(row)
            continue
        if observed.get("candidate_signature") != candidate.get("signature"):
            row["status"] = "candidate_signature_mismatch"
            results.append(row)
            continue
        for i in (ai, bi):
            if i not in memo:
                memo[i] = full_beat_cache(cache_root, str(tracks[i].get("path") or ""))[0]
        if memo[ai] is None or memo[bi] is None:
            row["status"] = "missing_full_beat_cache"
            results.append(row)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            bars = candidate.get("transition_bars")
            duration = candidate.get("transition_seconds")
            duration = float(duration) if duration is not None else 4*float(bars)*60/bpm
            args = {
                "exit_time_a": float(candidate["exit_time_a"]),
                "entry_time_b": float(candidate["entry_time_b"]),
                "duration": duration, "bpm": bpm,
                "rate_a": float(candidate["render_rate_a"]),
                "rate_b": float(candidate["render_rate_b"]),
                "nudge_b": float(candidate.get("phase_nudge_seconds_b") or 0.0),
                "samples": len((rejected.get("phase_integrity") or {}).get("points") or []) or 9,
            }
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            row["status"] = "incomplete_timing"
            results.append(row)
            continue
        # Audit windows must contain sufficient evidence; otherwise no
        # counterfactual claim is warranted even if a few peaks exist.
        cov_a = float(observed.get("deck_a", {}).get("support_coverage") or 0)
        cov_b = float(observed.get("deck_b", {}).get("support_coverage") or 0)
        if min(cov_a, cov_b) < MIN_OBSERVATION_COVERAGE:
            row["status"] = "insufficient_onset_coverage"
            results.append(row)
            continue
        row["nudge_assumed_zero"] = candidate.get("phase_nudge_seconds_b") is None
        row["reported_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
        row.update(evaluate(memo[ai]["beats_seconds"], memo[bi]["beats_seconds"],
            observed["deck_a"].get("observations") or [],
            observed["deck_b"].get("observations") or [], **args))
        results.append(row)
    return {
        "schema": "saund-beat-sequence-counterfactual-v1",
        "plan_id": plan.get("plan_id"),
        "conditions": {
            "maximum_single_beat_shift_ms": MAX_SHIFT_MS,
            "minimum_peak_to_median_ratio": MIN_SUPPORT_RATIO,
            "interval_ratio_range": [MIN_INTERVAL_FRACTION, MAX_INTERVAL_FRACTION],
            "minimum_coverage": MIN_OBSERVATION_COVERAGE,
            "preserve_beat_count_and_order": True,
        },
        "transitions": results,
        "production_changes": False,
        "limitations": [
            "Hypothetical onset shifts are not evidence of verified beat/downbeat or correct musical meter.",
            "This diagnostic does not alter real cached grids, plan, mixer, phase safeguards, or release policies.",
            "Reported uncorrected drift is not the safety-gated, bounded-correction residual.",
            "Onset offsets can be shared by both decks and may cancel; reducing one shift does not guarantee phase improvement.",
            "Large or ambiguous beat-indexing defects are deliberately not repaired here.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--cue-audit", type=Path, default=Path("output/milestone1-tempo-consistency/cue-onset-precision.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/beat-sequence-counterfactual.json"))
    args = p.parse_args()
    result = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")),
                      json.loads(args.cue_audit.read_text(encoding="utf-8")), args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()), "transitions": [
        {"number": x["transition"], "status": x["status"],
         "baseline_drift_ms": x.get("baseline_uncorrected_max_drift_ms"),
         "hypothetical_drift_ms": x.get("hypothetical_uncorrected_max_drift_ms")}
        for x in result["transitions"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
