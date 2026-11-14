#!/usr/bin/env python3
"""Localize exceptional beat events and test continuity hypotheses, read-only.

Uses the original complete beat arrays and the independent onset candidates
from beat-sequence-interpretation.json. A steady clock is fitted ONLY as a
mathematical reference using the saved planner source period and beat indices.
Event offsets, interval anomalies and persistent offsets are diagnostic
patterns, not music ground truth or authority for beat-index changes.

No plan, beat grid, cache, downbeat, phase gate or renderer is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
from scripts.diagnose_rejected_beat_grids import full_beat_cache

OUTLIER_MS = 45.0
INTERVAL_TOLERANCE = 0.12
AUDIO_SUPPORT_RADIUS_MS = 55.0
MIN_PERSISTENT_EVENTS = 3
LOCAL_NEIGHBORS = 3


def nearest_independent_onset(
    time_seconds: float, events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    valid = []
    for event in events:
        try:
            at = float(event["seconds"])
            prominence = float(event["prominence_ratio"])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(at) and math.isfinite(prominence) and prominence > 0:
            if abs(at-time_seconds)*1000 <= AUDIO_SUPPORT_RADIUS_MS:
                valid.append((abs(at-time_seconds), -prominence, at, prominence))
    if not valid:
        return None
    distance, _, when, prominence = min(valid)
    return {
        "seconds": round(when, 6),
        "distance_ms": round(distance*1000, 3),
        "prominence_ratio": round(prominence, 3),
    }


def inspect_grid(
    beat_times: list[float], onset_events: list[dict[str, Any]], *,
    start: float, end: float, expected_period: float,
) -> dict[str, Any]:
    """Find and explain anomalies without ever creating a replacement grid."""
    grid = np.asarray(beat_times, dtype=float)
    if (
        not math.isfinite(expected_period) or expected_period <= 0
        or not math.isfinite(start) or not math.isfinite(end) or end <= start
        or len(grid) < 8 or np.any(~np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        return {"status": "invalid_grid_or_window"}
    left = max(0, int(np.searchsorted(grid, start))-LOCAL_NEIGHBORS)
    right = min(len(grid), int(np.searchsorted(grid, end, side="right"))+LOCAL_NEIGHBORS)
    times = grid[left:right]
    if len(times) < 8:
        return {"status": "insufficient_context"}
    indices = np.arange(len(times), dtype=float)
    origin = float(np.median(times-indices*expected_period))
    ideal = origin+indices*expected_period
    delta_ms = (times-ideal)*1000
    intervals = np.diff(times)
    selected = [
        i for i, t in enumerate(times)
        if start <= t <= end
    ]
    findings: list[dict[str, Any]] = []
    for i in selected:
        interval_before = float(intervals[i-1]) if i else None
        interval_after = float(intervals[i]) if i < len(intervals) else None
        unexpected_before = (
            interval_before is not None
            and abs(interval_before/expected_period-1) > INTERVAL_TOLERANCE
        )
        unexpected_after = (
            interval_after is not None
            and abs(interval_after/expected_period-1) > INTERVAL_TOLERANCE
        )
        if abs(delta_ms[i]) <= OUTLIER_MS and not unexpected_before and not unexpected_after:
            continue
        prev = delta_ms[max(0, i-LOCAL_NEIGHBORS):i]
        after = delta_ms[i+1:min(len(times), i+1+LOCAL_NEIGHBORS)]
        before_median = float(np.median(prev)) if len(prev) else None
        after_median = float(np.median(after)) if len(after) else None
        jump = after_median-before_median if before_median is not None and after_median is not None else None
        neighbors_near = (
            before_median is not None and after_median is not None
            and abs(before_median) <= OUTLIER_MS and abs(after_median) <= OUTLIER_MS
        )
        isolated = bool(
            abs(delta_ms[i]) > OUTLIER_MS and neighbors_near
            and abs(after_median-before_median) <= OUTLIER_MS
        )
        # Important: re-fitting an origin across a missed index can make
        # the residual of an entire segment large. We report candidate
        # persistence; we never auto-insert or remove a beat.
        persistent = bool(
            jump is not None and abs(jump) >= OUTLIER_MS
            and len(after) >= MIN_PERSISTENT_EVENTS
            and sum(abs(float(v)-after_median) <= OUTLIER_MS for v in after) >= MIN_PERSISTENT_EVENTS
        )
        near_original = nearest_independent_onset(float(times[i]), onset_events)
        near_reference = nearest_independent_onset(float(ideal[i]), onset_events)
        if isolated:
            interpretation = "isolated_timestamp_displacement_candidate"
        elif persistent:
            interpretation = "persistent_phase_offset_candidate"
        elif unexpected_before or unexpected_after:
            interpretation = "local_interval_anomaly_unresolved"
        else:
            interpretation = "reference_offset_unresolved"
        # Report whether the observed interval suggests a whole-event
        # indexing change. Not enough evidence to authorize such a change.
        gap_hypotheses: list[str] = []
        for value in (interval_before, interval_after):
            if value is None:
                continue
            ratio = value/expected_period
            if 1.65 <= ratio <= 2.35:
                gap_hypotheses.append("potential_missing_beat")
            elif ratio < 0.82:
                gap_hypotheses.append("potential_extra_or_misplaced_beat")
        findings.append({
            "global_beat_index": left+i,
            "cached_beat_seconds": round(float(times[i]), 6),
            "reference_beat_seconds": round(float(ideal[i]), 6),
            "reference_offset_ms": round(float(delta_ms[i]), 3),
            "interval_before_ms": round(interval_before*1000, 3) if interval_before is not None else None,
            "interval_after_ms": round(interval_after*1000, 3) if interval_after is not None else None,
            "interval_before_ratio": round(interval_before/expected_period, 4) if interval_before is not None else None,
            "interval_after_ratio": round(interval_after/expected_period, 4) if interval_after is not None else None,
            "median_preceding_offset_ms": round(before_median, 3) if before_median is not None else None,
            "median_following_offset_ms": round(after_median, 3) if after_median is not None else None,
            "following_minus_preceding_offset_ms": round(jump, 3) if jump is not None else None,
            "independent_onset_near_cached": near_original,
            "independent_onset_near_reference": near_reference,
            "interpretation": interpretation,
            "indexing_hypotheses": sorted(set(gap_hypotheses)),
            "index_change_authorized": False,
        })
    return {
        "status": "audited",
        "beats_inside_overlap": len(selected),
        "expected_period_ms": round(expected_period*1000, 4),
        "reference": "median-origin fixed-rate grid using planner tempo and original indices; not independently verified tempo",
        "outlier_threshold_ms": OUTLIER_MS,
        "interval_tolerance_fraction": INTERVAL_TOLERANCE,
        "outliers": findings,
        "outlier_count": len(findings),
        "interpretation_counts": {
            category: sum(f["interpretation"] == category for f in findings)
            for category in sorted({f["interpretation"] for f in findings})
        },
        "index_change_authorized": False,
    }


def diagnose(plan: dict[str, Any], sequence_audit: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    if plan.get("plan_id") != sequence_audit.get("plan_id"):
        raise ValueError("Plan ID mismatch between saved plan and independent audio audit")
    tracks = plan.get("tracks") or []
    audits = {int(r["transition"]): r for r in sequence_audit.get("transitions") or []}
    memo: dict[int, dict[str, Any] | None] = {}
    results = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": number}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            row["status"] = "no_rejected_rhythmic_candidate"
            results.append(row)
            continue
        audit = audits.get(number) or {}
        if audit.get("status") != "audited" or audit.get("candidate_signature") != candidate.get("signature"):
            row["status"] = "missing_or_mismatched_audio_audit"
            results.append(row)
            continue
        a, b = edge.get("from_index"), edge.get("to_index")
        if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a < len(tracks) and 0 <= b < len(tracks)):
            row["status"] = "invalid_track_indices"
            results.append(row)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            rate_a = float(candidate["render_rate_a"])
            rate_b = float(candidate["render_rate_b"])
            duration = candidate.get("transition_seconds")
            duration = float(duration) if duration is not None else 4*float(candidate["transition_bars"])*60/bpm
            cue_a = float(candidate["exit_time_a"])
            cue_b = float(candidate["entry_time_b"]) + float(candidate.get("phase_nudge_seconds_b") or 0.0)
            if not all(math.isfinite(v) and v > 0 for v in (bpm, rate_a, rate_b, duration)):
                raise ValueError("Invalid transition duration/rate/tempo")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            row["status"] = "incomplete_candidate_timing"
            results.append(row)
            continue
        for idx in (a, b):
            if idx not in memo:
                memo[idx] = full_beat_cache(cache_root, str(tracks[idx].get("path") or ""))[0]
        if memo[a] is None or memo[b] is None:
            row["status"] = "full_beat_cache_missing"
            results.append(row)
            continue
        row["candidate_signature"] = candidate.get("signature")
        for deck, ti, start, rate in (
            ("deck_a", a, cue_a, rate_a),
            ("deck_b", b, cue_b, rate_b),
        ):
            evidence = (audit.get(deck) or {}).get("audio_onset_candidates_in_window") or []
            row[deck] = inspect_grid(
                memo[ti]["beats_seconds"], evidence,
                start=start, end=start+duration*rate,
                expected_period=60*rate/bpm,
            )
        row["reported_max_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
        row["status"] = (
            "audited" if all(row[d]["status"] == "audited" for d in ("deck_a", "deck_b"))
            else "insufficient_beat_context"
        )
        results.append(row)
    return {
        "schema": "saund-beat-grid-outlier-localization-v1",
        "plan_id": plan.get("plan_id"),
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "transitions": results,
        "limitations": [
            "The timing reference is fitted to original beat indices with planner tempo and is not independent beat ground truth.",
            "Nearby onset candidates may be hats, syncopated percussion or subdivisions rather than quarter-note beats.",
            "Classification is descriptive; persistent offsets do not prove a missing or duplicated beat.",
            "The script never inserts, removes or shifts beat events in production.",
            "No phase-lock safety decision or rendered audio quality is re-evaluated.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--sequence-audit", type=Path, default=Path("output/milestone1-tempo-consistency/beat-sequence-interpretation.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/beat-grid-outliers.json"))
    args = p.parse_args()
    report = diagnose(
        json.loads(args.set_plan.read_text(encoding="utf-8")),
        json.loads(args.sequence_audit.read_text(encoding="utf-8")),
        args.cache_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()),
                      "transitions": [
                          {"transition": row["transition"], "status": row["status"],
                           "deck_a_outliers": row.get("deck_a", {}).get("outlier_count"),
                           "deck_b_outliers": row.get("deck_b", {}).get("outlier_count")}
                          for row in report["transitions"]]}, indent=2))


if __name__ == "__main__":
    main()
