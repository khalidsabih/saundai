#!/usr/bin/env python3
"""Read-only event-level cached-vs-independent beat tracker disagreement audit.

Globally align two beat sequences in each already-audited source-audio window
using monotone dynamic programming with explicit skip penalties. Report timing
residuals and unmatched events, prioritizing the exact saved junctions. A
nearby beat within the previous 70 ms coverage tolerance is NOT a safety pass.
A discrepancy does not establish which tracker represents the musical beat.

No audio decode, beat insertion/deletion, retiming, production code change,
phase-gate change, transition decision or cached artifact mutation occurs.
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

MATCH_MAX_FRACTION = 0.30
SKIP_COST = 0.40
PHASE_LIMIT_MS = 25.0
LEGACY_MATCH_RADIUS_MS = 70.0
LOCAL_BEATS_EACH_SIDE = 6


def monotone_align(
    cached: list[float], tracked: list[float], *, period: float,
) -> dict[str, Any]:
    """Globally align ordered beat events, never matching across a long gap.

    Cost is |time difference| / nominal period. A skip costs 0.4; a match
    beyond 0.30 periods is prohibited. The alignment is descriptive;
    it does not relabel or rewrite musical beat indices.
    """
    a = np.asarray(cached, dtype=float)
    b = np.asarray(tracked, dtype=float)
    if (not math.isfinite(period) or period <= 0 or len(a) == 0 or len(b) == 0
        or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b))
        or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0)):
        return {"status": "invalid_sequences"}
    n, m = len(a), len(b)
    # Allowed transitions: matched pair, skip cached, skip tracked.
    cost = np.full((n+1, m+1), np.inf, dtype=float)
    move = np.zeros((n+1, m+1), dtype=np.uint8)
    cost[0, 0] = 0.0
    for i in range(1, n+1):
        cost[i, 0] = i*SKIP_COST
        move[i, 0] = 1
    for j in range(1, m+1):
        cost[0, j] = j*SKIP_COST
        move[0, j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = abs(a[i-1]-b[j-1])/period
            paired = (cost[i-1, j-1]+dist) if dist <= MATCH_MAX_FRACTION else np.inf
            skipped_a = cost[i-1, j]+SKIP_COST
            skipped_b = cost[i, j-1]+SKIP_COST
            best = min(paired, skipped_a, skipped_b)
            cost[i, j] = best
            move[i, j] = 3 if paired == best else (1 if skipped_a == best else 2)
    i, j = n, m
    pairs: list[tuple[int, int]] = []
    unmatched_cached: list[int] = []
    unmatched_tracked: list[int] = []
    while i or j:
        direction = int(move[i, j])
        if direction == 3:
            pairs.append((i-1, j-1))
            i -= 1
            j -= 1
        elif direction == 1:
            unmatched_cached.append(i-1)
            i -= 1
        elif direction == 2:
            unmatched_tracked.append(j-1)
            j -= 1
        else:
            return {"status": "alignment_failed"}
    pairs.reverse()
    unmatched_cached.reverse()
    unmatched_tracked.reverse()
    return {
        "status": "aligned",
        "pairs": pairs,
        "unmatched_cached_indices": unmatched_cached,
        "unmatched_tracked_indices": unmatched_tracked,
        "total_alignment_cost": round(float(cost[n, m]), 4),
        "matched": len(pairs),
        "unmatched_cached": len(unmatched_cached),
        "unmatched_independent": len(unmatched_tracked),
        "max_match_distance_ms": round(MATCH_MAX_FRACTION*period*1000, 3),
    }


def compare_junction(
    cached_full: list[float], independent: list[float],
    saved: dict[str, Any], *, expected_period: float,
) -> dict[str, Any]:
    boundary = saved.get("junction") or {}
    li, ri = boundary.get("left_index"), boundary.get("right_index")
    start = saved.get("source_start")
    end = saved.get("source_end")
    if not isinstance(li, int) or not isinstance(ri, int) or li < 0 or ri >= len(cached_full) or li >= ri:
        return {"status": "invalid_junction_indices"}
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return {"status": "missing_audio_window"}
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        return {"status": "invalid_audio_window"}
    full = np.asarray(cached_full, dtype=float)
    # Preserve original global beat indices, but compare only beats inside the
    # independently audited audio window to avoid false start/end mismatches.
    original_indices = [i for i,t in enumerate(full) if start <= t <= end]
    tracked = sorted(float(t) for t in independent if start <= float(t) <= end)
    if not original_indices or not tracked:
        return {"status": "missing_comparable_events"}
    ref = [float(full[i]) for i in original_indices]
    alignment = monotone_align(ref, tracked, period=expected_period)
    if alignment["status"] != "aligned":
        return alignment
    pairs = alignment.pop("pairs")
    u_cached = alignment.pop("unmatched_cached_indices")
    u_tracked = alignment.pop("unmatched_tracked_indices")
    begin_idx = max(0, li-LOCAL_BEATS_EACH_SIDE)
    finish_idx = min(len(full)-1, ri+LOCAL_BEATS_EACH_SIDE)
    left_time = float(full[begin_idx])
    right_time = float(full[finish_idx])
    # Treat local disagreements as information only. "25ms" is a screening
    # threshold here, not the deck controller's cross-deck residual.
    observed = []
    paired_local = 0
    screened = 0
    for ai, bi in pairs:
        global_index = original_indices[ai]
        ct, it = ref[ai], tracked[bi]
        signed_ms = (it-ct)*1000
        if global_index < begin_idx or global_index > finish_idx:
            continue
        paired_local += 1
        screened += int(abs(signed_ms) > PHASE_LIMIT_MS)
        observed.append({
            "global_cached_beat_index": global_index,
            "cached_seconds": round(ct, 6),
            "independent_seconds": round(it, 6),
            "independent_minus_cached_ms": round(signed_ms, 3),
            "over_25ms_screen": abs(signed_ms) > PHASE_LIMIT_MS,
            "within_previous_70ms_tolerance": abs(signed_ms) <= LEGACY_MATCH_RADIUS_MS,
        })
    unmatched_a = [{
        "global_cached_beat_index": original_indices[k],
        "cached_seconds": round(ref[k], 6),
    } for k in u_cached if begin_idx <= original_indices[k] <= finish_idx]
    unmatched_b = [{
        "independent_seconds": round(tracked[k], 6),
    } for k in u_tracked if left_time <= tracked[k] <= right_time]
    # Comparisons of adjacent aligned indices reveal a mismatch of event
    # counts but do not assert a true missing or duplicated musical beat.
    index_gaps = []
    for (ai, bi), (aj, bj) in zip(pairs, pairs[1:]):
        global_i, global_j = original_indices[ai], original_indices[aj]
        if global_j < begin_idx or global_i > finish_idx:
            continue
        cached_steps = global_j-global_i
        independent_steps = bj-bi
        if cached_steps != independent_steps:
            index_gaps.append({
                "after_cached_global_index": global_i,
                "before_cached_global_index": global_j,
                "cached_index_steps": cached_steps,
                "independent_index_steps": independent_steps,
                "discrepancy": independent_steps-cached_steps,
                "interpretation": "event_count_difference_not_verified_beat_index_error",
            })
    nearby_problem = bool(screened or unmatched_a or unmatched_b or index_gaps)
    if index_gaps:
        label = "tracker_event_count_disagreement"
    elif unmatched_a or unmatched_b:
        label = "unmatched_tracker_events"
    elif screened:
        label = "matched_event_timing_disagreement"
    else:
        label = "no_local_disagreement_over_25ms_screen"
    return {
        "status": "compared",
        "junction": boundary,
        "analysis_source_start": round(start, 6),
        "analysis_source_end": round(end, 6),
        "local_start_seconds": round(left_time, 6),
        "local_end_seconds": round(right_time, 6),
        "local_screen_threshold_ms": PHASE_LIMIT_MS,
        "prior_agreement_tolerance_ms": LEGACY_MATCH_RADIUS_MS,
        "alignment": alignment,
        "matched_local_events": paired_local,
        "matched_over_25ms": screened,
        "matched_over_25ms_but_within_70ms": sum(
            x["over_25ms_screen"] and x["within_previous_70ms_tolerance"]
            for x in observed
        ),
        "local_match_residuals": observed,
        "unmatched_cached_events": unmatched_a,
        "unmatched_independent_events": unmatched_b,
        "index_step_disagreements": index_gaps,
        "local_disagreement": nearby_problem,
        "classification": label,
        "beat_grid_edit_authorized": False,
    }


def diagnose(
    plan: dict[str, Any], independent: dict[str, Any], cache_root: Path,
) -> dict[str, Any]:
    if plan.get("plan_id") != independent.get("plan_id"):
        raise ValueError("Plan ID mismatch")
    tracks = plan.get("tracks") or []
    reports = {int(t["transition"]): t for t in independent.get("transitions") or []}
    memo: dict[int, dict[str, Any] | None] = {}
    output = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": number}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            row["status"] = "no_rejected_rhythmic_candidate"
            output.append(row)
            continue
        earlier = reports.get(number) or {}
        if earlier.get("status") != "audited" or earlier.get("candidate_signature") != candidate.get("signature"):
            row["status"] = "missing_or_mismatched_independent_beat_evidence"
            output.append(row)
            continue
        a, b = edge.get("from_index"), edge.get("to_index")
        if not isinstance(a,int) or not isinstance(b,int) or not (0 <= a < len(tracks) and 0 <= b < len(tracks)):
            row["status"] = "invalid_track_indices"
            output.append(row)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            ra, rb = float(candidate["render_rate_a"]), float(candidate["render_rate_b"])
            if not all(math.isfinite(v) and v > 0 for v in (bpm, ra, rb)):
                raise ValueError("invalid timing")
        except (KeyError, ValueError, TypeError):
            row["status"] = "incomplete_candidate_timing"
            output.append(row)
            continue
        for k in (a, b):
            if k not in memo:
                memo[k] = full_beat_cache(cache_root, str(tracks[k].get("path") or ""))[0]
        if memo[a] is None or memo[b] is None:
            row["status"] = "full_beat_cache_missing"
            output.append(row)
            continue
        row["candidate_signature"] = candidate.get("signature")
        for deck, index, rate in (("deck_a",a,ra), ("deck_b",b,rb)):
            junctions = (earlier.get(deck) or {}).get("junctions") or []
            analyzed = []
            for j in junctions:
                if j.get("status") != "audited":
                    analyzed.append({"status": "prior_junction_not_audited"})
                    continue
                analyzed.append(compare_junction(
                    memo[index]["beats_seconds"],
                    j.get("independent_beat_seconds") or [],
                    j, expected_period=60*rate/bpm,
                ))
            row[deck] = {
                "junction_count": len(junctions),
                "junctions": analyzed,
                "status": "no_junction" if not junctions else (
                    "compared" if all(x["status"] == "compared" for x in analyzed)
                    else "partial_evidence"),
            }
        row["status"] = ("compared" if all(row[d]["status"] in ("compared","no_junction")
                                         for d in ("deck_a","deck_b"))
                         else "partial_evidence")
        output.append(row)
    return {
        "schema": "saund-local-beat-disagreement-v1",
        "plan_id": plan.get("plan_id"),
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "transitions": output,
        "limitations": [
            "An event-time mismatch above 25ms is a diagnostic screen, NOT a measured cross-deck phase-integrity violation.",
            "Monotone matching can represent tracker insertions/deletions, not proof of musical beat loss or gain.",
            "Independent librosa tracking shares source audio and may share assumptions with the cached detector.",
            "Local results exclude unmatched beats outside the independent source-audio analysis window.",
            "The 70ms prior match tolerance is contextual; it does not authorize phase corrections or beat replacements.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--independent", type=Path, default=Path("output/milestone1-tempo-consistency/independent-beat-downbeat.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/local-beat-disagreements.json"))
    args = p.parse_args()
    report = diagnose(
        json.loads(args.set_plan.read_text(encoding="utf-8")),
        json.loads(args.independent.read_text(encoding="utf-8")),
        args.cache_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [{
            "transition": t["transition"], "status": t["status"],
            "incoming": [{
                "classification": j.get("classification"),
                "local_over_25ms": j.get("matched_over_25ms"),
                "unmatched_cached": len(j.get("unmatched_cached_events") or []),
                "unmatched_independent": len(j.get("unmatched_independent_events") or []),
            } for j in (t.get("deck_b") or {}).get("junctions") or []],
        } for t in report["transitions"]],
    }, indent=2))


if __name__ == "__main__":
    main()
