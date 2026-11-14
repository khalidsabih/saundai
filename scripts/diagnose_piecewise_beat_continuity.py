#!/usr/bin/env python3
"""Read-only piecewise beat-run continuity and rhythmic-phase ambiguity audit.

Separates steady runs (internally regular) from intervening abnormal beat
intervals. A stable run's absolute offset from a fitted reference is NOT
treated as a defect. Compare the change of offset across each discontinuity
and inspect onset support for both original and alternative half-beat phases.
This is descriptive evidence, NOT a proposed production beat replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for loc in (ROOT, ROOT / "src"):
    if str(loc) not in sys.path:
        sys.path.insert(0, str(loc))

import numpy as np
from scripts.diagnose_rejected_beat_grids import full_beat_cache

INTERVAL_TOLERANCE = 0.08
MIN_STABLE_BEATS = 3
AUDIO_RADIUS_S = 0.055
HALF_BEAT_FRACTION = 0.5
HALF_BEAT_AMBIGUITY_MARGIN = 0.15


def _nearby_onset(t: float, events: list[tuple[float, float]]) -> dict[str, float] | None:
    hits = [(abs(t-s), s, p) for s, p in events if abs(t-s) <= AUDIO_RADIUS_S]
    if not hits:
        return None
    distance, s, prominence = min(hits)
    return {"onset_seconds": round(s, 6),
            "distance_ms": round(distance*1000, 3),
            "prominence_ratio": round(prominence, 3)}


def _support(
    beats: np.ndarray, events: list[tuple[float, float]],
    shift: float = 0.0,
) -> dict[str, Any]:
    # Nearest events cannot be reused; one-to-one evidence avoids giving
    # adjacent beat predictions credit for a single onset.
    used: set[int] = set()
    matches = 0
    for t in beats:
        hits = [(abs(float(t)+shift-s), i) for i, (s, _) in enumerate(events)
                if i not in used and abs(float(t)+shift-s) <= AUDIO_RADIUS_S]
        if not hits:
            continue
        _, k = min(hits)
        used.add(k)
        matches += 1
    return {"matched": matches, "tested": len(beats),
            "coverage": round(matches/len(beats), 4) if len(beats) else 0.0}


def inspect_piecewise(
    beat_times: list[float], onset_candidates: list[dict[str, Any]], *,
    start: float, end: float, expected_period: float,
    context_beats: int = 8,
) -> dict[str, Any]:
    full = np.asarray(beat_times, dtype=float)
    if (expected_period <= 0 or not math.isfinite(expected_period)
        or not math.isfinite(start) or not math.isfinite(end) or start >= end
        or len(full) < 8 or not np.all(np.isfinite(full))
        or np.any(np.diff(full) <= 0)):
        return {"status": "invalid_input"}
    l = max(0, int(np.searchsorted(full, start))-context_beats)
    r = min(len(full), int(np.searchsorted(full, end, side="right"))+context_beats)
    grid = full[l:r]
    if len(grid) < 8:
        return {"status": "insufficient_context"}
    events: list[tuple[float, float]] = []
    for item in onset_candidates:
        try:
            t, p = float(item["seconds"]), float(item["prominence_ratio"])
        except (TypeError, KeyError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(p) and p > 0:
            events.append((t, p))
    events.sort()
    intervals = np.diff(grid)
    bad = np.abs(intervals/expected_period-1.0) > INTERVAL_TOLERANCE
    # Partition at every irregular interval. Runs of regular intervals must
    # contain at least three beats to count as stable.
    segments: list[tuple[int, int]] = []
    begin = 0
    for j, unusual in enumerate(bad):
        if unusual:
            if j+1-begin >= MIN_STABLE_BEATS:
                segments.append((begin, j+1))
            begin = j+1
    if len(grid)-begin >= MIN_STABLE_BEATS:
        segments.append((begin, len(grid)))
    runs: list[dict[str, Any]] = []
    for a, b in segments:
        slice_ = grid[a:b]
        if not any(start <= t <= end for t in slice_):
            # Keep contextual runs near the boundary because they can provide
            # a stable reference for the next overlap-local run.
            if slice_[-1] < start-2*expected_period or slice_[0] > end+2*expected_period:
                continue
        local_intervals = np.diff(slice_)
        original = _support(slice_, events)
        plus_half = _support(slice_, events, HALF_BEAT_FRACTION*expected_period)
        minus_half = _support(slice_, events, -HALF_BEAT_FRACTION*expected_period)
        alternative = max(plus_half, minus_half, key=lambda x: x["coverage"])
        margin = abs(original["coverage"]-alternative["coverage"])
        # Suppress confidence when few independent onsets are available or
        # both phases are supported by dense 8th-note/16th-note percussion.
        ambiguity = (
            len(slice_) < 4 or min(original["tested"], alternative["tested"]) < 4
            or margin < HALF_BEAT_AMBIGUITY_MARGIN
            or original["coverage"] < 0.5 and alternative["coverage"] < 0.5
        )
        runs.append({
            "start_global_beat_index": l+a, "end_global_beat_index": l+b-1,
            "start_seconds": round(float(slice_[0]), 6),
            "end_seconds": round(float(slice_[-1]), 6),
            "beat_count": len(slice_),
            "median_interval_ms": round(float(np.median(local_intervals))*1000, 3),
            "original_phase_audio_support": original,
            "half_beat_later_audio_support": plus_half,
            "half_beat_earlier_audio_support": minus_half,
            "half_beat_ambiguity": ambiguity,
            "phase_choice_authorized": False,
        })
    junctions = []
    for left, right in zip(runs, runs[1:]):
        ia, ib = left["end_global_beat_index"], right["start_global_beat_index"]
        # Gap may include >1 skipped irregular interval. Compare observed
        # elapsed time to the number of ORIGINAL beat-index steps to infer
        # the *relative* reference-phase shift, independent of global origin.
        delta_index = ib-ia
        elapsed = float(full[ib]-full[ia])
        difference_ms = (elapsed-delta_index*expected_period)*1000
        bridge = [
            {"left_global_index": i,
             "left_seconds": round(float(full[i]), 6),
             "right_seconds": round(float(full[i+1]), 6),
             "interval_ms": round(float(full[i+1]-full[i])*1000, 3),
             "ratio_to_expected": round(float((full[i+1]-full[i])/expected_period), 4)}
            for i in range(ia, ib)
        ]
        junctions.append({
            "left_run_final_index": ia,
            "right_run_first_index": ib,
            "relative_phase_shift_ms": round(difference_ms, 3),
            "shift_fraction_of_beat": round(difference_ms/(expected_period*1000), 4),
            "half_beat_scale_discontinuity": abs(abs(difference_ms)/(expected_period*1000)-0.5) < 0.10,
            "bridge_intervals": bridge,
            "beat_index_change_authorized": False,
            "interpretation": "phase_change_between_runs_only_not_proof_of_tracking_error",
        })
    # Include every abnormal original interval inside overlap even if
    # there are not stable runs on both sides.
    abnormalities = []
    for i, unusual in enumerate(bad):
        if not unusual or grid[i+1] < start or grid[i] > end:
            continue
        abnormalities.append({
            "left_global_index": l+i, "right_global_index": l+i+1,
            "left_seconds": round(float(grid[i]), 6),
            "right_seconds": round(float(grid[i+1]), 6),
            "interval_ms": round(float(intervals[i])*1000, 3),
            "interval_ratio": round(float(intervals[i]/expected_period), 4),
            "left_audio": _nearby_onset(float(grid[i]), events),
            "right_audio": _nearby_onset(float(grid[i+1]), events),
        })
    return {
        "status": "audited",
        "expected_period_ms": round(expected_period*1000, 4),
        "beat_indices_in_overlap": int(np.sum((grid >= start)&(grid <= end))),
        "stable_runs": runs,
        "junctions": junctions,
        "abnormal_intervals": abnormalities,
        "stable_run_count": len(runs),
        "abnormal_interval_count": len(abnormalities),
        "evidence_status": "no_stable_run_comparison" if len(runs) < 2 else "comparisons_available_not_verified",
        "beat_grid_change_authorized": False,
    }


def diagnose(plan: dict[str, Any], sequence_audit: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    if plan.get("plan_id") != sequence_audit.get("plan_id"):
        raise ValueError("Plan ID mismatch")
    tracks = plan.get("tracks") or []
    audited = {int(x["transition"]): x for x in sequence_audit.get("transitions") or []}
    memo: dict[int, dict[str, Any] | None] = {}
    output: list[dict[str, Any]] = []
    for n, edge in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": n}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        cand = rejected.get("candidate") or {}
        if not cand:
            row["status"] = "no_rejected_rhythmic_candidate"
            output.append(row)
            continue
        evidence = audited.get(n) or {}
        if evidence.get("status") != "audited" or evidence.get("candidate_signature") != cand.get("signature"):
            row["status"] = "missing_or_mismatched_independent_audio"
            output.append(row)
            continue
        ai, bi = edge.get("from_index"), edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            row["status"] = "invalid_track_indices"
            output.append(row)
            continue
        try:
            bpm = float(cand["target_bpm"])
            ra, rb = float(cand["render_rate_a"]), float(cand["render_rate_b"])
            dur = cand.get("transition_seconds")
            dur = float(dur) if dur is not None else 4*float(cand["transition_bars"])*60/bpm
            start_a = float(cand["exit_time_a"])
            start_b = float(cand["entry_time_b"])+float(cand.get("phase_nudge_seconds_b") or 0)
            if not all(math.isfinite(x) and x > 0 for x in (bpm, ra, rb, dur)):
                raise ValueError("nonpositive timing")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            row["status"] = "incomplete_timing"
            output.append(row)
            continue
        for index in (ai, bi):
            if index not in memo:
                memo[index] = full_beat_cache(cache_root, str(tracks[index].get("path") or ""))[0]
        if memo[ai] is None or memo[bi] is None:
            row["status"] = "missing_full_beat_cache"
            output.append(row)
            continue
        row["candidate_signature"] = cand.get("signature")
        for deck, index, start, rate in (
            ("deck_a", ai, start_a, ra), ("deck_b", bi, start_b, rb),
        ):
            events = (evidence.get(deck) or {}).get("audio_onset_candidates_in_window") or []
            row[deck] = inspect_piecewise(
                memo[index]["beats_seconds"], events, start=start,
                end=start+dur*rate, expected_period=60*rate/bpm,
            )
        row["reported_max_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
        row["status"] = "audited" if all(row[key]["status"] == "audited" for key in ("deck_a", "deck_b")) else "insufficient_context"
        output.append(row)
    return {
        "schema": "saund-piecewise-beat-phase-continuity-v1",
        "plan_id": plan.get("plan_id"),
        "production_changes": False,
        "beat_replacement_authorized": False,
        "phase_integrity_limit_changed": False,
        "transitions": output,
        "limitations": [
            "Audio onset events include subdivisions and non-beat percussion; match coverage does not verify beat identity.",
            "Audio events are only provided inside the audited overlap; nearby context runs have incomplete onset evidence.",
            "A run's absolute phase offset from a fitted clock is not treated as a beat error.",
            "The reference source period comes from planner BPM and playback rates, not an independently confirmed audio grid.",
            "Half-beat ambiguity can persist even when a phase has stronger onset coverage.",
            "No beat-grid/index/downbeat, cache, planning, rendering or safety behavior is changed.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--sequence-audit", type=Path, default=Path("output/milestone1-tempo-consistency/beat-sequence-interpretation.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/piecewise-beat-continuity.json"))
    args = p.parse_args()
    report = diagnose(
        json.loads(args.set_plan.read_text(encoding="utf-8")),
        json.loads(args.sequence_audit.read_text(encoding="utf-8")),
        args.cache_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [{
            "transition": x["transition"], "status": x["status"],
            "deck_a_runs": (x.get("deck_a") or {}).get("stable_run_count"),
            "deck_b_runs": (x.get("deck_b") or {}).get("stable_run_count"),
            "deck_a_abnormal": (x.get("deck_a") or {}).get("abnormal_interval_count"),
            "deck_b_abnormal": (x.get("deck_b") or {}).get("abnormal_interval_count"),
        } for x in report["transitions"]],
    }, indent=2))


if __name__ == "__main__":
    main()
