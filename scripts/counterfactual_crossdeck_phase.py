#!/usr/bin/env python3
"""Read-only cross-deck phase counterfactual from prior onset-coherence audit.

Replays the *same* cue times, rates, phase nudge, sample count, and bounded
warp correction as the production phase-integrity profile. Independent
onset-coherence scores are only prerequisites for candidate exploration;
they are not beat ground truth. Every proposed timestamp change is checked
against event movement, boundary continuity and existing downbeat anchors.
No source file or cache is modified. A numerical phase pass is NOT an
authorization to change the production beat grid.
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
from djai.performance.deck import _continuous_beat_position, _unwrapped_beat_numbers
from scripts.diagnose_rejected_beat_grids import full_beat_cache

MAX_MOVEMENT_MS = 30.0
MAX_DOWNBEAT_MOVEMENT_MS = 25.0
MAX_JOIN_INTERVAL_ERROR = 0.10
MIN_ONSET_SUPPORT = 0.55
MIN_SCORE_ADVANTAGE = 0.10
MAX_CORRECTION_MS = 45.0
MAX_STEP_MS = 18.0
RESIDUAL_LIMIT_MS = 25.0


def phase_profile(beats_a: list[float], beats_b: list[float], *,
                  exit_a: float, entry_b: float, duration: float,
                  bpm: float, rate_a: float, rate_b: float,
                  nudge_b: float = 0.0, samples: int = 9) -> dict[str, Any]:
    """Recompute *bounded* profile; does not use the 25ms limit to fit grids."""
    if min(duration, bpm, rate_a, rate_b) <= 0 or samples < 5:
        return {"status": "invalid_timing"}
    period = 60.0 / bpm
    n_a = _unwrapped_beat_numbers(beats_a, expected_source_beat_seconds=period*rate_a)
    n_b = _unwrapped_beat_numbers(beats_b, expected_source_beat_seconds=period*rate_b)
    baseline: float | None = None
    previous = 0.0
    raw_max = 0.0
    residual_max = 0.0
    points: list[dict[str, float]] = []
    for i in range(samples):
        frac = i / (samples-1)
        sa = exit_a + frac*duration*rate_a
        sb = entry_b + nudge_b + frac*duration*rate_b
        pa = _continuous_beat_position(
            beats_a, sa, expected_source_beat_seconds=period*rate_a, beat_numbers=n_a
        )
        pb = _continuous_beat_position(
            beats_b, sb, expected_source_beat_seconds=period*rate_b, beat_numbers=n_b
        )
        if pa is None or pb is None:
            return {"status": "missing_phase_coordinate", "sample": i}
        error = (pa-pb)*period
        if baseline is None:
            baseline = error
        drift = error-baseline
        correction = float(np.clip(-drift, -MAX_CORRECTION_MS/1000, MAX_CORRECTION_MS/1000))
        correction = float(np.clip(correction, previous-MAX_STEP_MS/1000,
                                   previous+MAX_STEP_MS/1000))
        previous = correction
        residual = drift+correction
        raw_max = max(raw_max, abs(drift)*1000)
        residual_max = max(residual_max, abs(residual)*1000)
        points.append({"fraction": round(frac, 6),
                       "raw_phase_error_ms": round(drift*1000, 3),
                       "bounded_correction_ms": round(correction*1000, 3),
                       "residual_ms": round(residual*1000, 3)})
    return {
        "status": "replayed", "samples": len(points),
        "max_raw_phase_error_ms": round(raw_max, 3),
        "max_bounded_residual_ms": round(residual_max, 3),
        "numeric_residual_below_25ms": residual_max <= RESIDUAL_LIMIT_MS,
        "points": points,
        "warning": "Does not validate available source samples, professional audio quality or authenticity of a hypothetical grid.",
    }


def hypothetical_incoming(
    beats: list[float], downbeats: list[float], *,
    cue_start: float, cue_end: float, expected_period: float,
    hypothesis: dict[str, Any],
) -> dict[str, Any]:
    """Create a hypothetical copy and reject unsupported/unsafe transformations.

    Candidate is a steady clock *derived from target BPM*, NOT independent
    audio tempo. The onset-coherence audit is its separate audio evidence.
    Only incoming deck is touched. Beat indexing and downbeats are retained.
    """
    original = np.asarray(beats, dtype=float)
    if len(original) < 6 or np.any(~np.isfinite(original)) or np.any(np.diff(original) <= 0):
        return {"status": "invalid_beat_grid"}
    if hypothesis.get("name") != "steady_pulse_at_independent_tempo" or hypothesis.get("status") != "scored":
        return {"status": "missing_steady_candidate"}
    if float(hypothesis.get("onset_support_score") or 0.0) < MIN_ONSET_SUPPORT:
        return {"status": "insufficient_independent_onset_support"}
    l = max(0, int(np.searchsorted(original, cue_start))-1)
    r = min(len(original), int(np.searchsorted(original, cue_end, side="right"))+1)
    region = original[l:r].copy()
    if len(region) < 6:
        return {"status": "too_few_beats"}
    n = np.arange(len(region), dtype=float)
    origin = float(np.median(region-n*expected_period))
    fitted = origin+n*expected_period
    moves = np.abs(fitted-region)*1000
    result: dict[str, Any] = {
        "max_displacement_ms": round(float(np.max(moves)), 3),
        "median_displacement_ms": round(float(np.median(moves)), 3),
        "beats_in_region": len(region),
        "original_beats_untouched": True,
        "proposed_beats": None,
    }
    # ALL events must remain within the conservative 30ms cap. A 100ms
    # indexing glitch cannot be smoothed away under a favorable score.
    if np.max(moves) > MAX_MOVEMENT_MS:
        result["status"] = "movement_exceeds_limit"
        return result
    # First/last event are boundary anchors; shifts large enough to introduce
    # a local discontinuity at either side make the alternative ineligible.
    candidate = original.copy()
    candidate[l:r] = fitted
    left_join = candidate[l]-candidate[l-1] if l else expected_period
    right_join = candidate[r]-candidate[r-1] if r < len(candidate) else expected_period
    if any(abs(x/expected_period-1) > MAX_JOIN_INTERVAL_ERROR
           for x in (left_join, right_join)):
        result["status"] = "boundary_interval_discontinuity"
        return result
    if np.any(np.diff(candidate) <= 0):
        result["status"] = "nonmonotonic_candidate"
        return result
    # Verify every *existing* downbeat in the covered section continues to
    # coincide with the corresponding beat after the change.
    for downbeat in downbeats:
        if not cue_start <= downbeat <= cue_end:
            continue
        j = int(np.argmin(np.abs(original-downbeat)))
        if abs(original[j]-downbeat) > 0.035:
            result["status"] = "downbeat_not_associated_with_original_beat"
            return result
        if abs(candidate[j]-downbeat)*1000 > MAX_DOWNBEAT_MOVEMENT_MS:
            result["status"] = "downbeat_anchor_disagreement"
            return result
    result["status"] = "eligible_for_numeric_counterfactual_only"
    result["original_beats_untouched"] = True
    result["proposed_beats"] = candidate.tolist()
    return result


def diagnose(plan: dict[str, Any], coherence: dict[str, Any],
             cache_root: Path) -> dict[str, Any]:
    if plan.get("plan_id") != coherence.get("plan_id"):
        raise ValueError("Plan ID mismatch")
    tracks = plan.get("tracks") or []
    prior = {int(x["transition"]): x for x in coherence.get("transitions") or []}
    cache: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": number}
        comparison = prior.get(number) or {}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        cand = rejected.get("candidate") or {}
        if not cand:
            row["status"] = "no_rejected_rhythmic_candidate"
            rows.append(row)
            continue
        if comparison.get("status") != "evaluated" or comparison.get("candidate_signature") != cand.get("signature"):
            row["status"] = "missing_or_mismatched_coherence_evidence"
            rows.append(row)
            continue
        ai, bi = edge.get("from_index"), edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            row["status"] = "invalid_track_indices"
            rows.append(row)
            continue
        try:
            bpm = float(cand["target_bpm"])
            ra, rb = float(cand["render_rate_a"]), float(cand["render_rate_b"])
            duration = cand.get("transition_seconds")
            duration = float(duration) if duration is not None else 4*float(cand["transition_bars"])*60/bpm
            exit_a, enter_b = float(cand["exit_time_a"]), float(cand["entry_time_b"])
            nudge = float(cand.get("phase_nudge_seconds_b") or 0.0)
            samples = len((rejected.get("phase_integrity") or {}).get("points") or []) or 9
            if not all(math.isfinite(x) and x > 0 for x in (bpm, ra, rb, duration)):
                raise ValueError("invalid candidate")
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            row["status"] = "incomplete_candidate_timing"
            rows.append(row)
            continue
        for ti in (ai, bi):
            if ti not in cache:
                cache[ti] = full_beat_cache(cache_root, str(tracks[ti].get("path") or ""))[0]
        a, b = cache[ai], cache[bi]
        if a is None or b is None:
            row["status"] = "missing_full_beat_cache"
            rows.append(row)
            continue
        args = dict(exit_a=exit_a, entry_b=enter_b, duration=duration,
                    bpm=bpm, rate_a=ra, rate_b=rb, nudge_b=nudge, samples=samples)
        baseline = phase_profile(a["beats_seconds"], b["beats_seconds"], **args)
        row["candidate_signature"] = cand.get("signature")
        row["baseline"] = baseline
        row["logged_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
        row["replay_vs_logged_residual_difference_ms"] = (
            round(abs(baseline["max_bounded_residual_ms"] - float(row["logged_residual_ms"])), 3)
            if baseline.get("status") == "replayed" and row["logged_residual_ms"] is not None else None
        )
        deck = comparison.get("deck_b") or {}
        hypotheses = {h.get("name"): h for h in deck.get("hypotheses") or []}
        best = deck.get("highest_scoring_hypothesis")
        steady = hypotheses.get("steady_pulse_at_independent_tempo") or {}
        original_score = float((hypotheses.get("original_event_grid") or {}).get("score") or 0)
        score_advantage = float(steady.get("score") or 0)-original_score
        if best != "steady_pulse_at_independent_tempo" or score_advantage < MIN_SCORE_ADVANTAGE:
            row["status"] = "incoming_steady_hypothesis_not_preferred"
            row["score_advantage"] = round(score_advantage, 5)
            rows.append(row)
            continue
        test = hypothetical_incoming(
            b["beats_seconds"], b.get("downbeats_seconds") or [],
            cue_start=enter_b+nudge, cue_end=enter_b+nudge+duration*rb,
            expected_period=rb*60.0/bpm, hypothesis=steady,
        )
        row["hypothetical_grid_evidence"] = {k:v for k,v in test.items() if k!="proposed_beats"}
        if test["status"] != "eligible_for_numeric_counterfactual_only":
            row["status"] = "hypothetical_grid_rejected"
            rows.append(row)
            continue
        hypothetical = phase_profile(a["beats_seconds"], test["proposed_beats"], **args)
        row["hypothetical"] = hypothetical
        row["status"] = "numeric_counterfactual_evaluated"
        if hypothetical.get("status") == "replayed" and baseline.get("status") == "replayed":
            row["change_in_bounded_residual_ms"] = round(
                hypothetical["max_bounded_residual_ms"] - baseline["max_bounded_residual_ms"], 3
            )
        rows.append(row)
    return {
        "schema": "saund-crossdeck-phase-counterfactual-v1",
        "plan_id": plan.get("plan_id"),
        "safety_residual_limit_ms": RESIDUAL_LIMIT_MS,
        "numeric_profile_default_limits_ms": {
            "warp_max_correction": MAX_CORRECTION_MS,
            "warp_max_step": MAX_STEP_MS,
        },
        "hypothetical_tempo_independent_of_planner": False,
        "production_changes": False,
        "beat_replacement_authorized": False,
        "transitions": rows,
        "limitations": [
            "The steady-pulse candidate is fitted to original grid indices using the planner tempo, NOT independently estimated audio tempo.",
            "Prior independent audio onsets serve only as support evidence, not ground truth beat/downbeat classification.",
            "Only incoming deck candidates are evaluated; this is not an optimization of both deck grids.",
            "Numerical bounded-phase replay does not prove a safe or musically correct rendered transition.",
            "Production phase gate, source availability, confidence and rendered audio must be independently validated.",
            "No beat positions, downbeats, cached analysis, decisions, or renderer settings were modified.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--coherence", type=Path, default=Path("output/milestone1-tempo-consistency/continuous-beat-phase.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/crossdeck-phase-counterfactual.json"))
    args = p.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")),
                      json.loads(args.coherence.read_text(encoding="utf-8")), args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()),
                      "transitions": [
                          {"number": x["transition"], "status": x["status"],
                           "baseline_residual_ms": x.get("baseline", {}).get("max_bounded_residual_ms"),
                           "hypothetical_residual_ms": x.get("hypothetical", {}).get("max_bounded_residual_ms")}
                          for x in report["transitions"]]}, indent=2))


if __name__ == "__main__":
    main()
