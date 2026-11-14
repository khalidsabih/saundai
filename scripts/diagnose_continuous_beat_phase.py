#!/usr/bin/env python3
"""Read-only competing beat-phase hypotheses on rejected transition windows.

Independent onset candidates come from validate_beat_sequence_interpretation,
NOT from peaks searched near an existing beat. Compare a steady fitted pulse,
original event times, small displacements, and single missed/extra events.
Scores are relative explanatory diagnostics, NOT calibrated likelihoods,
ground-truth beat validation, or an alternative Beat Integrity verdict.
No plan/cache/beat/index/renderer/safety data is mutated.
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

MAX_EVENT_SHIFT_FRACTION = 0.16
SUPPORT_RADIUS_FRACTION = 0.10
MISSING_INTERVAL_RANGE = (1.65, 2.35)
EXTRA_COMBINED_RANGE = (0.85, 1.15)
# Do not call tiny score differences evidence of a unique explanation.
MIN_WINNER_MARGIN = 0.10


def _independent_support(
    predicted: np.ndarray, audio_events: list[dict[str, Any]], period: float,
) -> tuple[float, int]:
    """Score one-to-one matching to independent onsets within 10% of a beat.

    A dense hi-hat pattern may support multiple equally plausible pulses.
    Only capped event prominence counts, and all predictions share the same
    reference onset list and score definition.
    """
    events = sorted(
        (float(e["seconds"]), float(e["prominence_ratio"]))
        for e in audio_events
        if "seconds" in e and "prominence_ratio" in e
        and math.isfinite(float(e["seconds"]))
        and math.isfinite(float(e["prominence_ratio"]))
    )
    if len(predicted) == 0:
        return 0.0, 0
    taken: set[int] = set()
    total = 0.0
    matches = 0
    radius = SUPPORT_RADIUS_FRACTION * period
    for beat in predicted:
        candidates = [
            (i, s, prominence)
            for i, (s, prominence) in enumerate(events)
            if i not in taken and abs(s - beat) <= radius
        ]
        if not candidates:
            continue
        i, t, strength = max(
            candidates,
            key=lambda item: math.exp(-0.5 * ((item[1] - beat) / (radius / 2)) ** 2)
                             * min(1.0, math.log1p(max(item[2], 0)) / math.log1p(8)),
        )
        taken.add(i)
        matches += 1
        total += math.exp(-0.5 * ((t - beat) / (radius / 2)) ** 2) * min(
            1.0, math.log1p(max(strength, 0)) / math.log1p(8)
        )
    return float(total / len(predicted)), matches


def _hypothesis(
    name: str, times: np.ndarray, reference: np.ndarray, period: float,
    onset_events: list[dict[str, Any]], *, edit_count: int = 0,
    hypothesis_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(times) < 4 or np.any(np.diff(times) <= 0):
        return {"name": name, "status": "invalid_monotonicity"}
    intervals = np.diff(times) / period
    onset_score, matches = _independent_support(times, onset_events, period)
    interval_median_error = float(np.median(np.abs(intervals - 1.0)))
    interval_p90_error = float(np.percentile(np.abs(intervals - 1.0), 90))
    # A slightly different grid length is allowed only in explicitly named
    # hypotheses. Penalize removal/insertion, even if it fits onsets better.
    # Do not punish onset-aligned fractional global shifts through an
    # arbitrary absolute zero, but do penalize changes in event placement.
    if len(times) == len(reference):
        displacement = float(np.median(np.abs(times - reference)) / period)
    else:
        displacement = 0.0
    score = (
        onset_score
        - 0.70 * interval_median_error
        - 0.35 * interval_p90_error
        - 0.25 * displacement
        - 0.13 * edit_count
    )
    return {
        "name": name,
        "status": "scored",
        "score": round(float(score), 5),
        "onset_support_score": round(onset_score, 5),
        "matched_onsets": matches,
        "predicted_beats": len(times),
        "median_interval_error_fraction": round(interval_median_error, 5),
        "p90_interval_error_fraction": round(interval_p90_error, 5),
        "median_displacement_ms": round(displacement * period * 1000, 3),
        "edit_count": edit_count,
        "detail": hypothesis_detail or {},
    }


def evaluate_sequence(
    beat_times: list[float], audio_events: list[dict[str, Any]], *,
    start: float, end: float, expected_period: float,
) -> dict[str, Any]:
    """Compare hypotheses over complete overlap, including edge context.

    Only original beats in the overlap plus one beat of context on each side
    are used. Both endpoints must be observed; never extrapolate a missing
    beat into the audio at the edge and imply it was verified.
    """
    beat_grid = np.asarray(beat_times, dtype=np.float64)
    if (
        end <= start or expected_period <= 0 or not math.isfinite(expected_period)
        or beat_grid.size < 6 or np.any(~np.isfinite(beat_grid))
        or np.any(np.diff(beat_grid) <= 0)
    ):
        return {"status": "invalid_input"}
    l = max(0, int(np.searchsorted(beat_grid, start))-1)
    r = min(len(beat_grid), int(np.searchsorted(beat_grid, end, side="right"))+1)
    original = beat_grid[l:r].copy()
    if len(original) < 6:
        return {"status": "insufficient_beat_context", "events": len(original)}

    # Fit the steady clock to the beat sequence using a robust median phase,
    # not by minimizing a cross-deck phase-lock residual.
    n = np.arange(len(original), dtype=float)
    origin = float(np.median(original - n * expected_period))
    ideal = origin + n * expected_period
    baseline = _hypothesis("original_event_grid", original, original, expected_period, audio_events)
    hypotheses = [baseline, _hypothesis("steady_pulse_at_independent_tempo", ideal,
                                       original, expected_period, audio_events)]

    moved = original.copy()
    movement = np.abs(original - ideal)
    can_move = movement <= MAX_EVENT_SHIFT_FRACTION * expected_period
    # Displaced event interpretation is constrained; do not shift the first
    # and last observed beats, which anchor the overlap's indexing.
    can_move[0] = False
    can_move[-1] = False
    moved[can_move] = ideal[can_move]
    hypotheses.append(_hypothesis("bounded_event_displacement", moved,
                                  original, expected_period, audio_events,
                                  hypothesis_detail={"moved_events": int(np.sum(can_move))}))

    missing_candidates = []
    extra_candidates = []
    for j, interval in enumerate(np.diff(original)):
        fraction = float(interval / expected_period)
        if MISSING_INTERVAL_RANGE[0] <= fraction <= MISSING_INTERVAL_RANGE[1]:
            new_t = float((original[j] + original[j+1]) / 2)
            proposed = np.insert(original, j+1, new_t)
            hyp = _hypothesis("single_missing_beat", proposed, original,
                              expected_period, audio_events, edit_count=1,
                              hypothesis_detail={"insert_after_local_index": j,
                                                 "candidate_seconds": round(new_t, 6)})
            missing_candidates.append(hyp)
        if j >= 1 and j + 1 < len(original)-1 and fraction < 0.85:
            # Only consider removing an interior event if the combined
            # neighbor span matches a normal period.
            combined = (original[j+1] - original[j-1]) / expected_period
            if EXTRA_COMBINED_RANGE[0] <= combined <= EXTRA_COMBINED_RANGE[1]:
                hyp = _hypothesis("single_extra_beat", np.delete(original, j),
                                  original, expected_period, audio_events,
                                  edit_count=1,
                                  hypothesis_detail={"remove_local_index": j})
                extra_candidates.append(hyp)
    if missing_candidates:
        hypotheses.append(max(missing_candidates, key=lambda h: h["score"]))
    if extra_candidates:
        hypotheses.append(max(extra_candidates, key=lambda h: h["score"]))
    valid = sorted((h for h in hypotheses if h["status"] == "scored"),
                   key=lambda h: h["score"], reverse=True)
    margin = valid[0]["score"]-valid[1]["score"] if len(valid)>1 else 0.0
    # Even a clear score winner cannot be promoted to ground truth. The
    # same audio-onset evidence can support multiple phase-offset grids.
    return {
        "status": "evaluated",
        "original_beat_count_with_context": len(original),
        "expected_period_ms": round(expected_period*1000, 4),
        "onset_candidates": len(audio_events),
        "hypotheses": hypotheses,
        "highest_scoring_hypothesis": valid[0]["name"],
        "runner_up_score_margin": round(margin, 5),
        "interpretation": "ambiguous" if margin < MIN_WINNER_MARGIN else "score_preference_only",
        "beat_correction_authorized": False,
        "note": "Scores are heuristic, not calibrated likelihood; an independent audio-onset sequence is not a verified beat grid.",
    }


def diagnose(plan: dict[str, Any], interpretation: dict[str, Any],
             cache_root: Path) -> dict[str, Any]:
    if plan.get("plan_id") != interpretation.get("plan_id"):
        raise ValueError("Plan ID does not match independent onset interpretation")
    tracks = plan.get("tracks") or []
    audit = {int(x["transition"]): x for x in interpretation.get("transitions", [])}
    cache: dict[int, dict[str, Any] | None] = {}
    rows = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        result: dict[str, Any] = {"transition": number}
        source = audit.get(number) or {}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            result["status"] = "no_rejected_rhythmic_candidate"
            rows.append(result)
            continue
        if source.get("status") != "audited" or source.get("candidate_signature") != candidate.get("signature"):
            result["status"] = "missing_or_mismatched_independent_audio_audit"
            rows.append(result)
            continue
        ai, bi = edge.get("from_index"), edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            result["status"] = "invalid_indices"
            rows.append(result)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            rate_a, rate_b = float(candidate["render_rate_a"]), float(candidate["render_rate_b"])
            duration = candidate.get("transition_seconds")
            duration = float(duration) if duration is not None else 4*float(candidate["transition_bars"])*60/bpm
            exit_a = float(candidate["exit_time_a"])
            enter_b = float(candidate["entry_time_b"]) + float(candidate.get("phase_nudge_seconds_b") or 0.0)
            if not all(math.isfinite(x) and x > 0 for x in (bpm, rate_a, rate_b, duration)):
                raise ValueError("invalid overlap")
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            result["status"] = "invalid_candidate_timing"
            rows.append(result)
            continue
        for idx in (ai, bi):
            if idx not in cache:
                cache[idx] = full_beat_cache(cache_root, str(tracks[idx].get("path") or ""))[0]
        if cache[ai] is None or cache[bi] is None:
            result["status"] = "missing_full_beat_cache"
            rows.append(result)
            continue
        result["candidate_signature"] = candidate.get("signature")
        for deck, idx, begin, rate in (
            ("deck_a", ai, exit_a, rate_a), ("deck_b", bi, enter_b, rate_b),
        ):
            evidence = (source.get(deck) or {}).get("audio_onset_candidates_in_window") or []
            result[deck] = evaluate_sequence(
                cache[idx]["beats_seconds"], evidence,
                start=begin, end=begin+duration*rate,
                expected_period=60.0*rate/bpm,
            )
        result["original_rejected_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
        result["status"] = "evaluated" if all(result[k]["status"] == "evaluated"
                                               for k in ("deck_a", "deck_b")) else "insufficient_evidence"
        rows.append(result)
    return {
        "schema": "saund-continuous-beat-phase-coherence-v1",
        "plan_id": plan.get("plan_id"),
        "hypothesis_score_is_calibrated_probability": False,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "transitions": rows,
        "limitations": [
            "Strong onset matches can reflect subdivision instruments rather than true beats.",
            "Score preference is not authority to edit beat positions or indices.",
            "This evaluates each deck independently, not a rendered cross-deck phase-lock counterfactual.",
            "An unconstrained steady pulse is a mathematical reference, not acoustic beat ground truth.",
            "No original safety residual is recomputed or weakened.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    parser.add_argument("--sequence-audit", type=Path, default=Path("output/milestone1-tempo-consistency/beat-sequence-interpretation.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/continuous-beat-phase.json"))
    args = parser.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")),
                      json.loads(args.sequence_audit.read_text(encoding="utf-8")), args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [
            {"transition": r["transition"], "status": r["status"],
             "deck_a_preference": r.get("deck_a", {}).get("highest_scoring_hypothesis"),
             "deck_b_preference": r.get("deck_b", {}).get("highest_scoring_hypothesis")}
            for r in report["transitions"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
