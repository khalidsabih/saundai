#!/usr/bin/env python3
"""Read-only phase reconstruction from cached beat arrays and rejected candidates.

Separates (a) the *detected-grid* index-vs-linear-time residual on each deck
from (b) relative phase error under the plan's render rates. A linear-time
reference is a mathematical counterfactual, NOT beat ground truth or proof of
an audible fix. Does not alter models, planning, rendering, or thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from djai.performance.deck import _continuous_beat_position, _unwrapped_beat_numbers


def reconstruct(
    beats_a: list[float], beats_b: list[float], *,
    exit_time_a: float, entry_time_b: float, duration: float,
    bpm: float, rate_a: float, rate_b: float, nudge_b: float = 0.0,
    samples: int = 9,
) -> dict[str, Any]:
    """Replay uncorrected phase coordinate, and decompose from nominal time."""
    if bpm <= 0 or duration <= 0 or min(rate_a, rate_b) <= 0 or samples < 2:
        raise ValueError("Invalid overlap timing")
    beat_period = 60.0 / bpm
    expected_a = beat_period * rate_a
    expected_b = beat_period * rate_b
    nums_a = _unwrapped_beat_numbers(beats_a, expected_source_beat_seconds=expected_a)
    nums_b = _unwrapped_beat_numbers(beats_b, expected_source_beat_seconds=expected_b)
    points: list[dict[str, Any]] = []
    baseline_phase = baseline_nominal = baseline_bias_a = baseline_bias_b = None
    for index in range(samples):
        fraction = index / (samples - 1)
        out = fraction * duration
        sa = exit_time_a + out * rate_a
        sb = entry_time_b + nudge_b + out * rate_b
        pa = _continuous_beat_position(beats_a, sa, expected_source_beat_seconds=expected_a, beat_numbers=nums_a)
        pb = _continuous_beat_position(beats_b, sb, expected_source_beat_seconds=expected_b, beat_numbers=nums_b)
        if pa is None or pb is None:
            points.append({"fraction": round(fraction, 6), "valid": False})
            continue
        # Time-based nominal grid (constant target BPM); initial origin cancels.
        na = (sa - beats_a[0]) / expected_a
        nb = (sb - beats_b[0]) / expected_b
        raw = (pa - pb) * beat_period
        nominal = (na - nb) * beat_period
        bias_a = (pa - na) * beat_period
        bias_b = (pb - nb) * beat_period
        if baseline_phase is None:
            baseline_phase = raw
            baseline_nominal = nominal
            baseline_bias_a = bias_a
            baseline_bias_b = bias_b
        measured = (raw - baseline_phase) * 1000
        nominal_drift = (nominal - baseline_nominal) * 1000
        deviation_a = (bias_a - baseline_bias_a) * 1000
        deviation_b = (bias_b - baseline_bias_b) * 1000
        points.append({
            "fraction": round(fraction, 6),
            "valid": True,
            "source_a_seconds": round(sa, 6),
            "source_b_seconds": round(sb, 6),
            "detected_grid_phase_drift_ms": round(measured, 3),
            "nominal_constant_tempo_drift_ms": round(nominal_drift, 3),
            "deck_a_grid_vs_nominal_change_ms": round(deviation_a, 3),
            "deck_b_grid_vs_nominal_change_ms": round(deviation_b, 3),
            "decomposition_error_ms": round(measured - (nominal_drift + deviation_a - deviation_b), 6),
        })
    valid = [p for p in points if p.get("valid")]
    return {
        "samples": samples,
        "valid_samples": len(valid),
        "expected_source_beat_a_ms": round(expected_a * 1000, 4),
        "expected_source_beat_b_ms": round(expected_b * 1000, 4),
        "phase_points": points,
        "max_abs_detected_grid_drift_ms": round(max((abs(p["detected_grid_phase_drift_ms"]) for p in valid), default=0.0), 3),
        "max_abs_nominal_drift_ms": round(max((abs(p["nominal_constant_tempo_drift_ms"]) for p in valid), default=0.0), 3),
        "max_abs_deck_a_deviation_ms": round(max((abs(p["deck_a_grid_vs_nominal_change_ms"]) for p in valid), default=0.0), 3),
        "max_abs_deck_b_deviation_ms": round(max((abs(p["deck_b_grid_vs_nominal_change_ms"]) for p in valid), default=0.0), 3),
        "decomposition_identity": "grid drift = nominal drift + deck A indexing deviation - deck B indexing deviation",
        "note": "Nominal constant tempo is an idealized mathematical reference, not verified music timing.",
    }


def _value(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def diagnose(plan: dict[str, Any], cache_root: Path) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    rows = []
    memo = {}
    for n, edge in enumerate(plan.get("transitions") or [], 1):
        ai, bi = edge.get("from_index"), edge.get("to_index")
        rejected = (edge.get("plan") or {}).get("metadata", {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        original = rejected.get("phase_integrity") or {}
        row = {"transition": n, "rejected_signature": candidate.get("signature")}
        if not candidate or not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            row["status"] = "missing_candidate_or_indices"
            rows.append(row)
            continue
        for idx in (ai, bi):
            if idx not in memo:
                memo[idx] = full_beat_cache(cache_root, str(tracks[idx].get("path") or ""))
        a, afiles = memo[ai]
        b, bfiles = memo[bi]
        row["cache_sources_a"] = afiles
        row["cache_sources_b"] = bfiles
        if a is None or b is None:
            row["status"] = "missing_or_ambiguous_full_cache"
            rows.append(row)
            continue
        bpm = _value(candidate.get("target_bpm"))
        bars = _value(candidate.get("transition_bars"))
        duration = _value(candidate.get("transition_seconds"))
        if duration is None and bpm is not None and bars is not None and bpm > 0:
            duration = bars * 4 * 60 / bpm
            row["duration_inferred_from_bars"] = True
        else:
            row["duration_inferred_from_bars"] = False
        xa, xb = _value(candidate.get("exit_time_a")), _value(candidate.get("entry_time_b"))
        ra, rb = _value(candidate.get("render_rate_a")), _value(candidate.get("render_rate_b"))
        nudge = _value(candidate.get("phase_nudge_seconds_b"))
        row["nudge_known"] = nudge is not None
        row["nudge_assumed_zero"] = nudge is None
        if None in (xa, xb, ra, rb, duration, bpm) or min(ra, rb, duration, bpm) <= 0:
            row["status"] = "incomplete_candidate_timing"
            rows.append(row)
            continue
        count = len(original.get("points") or []) or 9
        result = reconstruct(
            a["beats_seconds"], b["beats_seconds"],
            exit_time_a=xa, entry_time_b=xb, duration=duration, bpm=bpm,
            rate_a=ra, rate_b=rb, nudge_b=nudge or 0.0, samples=count,
        )
        row.update({
            "status": "reconstructed",
            "target_bpm": bpm,
            "render_rate_a": ra,
            "render_rate_b": rb,
            "nudge_b_seconds": nudge,
            "reported_max_phase_error_ms": original.get("max_phase_error_ms"),
            "reported_max_residual_ms": original.get("max_residual_ms"),
            "reported_residual_limit_ms": original.get("residual_limit_ms"),
            "reconstruction": result,
        })
        logged = [p for p in original.get("points") or [] if "phase_error_ms" in p]
        calculated = [p for p in result["phase_points"] if p.get("valid")]
        if len(logged) == len(calculated):
            discrepancies = [
                abs(float(a1["phase_error_ms"]) - float(b1["detected_grid_phase_drift_ms"]))
                for a1, b1 in zip(logged, calculated)
            ]
            row["max_abs_replay_difference_ms"] = round(max(discrepancies, default=0.0), 3)
            row["replay_matches_logged_1ms"] = max(discrepancies, default=float("inf")) <= 1.0
        else:
            row["max_abs_replay_difference_ms"] = None
            row["replay_matches_logged_1ms"] = None
        row["interpretation"] = (
            "If nominal drift is near zero but detected-grid drift is large, the discrepancy "
            "is in the detected grid's timing or indices relative to constant tempo; this "
            "does not determine whether the detected grid or constant-tempo reference reflects the music."
        )
        rows.append(row)
    return {
        "schema": "saund-phase-drift-decomposition-v1",
        "plan_id": plan.get("plan_id"),
        "transitions": rows,
        "safety_behavior_changed": False,
        "limitations": [
            "No audio ground truth: cannot distinguish actual musical tempo changes from beat detector timing errors.",
            "Old snapshots may omit initial phase nudge or exact duration; compare replay only when supported.",
            "Decomposition is descriptive and does not simulate safety correction or renderer audio.",
            "Original logged trajectory is the authority if a reconstruction differs.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-four-track/balanced/set-plan.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--output", type=Path, default=Path("output/milestone1-four-track/phase-drift-decomposition.json"))
    args = p.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")), args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [
            {"number": x["transition"], "status": x["status"],
             "max_drift_ms": x.get("reconstruction", {}).get("max_abs_detected_grid_drift_ms"),
             "replay_matches_logged_1ms": x.get("replay_matches_logged_1ms")}
            for x in report["transitions"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
