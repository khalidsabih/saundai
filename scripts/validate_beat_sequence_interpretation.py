#!/usr/bin/env python3
"""Read-only audio-backed audit of beat order, gaps, and local indexing.

Independently extracts onset peaks from the actual overlap audio; unlike the
near-beat precision audit, the search is NOT restricted to ±75 ms around
cached beats. Reports competing interval/index interpretations with supporting
onset evidence, without modifying beat grids, caches, plans or renderer.
Neither onset peaks nor an apparent missing beat establish musical ground truth.
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

import librosa
import numpy as np
from scipy.signal import find_peaks

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from scripts.reconstruct_phase_drift import reconstruct

SAMPLE_RATE = 22050
HOP_LENGTH = 128
SHORT_LIMIT = 0.82
LONG_LIMIT = 1.35
MISSING_MIN = 1.65
MISSING_MAX = 2.35
SUPPORT_RADIUS_SECONDS = 0.055
MIN_PROMINENCE_RATIO = 2.0


def onset_events(
    envelope: np.ndarray, *, window_start: float, fps: float,
    beat_period: float,
) -> list[dict[str, float]]:
    """Audio-only onset candidates; no guidance from cached beat positions."""
    values = np.asarray(envelope, dtype=np.float64)
    if values.size < 8 or not np.all(np.isfinite(values)):
        return []
    positive = values[values > 1e-8]
    if not len(positive):
        return []
    floor = float(np.median(positive))
    # Keep 16th-note-scale peaks: hihats may appear, but must not be
    # automatically interpreted as independent quarter-note beats.
    separation = max(2, int(beat_period * fps * 0.19))
    indices, properties = find_peaks(
        values, distance=separation, prominence=max(0.02, floor * 0.4),
    )
    result = []
    for i, prominence in zip(indices, properties["prominences"]):
        strength = float(prominence / max(floor, 1e-6))
        if strength < MIN_PROMINENCE_RATIO:
            continue
        result.append({
            "seconds": round(window_start + float(i) / fps, 6),
            "prominence_ratio": round(strength, 3),
        })
    return result


def _near_event(events: list[dict[str, float]], at: float, radius: float) -> dict[str, float] | None:
    hits = [p for p in events if abs(p["seconds"]-at) <= radius]
    return max(hits, key=lambda p: p["prominence_ratio"]) if hits else None


def inspect_sequence(
    beats: list[float], audio_events: list[dict[str, float]], *,
    start: float, end: float, expected_period: float,
) -> dict[str, Any]:
    """Find anomalous intervals and independent, non-circular support."""
    if expected_period <= 0 or not math.isfinite(expected_period):
        return {"status": "invalid_period"}
    grid = np.asarray(beats, dtype=np.float64)
    if grid.size < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        return {"status": "invalid_grid"}
    first = max(0, int(np.searchsorted(grid, start, side="left"))-1)
    last = min(len(grid)-1, int(np.searchsorted(grid, end, side="right")))
    anomalies: list[dict[str, Any]] = []
    regular: list[float] = []
    for j in range(first, last):
        left, right = float(grid[j]), float(grid[j+1])
        if right < start or left > end:
            continue
        interval = right-left
        ratio = interval/expected_period
        if SHORT_LIMIT <= ratio <= LONG_LIMIT:
            regular.append(interval)
            continue
        event: dict[str, Any] = {
            "left_beat_index": j,
            "left_seconds": round(left, 6),
            "right_seconds": round(right, 6),
            "interval_ms": round(interval*1000, 3),
            "expected_interval_ms": round(expected_period*1000, 3),
            "interval_ratio": round(ratio, 4),
            "left_onset_support": _near_event(audio_events, left, SUPPORT_RADIUS_SECONDS),
            "right_onset_support": _near_event(audio_events, right, SUPPORT_RADIUS_SECONDS),
            "interpretations": [],
        }
        if ratio < SHORT_LIMIT:
            event["interpretations"].append({
                "hypothesis": "extra_or_misplaced_cached_beat",
                "status": "unresolved",
                "reason": "Short beat interval; cannot distinguish extra beat from genuine syncopation or beat-placement error",
            })
        elif MISSING_MIN <= ratio <= MISSING_MAX:
            middle = (left+right)/2
            peak = _near_event(audio_events, middle, SUPPORT_RADIUS_SECONDS)
            event["interpretations"].append({
                "hypothesis": "one_missing_beat_between_cached_events",
                "status": "audio_supported_candidate" if peak else "unresolved",
                "candidate_seconds": round(middle, 6),
                "independent_midpoint_onset": peak,
                "reason": "Long interval near twice the expected period; midpoint onset is only a candidate, not proven beat identity",
            })
        else:
            event["interpretations"].append({
                "hypothesis": "local_beat_timing_or_index_disagreement",
                "status": "unresolved",
                "reason": "Interval outside normal tolerance; independent onset support is not enough to select a corrected beat index",
            })
        anomalies.append(event)
    return {
        "status": "audited",
        "beats_in_region": int(np.count_nonzero((grid >= start) & (grid <= end))),
        "audio_onset_candidates": len(audio_events),
        "expected_period_ms": round(expected_period*1000, 3),
        "regular_intervals": len(regular),
        "anomalous_intervals": len(anomalies),
        "anomalies": anomalies,
        "index_change_authorized": False,
    }


def inspect_audio_window(
    audio_path: Path, beats: list[float], *, start: float,
    end: float, expected_period: float,
) -> dict[str, Any]:
    if end <= start or not audio_path.is_file():
        return {"status": "audio_missing_or_invalid_window"}
    begin = max(0., start-1.0)
    stop = end+1.0
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                         offset=begin, duration=stop-begin)
    if len(y) < sr*max(1.0, 0.8*(end-start)):
        return {"status": "insufficient_audio"}
    envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.mean,
    )
    peaks = onset_events(envelope, window_start=begin,
                         fps=sr/HOP_LENGTH, beat_period=expected_period)
    result = inspect_sequence(beats, peaks, start=start, end=end,
                              expected_period=expected_period)
    result["audio_onset_candidates_in_window"] = [
        p for p in peaks if start <= p["seconds"] <= end
    ]
    return result


def diagnose(plan: dict[str, Any], cache_root: Path,
             source_dir: Path | None = None) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    memo: dict[int, dict[str, Any] | None] = {}
    results: list[dict[str, Any]] = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        item: dict[str, Any] = {"transition": number}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        ai, bi = edge.get("from_index"), edge.get("to_index")
        if not candidate:
            item["status"] = "no_rejected_rhythmic_candidate"
            results.append(item)
            continue
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            item["status"] = "invalid_indices"
            results.append(item)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            ra, rb = float(candidate["render_rate_a"]), float(candidate["render_rate_b"])
            duration = candidate.get("transition_seconds")
            duration = float(duration) if duration is not None else 4*float(candidate["transition_bars"])*60/bpm
            xa, xb = float(candidate["exit_time_a"]), float(candidate["entry_time_b"])
            nudge = float(candidate.get("phase_nudge_seconds_b") or 0.0)
            if not all(math.isfinite(x) and x > 0 for x in (bpm, ra, rb, duration)):
                raise ValueError("Invalid timing")
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            item["status"] = "incomplete_timing"
            results.append(item)
            continue
        for idx in (ai, bi):
            if idx not in memo:
                memo[idx] = full_beat_cache(cache_root, str(tracks[idx].get("path") or ""))[0]
        if memo[ai] is None or memo[bi] is None:
            item["status"] = "full_beat_cache_missing"
            results.append(item)
            continue
        item["candidate_signature"] = candidate.get("signature")
        item["nudge_assumed_zero"] = candidate.get("phase_nudge_seconds_b") is None
        for deck, idx, origin, rate in (("deck_a", ai, xa, ra), ("deck_b", bi, xb+nudge, rb)):
            path = Path(tracks[idx].get("path") or "")
            if source_dir is not None:
                path = source_dir/path.name
            item[deck] = inspect_audio_window(
                path, memo[idx]["beats_seconds"],
                start=origin, end=origin+duration*rate,
                expected_period=60*rate/bpm,
            )
        if item["deck_a"]["status"] == "audited" and item["deck_b"]["status"] == "audited":
            replay = reconstruct(
                memo[ai]["beats_seconds"], memo[bi]["beats_seconds"],
                exit_time_a=xa, entry_time_b=xb, duration=duration,
                bpm=bpm, rate_a=ra, rate_b=rb, nudge_b=nudge,
            )
            item["original_uncorrected_max_phase_drift_ms"] = replay["max_abs_detected_grid_drift_ms"]
            item["original_rejected_residual_ms"] = (rejected.get("phase_integrity") or {}).get("max_residual_ms")
            item["status"] = "audited"
        else:
            item["status"] = "audio_unavailable"
        results.append(item)
    return {
        "schema": "saund-beat-sequence-interpretation-v1",
        "plan_id": plan.get("plan_id"),
        "analysis_resolution_ms": round(1000*HOP_LENGTH/SAMPLE_RATE, 4),
        "transitions": results,
        "production_changes": False,
        "interpretation_only": True,
        "limitations": [
            "Independent onset candidates include percussion subdivisions, hats and syncopation, not verified quarter-note beats.",
            "Anomalous detected intervals may reflect actual rhythmic variation rather than tracker errors.",
            "Proposed missing, extra and misplaced beat hypotheses are never automatically accepted.",
            "No beat insertion, deletion, indexing or time shift is applied to planning, cache, rendering or phase safety.",
            "Original phase residual is only reported, not revalidated under an unproven alternative grid.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/beat-sequence-interpretation.json"))
    args = parser.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")), args.cache_root, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [
            {"transition": x["transition"], "status": x["status"],
             "deck_a_anomalies": x.get("deck_a", {}).get("anomalous_intervals"),
             "deck_b_anomalies": x.get("deck_b", {}).get("anomalous_intervals")}
            for x in report["transitions"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
