#!/usr/bin/env python3
"""Independent, extended-context rhythmic evidence around beat-run junctions.

Read-only verification of the saved piecewise discontinuities using decoded
source audio. Extract independently detected low-frequency ('kick-weighted')
and high-frequency ('hat-weighted') onset candidates over at least four
bars of context on each side when available. Compare observed cached beat
phase with half-beat alternatives and report ambiguities. A filtered
spectral onset is NOT an instrument classifier or musical beat ground truth.
Never alter beat grids, downbeats, plans, caches or phase safety.
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
from scipy.signal import butter, sosfilt

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from scripts.validate_beat_sequence_interpretation import onset_events

SAMPLE_RATE = 22050
HOP_LENGTH = 128
BARS_PER_SIDE = 4
BEATS_PER_BAR = 4
MIN_TESTED_BEATS = 6
MATCH_RADIUS_MS = 55.0
MIN_SUPPORT_COVERAGE = 0.55
AMBIGUITY_MARGIN = 0.20
LOW_PASS_HZ = 180.0
HIGH_PASS_HZ = 1800.0


def phase_support(
    predictions: list[float], events: list[dict[str, Any]], *, shift: float = 0.0,
) -> dict[str, Any]:
    """One-to-one matching of predicted beat positions to onset events."""
    event_times = sorted(
        (float(e["seconds"]), float(e["prominence_ratio"]))
        for e in events
        if math.isfinite(float(e.get("seconds", float("nan"))))
        and math.isfinite(float(e.get("prominence_ratio", float("nan"))))
        and float(e["prominence_ratio"]) > 0
    )
    taken = set()
    matched = 0
    total_strength = 0.0
    offsets = []
    for beat in predictions:
        options = [(abs(t-(beat+shift)), -strength, i, t, strength)
                   for i, (t, strength) in enumerate(event_times)
                   if i not in taken and abs(t-(beat+shift))*1000 <= MATCH_RADIUS_MS]
        if options:
            distance, _, i, t, strength = min(options)
            taken.add(i)
            matched += 1
            offsets.append(distance*1000)
            total_strength += min(strength, 20.0)
    return {
        "tested": len(predictions),
        "matched": matched,
        "coverage": round(matched/len(predictions), 4) if predictions else 0.0,
        "mean_capped_strength": round(total_strength/matched, 3) if matched else None,
        "median_abs_offset_ms": round(float(np.median(offsets)), 3) if offsets else None,
    }


def compare_phases(
    beats: list[float], band_events: dict[str, list[dict[str, Any]]],
    *, period: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for band in ("low", "high", "full"):
        events = band_events.get(band, [])
        original = phase_support(beats, events)
        ahead = phase_support(beats, events, shift=period/2)
        behind = phase_support(beats, events, shift=-period/2)
        alternate = max(ahead["coverage"], behind["coverage"])
        advantage = original["coverage"]-alternate
        # Full/high-band onsets can reflect subdivisions; never elevate one
        # strong low-band result alone to a verified beat interpretation.
        label = (
            "insufficient_data" if len(beats) < MIN_TESTED_BEATS
            or max(original["coverage"], alternate) < MIN_SUPPORT_COVERAGE
            else "ambiguous" if abs(advantage) < AMBIGUITY_MARGIN
            else "original_phase_onset_preferred" if advantage > 0
            else "half_beat_onset_preferred"
        )
        result[band] = {
            "original": original,
            "half_beat_later": ahead,
            "half_beat_earlier": behind,
            "original_minus_best_half_beat_coverage": round(advantage, 4),
            "interpretation": label,
        }
    return result


def extract_band_onsets(
    audio_path: Path, *, start: float, end: float, beat_period: float,
) -> dict[str, Any]:
    if not audio_path.is_file() or end <= start or beat_period <= 0:
        return {"status": "audio_missing_or_invalid_window"}
    # Load one continuous region to avoid favoring the pre- or post-
    # junction envelope, with additional context for onset filtering.
    begin = max(0.0, start-1.0)
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                         offset=begin, duration=end+1.0-begin)
    if len(y) < max(sr, int((end-start)*sr*0.85)):
        return {"status": "insufficient_decoded_audio"}
    sos_low = butter(4, LOW_PASS_HZ, btype="lowpass", fs=sr, output="sos")
    sos_high = butter(4, HIGH_PASS_HZ, btype="highpass", fs=sr, output="sos")
    channels = {
        "low": sosfilt(sos_low, y),
        "high": sosfilt(sos_high, y),
        "full": y,
    }
    detected = {}
    for name, signal in channels.items():
        envelope = librosa.onset.onset_strength(
            y=signal, sr=sr, hop_length=HOP_LENGTH, aggregate=np.mean,
        )
        candidates = onset_events(envelope, window_start=begin,
                                  fps=sr/HOP_LENGTH, beat_period=beat_period)
        detected[name] = [e for e in candidates if start <= e["seconds"] <= end]
    return {
        "status": "analyzed",
        "audio_start": round(start, 6),
        "audio_end": round(end, 6),
        "analysis_resolution_ms": round(1000*HOP_LENGTH/sr, 4),
        "independent_onset_candidates": {k: len(v) for k, v in detected.items()},
        "events": detected,
    }


def verify_junction(
    beats: list[float], audio_path: Path, junction: dict[str, Any], *,
    expected_period: float, bars_per_side: int = BARS_PER_SIDE,
) -> dict[str, Any]:
    if expected_period <= 0 or not math.isfinite(expected_period):
        return {"status": "invalid_period"}
    ia, ib = junction.get("left_run_final_index"), junction.get("right_run_first_index")
    if (not isinstance(ia, int) or not isinstance(ib, int)
        or ia < 0 or ib >= len(beats) or ia >= ib):
        return {"status": "invalid_junction_indices"}
    positions = np.asarray(beats, dtype=float)
    if np.any(~np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        return {"status": "invalid_beats"}
    count = max(BEATS_PER_BAR, int(bars_per_side)*BEATS_PER_BAR)
    first = max(0, ia-count+1)
    last = min(len(beats), ib+count)
    # Include the complete surrounding beat spans, not just previous
    # overlap's onset candidates.
    left_times = positions[first:ia+1].tolist()
    right_times = positions[ib:last].tolist()
    if len(left_times) < MIN_TESTED_BEATS or len(right_times) < MIN_TESTED_BEATS:
        return {"status": "insufficient_context_beats",
                "left_beats": len(left_times), "right_beats": len(right_times)}
    start = max(0., float(positions[first])-expected_period/2)
    end = float(positions[last-1])+expected_period/2
    audio = extract_band_onsets(audio_path, start=start, end=end,
                                beat_period=expected_period)
    if audio["status"] != "analyzed":
        return {"status": audio["status"]}
    events = audio.pop("events")
    # Compare only events in each side of the junction. A shared event at
    # the discontinuity must not silently support both sides.
    before_end = (float(positions[ia])+float(positions[ia+1]))/2
    after_start = (float(positions[ib-1])+float(positions[ib]))/2
    before_events = {k: [e for e in v if e["seconds"] <= before_end]
                     for k, v in events.items()}
    after_events = {k: [e for e in v if e["seconds"] >= after_start]
                    for k, v in events.items()}
    before = compare_phases(left_times, before_events, period=expected_period)
    after = compare_phases(right_times, after_events, period=expected_period)
    low_before = before["low"]["interpretation"]
    low_after = after["low"]["interpretation"]
    # Explicitly distinguish lack of evidence from a phase flip.
    possible_flip = (
        low_before in ("original_phase_onset_preferred", "half_beat_onset_preferred")
        and low_after in ("original_phase_onset_preferred", "half_beat_onset_preferred")
        and low_before != low_after
    )
    return {
        "status": "verified_audio_context_not_beat_ground_truth",
        "left_run_final_index": ia,
        "right_run_first_index": ib,
        "relative_phase_shift_ms": junction.get("relative_phase_shift_ms"),
        "half_beat_scale_discontinuity": junction.get("half_beat_scale_discontinuity"),
        "left_context_beats": len(left_times),
        "right_context_beats": len(right_times),
        "audio": audio,
        "before": before,
        "after": after,
        "possible_low_band_phase_preference_flip": possible_flip,
        "inference": "ambiguous_or_incomplete" if not possible_flip else "band_specific_phase_preference_flip_not_proof",
        "beat_grid_edit_authorized": False,
    }


def diagnose(
    plan: dict[str, Any], piecewise: dict[str, Any],
    cache_root: Path, source_dir: Path | None = None,
) -> dict[str, Any]:
    if plan.get("plan_id") != piecewise.get("plan_id"):
        raise ValueError("Plan ID mismatch")
    tracks = plan.get("tracks") or []
    evidence = {int(t["transition"]): t for t in piecewise.get("transitions") or []}
    memo: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    for transition, edge in enumerate(plan.get("transitions") or [], 1):
        row: dict[str, Any] = {"transition": transition}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            row["status"] = "no_rejected_rhythmic_candidate"
            rows.append(row)
            continue
        earlier = evidence.get(transition) or {}
        if earlier.get("status") != "audited" or earlier.get("candidate_signature") != candidate.get("signature"):
            row["status"] = "missing_or_mismatched_piecewise_evidence"
            rows.append(row)
            continue
        a, b = edge.get("from_index"), edge.get("to_index")
        if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a < len(tracks) and 0 <= b < len(tracks)):
            row["status"] = "invalid_indices"
            rows.append(row)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            ra, rb = float(candidate["render_rate_a"]), float(candidate["render_rate_b"])
            if not all(math.isfinite(v) and v > 0 for v in (bpm, ra, rb)):
                raise ValueError("nonpositive tempo/rate")
        except (KeyError, ValueError, TypeError):
            row["status"] = "incomplete_candidate_timing"
            rows.append(row)
            continue
        for index in (a, b):
            if index not in memo:
                memo[index] = full_beat_cache(cache_root, str(tracks[index].get("path") or ""))[0]
        if memo[a] is None or memo[b] is None:
            row["status"] = "missing_full_beat_cache"
            rows.append(row)
            continue
        row["candidate_signature"] = candidate.get("signature")
        for deck, idx, rate in (("deck_a", a, ra), ("deck_b", b, rb)):
            stored = Path(tracks[idx].get("path") or "")
            audio = source_dir/stored.name if source_dir else stored
            junctions = (earlier.get(deck) or {}).get("junctions") or []
            row[deck] = {
                "junctions": [
                    verify_junction(memo[idx]["beats_seconds"], audio, j,
                                    expected_period=60*rate/bpm)
                    for j in junctions
                ],
                "junction_count": len(junctions),
                "status": "no_piecewise_junction" if not junctions else "audited",
            }
            if any(j["status"] != "verified_audio_context_not_beat_ground_truth"
                   for j in row[deck]["junctions"]):
                row[deck]["status"] = "audio_or_context_unavailable"
        row["status"] = (
            "audited" if all(row[deck]["status"] in ("audited", "no_piecewise_junction")
                            for deck in ("deck_a", "deck_b")) else "partial_evidence"
        )
        rows.append(row)
    return {
        "schema": "saund-extended-band-rhythm-verification-v1",
        "plan_id": plan.get("plan_id"),
        "conditions": {
            "minimum_bars_per_side": BARS_PER_SIDE,
            "low_pass_hz": LOW_PASS_HZ,
            "high_pass_hz": HIGH_PASS_HZ,
            "onset_match_radius_ms": MATCH_RADIUS_MS,
            "minimum_onset_coverage": MIN_SUPPORT_COVERAGE,
            "ambiguous_coverage_margin": AMBIGUITY_MARGIN,
        },
        "transitions": rows,
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "limitations": [
            "Low/high spectral frequency bands are kick/hat-weighted proxies, not source-separated instrument identities.",
            "Onset matches, even in the low-frequency band, do not certify a musical beat or downbeat.",
            "All source beat times are unchanged; phase comparisons only test nearest onsets at original and half-beat phases.",
            "Half-beat phase preference flips can be arrangement effects and do not authorize editing stored beat indices.",
            "No cross-deck phase profile or rendered audio was recomputed, and the 25 ms safety gate is untouched.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    parser.add_argument("--piecewise", type=Path, default=Path("output/milestone1-tempo-consistency/piecewise-beat-continuity.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/extended-rhythmic-evidence.json"))
    args = parser.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")),
                      json.loads(args.piecewise.read_text(encoding="utf-8")),
                      args.cache_root, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [{"transition": t["transition"], "status": t["status"],
                         "deck_a_junctions": t.get("deck_a", {}).get("junction_count"),
                         "deck_b_junctions": t.get("deck_b", {}).get("junction_count")}
                        for t in report["transitions"]],
    }, indent=2))


if __name__ == "__main__":
    main()
