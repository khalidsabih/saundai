#!/usr/bin/env python3
"""Read-only audio-onset alignment audit at rejected phase-lock cue windows.

Refine only observations for comparison; never rewrite analyzed beats, downbeats,
plans, caches, or renderer inputs. Fine-resolution onset peaks are not
ground-truth kick or beat positions, and large apparent gains do not authorize
lowering Beat Integrity thresholds.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import librosa
import numpy as np

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from scripts.reconstruct_phase_drift import reconstruct

SAMPLE_RATE = 22050
HOP_LENGTH = 128
RADIUS_SECONDS = 0.075
MIN_PEAK_RATIO = 1.2


def nearest_supported_peaks(
    beat_seconds: list[float],
    onset_strength: np.ndarray,
    *,
    window_start_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    radius_seconds: float = RADIUS_SECONDS,
    min_peak_ratio: float = MIN_PEAK_RATIO,
) -> list[dict[str, Any]]:
    """One-to-one local onset observations within a capped 75 ms radius.

    A support ratio >1.2 compares the selected peak to median local strength,
    not an audio-level or beat correctness confidence probability.
    """
    env = np.asarray(onset_strength, dtype=float)
    fps = sample_rate / hop_length
    if not len(env) or not np.all(np.isfinite(env)):
        return []
    radius = max(1, round(radius_seconds * fps))
    observations: list[dict[str, Any]] = []
    for i, timestamp in enumerate(beat_seconds):
        center = (timestamp - window_start_seconds) * fps
        j = int(round(center))
        if j - radius < 1 or j + radius >= len(env) - 1:
            continue
        segment = env[j-radius:j+radius+1]
        maxima = [
            ix for ix in range(j-radius+1, j+radius)
            if env[ix] >= env[ix-1] and env[ix] >= env[ix+1]
        ]
        if not maxima:
            continue
        best = max(maxima, key=lambda ix: env[ix])
        baseline = float(np.median(segment))
        ratio = float(env[best] / max(baseline, 1e-6))
        if ratio < min_peak_ratio:
            continue
        # Sub-frame quadratic interpolation only when the peak is concave.
        denominator = float(env[best-1] - 2*env[best] + env[best+1])
        shift = 0.0
        if denominator < -1e-8:
            shift = float(np.clip(0.5*(env[best-1]-env[best+1])/denominator, -0.5, 0.5))
        peak_seconds = window_start_seconds + (best + shift) / fps
        observations.append({
            "beat_index": i,
            "cached_beat_seconds": round(timestamp, 6),
            "audio_onset_peak_seconds": round(peak_seconds, 6),
            "shift_ms": round((peak_seconds-timestamp)*1000, 3),
            "local_peak_to_median_ratio": round(ratio, 3),
        })
    return observations


def audit_window(
    audio_path: Path, beats: list[float], *, start: float, end: float,
    sample_rate: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH,
) -> dict[str, Any]:
    if end <= start or not audio_path.is_file():
        return {"status": "audio_missing_or_invalid_window"}
    # Keep 1s extra context for reliable onset envelope boundaries.
    region_start = max(0.0, start-1.0)
    region_end = end+1.0
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True,
                         offset=region_start, duration=region_end-region_start)
    if len(y) < sr * max(1.0, (end-start)*0.8):
        return {"status": "insufficient_decoded_audio"}
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                         aggregate=np.mean)
    chosen = [float(b) for b in beats if start <= b <= end]
    observations = nearest_supported_peaks(chosen, onset,
        window_start_seconds=region_start, sample_rate=sr, hop_length=hop_length)
    shifts = [float(x["shift_ms"]) for x in observations]
    return {
        "status": "audited",
        "beats_in_overlap": len(chosen),
        "supported_onset_observations": len(observations),
        "support_coverage": round(len(observations)/len(chosen), 4) if chosen else 0.0,
        "median_abs_shift_ms": round(float(np.median(np.abs(shifts))), 3) if shifts else None,
        "p90_abs_shift_ms": round(float(np.percentile(np.abs(shifts), 90)), 3) if shifts else None,
        "max_abs_shift_ms": round(max((abs(x) for x in shifts), default=0), 3),
        "observations": observations,
        "warning": "Onset peaks can follow hi-hats, percussion, or arrangement changes rather than beats. This is an audit, not permission to replace beat timestamps.",
    }


def diagnose(plan: dict[str, Any], cache_root: Path, source_dir: Path | None = None) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    cache: dict[int, tuple[dict[str, Any] | None, list[str]]] = {}
    rows = []
    for index, edge in enumerate(plan.get("transitions") or [], 1):
        ai = edge.get("from_index")
        bi = edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or not (0 <= ai < len(tracks) and 0 <= bi < len(tracks)):
            rows.append({"transition": index, "status": "unresolved_indices"})
            continue
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        cand = rejected.get("candidate") or {}
        if not cand:
            rows.append({"transition": index, "status": "no_rejected_rhythmic_candidate"})
            continue
        for ti in (ai, bi):
            if ti not in cache:
                cache[ti] = full_beat_cache(cache_root, str(tracks[ti].get("path") or ""))
        a, _ = cache[ai]
        b, _ = cache[bi]
        row: dict[str, Any] = {"transition": index, "candidate_signature": cand.get("signature")}
        if a is None or b is None:
            row["status"] = "missing_or_ambiguous_full_cache"
            rows.append(row)
            continue
        try:
            bpm = float(cand["target_bpm"])
            duration = cand.get("transition_seconds")
            duration = float(duration) if duration is not None else float(cand["transition_bars"])*4*60/bpm
            xa, xb = float(cand["exit_time_a"]), float(cand["entry_time_b"])
            ra, rb = float(cand["render_rate_a"]), float(cand["render_rate_b"])
            nudge = cand.get("phase_nudge_seconds_b")
            nudge = float(nudge) if nudge is not None else 0.0
            if min(bpm, duration, ra, rb) <= 0:
                raise ValueError("nonpositive timing")
        except (ValueError, TypeError, KeyError):
            row["status"] = "incomplete_timing"
            rows.append(row)
            continue
        row["nudge_assumed_zero"] = cand.get("phase_nudge_seconds_b") is None
        for key, ti, analysis, start, rate in (("deck_a", ai, a, xa, ra), ("deck_b", bi, b, xb+nudge, rb)):
            stored = Path(tracks[ti].get("path") or "")
            source = (source_dir / stored.name) if source_dir else stored
            row[key] = audit_window(source, analysis["beats_seconds"],
                start=start, end=start+duration*rate)
        # Report saved rejection separately; do NOT claim the observational
        # onset offsets are a counterfactual valid beat grid.
        row["reported_max_residual_ms"] = rejected.get("phase_integrity", {}).get("max_residual_ms")
        row["status"] = "audited" if all(row[key]["status"] == "audited" for key in ("deck_a", "deck_b")) else "audio_unavailable"
        rows.append(row)
    return {
        "schema": "saund-cue-onset-precision-audit-v1",
        "plan_id": plan.get("plan_id"),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "fine_time_step_ms": round(1000*HOP_LENGTH/SAMPLE_RATE, 4),
        "transitions": rows,
        "production_changes": False,
        "limitations": [
            "Peaks in spectral onset strength are not verified musical beat locations.",
            "A peak can be unrelated to a kick and local support ratio is not a correctness probability.",
            "The script deliberately does not replace any original beat timestamp or recalculate a safe-phase verdict.",
            "A missing rejected candidate does not imply good phase lock.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/cue-onset-precision.json"))
    args = parser.parse_args()
    report = diagnose(json.loads(args.set_plan.read_text(encoding="utf-8")), args.cache_root, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()), "statuses": [
        {"transition": x["transition"], "status": x["status"],
         "deck_a_coverage": x.get("deck_a", {}).get("support_coverage"),
         "deck_b_coverage": x.get("deck_b", {}).get("support_coverage")}
        for x in report["transitions"]
    ]}, indent=2))


if __name__ == "__main__":
    main()
