#!/usr/bin/env python3
"""Read-only second-method beat and downbeat/meter audit around discontinuities.

Separately run librosa's dynamic-programming beat tracker on extended source
audio (not the cached beat times). Compare to onset-ACF tempo and the cached
rhythmic phase, then report candidate 4/4 downbeat accent phases rather than
claiming a downbeat detector has verified musical meter. Export short,
unaltered source-audio WAV excerpts for HUMAN listening only when requested.

No stored beat grid, downbeat, transition, cache, render or safety gate changes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from scripts.diagnose_rejected_beat_grids import full_beat_cache
from djai.music.tempo_consistency import _acf_tempo

SAMPLE_RATE = 22050
HOP_LENGTH = 256
CONTEXT_BARS = 8
BEATS_PER_BAR = 4
MATCH_RADIUS_MS = 70.0
MIN_TRACKED_BEATS = 8
HALF_BEAT_AMBIGUITY_FRACTION = 0.12
LOW_PASS_HZ = 180.0
CLIP_SECONDS = 12.0


def compare_times(reference: list[float], detected: list[float], *, radius_ms: float = MATCH_RADIUS_MS) -> dict[str, Any]:
    """One-to-one matching; a detected beat cannot support two predictions."""
    refs = sorted(float(x) for x in reference)
    dets = sorted(float(x) for x in detected)
    unused = set(range(len(dets)))
    differences = []
    for r in refs:
        hits = [(abs(dets[i]-r), i) for i in unused if abs(dets[i]-r)*1000 <= radius_ms]
        if not hits:
            continue
        distance, i = min(hits)
        unused.remove(i)
        differences.append(distance*1000)
    return {
        "reference_count": len(refs), "detected_count": len(dets),
        "matches": len(differences),
        "coverage": round(len(differences)/len(refs), 4) if refs else 0.0,
        "median_abs_distance_ms": round(float(np.median(differences)), 3) if differences else None,
    }


def compare_pulse_phases(
    beats: list[float], tracked: list[float], *,
    period: float,
) -> dict[str, Any]:
    original = compare_times(beats, tracked)
    later = compare_times([t+period/2 for t in beats], tracked)
    earlier = compare_times([t-period/2 for t in beats], tracked)
    best_other = max(later["coverage"], earlier["coverage"])
    return {
        "original": original,
        "half_beat_later": later,
        "half_beat_earlier": earlier,
        "half_beat_ambiguous": abs(original["coverage"]-best_other) < HALF_BEAT_AMBIGUITY_FRACTION,
        "original_minus_best_half_beat_coverage": round(original["coverage"]-best_other, 4),
        "phase_choice_authorized": False,
    }


def four_beat_accent_hypotheses(
    beat_positions_seconds: list[float], low_onset: np.ndarray, *,
    window_start: float, frames_per_second: float,
) -> dict[str, Any]:
    """Report four possible metrical accents, never assert actual downbeat.

    Aggregate a smoothed low-frequency onset envelope at tracked beats.
    Strongest of four modulo-4 bins is NOT independent proof of 4/4 meter.
    """
    beat_positions = np.asarray(beat_positions_seconds, dtype=float)
    env = np.asarray(low_onset, dtype=float)
    if len(beat_positions) < 16 or len(env) < 8:
        return {"status": "insufficient_beat_context", "downbeat_verified": False}
    strengths = []
    for beat in beat_positions:
        index = int(round((beat-window_start)*frames_per_second))
        if index < 0 or index >= len(env):
            strengths.append(0.0)
        else:
            strengths.append(float(max(0.0, np.max(env[max(0, index-2):min(len(env), index+3)]))))
    x = np.asarray(strengths, dtype=float)
    scores = [float(np.mean(x[k::4])) for k in range(4)]
    order = sorted(range(4), key=lambda k: scores[k], reverse=True)
    top = scores[order[0]]
    second = scores[order[1]]
    return {
        "status": "accent_hypotheses_only",
        "candidate_metre": "four_beat_cycle_assumption_not_verified",
        "modulo_four_scores": [round(s, 5) for s in scores],
        "highest_accent_bin": int(order[0]),
        "highest_vs_second_ratio": round(top/max(second, 1e-9), 4),
        "ambiguous": second <= 1e-9 or top/max(second, 1e-9) < 1.3,
        "downbeat_verified": False,
    }


def run_independent_tracker(
    mono: np.ndarray, *, sr: int, start: float, target_bpm: float,
) -> dict[str, Any]:
    if len(mono) < sr*5:
        return {"status": "insufficient_audio"}
    hop = HOP_LENGTH
    # Beat tracker is independent of original detected beat timestamps.
    envelope = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    low = sosfilt(butter(4, LOW_PASS_HZ, btype="low", fs=sr, output="sos"), mono)
    low_env = librosa.onset.onset_strength(y=low, sr=sr, hop_length=hop)
    if len(envelope) < 32 or not np.any(envelope > 0):
        return {"status": "no_onset_evidence"}
    # The planner tempo is NOT passed as a start_bpm to the tracker. The
    # ACF source is independent too, but both methods share the same audio.
    try:
        tempo, frames = librosa.beat.beat_track(
            onset_envelope=envelope, sr=sr, hop_length=hop,
            trim=False, units="frames",
        )
    except (ValueError, RuntimeError) as exc:
        return {"status": "beat_tracker_unavailable", "reason": str(exc)}
    tempo = float(np.ravel(np.asarray(tempo))[0])
    positions = [
        round(start+float(v)*hop/sr, 6)
        for v in np.asarray(frames, dtype=int)
    ]
    acf = _acf_tempo(
        envelope, frames_per_second=sr/hop,
        low=max(65.0, target_bpm*0.6),
        high=min(205.0, target_bpm*1.6),
    )
    return {
        "status": "tracked" if len(positions) >= MIN_TRACKED_BEATS else "too_few_tracked_beats",
        "librosa_estimated_bpm": round(tempo, 4),
        "acf_estimated_bpm": round(acf[0], 4) if acf else None,
        "acf_peak_strength": round(acf[1], 4) if acf else None,
        "tracked_beat_seconds": positions,
        "low_onset_envelope": low_env,
        "hop_length": hop,
    }


def analyze_junction(
    audio_path: Path, beats: list[float], junction: dict[str, Any], *,
    expected_period: float, excerpts_dir: Path | None = None,
    excerpt_name: str = "junction",
) -> dict[str, Any]:
    left, right = junction.get("left_run_final_index"), junction.get("right_run_first_index")
    if not isinstance(left, int) or not isinstance(right, int) or left < 0 or right >= len(beats) or left >= right:
        return {"status": "invalid_junction_indices"}
    if expected_period <= 0 or not math.isfinite(expected_period):
        return {"status": "invalid_expected_period"}
    grid = np.asarray(beats, dtype=float)
    if np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        return {"status": "invalid_grid"}
    if not audio_path.is_file():
        return {"status": "source_audio_unavailable"}
    context = CONTEXT_BARS * BEATS_PER_BAR
    first = max(0, left-context+1)
    last = min(len(beats), right+context)
    begin = max(0.0, float(grid[first])-expected_period)
    finish = float(grid[last-1])+expected_period
    mono, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True,
                            offset=begin, duration=finish-begin)
    if len(mono) < sr*8:
        return {"status": "insufficient_source_audio"}
    tracked = run_independent_tracker(mono, sr=sr, start=begin, target_bpm=60/expected_period)
    if tracked["status"] not in ("tracked", "too_few_tracked_beats"):
        return tracked
    detected = tracked.pop("tracked_beat_seconds")
    low = tracked.pop("low_onset_envelope")
    beat_reference = [float(x) for x in grid[first:last] if begin <= x <= finish]
    result = {
        "status": "audited" if len(detected) >= MIN_TRACKED_BEATS else "insufficient_independent_beats",
        "source_start": round(begin, 6),
        "source_end": round(finish, 6),
        "junction": {"left_index": left, "right_index": right,
                     "relative_phase_shift_ms": junction.get("relative_phase_shift_ms")},
        "independent_tracker": tracked,
        "original_grid_vs_independent_tracker": compare_pulse_phases(
            beat_reference, detected, period=expected_period),
        "meter_accent_hypotheses": four_beat_accent_hypotheses(
            detected, low, window_start=begin, frames_per_second=sr/HOP_LENGTH),
        "independent_beat_seconds": detected,
        "production_changes": False,
    }
    # Only produce an UNALTERED audio excerpt for optional listening. Never
    # synthesize a new beat grid or re-render a purportedly corrected mix.
    if excerpts_dir is not None:
        center = (float(grid[left])+float(grid[right]))/2
        clip_start = max(begin, center-CLIP_SECONDS/2)
        clip_end = min(finish, clip_start+CLIP_SECONDS)
        a = max(0, int(round((clip_start-begin)*sr)))
        b = min(len(mono), int(round((clip_end-begin)*sr)))
        if b > a:
            excerpts_dir.mkdir(parents=True, exist_ok=True)
            out = excerpts_dir / (excerpt_name+".wav")
            sf.write(str(out), mono[a:b], sr, subtype="PCM_16")
            result["listening_excerpt"] = {
                "path": str(out),
                "source_audio_unchanged_except_mono_resampling_for_listening": True,
                "start_seconds": round(begin+a/sr, 6),
                "end_seconds": round(begin+b/sr, 6),
                "not_a_corrected_render": True,
            }
    return result


def diagnose(
    plan: dict[str, Any], piecewise: dict[str, Any], cache_root: Path,
    *, source_dir: Path | None = None, excerpts_dir: Path | None = None,
) -> dict[str, Any]:
    if plan.get("plan_id") != piecewise.get("plan_id"):
        raise ValueError("Plan ID mismatch")
    tracks = plan.get("tracks") or []
    prior = {int(t["transition"]): t for t in piecewise.get("transitions") or []}
    memo: dict[int, dict[str, Any] | None] = {}
    rows = []
    for number, edge in enumerate(plan.get("transitions") or [], 1):
        item: dict[str, Any] = {"transition": number}
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        candidate = rejected.get("candidate") or {}
        if not candidate:
            item["status"] = "no_rejected_rhythmic_candidate"
            rows.append(item)
            continue
        older = prior.get(number) or {}
        if older.get("status") != "audited" or older.get("candidate_signature") != candidate.get("signature"):
            item["status"] = "missing_or_mismatched_piecewise_evidence"
            rows.append(item)
            continue
        a, b = edge.get("from_index"), edge.get("to_index")
        if not isinstance(a, int) or not isinstance(b, int) or not (0 <= a < len(tracks) and 0 <= b < len(tracks)):
            item["status"] = "invalid_indices"
            rows.append(item)
            continue
        try:
            bpm = float(candidate["target_bpm"])
            rate_a = float(candidate["render_rate_a"])
            rate_b = float(candidate["render_rate_b"])
            if min(bpm, rate_a, rate_b) <= 0:
                raise ValueError("nonpositive")
        except (ValueError, KeyError, TypeError):
            item["status"] = "incomplete_timing"
            rows.append(item)
            continue
        for idx in (a, b):
            if idx not in memo:
                memo[idx] = full_beat_cache(cache_root, str(tracks[idx].get("path") or ""))[0]
        if memo[a] is None or memo[b] is None:
            item["status"] = "missing_full_beat_cache"
            rows.append(item)
            continue
        item["candidate_signature"] = candidate.get("signature")
        for deck, idx, rate in (("deck_a", a, rate_a), ("deck_b", b, rate_b)):
            path = Path(tracks[idx].get("path") or "")
            if source_dir is not None:
                path = source_dir/path.name
            junctions = (older.get(deck) or {}).get("junctions") or []
            item[deck] = {
                "junction_count": len(junctions),
                "junctions": [
                    analyze_junction(
                        path, memo[idx]["beats_seconds"], j,
                        expected_period=60*rate/bpm,
                        excerpts_dir=excerpts_dir,
                        excerpt_name=f"transition{number}-{deck}-junction{i+1}",
                    ) for i,j in enumerate(junctions)
                ],
                "status": "no_junction" if not junctions else "audited",
            }
            if any(x["status"] != "audited" for x in item[deck]["junctions"]):
                item[deck]["status"] = "partial_evidence"
        item["status"] = "audited" if all(item[d]["status"] in ("audited", "no_junction")
                                             for d in ("deck_a","deck_b")) else "partial_evidence"
        rows.append(item)
    return {
        "schema": "saund-independent-beat-downbeat-audit-v1",
        "plan_id": plan.get("plan_id"),
        "methods": ["librosa_dynamic_programming_beat_tracker",
                    "independent_spectral_onset_autocorrelation_tempo",
                    "four_beat_low_frequency_accent_hypotheses"],
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "downbeat_verified": False,
        "phase_integrity_limit_changed": False,
        "transitions": rows,
        "limitations": [
            "Librosa beat tracking and ACF use the same audio, so they are not statistically independent measurement sources.",
            "A four-beat low-band onset accent is not independent musical meter or downbeat verification.",
            "The beat tracker can select half/double tempo or offbeat pulses; manual listening and source-aware cross-validation remain necessary.",
            "Listening WAV excerpts are uncorrected mono resampled source audio, not A/B renders of beat-grid alternatives.",
            "No production changes or phase integrity re-evaluation are performed.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-plan", type=Path, default=Path("output/milestone1-tempo-consistency/balanced/set-plan.json"))
    p.add_argument("--piecewise", type=Path, default=Path("output/milestone1-tempo-consistency/piecewise-beat-continuity.json"))
    p.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--excerpts-dir", type=Path, help="Optional uncorrected WAV excerpts for manual listening")
    p.add_argument("--output", type=Path, default=Path("output/milestone1-tempo-consistency/independent-beat-downbeat.json"))
    args = p.parse_args()
    report = diagnose(
        json.loads(args.set_plan.read_text(encoding="utf-8")),
        json.loads(args.piecewise.read_text(encoding="utf-8")),
        args.cache_root, source_dir=args.source_dir, excerpts_dir=args.excerpts_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "transitions": [
            {"transition": x["transition"], "status": x["status"],
             "deck_a_junctions": x.get("deck_a", {}).get("junction_count"),
             "deck_b_junctions": x.get("deck_b", {}).get("junction_count")}
            for x in report["transitions"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
