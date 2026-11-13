#!/usr/bin/env python3
"""Independent audio-onset tempo validation at rejected DJ transition regions.

Read-only: derives tempo from waveform spectral flux/autocorrelation, NOT
cached beat timestamps, cached global BPM, or the original beat_track output.
A periodicity estimate is not ground-truth beat annotation. No model, cache
or audio is changed; disagreement alone must not overwrite global BPM.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import librosa
import numpy as np

from scripts.diagnose_rejected_beat_grids import full_beat_cache

SAMPLE_RATE = 22050
HOP_LENGTH = 128
TEMPO_MIN = 105.0
TEMPO_MAX = 140.0


def tempo_candidates_from_envelope(
    envelope: np.ndarray, *, frames_per_second: float,
    minimum_bpm: float = TEMPO_MIN, maximum_bpm: float = TEMPO_MAX,
) -> dict[str, Any]:
    """Rank interpolated ACF peaks in a constrained beat-rate range.

    The caller must report the range: restricting it to nearby DJ tempos
    eliminates half/double-time aliases but can also miss true tempos.
    """
    x = np.asarray(envelope, dtype=np.float64).reshape(-1)
    if len(x) < 32 or frames_per_second <= 0 or minimum_bpm <= 0 or maximum_bpm <= minimum_bpm:
        return {"status": "insufficient_evidence"}
    x = x - np.mean(x)
    if np.linalg.norm(x) < 1e-8:
        return {"status": "silent_or_flat_envelope"}
    # FFT autocorrelation, normalized by the number of overlapping frames.
    nfft = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:len(x)]
    counts = np.arange(len(x), 0, -1, dtype=float)
    acf /= counts
    low = max(2, int(math.floor(60 * frames_per_second / maximum_bpm)))
    high = min(len(x) - 3, int(math.ceil(60 * frames_per_second / minimum_bpm)))
    if high <= low:
        return {"status": "insufficient_evidence"}
    peaks = []
    for lag in range(low + 1, high):
        if acf[lag] < acf[lag - 1] or acf[lag] < acf[lag + 1]:
            continue
        denom = acf[lag - 1] - 2 * acf[lag] + acf[lag + 1]
        offset = 0.0 if abs(denom) < 1e-12 else 0.5 * (acf[lag - 1] - acf[lag + 1]) / denom
        fractional_lag = lag + max(-0.5, min(0.5, offset))
        bpm = 60 * frames_per_second / fractional_lag
        if minimum_bpm <= bpm <= maximum_bpm:
            peaks.append({"bpm": round(float(bpm), 4), "acf_strength": round(float(acf[lag] / max(acf[0], 1e-12)), 5)})
    peaks.sort(key=lambda item: item["acf_strength"], reverse=True)
    if not peaks:
        return {"status": "no_periodicity_peak"}
    return {
        "status": "estimated",
        "best_bpm": peaks[0]["bpm"],
        "relative_acf_strength": peaks[0]["acf_strength"],
        "alternatives": peaks[:5],
        "tempo_search_range_bpm": [minimum_bpm, maximum_bpm],
        "method": "audio_spectral_flux_autocorrelation_interpolated_lag",
        "limitations": "Tempo periodicity is not validated beat onset; competing meters, syncopation, and sparse percussion can mislead.",
    }


def estimate_audio_window(path: Path, *, start: float, end: float) -> dict[str, Any]:
    if end - start < 8:
        return {"status": "window_too_short"}
    # Load only selected audio range. Does not depend on previously cached beat
    # frames and does not run librosa.beat.beat_track.
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, offset=max(0.0, start), duration=max(0.0, end - max(0.0, start)))
    if len(y) < 8 * sr:
        return {"status": "insufficient_decoded_audio"}
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.mean)
    estimate = tempo_candidates_from_envelope(onset, frames_per_second=sr / HOP_LENGTH)
    return {**estimate, "window_start_seconds": round(start, 5), "window_end_seconds": round(end, 5), "decoded_samples": len(y)}


def _number(x: Any) -> float | None:
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def _local_grid_bpm(beats: list[float], *, start: float, end: float) -> dict[str, Any]:
    intervals = [
        b-a for a, b in zip(beats, beats[1:])
        if start <= a and b <= end and 0.30 <= b-a <= 0.8
    ]
    if len(intervals) < 6:
        return {"status": "insufficient_grid"}
    arr = np.array(intervals)
    return {
        "status": "available",
        "median_grid_bpm": round(60 / float(np.median(arr)), 5),
        "span_grid_bpm": round(60 * len(arr) / float(np.sum(arr)), 5),
        "sample_intervals": len(arr),
        "interval_cv": round(float(np.std(arr) / np.mean(arr)), 5),
        "note": "This comes from the original cached beat timestamps; not independent audio evidence.",
    }


def validate(plan: dict[str, Any], *, cache_root: Path, source_dir: Path | None = None) -> dict[str, Any]:
    tracks = plan.get("tracks") or []
    windows: dict[int, list[dict[str, float]]] = {}
    for edge in plan.get("transitions") or []:
        ai, bi = edge.get("from_index"), edge.get("to_index")
        if not isinstance(ai, int) or not isinstance(bi, int) or ai not in range(len(tracks)) or bi not in range(len(tracks)):
            continue
        rejected = ((edge.get("plan") or {}).get("metadata") or {}).get("decision_trace_rejected_phase") or {}
        c = rejected.get("candidate") or {}
        bpm = _number(c.get("target_bpm"))
        bars = _number(c.get("transition_bars"))
        duration = _number(c.get("transition_seconds"))
        if duration is None and bpm and bars:
            duration = bars * 4 * 60 / bpm
        xa, xb = _number(c.get("exit_time_a")), _number(c.get("entry_time_b"))
        ra, rb = _number(c.get("render_rate_a")), _number(c.get("render_rate_b"))
        if None in (xa, xb, ra, rb, duration) or min(ra, rb, duration) <= 0:
            continue
        for idx, start, rate in ((ai, xa, ra), (bi, xb, rb)):
            windows.setdefault(idx, []).append({
                "start": max(0.0, start),
                "end": start + duration * rate,
                "transition": len(windows.get(idx, [])) + 1,
            })
    results = []
    for idx, track in enumerate(tracks):
        if idx not in windows:
            continue
        stored_path = Path(str(track.get("path", "")))
        audio_path = source_dir / stored_path.name if source_dir else stored_path
        analysis, paths = full_beat_cache(cache_root, str(stored_path))
        item = {
            "track": stored_path.name,
            "saved_global_bpm": _number(track.get("bpm")),
            "audio_path": str(audio_path),
            "audio_available": audio_path.is_file(),
            "full_cached_grid_available": analysis is not None,
            "cached_grid_sources": paths,
            "regions": [],
        }
        for window in windows[idx]:
            start, end = window["start"], window["end"]
            # Pad the window to improve ACF frequency resolution. Check also
            # narrower region so disagreements are visible rather than hidden.
            wide_start = max(0.0, start - 12.0)
            wide_end = end + 12.0
            region = {"overlap_start_seconds": round(start, 6), "overlap_end_seconds": round(end, 6)}
            if analysis is not None:
                region["cached_grid_tempo"] = _local_grid_bpm(analysis["beats_seconds"], start=start, end=end)
            if not audio_path.is_file():
                region["audio_periodicity"] = {"status": "audio_unavailable"}
            else:
                region["audio_periodicity"] = estimate_audio_window(audio_path, start=wide_start, end=wide_end)
                # The unpadded overlap is an independent stability cross-check
                # only when it is long enough for a useful estimate.
                region["overlap_only_periodicity"] = (
                    estimate_audio_window(audio_path, start=start, end=end)
                    if end-start >= 12 else {"status": "too_short_for_cross_check"}
                )
            item["regions"].append(region)
        results.append(item)
    return {
        "schema": "saund-independent-overlap-audio-tempo-v1",
        "plan_id": plan.get("plan_id"),
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "source_of_tempo": "fresh audio-only onset envelope autocorrelation, not cached beats or beat_track",
        "tracks": results,
        "validation_limits": [
            "Automatic tempo extraction from audio is not ground truth.",
            "The ACF search is deliberately restricted to 105–140 BPM and cannot detect true tempos outside it.",
            "The 12-second padding gives context but is not limited exactly to the overlap.",
            "Comparing cached grid with audio periodicity is diagnostic; no global BPM should be changed without independent validation.",
            "The script never writes audio, cache, or production configurations.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-four-track/balanced/set-plan.json"))
    parser.add_argument("--cache-root", type=Path, default=Path(".djai-cache/tracks"))
    parser.add_argument("--source-dir", type=Path, default=None, help="Optional directory containing the four original audio basenames")
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-four-track/independent-audio-tempo.json"))
    args = parser.parse_args()
    report = validate(json.loads(args.set_plan.read_text(encoding="utf-8")), cache_root=args.cache_root, source_dir=args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "tracks": [
            {"name": track["track"], "audio_available": track["audio_available"],
             "best_periodicities": [reg.get("audio_periodicity", {}).get("best_bpm") for reg in track["regions"]]}
            for track in report["tracks"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
