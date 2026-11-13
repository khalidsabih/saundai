"""Conservative global-BPM reconciliation using independent onset periodicity.

Do not rewrite beat timestamps. This module only authorizes changing the
scalar BPM when several audio windows and the full beat-grid span agree.
If they disagree, retain the original estimate and surface diagnostics.
"""
from __future__ import annotations

import numpy as np


def _acf_tempo(onsets: np.ndarray, *, frames_per_second: float, low: float, high: float) -> tuple[float, float] | None:
    x = np.asarray(onsets, dtype=np.float64)
    if len(x) < 256 or not np.all(np.isfinite(x)):
        return None
    x = x - np.mean(x)
    if float(np.std(x)) < 1e-8:
        return None
    lag_min = max(2, int(np.floor(60.0 * frames_per_second / high)))
    lag_max = min(len(x) - 3, int(np.ceil(60.0 * frames_per_second / low)))
    if lag_max <= lag_min:
        return None
    n = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n)
    acf = np.fft.irfft(spectrum * spectrum.conjugate(), n=n)[:len(x)]
    acf /= np.arange(len(x), 0, -1, dtype=np.float64)
    if acf[0] <= 1e-10:
        return None
    peaks: list[tuple[float, float]] = []
    for lag in range(lag_min + 1, lag_max):
        if acf[lag] < acf[lag - 1] or acf[lag] < acf[lag + 1]:
            continue
        denominator = acf[lag-1] - 2*acf[lag] + acf[lag+1]
        offset = 0.0 if abs(denominator) < 1e-12 else 0.5*(acf[lag-1] - acf[lag+1])/denominator
        bpm = 60.0*frames_per_second/(lag + float(np.clip(offset, -0.5, 0.5)))
        if low <= bpm <= high:
            peaks.append((float(bpm), float(acf[lag]/acf[0])))
    return max(peaks, key=lambda item: item[1]) if peaks else None


def reconcile_bpm(
    original_bpm: float,
    beat_times: np.ndarray,
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[float, dict[str, object]]:
    """Return selected BPM and explicit evidence; fail closed on ambiguity.

    Designed for approximately steady 105-140 BPM electronic dance music.
    Never change BPM based on median local intervals alone: the failure we
    observed had a misleading beat-interval median but correct span tempo.
    """
    trace: dict[str, object] = {"original_bpm": round(float(original_bpm), 5), "decision": "retain"}
    beats = np.asarray(beat_times, dtype=np.float64)
    if not 105 <= original_bpm <= 140 or beats.size < 64 or len(onset_envelope) < 1024:
        trace["reason"] = "outside_supported_range_or_insufficient_evidence"
        return original_bpm, trace
    differences = np.diff(beats)
    if np.any(~np.isfinite(differences)) or np.any(differences <= 0):
        trace["reason"] = "invalid_beat_grid"
        return original_bpm, trace
    # Whole-span estimate is resistant to quantized local interval modes.
    grid_bpm = float(60.0*(len(beats)-1)/(beats[-1]-beats[0]))
    trace["full_grid_span_bpm"] = round(grid_bpm, 5)
    if not 105 <= grid_bpm <= 140 or abs(grid_bpm/original_bpm - 1.0) < 0.008:
        trace["reason"] = "no_material_global_disagreement"
        return original_bpm, trace
    if abs(grid_bpm/original_bpm - 1.0) > 0.07:
        trace["reason"] = "large_disagreement_requires_manual_review"
        return original_bpm, trace

    x = np.asarray(onset_envelope, dtype=np.float64)
    frames_per_second = sample_rate / hop_length
    window = int(round(36*frames_per_second))
    # Distributed windows protect against a single local arrangement or break.
    starts = [int(0.15*len(x)), int(0.45*len(x)), int(0.72*len(x))]
    audio = []
    for start in starts:
        segment = x[start:min(len(x), start+window)]
        result = _acf_tempo(segment, frames_per_second=frames_per_second, low=105.0, high=140.0)
        if result is not None:
            audio.append(result)
    trace["audio_windows"] = [{"bpm": round(v, 5), "strength": round(strength, 4)} for v, strength in audio]
    # Require 3 independent pieces of audio evidence. No inferred certainty
    # from the autocorrelation's peak when it is weak or conflicting.
    if len(audio) != 3 or any(strength < 0.5 for _, strength in audio):
        trace["reason"] = "insufficient_independent_audio_confidence"
        return original_bpm, trace
    estimates = [v for v, _ in audio]
    audio_median = float(np.median(estimates))
    trace["audio_median_bpm"] = round(audio_median, 5)
    if max(abs(v/audio_median - 1.0) for v in estimates) > 0.006:
        trace["reason"] = "audio_windows_disagree"
        return original_bpm, trace
    if abs(audio_median/grid_bpm - 1.0) > 0.006:
        trace["reason"] = "audio_and_beat_grid_disagree"
        return original_bpm, trace

    # The beat grid and audio agree; taking their median avoids claiming
    # sub-frame precision not supplied by the original beat timestamps.
    corrected = float(np.median([grid_bpm, audio_median]))
    trace["decision"] = "reconcile"
    trace["reason"] = "three_audio_windows_and_full_beat_grid_span_agree"
    trace["selected_bpm"] = round(corrected, 5)
    return corrected, trace
