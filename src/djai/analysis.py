from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hashlib
import json

import math
import os

import librosa
import numpy as np

from .audio import load_audio, rms_dbfs
from .loudness import integrated_lufs, true_peak_dbtp
from .key_detection import estimate_key_from_chroma
from .models import (
    BarFeature,
    KeyEstimate,
    SectionLabel,
    TempoPoint,
    TrackAnalysis,
    TransitionCandidate,
)


ANALYSIS_SAMPLE_RATE = 22_050
HOP_LENGTH = 512
BEATS_PER_BAR = 4
BOUNDARY_CONTEXT_BARS = 4
MAX_CANDIDATES = 12
PHRASE_LENGTH_BARS = 8
SECTION_CONTEXT_BARS = 8
LOCAL_HARMONIC_BARS = 8
LOCAL_SOURCE_BARS = 8
MIN_LOCAL_HARMONIC_FRAMES = 12
MIN_LOCAL_SOURCE_FRAMES = 12
SOURCE_N_FFT = 2048
ANALYSIS_VERSION = "5.2.1"


@dataclass(slots=True)
class _BoundaryFeatures:
    time_seconds: float
    bar_index: int
    available_bars: int
    boundary_strength: float
    phrase_strength: float
    beat_grid_confidence: float
    local_bpm: float
    section_label: SectionLabel
    energy_before: float
    energy_after: float
    rhythm_change: float
    harmonic_change: float
    boundary_peak: float = 0.0
    phrase_alignment: float = 0.0


@dataclass(slots=True)
class _SourceFrameFeatures:
    """Frame-aligned evidence used for candidate-level source analysis.

    The vocal value remains a conservative signal-processing proxy, not a
    stem-separation result. Bass activity is more directly observable from the
    low-frequency power distribution.
    """

    frame_times: np.ndarray
    energy_presence: np.ndarray
    bass_likelihood: np.ndarray
    harmonic_share: np.ndarray
    harmonic_mid_fraction: np.ndarray
    mid_mix_share: np.ndarray
    tonality: np.ndarray
    harmonic_centroid: np.ndarray
    harmonic_bandwidth: np.ndarray
    chroma_concentration: np.ndarray


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _normalized_energy_from_db(dbfs: float) -> float:
    # A pragmatic music-level scale: -36 dBFS ~= quiet, -7 dBFS ~= dense/loud.
    return _clamp((dbfs + 36.0) / 29.0)



def _waveform_peaks(y: np.ndarray, bins: int = 512) -> list[float]:
    values = np.abs(np.asarray(y, dtype=np.float32).reshape(-1))
    if values.size == 0:
        return []
    bins = max(32, min(int(bins), values.size))
    edges = np.linspace(0, values.size, bins + 1, dtype=int)
    peaks = [
        float(np.max(values[edges[index] : max(edges[index] + 1, edges[index + 1])]))
        for index in range(bins)
    ]
    maximum = max(peaks, default=0.0)
    if maximum > 1e-8:
        peaks = [value / maximum for value in peaks]
    return [round(value, 4) for value in peaks]


def _tempo_curve(
    beat_times: np.ndarray,
    global_bpm: float,
    duration: float,
) -> tuple[list[TempoPoint], bool]:
    if beat_times.size < 9:
        return [TempoPoint(0.0, round(global_bpm, 4), 0.25)], False
    intervals = np.diff(beat_times)
    points: list[TempoPoint] = []
    window = 16
    step = 8
    for start in range(0, max(1, intervals.size - 3), step):
        local = intervals[start : start + window]
        local = local[np.isfinite(local) & (local > 1e-5)]
        if local.size < 4:
            continue
        median = float(np.median(local))
        bpm = 60.0 / max(median, 1e-8)
        # Resolve common half/double-time ambiguity against the global estimate.
        options = [bpm * 0.5, bpm, bpm * 2.0]
        bpm = min(options, key=lambda value: abs(value - global_bpm))
        cv = float(np.std(local) / max(median, 1e-8))
        confidence = _clamp(np.exp(-10.0 * cv))
        center_index = min(start + local.size // 2, beat_times.size - 1)
        points.append(
            TempoPoint(
                time_seconds=round(float(beat_times[center_index]), 4),
                bpm=round(float(bpm), 4),
                confidence=round(confidence, 4),
            )
        )
    if not points:
        points = [TempoPoint(0.0, round(global_bpm, 4), 0.25)]
    confident = [point.bpm for point in points if point.confidence >= 0.35]
    if len(confident) < 2:
        return points[:64], False
    median_bpm = float(np.median(confident))
    relative_spread = float(np.std(confident) / max(median_bpm, 1e-8))
    percentile_spread = (
        float(np.percentile(confident, 90) - np.percentile(confident, 10))
        / max(median_bpm, 1e-8)
    )
    return points[:64], bool(relative_spread > 0.025 or percentile_spread > 0.055)


def _estimate_time_signature(
    beat_frames: np.ndarray,
    onset_envelope: np.ndarray,
    beat_grid_confidence: float,
) -> tuple[int, float]:
    """Estimate meter before any bar/downbeat construction.

    A return value of 0 means unknown/irregular. 4/4 remains the conservative
    default for a stable but weakly accented grid, while 3/4 and 6/8 require
    clearer periodic accent evidence.
    """
    if beat_frames.size < 12:
        if beat_grid_confidence < 0.30:
            return 0, 0.12
        return 4, 0.20

    def meter_score(meter: int) -> float:
        phase_scores: list[float] = []
        for phase in range(meter):
            frames = beat_frames[phase::meter]
            frames = frames[frames < onset_envelope.size]
            phase_scores.append(float(np.mean(onset_envelope[frames])) if frames.size else 0.0)
        best = max(phase_scores, default=0.0)
        mean = float(np.mean(phase_scores)) if phase_scores else 0.0
        return max(0.0, (best - mean) / max(abs(best) + abs(mean), 1e-8))

    score_3 = meter_score(3)
    score_4 = meter_score(4)
    score_6 = meter_score(6)
    strongest = max(score_3, score_4, score_6)

    if beat_grid_confidence < 0.28 and strongest < 0.08:
        return 0, _clamp(0.35 * beat_grid_confidence)

    if (
        score_6 > 0.08
        and score_6 > score_4 * 1.15
        and score_6 > score_3 * 1.08
    ):
        meter = 6
        runner_up = max(score_3, score_4)
    elif score_3 > 0.08 and score_3 > score_4 * 1.18 and score_3 >= score_6 * 0.96:
        meter = 3
        runner_up = max(score_4, score_6)
    else:
        meter = 4
        runner_up = max(score_3, score_6)

    selected = {3: score_3, 4: score_4, 6: score_6}[meter]
    contrast = max(0.0, (selected - runner_up) / max(selected, 1e-8))
    confidence = _clamp(0.55 * contrast + 0.45 * beat_grid_confidence)
    return meter, confidence


def _relative_scale(values: np.ndarray, low_percentile: float = 20.0, high_percentile: float = 85.0) -> np.ndarray:
    """Robustly scale a one-dimensional feature inside the current track."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros_like(array)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high - low <= 1e-10:
        return np.full_like(array, 0.5, dtype=np.float64)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _build_source_frame_features(
    *,
    y: np.ndarray,
    sr: int,
    harmonic: np.ndarray,
    percussive: np.ndarray,
    chroma: np.ndarray,
) -> _SourceFrameFeatures:
    """Build frame-level bass and vocal-proxy evidence once per track.

    This intentionally avoids claiming true vocal separation. It combines
    harmonic/percussive balance, mid-band harmonic content, spectral tonality,
    spectral motion, and pitch concentration. A stem or trained vocal model can
    replace this proxy later without changing TransitionCandidate fields.
    """
    mix_stft = librosa.stft(
        np.asarray(y, dtype=np.float32),
        n_fft=SOURCE_N_FFT,
        hop_length=HOP_LENGTH,
    )
    harmonic_stft = librosa.stft(
        np.asarray(harmonic, dtype=np.float32),
        n_fft=SOURCE_N_FFT,
        hop_length=HOP_LENGTH,
    )
    mix_power = np.square(np.abs(mix_stft), dtype=np.float32)
    harmonic_power = np.square(np.abs(harmonic_stft), dtype=np.float32)

    harmonic_rms = librosa.feature.rms(
        y=harmonic,
        frame_length=SOURCE_N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    percussive_rms = librosa.feature.rms(
        y=percussive,
        frame_length=SOURCE_N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    frame_count = min(
        mix_power.shape[1],
        harmonic_power.shape[1],
        harmonic_rms.size,
        percussive_rms.size,
        chroma.shape[1] if chroma.ndim == 2 else 0,
    )
    if frame_count <= 0:
        empty = np.empty(0, dtype=np.float64)
        return _SourceFrameFeatures(
            frame_times=empty,
            energy_presence=empty,
            bass_likelihood=empty,
            harmonic_share=empty,
            harmonic_mid_fraction=empty,
            mid_mix_share=empty,
            tonality=empty,
            harmonic_centroid=empty,
            harmonic_bandwidth=empty,
            chroma_concentration=empty,
        )

    mix_power = np.asarray(mix_power[:, :frame_count], dtype=np.float64)
    harmonic_power = np.asarray(harmonic_power[:, :frame_count], dtype=np.float64)
    harmonic_rms = np.asarray(harmonic_rms[:frame_count], dtype=np.float64)
    percussive_rms = np.asarray(percussive_rms[:frame_count], dtype=np.float64)
    local_chroma = np.asarray(chroma[:, :frame_count], dtype=np.float64)

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=SOURCE_N_FFT)
    audible_mask = (frequencies >= 25.0) & (frequencies <= min(12_000.0, sr / 2.0))
    bass_mask = (frequencies >= 30.0) & (frequencies <= 220.0)
    mid_mask = (frequencies >= 250.0) & (frequencies <= 4_200.0)

    epsilon = 1e-12
    total_power = np.sum(mix_power[audible_mask], axis=0) + epsilon
    harmonic_total = np.sum(harmonic_power[audible_mask], axis=0) + epsilon
    bass_mix = np.sum(mix_power[bass_mask], axis=0) / total_power
    bass_harmonic = np.sum(harmonic_power[bass_mask], axis=0) / total_power
    mid_mix_share = np.sum(mix_power[mid_mask], axis=0) / total_power
    harmonic_mid_fraction = np.sum(harmonic_power[mid_mask], axis=0) / harmonic_total

    harmonic_share = harmonic_rms / np.maximum(
        harmonic_rms + percussive_rms,
        1e-10,
    )

    # Spectral flatness is computed directly on power to avoid another full
    # spectrogram allocation. Low flatness means more tonal/harmonic content.
    log_power = np.log(np.maximum(mix_power, epsilon))
    geometric_mean = np.exp(np.mean(log_power, axis=0))
    arithmetic_mean = np.mean(mix_power, axis=0) + epsilon
    flatness = geometric_mean / arithmetic_mean
    tonality = 1.0 - np.clip(flatness / 0.16, 0.0, 1.0)

    harmonic_centroid = np.sum(
        frequencies[:, None] * harmonic_power,
        axis=0,
    ) / harmonic_total
    harmonic_variance = np.sum(
        np.square(frequencies[:, None] - harmonic_centroid[None, :])
        * harmonic_power,
        axis=0,
    ) / harmonic_total
    harmonic_bandwidth = np.sqrt(np.maximum(harmonic_variance, 0.0))

    chroma_sum = np.sum(local_chroma, axis=0)
    normalized_chroma = local_chroma / np.maximum(chroma_sum[None, :], 1e-10)
    sorted_chroma = np.sort(normalized_chroma, axis=0)
    chroma_concentration = np.sum(sorted_chroma[-3:, :], axis=0)

    frame_db = 10.0 * np.log10(total_power)
    reference_db = float(np.percentile(frame_db, 90.0))
    energy_presence = np.clip(
        (frame_db - (reference_db - 38.0)) / 32.0,
        0.0,
        1.0,
    )

    bass_feature = 0.72 * bass_mix + 0.28 * bass_harmonic
    bass_absolute = np.clip((bass_feature - 0.025) / 0.285, 0.0, 1.0)
    bass_relative = _relative_scale(bass_feature)
    bass_likelihood = np.clip(
        (0.78 * bass_absolute + 0.22 * bass_relative)
        * (0.68 + 0.32 * energy_presence),
        0.0,
        1.0,
    )

    frame_times = librosa.frames_to_time(
        np.arange(frame_count),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    return _SourceFrameFeatures(
        frame_times=np.asarray(frame_times, dtype=np.float64),
        energy_presence=np.asarray(energy_presence, dtype=np.float64),
        bass_likelihood=np.asarray(bass_likelihood, dtype=np.float64),
        harmonic_share=np.clip(harmonic_share, 0.0, 1.0),
        harmonic_mid_fraction=np.clip(harmonic_mid_fraction, 0.0, 1.0),
        mid_mix_share=np.clip(mid_mix_share, 0.0, 1.0),
        tonality=np.clip(tonality, 0.0, 1.0),
        harmonic_centroid=np.nan_to_num(harmonic_centroid, nan=0.0, posinf=0.0, neginf=0.0),
        harmonic_bandwidth=np.nan_to_num(harmonic_bandwidth, nan=0.0, posinf=0.0, neginf=0.0),
        chroma_concentration=np.clip(chroma_concentration, 0.0, 1.0),
    )


def _robust_window_summary(values: np.ndarray, *, upper_weight: float = 0.30) -> float:
    """Summarize intermittent musical activity without one-frame spikes."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    upper = float(np.percentile(array, 75.0))
    mean = float(np.mean(array))
    upper_weight = _clamp(upper_weight)
    return _clamp(
        (0.70 - upper_weight / 2.0) * median
        + upper_weight * upper
        + (0.30 - upper_weight / 2.0) * mean
    )


def _source_window_scores(
    features: _SourceFrameFeatures,
    start_frame: int,
    end_frame: int,
) -> tuple[float, float]:
    """Return local vocal-likelihood and bass-activity scores."""
    frame_count = features.frame_times.size
    if frame_count == 0:
        return 0.0, 0.0
    start = max(0, min(int(start_frame), frame_count - 1))
    end = max(start + 1, min(int(end_frame), frame_count))
    if end - start < MIN_LOCAL_SOURCE_FRAMES:
        half = MIN_LOCAL_SOURCE_FRAMES // 2
        center = (start + end) // 2
        start = max(0, center - half)
        end = min(frame_count, start + MIN_LOCAL_SOURCE_FRAMES)
        start = max(0, end - MIN_LOCAL_SOURCE_FRAMES)

    window = slice(start, end)
    harmonic_share = _robust_window_summary(features.harmonic_share[window])
    harmonic_mid = _robust_window_summary(features.harmonic_mid_fraction[window])
    mid_presence = _robust_window_summary(features.mid_mix_share[window])
    tonality = _robust_window_summary(features.tonality[window])
    chroma_concentration = _robust_window_summary(features.chroma_concentration[window])
    energy_presence = _robust_window_summary(features.energy_presence[window])

    centroid_values = features.harmonic_centroid[window]
    centroid_values = centroid_values[np.isfinite(centroid_values) & (centroid_values > 40.0)]
    if centroid_values.size:
        centroid_median = float(np.median(centroid_values))
        centroid_score = math.exp(-(((centroid_median - 1_850.0) / 2_450.0) ** 2))
        log_centroid = np.log(np.maximum(centroid_values, 40.0))
        centroid_motion = (
            float(np.median(np.abs(np.diff(log_centroid))))
            if log_centroid.size >= 3
            else 0.0
        )
        motion_score = _clamp(centroid_motion / 0.115)
    else:
        centroid_score = 0.0
        motion_score = 0.0

    bandwidth_values = features.harmonic_bandwidth[window]
    bandwidth_values = bandwidth_values[np.isfinite(bandwidth_values) & (bandwidth_values > 0.0)]
    if bandwidth_values.size:
        bandwidth_median = float(np.median(bandwidth_values))
        bandwidth_score = math.exp(-(((bandwidth_median - 1_600.0) / 1_900.0) ** 2))
    else:
        bandwidth_score = 0.0

    harmonic_component = _clamp((harmonic_share - 0.28) / 0.55)
    harmonic_mid_component = _clamp((harmonic_mid - 0.30) / 0.56)
    mid_presence_component = _clamp((mid_presence - 0.10) / 0.42)
    chroma_component = _clamp((chroma_concentration - 0.28) / 0.55)

    # This is a conservative vocal-likelihood proxy. Spectral motion acts as a
    # gate so static pads/leads are less likely to be labeled as prominent vocals.
    vocal_evidence = (
        0.23 * harmonic_component
        + 0.23 * harmonic_mid_component
        + 0.16 * mid_presence_component
        + 0.10 * tonality
        + 0.08 * _clamp(centroid_score)
        + 0.08 * _clamp(bandwidth_score)
        + 0.07 * motion_score
        + 0.05 * chroma_component
    )
    vocal_probability = _clamp((vocal_evidence - 0.30) / 0.52)
    vocal_probability *= 0.50 + 0.50 * motion_score
    vocal_probability *= 0.72 + 0.28 * energy_presence

    bass_activity = _robust_window_summary(
        features.bass_likelihood[window],
        upper_weight=0.22,
    )
    return _clamp(vocal_probability), _clamp(bass_activity)


def _source_characteristics(
    y: np.ndarray,
    sr: int,
    harmonic: np.ndarray,
    percussive: np.ndarray,
    onset_envelope: np.ndarray,
    duration: float,
    *,
    frame_features: _SourceFrameFeatures | None = None,
) -> tuple[float, float, float, float]:
    """Return global source descriptors, reusing frame analysis when available."""
    if frame_features is not None and frame_features.frame_times.size:
        vocal_probability, bass_activity = _source_window_scores(
            frame_features,
            0,
            frame_features.frame_times.size,
        )
    else:
        stft = np.abs(librosa.stft(y, n_fft=SOURCE_N_FFT, hop_length=HOP_LENGTH)) ** 2
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=SOURCE_N_FFT)
        total_power = float(np.sum(stft)) + 1e-12
        bass_power = float(np.sum(stft[frequencies <= 220.0]))
        mid_power = float(np.sum(stft[(frequencies >= 250.0) & (frequencies <= 4_200.0)]))
        bass_activity = _clamp((bass_power / total_power) * 3.2)
        harmonic_power = float(np.mean(np.square(harmonic), dtype=np.float64))
        percussive_power = float(np.mean(np.square(percussive), dtype=np.float64))
        harmonic_share = harmonic_power / max(harmonic_power + percussive_power, 1e-12)
        mid_share = mid_power / total_power
        flatness = float(np.median(librosa.feature.spectral_flatness(y=y)))
        vocal_probability = _clamp(
            0.46 * harmonic_share
            + 0.42 * min(1.0, mid_share * 2.0)
            + 0.12 * (1.0 - min(1.0, flatness * 8.0))
        )

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        roll_percent=0.85,
    )[0]
    rolloff_normalized = _clamp(float(np.median(rolloff)) / max(sr / 2.0, 1.0))
    onset_events = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="time",
    )
    onset_density = float(len(onset_events) / max(duration, 1.0))
    return vocal_probability, bass_activity, rolloff_normalized, onset_density


def _populate_local_source_context(
    candidates: list[TransitionCandidate],
    *,
    features: _SourceFrameFeatures,
    sr: int,
    duration: float,
    global_bpm: float,
    fallback_vocal: float,
    fallback_bass: float,
    beats_per_bar: int = BEATS_PER_BAR,
) -> None:
    """Populate source activity for the actual material after every cue."""
    frame_count = features.frame_times.size
    if frame_count == 0:
        for candidate in candidates:
            candidate.vocal_probability = round(max(0.0001, fallback_vocal), 4)
            candidate.bass_activity = round(max(0.0001, fallback_bass), 4)
        return

    for candidate in candidates:
        bpm = candidate.local_bpm or global_bpm
        meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
        bar_seconds = meter * 60.0 / max(float(bpm), 1.0)
        usable_bars = max(1, min(LOCAL_SOURCE_BARS, int(candidate.available_bars)))
        start_seconds = max(0.0, min(float(candidate.time_seconds), duration))
        end_seconds = min(duration, start_seconds + usable_bars * bar_seconds)
        start_frame = int(
            librosa.time_to_frames(
                start_seconds,
                sr=sr,
                hop_length=HOP_LENGTH,
            )
        )
        end_frame = int(
            librosa.time_to_frames(
                end_seconds,
                sr=sr,
                hop_length=HOP_LENGTH,
            )
        )
        vocal_probability, bass_activity = _source_window_scores(
            features,
            start_frame,
            end_frame,
        )

        # Compatibility currently uses zero as "missing". Preserve a tiny
        # non-zero analyzed value so a genuinely clean cue does not fall back to
        # the whole-track estimate.
        candidate.vocal_probability = round(max(0.0001, vocal_probability), 4)
        candidate.bass_activity = round(max(0.0001, bass_activity), 4)


def _estimate_downbeats(
    beat_frames: np.ndarray,
    onset_envelope: np.ndarray,
    *,
    beats_per_bar: int = BEATS_PER_BAR,
) -> tuple[np.ndarray, float]:
    meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
    if beat_frames.size == 0:
        return np.array([], dtype=int), 0.0
    if beat_frames.size < meter:
        return beat_frames.astype(int), 0.25

    phase_scores: list[float] = []
    for phase in range(meter):
        phase_beats = beat_frames[phase::meter]
        valid = phase_beats[phase_beats < len(onset_envelope)]
        score = float(np.mean(onset_envelope[valid])) if valid.size else 0.0
        phase_scores.append(score)

    order = np.argsort(phase_scores)[::-1]
    best_phase = int(order[0])
    best = float(phase_scores[best_phase])
    second = float(phase_scores[int(order[1])]) if len(order) > 1 else 0.0
    contrast = (best - second) / max(best, 1e-8)
    mean_score = float(np.mean(phase_scores))
    prominence = (best - mean_score) / max(abs(best) + abs(mean_score), 1e-8)
    confidence = _clamp(0.70 * contrast + 0.30 * max(0.0, prominence))
    return beat_frames[best_phase::meter].astype(int), confidence


def _beat_grid_diagnostics(
    beat_times: np.ndarray,
    duration: float,
) -> tuple[float, float]:
    if beat_times.size < 4:
        return 0.0, 1.0
    intervals = np.diff(beat_times)
    intervals = intervals[np.isfinite(intervals) & (intervals > 1e-5)]
    if intervals.size < 3:
        return 0.0, 1.0
    median = float(np.median(intervals))
    cv = float(np.std(intervals) / max(median, 1e-8))
    regularity = float(np.exp(-9.0 * cv))
    expected_beats = max(1.0, duration / median)
    coverage = min(1.0, beat_times.size / expected_beats)
    confidence = _clamp(0.78 * regularity + 0.22 * coverage)
    return confidence, cv


def _local_beat_grid(
    beat_times: np.ndarray,
    center_seconds: float,
    global_bpm: float,
    *,
    window_bars: int = 8,
    beats_per_bar: int = BEATS_PER_BAR,
) -> tuple[float, float]:
    if beat_times.size < 4:
        return global_bpm, 0.0

    beat_seconds = 60.0 / max(global_bpm, 1.0)
    meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
    half_window = max(4.0, window_bars * meter * beat_seconds / 2.0)
    local = beat_times[
        (beat_times >= center_seconds - half_window)
        & (beat_times <= center_seconds + half_window)
    ]
    if local.size < 5:
        nearest = np.argsort(np.abs(beat_times - center_seconds))[:12]
        local = np.sort(beat_times[nearest])
    if local.size < 4:
        return global_bpm, 0.0

    intervals = np.diff(local)
    intervals = intervals[np.isfinite(intervals) & (intervals > 1e-5)]
    if intervals.size < 3:
        return global_bpm, 0.0

    median = float(np.median(intervals))
    local_bpm = 60.0 / max(median, 1e-8)
    cv = float(np.std(intervals) / max(median, 1e-8))
    regularity = float(np.exp(-10.0 * cv))
    relative_error = abs(local_bpm - global_bpm) / max(global_bpm, 1e-8)
    consistency = float(np.exp(-((relative_error / 0.035) ** 2)))
    confidence = _clamp(0.72 * regularity + 0.28 * consistency)
    return float(local_bpm), confidence


def _local_rms_stats(
    frame_times: np.ndarray,
    rms_values: np.ndarray,
    start_seconds: float,
    end_seconds: float,
) -> tuple[float, float, float]:
    mask = (frame_times >= start_seconds) & (frame_times <= end_seconds)
    values = rms_values[mask]
    if values.size == 0:
        return 0.0, 0.0, 0.0

    mean_rms = float(np.mean(values))
    local_db = float(20.0 * np.log10(max(mean_rms, 1e-8)))
    local_energy = _normalized_energy_from_db(local_db)
    coefficient = float(np.std(values) / max(mean_rms, 1e-8))
    stability = _clamp(np.exp(-3.5 * coefficient))

    third = max(1, values.size // 3)
    early = float(np.mean(values[:third]))
    late = float(np.mean(values[-third:]))
    trend = (late - early) / max(mean_rms, 1e-8)
    return local_energy, stability, float(trend)


def _robust_scale(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    median = np.median(matrix, axis=0, keepdims=True)
    mad = np.median(np.abs(matrix - median), axis=0, keepdims=True)
    scaled = (matrix - median) / np.maximum(1.4826 * mad, 1e-5)
    return np.clip(scaled, -4.0, 4.0)


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-10:
        return 0.0
    return _clamp(1.0 - float(np.dot(a, b) / denominator))



def _normalize_chroma_profile(values: np.ndarray) -> np.ndarray:
    """Return a finite, non-negative, L1-normalized 12-bin chroma vector."""
    profile = np.asarray(values, dtype=np.float64).reshape(-1)
    if profile.size != 12:
        return np.zeros(12, dtype=np.float64)
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    profile = np.maximum(profile, 0.0)
    total = float(np.sum(profile))
    if total <= 1e-10:
        return np.zeros(12, dtype=np.float64)
    return profile / total


def _local_harmonic_context(
    *,
    chroma: np.ndarray,
    sr: int,
    cue_seconds: float,
    duration: float,
    bpm: float,
    available_bars: int,
    beats_per_bar: int = BEATS_PER_BAR,
) -> tuple[KeyEstimate | None, list[float], float]:
    """Estimate key, chroma profile, and harmonic stability after a cue.

    The analyzed window follows the cue because this is the material that will
    actually overlap during a transition. Near the end of a track, the window
    is shortened honestly instead of borrowing unrelated earlier material.
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
        return None, [], 0.0

    meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
    bar_seconds = meter * 60.0 / max(float(bpm), 1.0)
    usable_bars = max(1, min(LOCAL_HARMONIC_BARS, int(available_bars)))
    start_seconds = max(0.0, min(float(cue_seconds), duration))
    end_seconds = min(duration, start_seconds + usable_bars * bar_seconds)

    start_frame = int(
        librosa.time_to_frames(
            start_seconds,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )
    end_frame = int(
        librosa.time_to_frames(
            end_seconds,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    )
    start_frame = max(0, min(start_frame, chroma.shape[1] - 1))
    end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
    local = np.asarray(chroma[:, start_frame:end_frame], dtype=np.float64)
    local = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
    local = np.maximum(local, 0.0)

    frame_energy = np.sum(local, axis=0)
    valid = frame_energy > 1e-8
    local = local[:, valid]
    if local.shape[1] < MIN_LOCAL_HARMONIC_FRAMES:
        return None, [], 0.0

    profile = _normalize_chroma_profile(np.mean(local, axis=1))
    if float(np.sum(profile)) <= 1e-8:
        return None, [], 0.0

    local_key = estimate_key_from_chroma(local)

    normalized_frames = local / np.maximum(np.sum(local, axis=0, keepdims=True), 1e-10)
    profile_norm = float(np.linalg.norm(profile))
    frame_norms = np.linalg.norm(normalized_frames, axis=0)
    denominators = np.maximum(profile_norm * frame_norms, 1e-10)
    similarities = np.sum(normalized_frames * profile[:, None], axis=0) / denominators
    similarities = np.clip(similarities, 0.0, 1.0)

    temporal_consistency = float(np.median(similarities))
    lower_consistency = float(np.percentile(similarities, 25.0))
    sorted_profile = np.sort(profile)[::-1]
    pitch_concentration = float(np.sum(sorted_profile[:4]))

    key_confidence = _clamp(float(local_key.confidence))
    stability = _clamp(
        0.42 * temporal_consistency
        + 0.23 * lower_consistency
        + 0.23 * key_confidence
        + 0.12 * pitch_concentration
    )

    return (
        local_key,
        [round(float(value), 6) for value in profile],
        stability,
    )


def _populate_local_harmony(
    candidates: list[TransitionCandidate],
    *,
    chroma: np.ndarray,
    sr: int,
    duration: float,
    global_bpm: float,
    beats_per_bar: int = BEATS_PER_BAR,
) -> None:
    """Populate local harmonic fields for all transition candidates in place."""
    for candidate in candidates:
        local_key, local_chroma, stability = _local_harmonic_context(
            chroma=chroma,
            sr=sr,
            cue_seconds=candidate.time_seconds,
            duration=duration,
            bpm=candidate.local_bpm or global_bpm,
            available_bars=candidate.available_bars,
            beats_per_bar=beats_per_bar,
        )
        stability *= 1.0 - 0.22 * _clamp(candidate.harmonic_change)

        candidate.local_key = local_key
        candidate.local_chroma = local_chroma
        candidate.local_harmonic_stability = round(_clamp(stability), 4)


def _aggregate_bar_features(
    *,
    downbeat_frames: np.ndarray,
    total_frames: int,
    rms_frames: np.ndarray,
    onset_envelope: np.ndarray,
    centroid: np.ndarray,
    bandwidth: np.ndarray,
    percussive_rms: np.ndarray,
    chroma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return feature matrix plus energy, rhythm, and chroma summaries per bar."""
    if downbeat_frames.size == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty((0, 12), dtype=np.float64),
        )

    starts = np.asarray(downbeat_frames, dtype=int)
    ends = np.concatenate([starts[1:], np.array([total_frames], dtype=int)])
    feature_rows: list[np.ndarray] = []
    energies: list[float] = []
    rhythms: list[float] = []
    chroma_rows: list[np.ndarray] = []

    def _slice_mean(values: np.ndarray, start: int, end: int) -> float:
        length = values.shape[-1]
        if length == 0:
            return 0.0
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        return float(np.mean(values[..., start:end]))

    def _slice_std(values: np.ndarray, start: int, end: int) -> float:
        length = values.shape[-1]
        if length == 0:
            return 0.0
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        return float(np.std(values[..., start:end]))

    for start, end in zip(starts, ends, strict=True):
        end = max(start + 1, int(end))
        rms_mean = _slice_mean(rms_frames, start, end)
        energy_db = float(20.0 * np.log10(max(rms_mean, 1e-8)))
        energy = _normalized_energy_from_db(energy_db)
        onset_mean = _slice_mean(onset_envelope, start, end)
        onset_std = _slice_std(onset_envelope, start, end)
        centroid_mean = _slice_mean(centroid, start, end)
        bandwidth_mean = _slice_mean(bandwidth, start, end)
        percussion_mean = _slice_mean(percussive_rms, start, end)

        chroma_length = chroma.shape[-1]
        if chroma_length == 0:
            chroma_mean = np.zeros(12, dtype=np.float64)
        else:
            chroma_start = max(0, min(start, chroma_length - 1))
            chroma_end = max(chroma_start + 1, min(end, chroma_length))
            chroma_mean = np.mean(chroma[:, chroma_start:chroma_end], axis=1)
        chroma_sum = float(np.sum(chroma_mean))
        if chroma_sum > 1e-8:
            chroma_mean = chroma_mean / chroma_sum

        rhythm = onset_mean + 0.35 * onset_std + 0.45 * percussion_mean
        row = np.concatenate(
            [
                np.array(
                    [
                        energy,
                        onset_mean,
                        onset_std,
                        centroid_mean,
                        bandwidth_mean,
                        percussion_mean,
                    ],
                    dtype=np.float64,
                ),
                chroma_mean.astype(np.float64),
            ]
        )
        feature_rows.append(row)
        energies.append(energy)
        rhythms.append(float(rhythm))
        chroma_rows.append(chroma_mean.astype(np.float64))

    return (
        np.vstack(feature_rows),
        np.asarray(energies, dtype=np.float64),
        np.asarray(rhythms, dtype=np.float64),
        np.vstack(chroma_rows),
    )


def _normalize_novelty(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low = float(np.percentile(finite, 15.0))
    high = float(np.percentile(finite, 90.0))
    if high - low < 1e-8:
        maximum = float(np.max(finite))
        if maximum <= 1e-8:
            return np.zeros_like(values)
        normalized = values / maximum
    else:
        normalized = (values - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    if normalized.size >= 3:
        normalized = np.convolve(normalized, np.array([0.2, 0.6, 0.2]), mode="same")
    return np.clip(normalized, 0.0, 1.0)



def _local_peak_strength(values: np.ndarray, index: int, radius: int = 2) -> float:
    """Return how convincingly `index` is a local structural novelty peak."""
    if values.size == 0 or index < 0 or index >= values.size:
        return 0.0
    start = max(0, index - radius)
    end = min(values.size, index + radius + 1)
    neighborhood = np.asarray(values[start:end], dtype=np.float64)
    center = float(values[index])
    if neighborhood.size <= 1:
        return _clamp(center)
    others = np.delete(neighborhood, index - start)
    competitor = float(np.max(others)) if others.size else 0.0
    prominence = max(0.0, center - competitor)
    return _clamp(0.72 * center + 0.28 * min(1.0, prominence * 2.5))


def _phrase_alignment_score(
    index: int,
    novelty: np.ndarray,
    *,
    phrase_bars: int = PHRASE_LENGTH_BARS,
) -> float:
    """Estimate whether a boundary behaves like a musical phrase boundary.

    Phrase-grid position is only a weak prior. Audible novelty and local peak
    evidence dominate so DJAI does not blindly choose every eighth bar.
    """
    if novelty.size == 0:
        return 0.0
    metric_distance = min(index % phrase_bars, (-index) % phrase_bars)
    grid_prior = 1.0 - min(1.0, metric_distance / max(phrase_bars / 2.0, 1.0))
    peak = _local_peak_strength(novelty, index, radius=2)
    four_bar_prior = 1.0 if index % 4 == 0 else 0.35
    return _clamp(0.58 * peak + 0.27 * grid_prior + 0.15 * four_bar_prior)


def _section_context(
    bar_energies: np.ndarray,
    bar_rhythms: np.ndarray,
    index: int,
    *,
    radius: int = SECTION_CONTEXT_BARS,
) -> tuple[float, float, float, float]:
    """Return energy/rhythm context before and after a boundary."""
    total = len(bar_energies)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    left_start = max(0, index - radius)
    right_end = min(total, index + radius)
    left_energy = bar_energies[left_start:index]
    right_energy = bar_energies[index:right_end]
    left_rhythm = bar_rhythms[left_start:index]
    right_rhythm = bar_rhythms[index:right_end]

    fallback_energy = float(bar_energies[min(index, total - 1)])
    fallback_rhythm = float(bar_rhythms[min(index, len(bar_rhythms) - 1)])
    return (
        float(np.mean(left_energy)) if left_energy.size else fallback_energy,
        float(np.mean(right_energy)) if right_energy.size else fallback_energy,
        float(np.mean(left_rhythm)) if left_rhythm.size else fallback_rhythm,
        float(np.mean(right_rhythm)) if right_rhythm.size else fallback_rhythm,
    )


def _section_label(
    *,
    bar_index: int,
    total_bars: int,
    energy_before: float,
    energy_after: float,
    earlier_energy: float,
    median_energy: float,
    rhythm_change: float,
    rhythm_before: float = 0.0,
    rhythm_after: float = 0.0,
    boundary_strength: float = 0.0,
) -> SectionLabel:
    """Classify the musical role around a structural boundary.

    This remains signal-processing inference, but uses sustained context rather
    than a single energy delta. Ambiguous interior regions deliberately become
    groove instead of overclaiming a drop/breakdown.
    """
    position = bar_index / max(total_bars - 1, 1)
    energy_delta = energy_after - energy_before
    rhythm_delta = (rhythm_after - rhythm_before) / max(
        abs(rhythm_before), abs(rhythm_after), 1e-8
    )
    build_delta = energy_before - earlier_energy

    # Edge sections need positional evidence as well as musical evidence.
    if position <= 0.18 and (
        energy_before <= median_energy + 0.05
        or energy_after >= energy_before - 0.04
    ):
        return "intro"
    if position >= 0.78 and (
        energy_after <= median_energy
        or energy_delta <= -0.035
        or rhythm_delta <= -0.08
    ):
        return "outro"

    # A drop should be a strong boundary into materially denser/louder content.
    if (
        boundary_strength >= 0.42
        and energy_delta >= 0.075
        and (energy_after >= median_energy or rhythm_delta >= 0.10)
    ):
        return "drop"

    # A breakdown should represent a sustained reduction, not a one-bar dip.
    if (
        boundary_strength >= 0.34
        and energy_delta <= -0.075
        and (energy_after <= median_energy - 0.06 or rhythm_delta <= -0.10)
    ):
        return "breakdown"

    # Build: preceding bars are rising and the post-boundary region stays active.
    if (
        build_delta >= 0.055
        and energy_delta >= -0.035
        and rhythm_delta >= -0.08
        and energy_after >= median_energy - 0.10
    ):
        return "build"

    return "groove"


def _structural_boundaries(
    *,
    downbeat_frames: np.ndarray,
    downbeat_times: np.ndarray,
    beat_times: np.ndarray,
    bpm: float,
    duration: float,
    feature_matrix: np.ndarray,
    bar_energies: np.ndarray,
    bar_rhythms: np.ndarray,
    bar_chroma: np.ndarray,
    beats_per_bar: int = BEATS_PER_BAR,
) -> list[_BoundaryFeatures]:
    total_bars = len(downbeat_times)
    if total_bars == 0:
        return []

    scaled = _robust_scale(feature_matrix)
    raw_novelty = np.zeros(total_bars, dtype=np.float64)
    rhythm_changes = np.zeros(total_bars, dtype=np.float64)
    harmonic_changes = np.zeros(total_bars, dtype=np.float64)

    for index in range(total_bars):
        left_start = max(0, index - BOUNDARY_CONTEXT_BARS)
        right_end = min(total_bars, index + BOUNDARY_CONTEXT_BARS)
        left = scaled[left_start:index]
        right = scaled[index:right_end]
        if left.size == 0 or right.size == 0:
            raw_novelty[index] = 0.0
            continue

        left_mean = np.mean(left, axis=0)
        right_mean = np.mean(right, axis=0)
        raw_novelty[index] = float(
            np.linalg.norm(right_mean - left_mean) / np.sqrt(max(scaled.shape[1], 1))
        )

        rhythm_left = float(np.mean(bar_rhythms[left_start:index]))
        rhythm_right = float(np.mean(bar_rhythms[index:right_end]))
        rhythm_scale = max(abs(rhythm_left), abs(rhythm_right), 1e-8)
        rhythm_changes[index] = abs(rhythm_right - rhythm_left) / rhythm_scale

        chroma_left = np.mean(bar_chroma[left_start:index], axis=0)
        chroma_right = np.mean(bar_chroma[index:right_end], axis=0)
        harmonic_changes[index] = _safe_cosine_distance(chroma_left, chroma_right)

    novelty = _normalize_novelty(raw_novelty)
    rhythm_novelty = _normalize_novelty(rhythm_changes)
    harmonic_novelty = _normalize_novelty(harmonic_changes)
    median_energy = float(np.median(bar_energies)) if bar_energies.size else 0.5
    meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
    bar_duration = meter * 60.0 / max(bpm, 1.0)
    boundaries: list[_BoundaryFeatures] = []

    for index, time_seconds in enumerate(downbeat_times):
        left_start = max(0, index - BOUNDARY_CONTEXT_BARS)
        right_end = min(total_bars, index + BOUNDARY_CONTEXT_BARS)
        left_energy = bar_energies[left_start:index]
        right_energy = bar_energies[index:right_end]
        energy_before = (
            float(np.mean(left_energy)) if left_energy.size else float(bar_energies[index])
        )
        energy_after = (
            float(np.mean(right_energy)) if right_energy.size else float(bar_energies[index])
        )
        earlier_slice = bar_energies[max(0, left_start - BOUNDARY_CONTEXT_BARS):left_start]
        earlier_energy = (
            float(np.mean(earlier_slice)) if earlier_slice.size else energy_before
        )

        metric_novelty = _clamp(
            0.62 * novelty[index]
            + 0.20 * rhythm_novelty[index]
            + 0.18 * harmonic_novelty[index]
        )
        boundary_peak = _local_peak_strength(novelty, index, radius=2)
        phrase_alignment = _phrase_alignment_score(index, novelty)
        # Audible evidence dominates. Metric position is deliberately secondary.
        boundary_strength = _clamp(
            0.64 * metric_novelty
            + 0.24 * boundary_peak
            + 0.12 * phrase_alignment
        )
        phrase_strength = _clamp(
            0.50 * boundary_peak
            + 0.35 * phrase_alignment
            + 0.15 * boundary_strength
        )
        local_bpm, local_grid = _local_beat_grid(
            beat_times,
            float(time_seconds),
            bpm,
            beats_per_bar=meter,
        )
        available_bars = max(0, int((duration - float(time_seconds)) // bar_duration))
        context_energy_before, context_energy_after, rhythm_before, rhythm_after = (
            _section_context(bar_energies, bar_rhythms, index)
        )
        section = _section_label(
            bar_index=index,
            total_bars=total_bars,
            energy_before=context_energy_before,
            energy_after=context_energy_after,
            earlier_energy=earlier_energy,
            median_energy=median_energy,
            rhythm_change=float(rhythm_novelty[index]),
            rhythm_before=rhythm_before,
            rhythm_after=rhythm_after,
            boundary_strength=boundary_strength,
        )
        boundaries.append(
            _BoundaryFeatures(
                time_seconds=round(float(time_seconds), 6),
                bar_index=index,
                available_bars=available_bars,
                boundary_strength=boundary_strength,
                phrase_strength=phrase_strength,
                beat_grid_confidence=local_grid,
                local_bpm=local_bpm,
                section_label=section,
                energy_before=_clamp(energy_before),
                energy_after=_clamp(energy_after),
                rhythm_change=_clamp(float(rhythm_novelty[index])),
                harmonic_change=_clamp(float(harmonic_novelty[index])),
                boundary_peak=boundary_peak,
                phrase_alignment=phrase_alignment,
            )
        )
    return boundaries



def _normalize_bar_series(values: np.ndarray) -> np.ndarray:
    """Robustly map one bar-level feature sequence to 0..1."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    low = float(np.percentile(finite, 10.0))
    high = float(np.percentile(finite, 90.0))
    if high - low < 1e-9:
        maximum = float(np.max(np.abs(finite)))
        if maximum < 1e-9:
            return np.zeros_like(array)
        return np.clip(array / maximum, 0.0, 1.0)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _build_bar_features(
    *,
    boundaries: list[_BoundaryFeatures],
    downbeat_frames: np.ndarray,
    downbeat_times: np.ndarray,
    duration: float,
    sr: int,
    bar_energies: np.ndarray,
    bar_rhythms: np.ndarray,
    bar_chroma: np.ndarray,
    centroid: np.ndarray,
    percussive_rms: np.ndarray,
    source_features: _SourceFrameFeatures,
) -> list[BarFeature]:
    """Build the bar-synchronous TrackMap representation used by the brain."""
    total = min(len(downbeat_frames), len(downbeat_times), len(bar_energies))
    if total <= 0:
        return []
    boundary_by_index = {item.bar_index: item for item in boundaries}
    rhythm_norm = _normalize_bar_series(np.asarray(bar_rhythms[:total], dtype=np.float64))
    percussive_values: list[float] = []
    brightness_values: list[float] = []
    raw_windows: list[tuple[int, int]] = []
    total_frames = max(len(centroid), len(percussive_rms), source_features.frame_times.size)
    for index in range(total):
        start = int(downbeat_frames[index])
        end = int(downbeat_frames[index + 1]) if index + 1 < total else total_frames
        start = max(0, min(start, max(total_frames - 1, 0)))
        end = max(start + 1, min(end, max(total_frames, start + 1)))
        raw_windows.append((start, end))
        if percussive_rms.size:
            ps = max(0, min(start, percussive_rms.size - 1))
            pe = max(ps + 1, min(end, percussive_rms.size))
            percussive_values.append(float(np.mean(percussive_rms[ps:pe])))
        else:
            percussive_values.append(0.0)
        if centroid.size:
            cs = max(0, min(start, centroid.size - 1))
            ce = max(cs + 1, min(end, centroid.size))
            brightness_values.append(_clamp(float(np.median(centroid[cs:ce])) / max(sr / 2.0, 1.0)))
        else:
            brightness_values.append(0.0)
    percussive_norm = _normalize_bar_series(np.asarray(percussive_values, dtype=np.float64))

    output: list[BarFeature] = []
    for index in range(total):
        start_frame, end_frame = raw_windows[index]
        start_seconds = float(downbeat_times[index])
        end_seconds = (
            float(downbeat_times[index + 1])
            if index + 1 < total
            else float(duration)
        )
        boundary = boundary_by_index.get(index)
        vocal, bass = _source_window_scores(source_features, start_frame, end_frame)
        chroma_profile = (
            np.asarray(bar_chroma[index], dtype=np.float64)
            if index < len(bar_chroma)
            else np.zeros(12, dtype=np.float64)
        )
        chroma_profile = np.nan_to_num(chroma_profile, nan=0.0, posinf=0.0, neginf=0.0)
        chroma_profile = np.maximum(chroma_profile, 0.0)
        total_chroma = float(np.sum(chroma_profile))
        if total_chroma > 1e-9:
            chroma_profile /= total_chroma
        sorted_chroma = np.sort(chroma_profile)[::-1]
        concentration = float(np.sum(sorted_chroma[:3])) if sorted_chroma.size else 0.0
        local_tension = _clamp(1.0 - concentration)
        if boundary is not None:
            local_tension = _clamp(0.72 * local_tension + 0.28 * boundary.harmonic_change)
        output.append(
            BarFeature(
                bar_index=index,
                start_seconds=round(start_seconds, 6),
                end_seconds=round(max(start_seconds, end_seconds), 6),
                section_label=boundary.section_label if boundary else "unknown",
                energy=round(_clamp(float(bar_energies[index])), 4),
                rhythm=round(_clamp(float(rhythm_norm[index])), 4),
                brightness=round(_clamp(float(brightness_values[index])), 4),
                percussive_activity=round(_clamp(float(percussive_norm[index])), 4),
                vocal_probability=round(_clamp(vocal), 4),
                bass_activity=round(_clamp(bass), 4),
                harmonic_tension=round(local_tension, 4),
                boundary_strength=round(boundary.boundary_strength if boundary else 0.0, 4),
                phrase_strength=round(boundary.phrase_strength if boundary else 0.0, 4),
                beat_grid_confidence=round(boundary.beat_grid_confidence if boundary else 0.0, 4),
                local_bpm=round(boundary.local_bpm if boundary else 0.0, 4),
                chroma=[round(float(value), 6) for value in chroma_profile.tolist()],
            )
        )
    return output


def _section_prior(kind: str, label: SectionLabel) -> float:
    """Context-neutral structural usefulness for DJ cue discovery.

    This deliberately does *not* encode the old "play A to the outro, then
    start B at the intro" behavior.  Transition intent and set context decide
    which section role is best later in the pipeline.  Analysis only asks
    whether the boundary is a musically useful place to act.
    """
    if kind == "entry":
        return {
            "groove": 1.00,
            "drop": 0.94,
            "build": 0.88,
            "intro": 0.76,
            "breakdown": 0.70,
            "outro": 0.16,
            "unknown": 0.48,
        }[label]
    return {
        "breakdown": 1.00,
        "groove": 0.96,
        "build": 0.90,
        "outro": 0.72,
        "drop": 0.58,
        "intro": 0.20,
        "unknown": 0.48,
    }[label]


def _entry_position_score(time_seconds: float, bar_duration: float, label: SectionLabel) -> float:
    """Favor useful DJ entry neighborhoods, not literal track start."""
    bar = max(0.0, time_seconds / max(bar_duration, 1e-9))
    target_width = {
        "intro": (6.0, 8.0),
        "groove": (14.0, 14.0),
        "build": (22.0, 18.0),
        "drop": (28.0, 22.0),
        "breakdown": (22.0, 20.0),
        "outro": (28.0, 18.0),
        "unknown": (14.0, 16.0),
    }[label]
    target, width = target_width
    return _clamp(math.exp(-(((bar - target) / max(width, 1.0)) ** 2)))


def _exit_position_score(time_seconds: float, duration: float) -> float:
    """Keep cue discovery broad while avoiding an end-of-track bias."""
    fraction = _clamp(time_seconds / max(duration, 1.0))
    return _clamp(math.exp(-(((fraction - 0.58) / 0.30) ** 2)))


def _select_diverse_candidates(
    candidates: list[TransitionCandidate],
    *,
    minimum_bar_distance: int = 2,
) -> list[TransitionCandidate]:
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    selected: list[TransitionCandidate] = []
    for candidate in candidates:
        if all(
            abs(candidate.bar_index - existing.bar_index) >= minimum_bar_distance
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= MAX_CANDIDATES:
            break
    if len(selected) < min(MAX_CANDIDATES, len(candidates)):
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= MAX_CANDIDATES:
                break
    return selected


def _make_candidates(
    *,
    kind: str,
    boundaries: list[_BoundaryFeatures],
    duration: float,
    bpm: float,
    rms_times: np.ndarray,
    rms_values: np.ndarray,
    beats_per_bar: int = BEATS_PER_BAR,
) -> list[TransitionCandidate]:
    meter = beats_per_bar if beats_per_bar in {3, 4, 6} else BEATS_PER_BAR
    bar_duration = meter * 60.0 / max(bpm, 1.0)
    window_seconds = min(16.0 * bar_duration, max(4.0, duration * 0.35))
    candidates: list[TransitionCandidate] = []

    for boundary in boundaries:
        time_seconds = boundary.time_seconds
        available_bars = boundary.available_bars

        # Weak structural points should not compete with genuinely audible
        # phrase/section boundaries. Edge sections get a small exception.
        edge_section = boundary.section_label in {"intro", "outro"}
        if (
            not edge_section
            and boundary.boundary_strength < 0.24
            and boundary.phrase_strength < 0.34
        ):
            continue
        if kind == "entry":
            if time_seconds > min(150.0, duration * 0.55) or available_bars < 8:
                continue
        else:
            if time_seconds < duration * 0.25 or available_bars < 4:
                continue

        end_seconds = min(duration, time_seconds + window_seconds)
        local_energy, stability, trend = _local_rms_stats(
            rms_times,
            rms_values,
            time_seconds,
            end_seconds,
        )
        section_score = _section_prior(kind, boundary.section_label)

        if kind == "entry":
            position_score = _entry_position_score(time_seconds, bar_duration, boundary.section_label)
            energy_score = _clamp(1.0 - abs(local_energy - 0.50) / 0.60)
            availability_score = _clamp(available_bars / 24.0)
            score = (
                0.09 * position_score
                + 0.09 * stability
                + 0.05 * energy_score
                + 0.07 * availability_score
                + 0.27 * boundary.boundary_strength
                + 0.12 * boundary.beat_grid_confidence
                + 0.18 * boundary.phrase_strength
                + 0.13 * section_score
            )
        else:
            position_score = _exit_position_score(time_seconds, duration)
            decline_score = 1.0 if trend <= 0.0 else _clamp(np.exp(-3.0 * trend))
            energy_score = _clamp(1.0 - abs(local_energy - 0.50) / 0.65)
            availability_score = _clamp(available_bars / 16.0)
            score = (
                0.07 * position_score
                + 0.09 * stability
                + 0.06 * decline_score
                + 0.05 * energy_score
                + 0.06 * availability_score
                + 0.29 * boundary.boundary_strength
                + 0.12 * boundary.beat_grid_confidence
                + 0.17 * boundary.phrase_strength
                + 0.09 * section_score
            )

        candidates.append(
            TransitionCandidate(
                kind=kind,  # type: ignore[arg-type]
                time_seconds=round(time_seconds, 6),
                bar_index=boundary.bar_index,
                score=_clamp(score),
                local_energy=_clamp(local_energy),
                stability=_clamp(stability),
                available_bars=available_bars,
                boundary_strength=round(boundary.boundary_strength, 4),
                phrase_strength=round(boundary.phrase_strength, 4),
                beat_grid_confidence=round(boundary.beat_grid_confidence, 4),
                local_bpm=round(boundary.local_bpm, 4),
                section_label=boundary.section_label,
                energy_before=round(boundary.energy_before, 4),
                energy_after=round(boundary.energy_after, 4),
                rhythm_change=round(boundary.rhythm_change, 4),
                harmonic_change=round(boundary.harmonic_change, 4),
            )
        )

    return _select_diverse_candidates(candidates)


def analyze_track(path: str | Path) -> TrackAnalysis:
    """Analyze one track using local signal processing and structural novelty."""
    resolved = Path(path).expanduser().resolve()
    source_audio, sr = load_audio(resolved, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    y = librosa.to_mono(source_audio)
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration < 4.0:
        raise ValueError("Track is too short. DJAI requires at least 4 seconds of audio.")

    onset_envelope = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        aggregate=np.median,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        sparse=True,
    )
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    if bpm <= 0.0:
        bpm = 120.0
    beat_grid_confidence, beat_interval_cv = _beat_grid_diagnostics(beat_times, duration)
    time_signature, time_signature_confidence = _estimate_time_signature(
        beat_frames,
        onset_envelope,
        beat_grid_confidence,
    )
    beats_per_bar = time_signature if time_signature in {3, 4, 6} else BEATS_PER_BAR
    downbeat_frames, downbeat_confidence = _estimate_downbeats(
        beat_frames,
        onset_envelope,
        beats_per_bar=beats_per_bar,
    )
    downbeat_times = librosa.frames_to_time(
        downbeat_frames,
        sr=sr,
        hop_length=HOP_LENGTH,
    )

    # DJAI 5.0 Neural Ears. Beat This! is optional and registry-gated: shadow
    # predictions are retained as evidence, while an advisory/production grid
    # replaces the deterministic estimate only after confidence validation.
    neural_evidence: dict[str, object] = {}
    try:
        from .intelligence.neural_ears import predict_beats
        neural_beats = predict_beats(resolved, fallback_beats=beat_times, fallback_downbeats=downbeat_times)
        neural_evidence["beat_downbeat"] = neural_beats.to_dict()
        if neural_beats.used:
            beat_times = np.asarray(neural_beats.beats_seconds, dtype=float)
            downbeat_times = np.asarray(neural_beats.downbeats_seconds, dtype=float)
            beat_frames = np.asarray(librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP_LENGTH), dtype=int)
            downbeat_frames = np.asarray(librosa.time_to_frames(downbeat_times, sr=sr, hop_length=HOP_LENGTH), dtype=int)
            if neural_beats.bpm > 0.0:
                bpm = float(neural_beats.bpm)
            beat_grid_confidence, beat_interval_cv = _beat_grid_diagnostics(beat_times, duration)
            beat_grid_confidence = max(beat_grid_confidence, float(neural_beats.confidence))
            downbeat_confidence = max(downbeat_confidence, float(neural_beats.confidence))
            time_signature, time_signature_confidence = _estimate_time_signature(
                beat_frames,
                onset_envelope,
                beat_grid_confidence,
            )
            beats_per_bar = time_signature if time_signature in {3, 4, 6} else BEATS_PER_BAR
    except Exception as exc:
        neural_evidence["beat_downbeat"] = {"available": False, "used": False, "reason": f"neural beat adapter unavailable: {exc}"}

    # Opt-in consistency reconciliation: never changes beat positions or
    # downbeat annotations. Three independent onset windows and the complete
    # beat-grid span must agree before replacing the scalar global tempo.
    # Strictly limited to steady dance-tempo evidence; not a generic tempo fix.
    if os.environ.get("DJAI_TEMPO_CONSISTENCY", "0").strip().lower() in {"1", "true", "yes", "on"}:
        from .music.tempo_consistency import reconcile_bpm

        bpm, tempo_consistency_evidence = reconcile_bpm(
            bpm,
            beat_times,
            onset_envelope,
            sample_rate=sr,
            hop_length=HOP_LENGTH,
        )
        neural_evidence["tempo_consistency"] = tempo_consistency_evidence

    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
        bins_per_octave=36,
    )
    key = estimate_key_from_chroma(chroma)
    source_frame_features = _build_source_frame_features(
        y=y,
        sr=sr,
        harmonic=harmonic,
        percussive=percussive,
        chroma=chroma,
    )

    rms_frames = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    percussive_rms_frames = librosa.feature.rms(y=percussive, hop_length=HOP_LENGTH)[0]
    rms_times = librosa.times_like(rms_frames, sr=sr, hop_length=HOP_LENGTH)
    loudness = rms_dbfs(y)
    track_lufs = integrated_lufs(source_audio, sr)
    track_true_peak = true_peak_dbtp(source_audio)
    energy = _normalized_energy_from_db(loudness)

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )[0]
    brightness = _clamp(float(np.median(centroid)) / (sr / 2.0))

    harmonic_rms = float(np.sqrt(np.mean(np.square(harmonic), dtype=np.float64) + 1e-12))
    percussive_rms = float(np.sqrt(np.mean(np.square(percussive), dtype=np.float64) + 1e-12))
    percussive_ratio = _clamp(percussive_rms / max(harmonic_rms + percussive_rms, 1e-8))

    if downbeat_frames.size < 2:
        bar_duration = beats_per_bar * 60.0 / max(bpm, 1.0)
        synthetic_times = np.arange(0.0, duration, bar_duration)
        downbeat_times = synthetic_times
        downbeat_frames = librosa.time_to_frames(
            synthetic_times,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
        downbeat_frames = np.asarray(downbeat_frames, dtype=int)
        downbeat_confidence = min(downbeat_confidence, 0.2)

    total_frames = max(
        len(rms_frames),
        len(onset_envelope),
        len(centroid),
        len(bandwidth),
        chroma.shape[-1],
    )
    feature_matrix, bar_energies, bar_rhythms, bar_chroma = _aggregate_bar_features(
        downbeat_frames=downbeat_frames,
        total_frames=total_frames,
        rms_frames=rms_frames,
        onset_envelope=onset_envelope,
        centroid=centroid,
        bandwidth=bandwidth,
        percussive_rms=percussive_rms_frames,
        chroma=chroma,
    )
    boundaries = _structural_boundaries(
        downbeat_frames=downbeat_frames,
        downbeat_times=downbeat_times,
        beat_times=beat_times,
        bpm=bpm,
        duration=duration,
        feature_matrix=feature_matrix,
        bar_energies=bar_energies,
        bar_rhythms=bar_rhythms,
        bar_chroma=bar_chroma,
        beats_per_bar=beats_per_bar,
    )
    bar_features = _build_bar_features(
        boundaries=boundaries,
        downbeat_frames=downbeat_frames,
        downbeat_times=downbeat_times,
        duration=duration,
        sr=sr,
        bar_energies=bar_energies,
        bar_rhythms=bar_rhythms,
        bar_chroma=bar_chroma,
        centroid=centroid,
        percussive_rms=percussive_rms_frames,
        source_features=source_frame_features,
    )
    entry_candidates = _make_candidates(
        kind="entry",
        boundaries=boundaries,
        duration=duration,
        bpm=bpm,
        rms_times=rms_times,
        rms_values=rms_frames,
        beats_per_bar=beats_per_bar,
    )
    exit_candidates = _make_candidates(
        kind="exit",
        boundaries=boundaries,
        duration=duration,
        bpm=bpm,
        rms_times=rms_times,
        rms_values=rms_frames,
        beats_per_bar=beats_per_bar,
    )

    bar_duration = beats_per_bar * 60.0 / bpm
    if not entry_candidates:
        desired_entry = min(max(4.0 * bar_duration, 0.0), max(0.0, duration * 0.18))
        if downbeat_times.size:
            entry_index = min(
                range(len(downbeat_times)),
                key=lambda index: abs(float(downbeat_times[index]) - desired_entry),
            )
            fallback_entry = float(downbeat_times[entry_index])
        else:
            entry_index = max(0, int(round(desired_entry / max(bar_duration, 1e-9))))
            fallback_entry = desired_entry
        entry_candidates = [
            TransitionCandidate(
                kind="entry",
                time_seconds=round(fallback_entry, 6),
                bar_index=entry_index,
                score=0.30,
                local_energy=energy,
                stability=0.45,
                available_bars=max(4, int(duration // bar_duration)),
                boundary_strength=0.15,
                phrase_strength=0.25,
                beat_grid_confidence=beat_grid_confidence,
                local_bpm=bpm,
                section_label="intro",
                energy_before=energy,
                energy_after=energy,
            )
        ]
    if not exit_candidates:
        latest = max(0.0, duration - 8.0 * bar_duration)
        target = min(latest, max(duration * 0.55, 8.0 * bar_duration))
        if downbeat_times.size:
            eligible = [float(value) for value in downbeat_times if float(value) <= latest + 1e-6]
            fallback = min(eligible, key=lambda value: abs(value - target)) if eligible else target
        else:
            fallback = target
        exit_candidates = [
            TransitionCandidate(
                kind="exit",
                time_seconds=fallback,
                bar_index=max(0, int(fallback // bar_duration)),
                score=0.28,
                local_energy=energy,
                stability=0.45,
                available_bars=max(4, int((duration - fallback) // bar_duration)),
                boundary_strength=0.15,
                phrase_strength=0.25,
                beat_grid_confidence=beat_grid_confidence,
                local_bpm=bpm,
                section_label="outro",
                energy_before=energy,
                energy_after=energy,
            )
        ]

    top_boundary_values = sorted(
        (boundary.boundary_strength for boundary in boundaries),
        reverse=True,
    )[: min(8, len(boundaries))]
    structure_confidence = (
        float(np.mean(top_boundary_values)) if top_boundary_values else 0.0
    )
    structure_confidence = _clamp(
        0.70 * structure_confidence
        + 0.20 * beat_grid_confidence
        + 0.10 * downbeat_confidence
    )

    tempo_curve, variable_tempo = _tempo_curve(beat_times, bpm, duration)
    vocal_probability, bass_activity, spectral_rolloff, onset_density = (
        _source_characteristics(
            y,
            sr,
            harmonic,
            percussive,
            onset_envelope,
            duration,
            frame_features=source_frame_features,
        )
    )
    waveform_peaks = _waveform_peaks(y)
    all_candidates = [*entry_candidates, *exit_candidates]

    _populate_local_source_context(
        all_candidates,
        features=source_frame_features,
        sr=sr,
        duration=duration,
        global_bpm=bpm,
        fallback_vocal=vocal_probability,
        fallback_bass=bass_activity,
        beats_per_bar=beats_per_bar,
    )

    _populate_local_harmony(
        all_candidates,
        chroma=chroma,
        sr=sr,
        duration=duration,
        global_bpm=bpm,
        beats_per_bar=beats_per_bar,
    )

    analysis = TrackAnalysis(
        path=str(resolved),
        duration_seconds=round(duration, 6),
        sample_rate=sr,
        bpm=round(bpm, 4),
        bpm_confidence=round(beat_grid_confidence, 4),
        beats_seconds=[round(float(value), 6) for value in beat_times],
        downbeats_seconds=[round(float(value), 6) for value in downbeat_times],
        key=key,
        loudness_dbfs=round(loudness, 4),
        energy=round(energy, 4),
        brightness=round(brightness, 4),
        percussive_ratio=round(percussive_ratio, 4),
        integrated_lufs=round(track_lufs, 4),
        true_peak_dbtp=round(track_true_peak, 4),
        beat_grid_confidence=round(beat_grid_confidence, 4),
        beat_interval_cv=round(beat_interval_cv, 6),
        downbeat_confidence=round(downbeat_confidence, 4),
        structure_confidence=round(structure_confidence, 4),
        entry_candidates=entry_candidates,
        exit_candidates=exit_candidates,
        time_signature=time_signature,
        time_signature_confidence=round(time_signature_confidence, 4),
        tempo_curve=tempo_curve,
        variable_tempo=variable_tempo,
        vocal_probability=round(vocal_probability, 4),
        bass_activity=round(bass_activity, 4),
        spectral_rolloff=round(spectral_rolloff, 4),
        onset_density=round(onset_density, 4),
        waveform_peaks=waveform_peaks,
        analysis_version=ANALYSIS_VERSION,
        neural_evidence=neural_evidence,
        neural_evidence_fingerprint=(hashlib.sha256(json.dumps(neural_evidence, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24] if neural_evidence else ""),
        bar_features=bar_features,
        perception_confidence=round(_clamp(0.38 * beat_grid_confidence + 0.22 * downbeat_confidence + 0.20 * structure_confidence + 0.20 * key.confidence), 4),
    )
    from .genre import classify_genre

    analysis.genre = classify_genre(analysis)

    # DJAI 3.1 musical representation. The default auto backend uses a local
    # pretrained MERT model when installed/cached, otherwise the deterministic
    # statistical representation. Set DJAI_EMBEDDINGS_ENABLED=0 to disable.
    if os.environ.get("DJAI_EMBEDDINGS_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}:
        from .intelligence.embeddings import embed_track_array, section_embeddings_from_array
        try:
            embedded = embed_track_array(y, sr)
            analysis.music_embedding = list(embedded.vector)
            analysis.embedding_model = embedded.model
            analysis.embedding_backend = embedded.backend
            analysis.embedding_fingerprint = embedded.fingerprint
            section_points = [(candidate.section_label, candidate.time_seconds) for candidate in all_candidates]
            # Reuse the backend that already succeeded for the whole-track
            # representation. In auto mode this avoids retrying unavailable
            # MusicFM/MERT backends once per section, and a loaded foundation
            # model is process-cached by the embedding adapter.
            analysis.section_embeddings = section_embeddings_from_array(
                y,
                sr,
                section_points,
                track_duration=duration,
                backend=embedded.backend,
            )
            analysis.neural_evidence["music_encoder"] = {
                "available": embedded.backend in {"mert", "musicfm"},
                "backend": embedded.backend,
                "model": embedded.model,
                "fingerprint": embedded.fingerprint,
                "used_as_sensor": True,
                "note": "Foundation representation is evidence, not a DJ-taste score.",
            }
            analysis.neural_evidence_fingerprint = hashlib.sha256(
                json.dumps(analysis.neural_evidence, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:24]
        except (RuntimeError, OSError, ValueError):
            # Embeddings enrich judgment but must never make baseline analysis fail.
            analysis.embedding_backend = "unavailable"

    # Optional expert-cue prior from the public EDM-CUE/CUE-DETR work. It is
    # disabled by default because the checkpoint is large and optional. When
    # enabled, it reranks DJAI's existing quantized candidates rather than
    # bypassing beat/downbeat and source-duration safety.
    if os.environ.get("DJAI_CUE_MODEL_ENABLED", "0").strip().lower() in {"1", "true", "on", "yes"}:
        from .intelligence.cue_detr import (
            cue_model_fingerprint,
            predict_cue_points,
            rerank_analysis_candidates_with_cue_model,
        )
        cue_model_id = os.environ.get("DJAI_CUE_MODEL_ID", "disco-eth/cue-detr")
        try:
            cue_predictions = predict_cue_points(
                resolved,
                model_id=cue_model_id,
                bpm=bpm,
                allow_download=os.environ.get("DJAI_CUE_MODEL_ALLOW_DOWNLOAD", "0").strip().lower()
                in {"1", "true", "on", "yes"},
                device=os.environ.get("DJAI_CUE_MODEL_DEVICE", "auto"),
            )
            rerank_analysis_candidates_with_cue_model(analysis, cue_predictions)
            analysis.cue_model = cue_model_id
            analysis.cue_model_fingerprint = cue_model_fingerprint(cue_model_id)
            analysis.cue_model_predictions = [
                {"time_seconds": item.time_seconds, "confidence": item.confidence}
                for item in cue_predictions
            ]
        except (RuntimeError, OSError, ValueError):
            # The pretrained cue model is an optional expert prior; baseline
            # analysis must remain available when weights/dependencies are absent.
            analysis.cue_model = "unavailable"

    return analysis