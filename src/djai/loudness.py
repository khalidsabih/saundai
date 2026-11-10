from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly


def _as_samples_channels(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        return values.reshape(-1)
    # SAUND audio is channel-first. pyloudnorm expects samples x channels.
    if values.shape[0] <= 8 and values.shape[0] < values.shape[1]:
        return values.T
    return values


def integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure BS.1770-style integrated loudness in LUFS.

    Very short or effectively silent buffers return a conservative floor
    instead of leaking infinities into planning metadata.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = _as_samples_channels(audio)
    if values.size == 0 or float(np.max(np.abs(values))) < 1e-8:
        return -80.0
    minimum_frames = max(1, int(round(0.4 * sample_rate)))
    if values.shape[0] < minimum_frames:
        if values.ndim == 1:
            values = np.pad(values, (0, minimum_frames - values.shape[0]))
        else:
            values = np.pad(values, ((0, minimum_frames - values.shape[0]), (0, 0)))
    try:
        result = float(pyln.Meter(sample_rate).integrated_loudness(values))
    except Exception:
        rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64) + 1e-12))
        result = 20.0 * math.log10(max(rms, 1e-8)) - 0.691
    return float(np.clip(result, -80.0, 6.0))


def true_peak_dbtp(audio: np.ndarray, *, oversample: int = 4) -> float:
    """Approximate inter-sample true peak using polyphase oversampling."""
    values = np.asarray(audio, dtype=np.float64)
    if values.size == 0:
        return -80.0
    if values.ndim == 1:
        values = values[None, :]
    factor = max(1, int(oversample))
    if factor > 1 and values.shape[-1] >= 8:
        values = resample_poly(values, factor, 1, axis=-1)
    peak = float(np.max(np.abs(values)))
    return float(20.0 * math.log10(max(peak, 1e-8)))


def loudness_normalization_gain(
    integrated_loudness_lufs: float,
    true_peak_db: float,
    *,
    target_lufs: float = -12.0,
    true_peak_ceiling_dbtp: float = -1.0,
    maximum_boost_db: float = 6.0,
    maximum_cut_db: float = 9.0,
) -> float:
    """Return loudness gain bounded by both LUFS target and true-peak headroom."""
    loudness_gain_db = float(target_lufs) - float(integrated_loudness_lufs)
    loudness_gain_db = float(np.clip(loudness_gain_db, -abs(maximum_cut_db), abs(maximum_boost_db)))
    peak_gain_db = float(true_peak_ceiling_dbtp) - float(true_peak_db)
    gain_db = min(loudness_gain_db, peak_gain_db)
    return float(10.0 ** (gain_db / 20.0))


def loudness_headroom_db(true_peak_db: float, ceiling_dbtp: float = -1.0) -> float:
    return float(ceiling_dbtp - true_peak_db)
