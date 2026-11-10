from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .loudness import integrated_lufs, loudness_headroom_db, true_peak_dbtp
from .models import TransitionMeasurement, TransitionPlan
from .version import MEASUREMENT_VERSION

_EPSILON = 1e-12
_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (20.0, 200.0),
    (200.0, 800.0),
    (800.0, 2_500.0),
    (2_500.0, 6_000.0),
    (6_000.0, 20_000.0),
)


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    loudness_weight: float = 0.27
    spectral_weight: float = 0.22
    bass_weight: float = 0.18
    phase_weight: float = 0.12
    artifact_weight: float = 0.14
    silence_weight: float = 0.07
    maximum_loudness_jump_db: float = 14.0
    maximum_loudness_jump_lu: float = 8.0
    preferred_true_peak_dbtp: float = -1.0
    maximum_true_peak_dbtp: float = 0.5
    maximum_low_buildup_db: float = 12.0
    maximum_clipping_fraction: float = 0.001
    minimum_phase_correlation: float = -0.85
    maximum_silence_fraction: float = 0.60


DEFAULT_MEASUREMENT_POLICY = MeasurementPolicy()


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mono(values: np.ndarray) -> np.ndarray:
    audio = np.asarray(values, dtype=np.float64)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        return audio.reshape(-1)
    return np.mean(audio, axis=0)


def _rms_db(values: np.ndarray) -> float:
    mono = _mono(values)
    if mono.size == 0:
        return -120.0
    level = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64) + _EPSILON))
    return float(20.0 * math.log10(max(level, 1e-9)))


def _head(values: np.ndarray, frames: int) -> np.ndarray:
    return np.asarray(values)[..., : max(1, min(frames, values.shape[-1]))]


def _tail(values: np.ndarray, frames: int) -> np.ndarray:
    return np.asarray(values)[..., -max(1, min(frames, values.shape[-1])) :]


def _band_energy(values: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = _mono(values)
    if mono.size < 32:
        return np.zeros(len(_BANDS_HZ), dtype=np.float64)
    window = np.hanning(mono.size)
    spectrum = np.square(np.abs(np.fft.rfft(mono * window)), dtype=np.float64)
    # Normalize by window length so band power remains comparable when the
    # pre-roll, transition, and post-roll windows have different durations.
    spectrum /= max(float(mono.size * mono.size), 1.0)
    frequencies = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    energies: list[float] = []
    for low, high in _BANDS_HZ:
        upper = min(high, nyquist)
        mask = (frequencies >= low) & (frequencies < upper)
        energies.append(float(np.sum(spectrum[mask])) if np.any(mask) else 0.0)
    return np.asarray(energies, dtype=np.float64) + _EPSILON


def _spectral_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_total = float(np.sum(left))
    right_total = float(np.sum(right))
    if left_total <= _EPSILON or right_total <= _EPSILON:
        return 0.0
    left_profile = left / left_total
    right_profile = right / right_total
    return _clamp(float(np.sum(np.sqrt(left_profile * right_profile))))


def _median_phase_correlation(values: np.ndarray, sample_rate: int) -> float:
    audio = np.asarray(values, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] < 2 or audio.shape[-1] < 64:
        return 1.0
    frame = max(256, int(round(sample_rate * 0.25)))
    correlations: list[float] = []
    for start in range(0, audio.shape[-1] - frame + 1, frame):
        left = audio[0, start : start + frame]
        right = audio[1, start : start + frame]
        left = left - float(np.mean(left))
        right = right - float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-10:
            continue
        correlations.append(float(np.dot(left, right) / denominator))
    return float(np.median(correlations)) if correlations else 1.0


def _silence_fraction(values: np.ndarray, sample_rate: int) -> float:
    mono = _mono(values)
    if mono.size == 0:
        return 1.0
    frame = max(128, int(round(sample_rate * 0.10)))
    levels: list[float] = []
    for start in range(0, mono.size, frame):
        segment = mono[start : start + frame]
        if segment.size:
            levels.append(_rms_db(segment))
    if not levels:
        return 1.0
    return float(np.mean(np.asarray(levels) < -55.0))


def _boundary_jump(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 1.0
    left_sample = np.asarray(left, dtype=np.float64)[..., -1]
    right_sample = np.asarray(right, dtype=np.float64)[..., 0]
    jump = float(np.max(np.abs(right_sample - left_sample)))
    local = np.concatenate(
        [
            np.asarray(left, dtype=np.float64)[..., -min(2048, left.shape[-1]) :].reshape(-1),
            np.asarray(right, dtype=np.float64)[..., : min(2048, right.shape[-1])].reshape(-1),
        ]
    )
    scale = float(np.sqrt(np.mean(np.square(local)) + _EPSILON))
    return float(jump / max(scale, 1e-4))


def measure_transition_audio(
    *,
    pre: np.ndarray,
    transition: np.ndarray,
    post: np.ndarray,
    sample_rate: int,
    plan: TransitionPlan,
    policy: MeasurementPolicy = DEFAULT_MEASUREMENT_POLICY,
) -> TransitionMeasurement:
    """Measure the rendered audio that would actually be used for a transition.

    The result is a deterministic signal-quality estimate, not a claim of human
    taste. It catches measurable loudness, spectrum, low-end, phase, clipping,
    silence, and boundary-discontinuity problems.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    audio = np.asarray(transition, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.size == 0 or audio.shape[-1] == 0:
        return TransitionMeasurement(
            version=MEASUREMENT_VERSION,
            sample_rate=sample_rate,
            duration_seconds=0.0,
            overall=0.0,
            hard_failures=["Rendered transition contains no audio."],
        )

    context_frames = max(256, int(round(sample_rate * 1.5)))
    transition_frames = max(256, min(context_frames, max(1, audio.shape[-1] // 3)))
    pre_tail = _tail(pre, context_frames)
    post_head = _head(post, context_frames)
    transition_head = _head(audio, transition_frames)
    transition_tail = _tail(audio, transition_frames)

    pre_db = _rms_db(pre_tail)
    transition_head_db = _rms_db(transition_head)
    transition_tail_db = _rms_db(transition_tail)
    post_db = _rms_db(post_head)
    loudness_jump_db = max(
        abs(transition_head_db - pre_db),
        abs(post_db - transition_tail_db),
    )

    pre_lufs = integrated_lufs(pre_tail, sample_rate)
    transition_head_lufs = integrated_lufs(transition_head, sample_rate)
    transition_tail_lufs = integrated_lufs(transition_tail, sample_rate)
    post_lufs = integrated_lufs(post_head, sample_rate)
    transition_lufs = integrated_lufs(audio, sample_rate)
    loudness_jump_lu = max(
        abs(transition_head_lufs - pre_lufs),
        abs(post_lufs - transition_tail_lufs),
    )
    loudness_tolerance = 5.0 if plan.transition_mode == "emergency_cut" else 3.0
    loudness_continuity = _clamp(
        math.exp(-((max(0.0, loudness_jump_lu - loudness_tolerance) / 7.0) ** 2))
    )

    pre_spectrum = _band_energy(pre_tail, sample_rate)
    head_spectrum = _band_energy(transition_head, sample_rate)
    tail_spectrum = _band_energy(transition_tail, sample_rate)
    post_spectrum = _band_energy(post_head, sample_rate)
    spectral_continuity = _clamp(
        0.5 * _spectral_similarity(pre_spectrum, head_spectrum)
        + 0.5 * _spectral_similarity(tail_spectrum, post_spectrum)
    )

    pre_low_share = float(pre_spectrum[0] / max(float(np.sum(pre_spectrum)), _EPSILON))
    post_low_share = float(post_spectrum[0] / max(float(np.sum(post_spectrum)), _EPSILON))
    head_low_share = float(head_spectrum[0] / max(float(np.sum(head_spectrum)), _EPSILON))
    tail_low_share = float(tail_spectrum[0] / max(float(np.sum(tail_spectrum)), _EPSILON))
    context_low = max(pre_low_share, post_low_share, _EPSILON)
    transition_low = max(head_low_share, tail_low_share, _EPSILON)
    # Do not turn numerical leakage in effectively bass-free material into a
    # false low-frequency collision. Bass buildup is meaningful only when the
    # transition contains a measurable low-band share.
    low_buildup_db = (
        0.0
        if transition_low < 1e-4 and context_low < 1e-4
        else float(10.0 * math.log10(transition_low / context_low))
    )
    bass_tolerance = 4.0 if plan.technique in {"filter_build", "double_drop"} else 2.0
    bass_cleanliness = _clamp(
        math.exp(-((max(0.0, low_buildup_db - bass_tolerance) / 8.0) ** 2))
    )

    phase_correlation = _median_phase_correlation(audio, sample_rate)
    phase_safety = _clamp((phase_correlation + 0.80) / 1.80)

    peak = float(np.max(np.abs(audio)))
    transition_true_peak_dbtp = true_peak_dbtp(audio)
    true_peak_headroom = loudness_headroom_db(
        transition_true_peak_dbtp,
        policy.preferred_true_peak_dbtp,
    )
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
    boundary_jump = max(_boundary_jump(pre, audio), _boundary_jump(audio, post))
    clipping_score = _clamp(1.0 - min(1.0, clipping_fraction / 0.002))
    peak_score = _clamp(
        math.exp(
            -(
                max(0.0, transition_true_peak_dbtp - policy.preferred_true_peak_dbtp)
                / 2.5
            )
            ** 2
        )
    )
    boundary_score = _clamp(
        math.exp(-((max(0.0, boundary_jump - 3.0) / 8.0) ** 2))
    )
    artifact_safety = _clamp(
        0.45 * clipping_score + 0.20 * peak_score + 0.35 * boundary_score
    )

    silence_fraction = _silence_fraction(audio, sample_rate)
    silence_tolerance = 0.28 if plan.transition_mode == "emergency_cut" else 0.12
    silence_safety = _clamp(
        1.0
        - max(0.0, silence_fraction - silence_tolerance)
        / max(1e-9, 1.0 - silence_tolerance)
    )

    overall = _clamp(
        policy.loudness_weight * loudness_continuity
        + policy.spectral_weight * spectral_continuity
        + policy.bass_weight * bass_cleanliness
        + policy.phase_weight * phase_safety
        + policy.artifact_weight * artifact_safety
        + policy.silence_weight * silence_safety
    )

    hard_failures: list[str] = []
    warnings: list[str] = []
    if loudness_jump_db > policy.maximum_loudness_jump_db:
        hard_failures.append(
            f"Rendered RMS level changes by {loudness_jump_db:.1f} dB at a transition boundary."
        )
    elif loudness_jump_db > 8.0:
        warnings.append(f"Rendered RMS level changes by {loudness_jump_db:.1f} dB.")

    loudness_jump_limit = (
        policy.maximum_loudness_jump_lu + 2.0
        if plan.transition_mode == "emergency_cut"
        else policy.maximum_loudness_jump_lu
    )
    if loudness_jump_lu > loudness_jump_limit:
        hard_failures.append(
            f"Rendered short-term loudness changes by {loudness_jump_lu:.1f} LU at a transition boundary."
        )
    elif loudness_jump_lu > 4.5:
        warnings.append(
            f"Rendered short-term loudness changes by {loudness_jump_lu:.1f} LU."
        )

    if transition_true_peak_dbtp > policy.maximum_true_peak_dbtp:
        hard_failures.append(
            f"Rendered true peak reaches {transition_true_peak_dbtp:.2f} dBTP."
        )
    elif transition_true_peak_dbtp > -0.5:
        warnings.append(
            f"Rendered true peak is {transition_true_peak_dbtp:.2f} dBTP; preferred headroom is {policy.preferred_true_peak_dbtp:.1f} dBTP."
        )
    if low_buildup_db > policy.maximum_low_buildup_db:
        hard_failures.append(
            f"Low-frequency buildup reaches {low_buildup_db:.1f} dB above surrounding audio."
        )
    elif low_buildup_db > 7.0:
        warnings.append(
            f"Low-frequency buildup is {low_buildup_db:.1f} dB above surrounding audio."
        )
    if clipping_fraction > policy.maximum_clipping_fraction:
        hard_failures.append(f"Clipping affects {clipping_fraction:.3%} of rendered samples.")
    if phase_correlation < policy.minimum_phase_correlation:
        hard_failures.append(f"Stereo phase correlation is unsafe at {phase_correlation:.2f}.")
    elif phase_correlation < -0.35:
        warnings.append(f"Stereo phase correlation is low at {phase_correlation:.2f}.")
    if silence_fraction > policy.maximum_silence_fraction:
        hard_failures.append(
            f"Unexpected silence occupies {silence_fraction:.1%} of the rendered transition."
        )
    elif silence_fraction > 0.30:
        warnings.append(
            f"The rendered transition contains {silence_fraction:.1%} near-silence."
        )
    if boundary_jump > 18.0:
        hard_failures.append(
            "A large sample discontinuity was detected at a transition boundary."
        )
    elif boundary_jump > 8.0:
        warnings.append(
            "A noticeable sample discontinuity may produce a click at a transition boundary."
        )

    return TransitionMeasurement(
        version=MEASUREMENT_VERSION,
        sample_rate=sample_rate,
        duration_seconds=round(audio.shape[-1] / sample_rate, 6),
        loudness_continuity=round(loudness_continuity, 4),
        spectral_continuity=round(spectral_continuity, 4),
        bass_cleanliness=round(bass_cleanliness, 4),
        phase_safety=round(phase_safety, 4),
        artifact_safety=round(artifact_safety, 4),
        silence_safety=round(silence_safety, 4),
        overall=round(overall, 4),
        peak=round(peak, 6),
        clipping_fraction=round(clipping_fraction, 8),
        phase_correlation=round(phase_correlation, 4),
        loudness_jump_db=round(loudness_jump_db, 4),
        loudness_jump_lu=round(loudness_jump_lu, 4),
        transition_lufs=round(transition_lufs, 4),
        true_peak_dbtp=round(transition_true_peak_dbtp, 4),
        true_peak_headroom_db=round(true_peak_headroom, 4),
        low_frequency_buildup_db=round(low_buildup_db, 4),
        silence_fraction=round(silence_fraction, 6),
        boundary_jump=round(boundary_jump, 4),
        hard_failures=hard_failures,
        warnings=warnings,
    )
