from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks

from ..models import TransitionPlan

_EPS = 1e-9


@dataclass(slots=True)
class MusicalExecutionMeasurement:
    rhythm_coherence: float = 0.0
    transient_clarity: float = 0.0
    groove_continuity: float = 0.0
    harmonic_continuity: float = 0.0
    energy_intent_fit: float = 0.0
    drop_payoff: float = 0.5
    overall: float = 0.0
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mono(values: np.ndarray) -> np.ndarray:
    audio = np.asarray(values, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0, dtype=np.float32)
    return audio.reshape(-1).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.5
    return _clamp((float(np.dot(left, right) / denominator) + 1.0) / 2.0)


def _onset(values: np.ndarray, sample_rate: int, hop: int = 512) -> np.ndarray:
    mono = _mono(values)
    if mono.size < hop * 2:
        return np.zeros(1, dtype=np.float64)
    envelope = librosa.onset.onset_strength(y=mono, sr=sample_rate, hop_length=hop)
    return np.nan_to_num(np.asarray(envelope, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _rhythm_coherence(onset: np.ndarray, sample_rate: int, bpm: float, hop: int = 512) -> float:
    if onset.size < 8:
        return 0.5
    beat_frames = sample_rate * 60.0 / max(float(bpm), 1.0) / hop
    lag = max(1, int(round(beat_frames)))
    scores: list[float] = []
    for multiplier, weight in ((0.5, 0.18), (1.0, 0.50), (2.0, 0.20), (4.0, 0.12)):
        actual = max(1, int(round(lag * multiplier)))
        if onset.size <= actual + 3:
            continue
        left = onset[:-actual]
        right = onset[actual:]
        if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        scores.append(weight * _clamp((correlation + 0.35) / 1.35))
    if not scores:
        return 0.5
    total_weight = sum(weight for multiplier, weight in ((0.5, 0.18), (1.0, 0.50), (2.0, 0.20), (4.0, 0.12)) if onset.size > max(1, int(round(lag * multiplier))) + 3)
    return _clamp(sum(scores) / max(total_weight, _EPS))


def _transient_clarity(onset: np.ndarray, sample_rate: int, bpm: float, hop: int = 512) -> float:
    if onset.size < 4 or float(np.max(onset)) <= _EPS:
        return 0.5
    threshold = max(float(np.percentile(onset, 68.0)), float(np.max(onset)) * 0.18)
    peaks, props = find_peaks(onset, height=threshold, distance=1)
    if len(peaks) < 2:
        return 0.65
    beat_frames = sample_rate * 60.0 / max(float(bpm), 1.0) / hop
    minimum_separation = max(1.0, beat_frames * 0.18)
    gaps = np.diff(peaks).astype(np.float64)
    close_fraction = float(np.mean(gaps < minimum_separation)) if gaps.size else 0.0
    # A second onset very close to a strong onset is a useful proxy for kick/
    # snare flam or transient doubling in a transition.
    return _clamp(1.0 - close_fraction * 1.35)


def _phase_profile(onset: np.ndarray, sample_rate: int, bpm: float, hop: int = 512, bins: int = 16) -> np.ndarray:
    if onset.size == 0:
        return np.zeros(bins, dtype=np.float64)
    beat_frames = max(1.0, sample_rate * 60.0 / max(float(bpm), 1.0) / hop)
    indices = np.arange(onset.size, dtype=np.float64)
    phase = np.mod(indices / beat_frames, 1.0)
    bucket = np.minimum(bins - 1, (phase * bins).astype(int))
    profile = np.zeros(bins, dtype=np.float64)
    for index, value in zip(bucket, onset, strict=True):
        profile[index] += max(0.0, float(value))
    total = float(np.sum(profile))
    return profile / total if total > _EPS else profile


def _groove_continuity(pre_onset: np.ndarray, transition_onset: np.ndarray, post_onset: np.ndarray, sample_rate: int, bpm: float) -> float:
    transition_profile = _phase_profile(transition_onset, sample_rate, bpm)
    scores = []
    if pre_onset.size:
        scores.append(_cosine(_phase_profile(pre_onset, sample_rate, bpm), transition_profile))
    if post_onset.size:
        scores.append(_cosine(transition_profile, _phase_profile(post_onset, sample_rate, bpm)))
    return _clamp(float(np.mean(scores))) if scores else 0.5


def _chroma_profile(values: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = _mono(values)
    if mono.size < 2048:
        return np.zeros(12, dtype=np.float64)
    chroma = librosa.feature.chroma_stft(y=mono, sr=sample_rate, n_fft=2048, hop_length=1024)
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float64)
    profile = np.nan_to_num(np.mean(chroma, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(profile))
    return profile / total if total > _EPS else profile


def _harmonic_continuity(pre: np.ndarray, transition: np.ndarray, post: np.ndarray, sample_rate: int) -> float:
    seconds = 3.0
    frames = max(2048, int(round(seconds * sample_rate)))
    head = transition[..., : min(frames, transition.shape[-1])]
    tail = transition[..., -min(frames, transition.shape[-1]) :]
    pre_tail = pre[..., -min(frames, pre.shape[-1]) :] if pre.size else pre
    post_head = post[..., : min(frames, post.shape[-1])] if post.size else post
    scores: list[float] = []
    if pre_tail.size:
        scores.append(_cosine(_chroma_profile(pre_tail, sample_rate), _chroma_profile(head, sample_rate)))
    if post_head.size:
        scores.append(_cosine(_chroma_profile(tail, sample_rate), _chroma_profile(post_head, sample_rate)))
    return _clamp(float(np.mean(scores))) if scores else 0.5


def _rms_curve(values: np.ndarray, sample_rate: int, windows: int = 12) -> np.ndarray:
    mono = _mono(values).astype(np.float64)
    if mono.size == 0:
        return np.zeros(windows, dtype=np.float64)
    edges = np.linspace(0, mono.size, windows + 1, dtype=int)
    levels: list[float] = []
    for index in range(windows):
        segment = mono[edges[index] : max(edges[index] + 1, edges[index + 1])]
        levels.append(float(np.sqrt(np.mean(segment * segment) + _EPS)))
    curve = np.asarray(levels, dtype=np.float64)
    maximum = float(np.max(curve))
    return curve / maximum if maximum > _EPS else curve


def _energy_intent_fit(transition: np.ndarray, plan: TransitionPlan, sample_rate: int) -> float:
    curve = _rms_curve(transition, sample_rate)
    if curve.size < 3:
        return 0.5
    x = np.linspace(-1.0, 1.0, curve.size)
    slope = float(np.polyfit(x, curve, 1)[0])
    intent = plan.intent
    if intent in {"energy_lift", "breakdown_to_drop"}:
        target = 0.18
        return _clamp(math.exp(-((slope - target) / 0.32) ** 2))
    if intent == "energy_release":
        target = -0.15
        return _clamp(math.exp(-((slope - target) / 0.32) ** 2))
    if intent in {"smooth_continuation", "groove_handoff", "vocal_handoff"}:
        return _clamp(math.exp(-((slope - 0.0) / 0.28) ** 2))
    return _clamp(math.exp(-((abs(slope) - 0.10) / 0.45) ** 2))


def _db_rms(values: np.ndarray) -> float:
    mono = _mono(values).astype(np.float64)
    if mono.size == 0:
        return -80.0
    return float(20.0 * np.log10(max(float(np.sqrt(np.mean(mono * mono) + _EPS)), 1e-8)))


def _drop_payoff(pre: np.ndarray, transition: np.ndarray, post: np.ndarray, plan: TransitionPlan, sample_rate: int) -> float:
    if plan.intent not in {"energy_lift", "breakdown_to_drop"} and plan.technique != "double_drop":
        return 0.5
    probe = max(512, int(round(2.0 * sample_rate)))
    pre_db = _db_rms(pre[..., -min(probe, pre.shape[-1]) :] if pre.size else pre)
    post_db = _db_rms(post[..., : min(probe, post.shape[-1])] if post.size else post)
    curve = _rms_curve(transition, sample_rate, windows=16)
    trough = float(np.min(curve[max(0, len(curve) // 3) :])) if curve.size else 0.5
    contrast = _clamp((post_db - pre_db + 4.0) / 10.0)
    tension_release = _clamp((1.0 - trough - 0.08) / 0.65)
    return _clamp(0.62 * contrast + 0.38 * tension_release)


def _musical_hard_failures(
    plan: TransitionPlan,
    *,
    rhythm: float,
    transient: float,
    groove: float,
    harmony: float,
    payoff: float,
) -> list[str]:
    """Technique-aware musical failures that should block automatic delivery.

    Reset/cut techniques intentionally break continuity, so blend-style rhythm
    and groove gates do not apply to them. The hard gates focus on sustained
    rhythmic overlaps where flam, drift, phrase failure, or harmonic collapse
    would sound like a mistake rather than an artistic reset.
    """
    failures: list[str] = []
    reset = (
        plan.technique in {"quick_cut", "backspin", "echo_out_drop_in"}
        or plan.transition_mode == "emergency_cut"
    )
    long_overlap = plan.beatmatched and plan.transition_bars >= 4 and not reset
    extended_overlap = plan.beatmatched and plan.transition_bars >= 8 and not reset

    if long_overlap and rhythm < 0.20:
        failures.append(
            "Rendered beatmatched overlap has severe rhythmic incoherence; automatic delivery must replan."
        )
    if long_overlap and transient < 0.16:
        failures.append(
            "Rendered overlap contains persistent doubled transients/flam consistent with kick or snare competition."
        )
    if extended_overlap and groove < 0.18:
        failures.append(
            "Rendered long overlap loses groove continuity relative to the surrounding phrases."
        )
    if extended_overlap and harmony < 0.16:
        failures.append(
            "Rendered long overlap has severe harmonic discontinuity inconsistent with a transparent blend."
        )
    if (
        payoff < 0.12
        and (plan.intent in {"energy_lift", "breakdown_to_drop"} or plan.technique == "double_drop")
        and not reset
    ):
        failures.append(
            "The committed build/drop transition has insufficient rendered payoff for its musical intent."
        )
    return failures


def measure_musical_execution(
    *,
    pre: np.ndarray,
    transition: np.ndarray,
    post: np.ndarray,
    sample_rate: int,
    plan: TransitionPlan,
) -> MusicalExecutionMeasurement:
    """Judge musical execution from the *actual rendered transition audio*.

    This remains a deterministic signal judge, not a human-taste oracle. It is
    deliberately separate from expert preference learning.
    """
    transition_onset = _onset(transition, sample_rate)
    pre_onset = _onset(pre, sample_rate)
    post_onset = _onset(post, sample_rate)
    rhythm = _rhythm_coherence(transition_onset, sample_rate, plan.target_bpm)
    transient = _transient_clarity(transition_onset, sample_rate, plan.target_bpm)
    groove = _groove_continuity(pre_onset, transition_onset, post_onset, sample_rate, plan.target_bpm)
    harmony = _harmonic_continuity(pre, transition, post, sample_rate)
    energy_fit = _energy_intent_fit(transition, plan, sample_rate)
    payoff = _drop_payoff(pre, transition, post, plan, sample_rate)

    # Harmony receives deliberately limited authority: intentional tension is
    # a legitimate professional move and should not be treated as a hard error.
    overall = _clamp(
        0.25 * rhythm
        + 0.18 * transient
        + 0.22 * groove
        + 0.10 * harmony
        + 0.17 * energy_fit
        + 0.08 * payoff
    )
    hard_failures = _musical_hard_failures(
        plan,
        rhythm=rhythm,
        transient=transient,
        groove=groove,
        harmony=harmony,
        payoff=payoff,
    )
    warnings: list[str] = []
    if rhythm < 0.30 and plan.beatmatched:
        warnings.append("Rendered rhythmic periodicity is weak for a beatmatched transition.")
    if transient < 0.28:
        warnings.append("Closely doubled transients may create audible flam or kick competition.")
    if groove < 0.28 and plan.transition_bars >= 4:
        warnings.append("The transition changes rhythmic feel abruptly relative to its surrounding audio.")
    if energy_fit < 0.22:
        warnings.append("The rendered energy movement does not strongly match the intended DJ move.")
    if payoff < 0.22 and (plan.intent in {"energy_lift", "breakdown_to_drop"} or plan.technique == "double_drop"):
        warnings.append("The intended build/drop has weak measurable payoff.")
    return MusicalExecutionMeasurement(
        rhythm_coherence=round(rhythm, 4),
        transient_clarity=round(transient, 4),
        groove_continuity=round(groove, 4),
        harmonic_continuity=round(harmony, 4),
        energy_intent_fit=round(energy_fit, 4),
        drop_payoff=round(payoff, 4),
        overall=round(overall, 4),
        hard_failures=hard_failures,
        warnings=warnings,
    )
