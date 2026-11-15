"""Conservative, observation-only tempo arbitration for SAUND.AI.

A candidate is never accepted solely because two estimators agree, because
those estimators can agree on the same octave error. Require three distributed
onset-periodicity windows as additional evidence. This module never writes to
track analysis or changes beat positions, bar phase, or cue placement.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..music.tempo_consistency import _acf_tempo


@dataclass(frozen=True)
class TempoArbitration:
    decision: str
    suggested_bpm: float | None
    deterministic_bpm: float | None
    neural_bpm: float | None
    candidates_bpm: tuple[float, ...]
    window_bpms: tuple[float, ...]
    window_strengths: tuple[float, ...]
    reason: str
    changes_dj_decisions: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _valid(bpm: float | None) -> bool:
    return bpm is not None and math.isfinite(bpm) and bpm > 0


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / b


def tempo_candidates(
    deterministic_bpm: float | None, neural_bpm: float | None,
    *, low: float = 60.0, high: float = 180.0,
) -> tuple[float, ...]:
    """Enumerate original and half/double tempo hypotheses, not corrections."""
    values = set()
    for bpm in (deterministic_bpm, neural_bpm):
        if _valid(bpm):
            for scale in (0.5, 1.0, 2.0):
                candidate = float(bpm) * scale
                if low <= candidate <= high:
                    values.add(round(candidate, 4))
    return tuple(sorted(values))


def arbitrate_tempo(
    deterministic_bpm: float | None,
    neural_bpm: float | None,
    onset_windows: tuple[tuple[float, float], ...] = (),
    *,
    minimum_strength: float = 0.35,
    agreement_tolerance: float = 0.015,
) -> TempoArbitration:
    """Suggest only with three strong distributed windows and neural support.

    A suggestion remains research evidence; it never changes the BPM.
    Window onset estimates are corroboration, not independent annotations.
    """
    candidates = tempo_candidates(deterministic_bpm, neural_bpm)
    window_bpms = tuple(round(float(b), 4) for b, _ in onset_windows)
    strengths = tuple(round(float(s), 4) for _, s in onset_windows)

    def outcome(decision: str, reason: str, suggestion: float | None = None):
        return TempoArbitration(
            decision=decision, suggested_bpm=suggestion,
            deterministic_bpm=deterministic_bpm, neural_bpm=neural_bpm,
            candidates_bpm=candidates, window_bpms=window_bpms,
            window_strengths=strengths, reason=reason,
        )

    if not _valid(deterministic_bpm) or not _valid(neural_bpm):
        return outcome("abstain", "both BPM estimators are required")
    if len(onset_windows) != 3 or any(
        not _valid(b) or not math.isfinite(s) or s < minimum_strength
        for b, s in onset_windows
    ):
        return outcome("abstain", "three sufficiently strong onset windows required")

    values = [float(b) for b, _ in onset_windows]
    window_median = float(np.median(values))
    if any(_relative_error(b, window_median) > agreement_tolerance for b in values):
        return outcome("abstain", "onset windows disagree")
    if not candidates:
        return outcome("abstain", "no tempo candidates in supported range")

    # Neural corroboration is mandatory: a plausible deterministic octave
    # candidate by itself cannot establish the correct perceived pulse.
    neural_supported = [v for v in candidates
                        if _relative_error(v, window_median) <= agreement_tolerance
                        and min(_relative_error(v, neural_bpm * scale)
                                for scale in (0.5, 1.0, 2.0)) <= agreement_tolerance]
    if not neural_supported:
        return outcome("abstain", "onset windows do not corroborate neural tempo hypotheses")
    best = min(neural_supported, key=lambda v: abs(v - window_median))

    # Never 'correct' an already agreeing deterministic BPM. Selecting an
    # octave changes perceived metrical pulse; human review still required.
    if _relative_error(deterministic_bpm, best) <= agreement_tolerance:
        return outcome("retain", "deterministic tempo corroborated; no change suggested")
    return outcome(
        "review_candidate",
        "onset windows and neural tempo corroborate a non-default pulse; human validation required",
        round(best, 4),
    )


def onset_tempo_windows(path: Path) -> tuple[tuple[float, float], ...]:
    """Three disjoint ~24-second onset windows (where duration permits).

    The returned evidence comes from a signal-processing view of the same
    recording; it is not ground truth or completely independent of librosa.
    """
    import librosa
    from ..analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH
    from ..audio import load_audio

    audio, sr = load_audio(path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    y = librosa.to_mono(audio)
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    frames_per_second = sr / HOP_LENGTH
    window = round(24 * frames_per_second)
    if len(onset) < 3 * max(window, 256):
        return ()
    # Non-overlapping windows distributed across the full recording.
    starts = [
        round(0.08 * (len(onset) - window)),
        round(0.47 * (len(onset) - window)),
        round(0.87 * (len(onset) - window)),
    ]
    estimates = []
    for start in starts:
        result = _acf_tempo(
            onset[start:start + window],
            frames_per_second=frames_per_second,
            low=60.0, high=180.0,
        )
        if result is None:
            return ()
        estimates.append((round(result[0], 4), round(result[1], 4)))
    return tuple(estimates)
