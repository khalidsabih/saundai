"""Conservative, observation-only tempo arbitration for SAUND.AI.

A candidate is never accepted solely because two estimators agree, because
those estimators can agree on the same octave error. Require three distributed
onset-periodicity windows as additional evidence. This module never writes to
track analysis or changes beat positions, bar phase, or cue placement.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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
    octave_family_agreement: bool = False
    window_diagnostics: tuple[dict, ...] = ()
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


def octave_family_agreement(
    bpms: tuple[float, ...], *, tolerance: float = 0.015
) -> bool:
    """Does each window support one pulse family at half/normal/double time?

    This describes rhythmic equivalence, NOT the correct perceived BPM level.
    """
    if len(bpms) != 3 or any(not _valid(b) for b in bpms):
        return False
    for reference in bpms:
        if all(
            min(_relative_error(value, reference * factor) for factor in (0.5, 1.0, 2.0))
            <= tolerance
            for value in bpms
        ):
            return True
    return False


def onset_window_diagnostics(path: Path) -> dict:
    """Report each ACF window result and explicit reason for insufficient data.

    An observed peak is not a confidence guarantee; capture weak estimates.
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
    minimum_frames = 3 * max(window, 256)
    if len(onset) < minimum_frames:
        return {
            "status": "insufficient_duration",
            "duration_seconds": round(len(onset) / frames_per_second, 3),
            "minimum_seconds": round(minimum_frames / frames_per_second, 3),
            "windows": [],
        }
    starts = [
        round(0.08 * (len(onset) - window)),
        round(0.47 * (len(onset) - window)),
        round(0.87 * (len(onset) - window)),
    ]
    windows = []
    for index, start in enumerate(starts, start=1):
        result = _acf_tempo(
            onset[start:start + window],
            frames_per_second=frames_per_second,
            low=60.0, high=180.0,
        )
        if result is None:
            windows.append({
                "index": index, "start_seconds": round(start / frames_per_second, 3),
                "status": "no_periodicity_peak", "bpm": None, "strength": None,
            })
        else:
            windows.append({
                "index": index, "start_seconds": round(start / frames_per_second, 3),
                "status": "ok", "bpm": round(result[0], 4),
                "strength": round(result[1], 4),
            })
    return {
        "status": "complete" if all(w["status"] == "ok" for w in windows)
                  else "incomplete_periodicity",
        "duration_seconds": round(len(onset) / frames_per_second, 3),
        "windows": windows,
    }


def arbitrate_tempo(
    deterministic_bpm: float | None,
    neural_bpm: float | None,
    onset_windows: tuple[tuple[float, float], ...] = (),
    *,
    minimum_strength: float = 0.35,
    agreement_tolerance: float = 0.015,
    window_diagnostics: tuple[dict, ...] = (),
) -> TempoArbitration:
    """Suggest only with three strong distributed windows and neural support.

    A suggestion remains research evidence; it never changes the BPM.
    Window onset estimates are corroboration, not independent annotations.
    """
    candidates = tempo_candidates(deterministic_bpm, neural_bpm)
    window_bpms = tuple(round(float(b), 4) for b, _ in onset_windows)
    strengths = tuple(round(float(s), 4) for _, s in onset_windows)
    family_agrees = octave_family_agreement(tuple(float(b) for b, _ in onset_windows), tolerance=agreement_tolerance)

    def outcome(decision: str, reason: str, suggestion: float | None = None):
        return TempoArbitration(
            decision=decision, suggested_bpm=suggestion,
            deterministic_bpm=deterministic_bpm, neural_bpm=neural_bpm,
            candidates_bpm=candidates, window_bpms=window_bpms,
            window_strengths=strengths, reason=reason,
            octave_family_agreement=family_agrees, window_diagnostics=window_diagnostics,
        )

    if not _valid(deterministic_bpm) or not _valid(neural_bpm):
        return outcome("abstain", "both BPM estimators are required")
    if len(onset_windows) != 3 or any(
        not _valid(b) or not math.isfinite(s) or s < minimum_strength
        for b, s in onset_windows
    ):
        return outcome("abstain", "three sufficiently strong onset windows required")

    values = [float(b) for b, _ in onset_windows]
    if not family_agrees:
        return outcome("abstain", "onset windows disagree, including octave families")
    if not candidates:
        return outcome("abstain", "no tempo candidates in supported range")

    # Octave-family agreement does not establish the correct musical pulse.
    # Require one single candidate to be independently supported by the
    # neural estimate and EVERY sufficiently strong window, allowing /2 or x2.
    supported = [
        v for v in candidates
        if min(_relative_error(v, neural_bpm * factor) for factor in (0.5, 1.0, 2.0))
           <= agreement_tolerance
        and all(
            min(_relative_error(v, w * factor) for factor in (0.5, 1.0, 2.0))
            <= agreement_tolerance for w in values
        )
    ]
    if not supported:
        return outcome("abstain", "no candidate is supported by all onset windows and neural tempo")
    # Two octave-related candidates can be equally supported. This is a pulse-
    # family finding, not enough evidence to select a metrical level.
    families = sorted(supported, key=lambda v: (abs(v - neural_bpm), v))
    if len(families) > 1 and any(
        _relative_error(v, families[0]) <= agreement_tolerance or
        min(_relative_error(v, families[0] * scale) for scale in (0.5, 2.0))
            <= agreement_tolerance
        for v in families[1:]
    ):
        if any(_relative_error(v, deterministic_bpm) <= agreement_tolerance for v in supported):
            return outcome("retain", "octave family corroborated but musical pulse level ambiguous; retain deterministic")
        return outcome("abstain", "octave family corroborated but musical pulse level ambiguous")
    best = families[0]

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
    """Backwards-compatible complete-window view; detailed diagnostics are preferred."""
    diagnostics = onset_window_diagnostics(path)
    if diagnostics["status"] != "complete":
        return ()
    return tuple(
        (float(w["bpm"]), float(w["strength"])) for w in diagnostics["windows"]
    )
