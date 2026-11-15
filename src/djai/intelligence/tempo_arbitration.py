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
    octave_family_agreement: bool = False
    window_diagnostics: tuple[dict, ...] = ()
    coverage: dict | None = None
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
    if len(bpms) < 3 or any(not _valid(b) for b in bpms):
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
    # Seven evenly distributed windows, but never count duplicated/overlapping
    # windows as independent evidence. Very short tracks still yield three.
    count = min(7, max(3, len(onset) // max(window, 256)))
    span = len(onset) - window
    starts = [round(i * span / (count - 1)) for i in range(count)]
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


def _in_family(bpm: float, anchor: float, tolerance: float) -> bool:
    """Check measured pulse-family similarity; no meter-level inference."""
    return min(
        _relative_error(bpm, anchor * scale)
        for scale in (0.5, 1.0, 2.0)
    ) <= tolerance


def _subdivision_alias(bpm: float, anchor: float, tolerance: float) -> bool:
    """Flag a possible 2:3 rhythmic subdivision, never treat it as tempo equivalence."""
    return any(
        _relative_error(bpm, anchor * factor) <= tolerance
        for factor in (2.0 / 3.0, 3.0 / 2.0)
    )


def evidence_coverage(
    diagnostics: tuple[dict, ...],
    neural_bpm: float | None,
    *,
    deterministic_bpm: float | None = None,
    minimum_strength: float = 0.35,
    tolerance: float = 0.015,
) -> dict:
    """Classify strong *audio* windows into a dominant octave family.

    Detector agreement is recorded separately. Neither model defines which
    measured windows count as conflicts, and octave agreement cannot establish
    the musically intended metrical level.
    """
    total = len(diagnostics)
    valid = [
        w for w in diagnostics
        if w.get("status") == "ok" and _valid(w.get("bpm"))
        and isinstance(w.get("strength"), (int, float))
        and math.isfinite(w["strength"]) and w["strength"] >= 0
    ]
    strong = [w for w in valid if w["strength"] >= minimum_strength]
    weak = [w for w in valid if w["strength"] < minimum_strength]
    anchor = None
    family = []
    if strong:
        # Choose family from measured strong windows only; model estimates
        # must not influence which audio peaks are called contradictory.
        # Report ties explicitly rather than silently choosing a winner.
        rankings = []
        for seed in strong:
            members = [w for w in strong if _in_family(w["bpm"], seed["bpm"], tolerance)]
            rankings.append((len(members), sum(w["strength"] for w in members), seed["strength"], seed["bpm"], members))
        best = max(rankings, key=lambda rank: (rank[0], rank[1], rank[2], rank[3]))
        anchor = float(best[3])
        family = best[4]
    conflicts = [w for w in strong if w not in family]
    subdivision_aliases = [
        w for w in conflicts if anchor is not None
        and _subdivision_alias(w["bpm"], anchor, tolerance)
    ]
    det_support = (
        _in_family(float(deterministic_bpm), anchor, tolerance)
        if anchor is not None and _valid(deterministic_bpm) else None
    )
    neural_support = (
        _in_family(float(neural_bpm), anchor, tolerance)
        if anchor is not None and _valid(neural_bpm) else None
    )
    # Legacy field is retained as separately measured model agreement.
    neural_supported = [
        w for w in strong if _in_family(w["bpm"], neural_bpm, tolerance)
    ] if _valid(neural_bpm) else []
    return {
        "total_windows": total,
        "valid_windows": len(valid),
        "strong_windows": len(strong),
        "weak_windows": len(weak),
        "missing_windows": total - len(valid),
        "dominant_family_anchor_bpm": round(anchor, 4) if anchor is not None else None,
        "dominant_family_supported_windows": len(family),
        "conflicting_strong_windows": len(conflicts),
        "possible_two_to_three_subdivision_windows": len(subdivision_aliases),
        "possible_two_to_three_subdivision_indices": [w.get("index") for w in subdivision_aliases],
        "subdivision_aliases_are_corroboration": False,
        "deterministic_supports_dominant_family": det_support,
        "neural_supports_dominant_family": neural_support,
        "detector_disagreement": (
            det_support != neural_support
            if det_support is not None and neural_support is not None else None
        ),
        "neural_family_supported_windows": len(neural_supported),
        "strong_coverage_fraction": round(len(strong) / total, 4) if total else 0.0,
        "family_coverage_fraction": round(len(family) / total, 4) if total else 0.0,
        "support_strength_mean": (
            round(float(np.mean([w["strength"] for w in family])), 4)
            if family else None
        ),
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
    """Suggest only with sufficient distributed strong windows and neural support.

    A suggestion remains research evidence; it never changes the BPM.
    Window onset estimates are corroboration, not independent annotations.
    """
    candidates = tempo_candidates(deterministic_bpm, neural_bpm)
    window_bpms = tuple(round(float(b), 4) for b, _ in onset_windows)
    strengths = tuple(round(float(s), 4) for _, s in onset_windows)
    family_agrees = octave_family_agreement(tuple(float(b) for b, _ in onset_windows), tolerance=agreement_tolerance)
    coverage = evidence_coverage(window_diagnostics, neural_bpm, deterministic_bpm=deterministic_bpm, minimum_strength=minimum_strength, tolerance=agreement_tolerance) if window_diagnostics else None

    def outcome(decision: str, reason: str, suggestion: float | None = None):
        return TempoArbitration(
            decision=decision, suggested_bpm=suggestion,
            deterministic_bpm=deterministic_bpm, neural_bpm=neural_bpm,
            candidates_bpm=candidates, window_bpms=window_bpms,
            window_strengths=strengths, reason=reason,
            octave_family_agreement=family_agrees, window_diagnostics=window_diagnostics,
            coverage=coverage,
        )

    if not _valid(deterministic_bpm) or not _valid(neural_bpm):
        return outcome("abstain", "both BPM estimators are required")
    # Legacy direct calls without per-window data retain strict three-window
    # behavior; the benchmark supplies full coverage, including missing peaks.
    if coverage is None:
        if len(onset_windows) != 3 or any(
            not _valid(b) or not math.isfinite(s) or s < minimum_strength
            for b, s in onset_windows
        ):
            return outcome("abstain", "three sufficiently strong onset windows required")
        values = [float(b) for b, _ in onset_windows]
    else:
        if (coverage["total_windows"] < 3 or coverage["strong_windows"] < 3
            or coverage["strong_coverage_fraction"] < 0.60):
            return outcome("abstain", "insufficient strong evidence coverage across song")
        if coverage["conflicting_strong_windows"] > 0:
            return outcome("abstain", "contradictory strong onset evidence")
        if coverage["family_coverage_fraction"] < 0.60:
            return outcome("abstain", "insufficient dominant audio pulse-family coverage")
        values = [float(w["bpm"]) for w in window_diagnostics
                  if w.get("status") == "ok"
                  and _valid(w.get("bpm"))
                  and w.get("strength") is not None
                  and math.isfinite(w["strength"])
                  and w["strength"] >= minimum_strength]
        family_agrees = octave_family_agreement(tuple(values), tolerance=agreement_tolerance)
        # Strong conflict-free windows can corroborate the existing tempo at
        # its own metrical level. No alternative octave candidate is needed
        # to justify *retaining* an already-supported deterministic BPM.
        if (family_agrees and coverage["deterministic_supports_dominant_family"]
            and coverage["neural_supports_dominant_family"]
            and _in_family(float(deterministic_bpm), float(coverage["dominant_family_anchor_bpm"]), agreement_tolerance)
            and abs(float(deterministic_bpm) - float(coverage["dominant_family_anchor_bpm"]))
                / float(coverage["dominant_family_anchor_bpm"]) <= agreement_tolerance):
            return outcome("retain", "strong audio windows corroborate existing deterministic BPM; retain without selecting octave")
        # Both detectors are hypotheses, not the reference by which we decide
        # whether measured audio windows contradict one another.
        if coverage["deterministic_supports_dominant_family"] and not coverage["neural_supports_dominant_family"]:
            return outcome("retain", "strong audio-window family supports deterministic tempo; neural estimator disagrees")
        if not coverage["neural_supports_dominant_family"]:
            return outcome("abstain", "neither estimator supplies sufficient neural corroboration of audio family")

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
