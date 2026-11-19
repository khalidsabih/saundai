"""Conservative opt-in bar-phase resolver for neural beat grid transition trials.

No reference annotations, known demo BPM, model downbeat labels or generator
metadata enter phase selection. Phase is inferred only from the measured onset
envelope, then independently checked against the deterministic bar landmarks.
A regular grid alone is not sufficient. Uncertain music abstains.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
from typing import Mapping, Sequence

import numpy as np

from .model_registry import load_model_registry


def trial_governance(evidence: Mapping[str, object]) -> tuple[bool, str]:
    """Never let an experimental switch override the model registry.

    This trial is not a production release path, even if a user exports its
    flag on a deployed instance. Its confidence threshold cannot be lowered
    below the current normal beat-grid threshold by changing environment.
    """
    if os.environ.get("SAUND_ENV", "").strip().lower() in {"production", "prod"} or (
        os.environ.get("SAUND_HOSTED_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
    ):
        return False, "experimental neural bar-phase trial disabled in production/hosted mode"
    if evidence.get("available") is not True:
        return False, "neural candidate unavailable"
    try:
        confidence = float(evidence.get("confidence", 0))
        claimed_authority = float(evidence.get("authority", 0))
        configured = float(os.environ.get("DJAI_NEURAL_BEAT_USE_THRESHOLD", "0.70"))
    except (TypeError, ValueError, OverflowError):
        return False, "invalid neural candidate governance evidence"
    if not all(math.isfinite(v) for v in (confidence, claimed_authority, configured)):
        return False, "nonfinite neural candidate governance evidence"
    if confidence < max(.70, configured) or claimed_authority <= 0:
        return False, "neural confidence or candidate authority below normal selection gate"
    registry = load_model_registry()
    manifest = registry.get("beat_this")
    if manifest is None or manifest.authority in {"disabled", "shadow"}:
        return False, "Beat This! model registry does not authorize neural selection"
    if registry.authority("beat_this", confidence) <= 0:
        return False, "Beat This! model registry rejects candidate confidence"
    return True, "neural candidate passes ordinary authority gate; bar phase still unverified"


@dataclass(frozen=True)
class BarPhaseResolution:
    eligible: bool
    reason: str
    meter: int
    phase: int
    acoustic_margin: float
    bar_landmark_agreement: float
    beats_seconds: tuple[float, ...]
    downbeats_seconds: tuple[float, ...]
    bpm: float

    def evidence(self) -> dict[str, object]:
        # Keep output small; raw neural beat candidate already lives in
        # neural_evidence["beat_downbeat"].
        return {k: v for k, v in asdict(self).items()
                if k not in {"beats_seconds", "downbeats_seconds"}}


def resolve_bar_phase(
    neural_beats: Sequence[float],
    deterministic_downbeats: Sequence[float],
    onset_envelope: Sequence[float],
    *,
    sample_rate: int,
    hop_length: int,
    meter: int,
    meter_confidence: float,
    duration_seconds: float,
) -> BarPhaseResolution:
    def reject(reason: str, phase: int = -1, margin: float = 0.,
               agreement: float = 0.) -> BarPhaseResolution:
        return BarPhaseResolution(False, reason, meter, phase, margin,
                                  agreement, (), (), 0.)

    if meter not in (3, 4, 6) or meter_confidence < 0.80:
        return reject("meter not independently confident")
    try:
        beats = np.asarray(neural_beats, dtype=float)
        anchors = np.asarray(deterministic_downbeats, dtype=float)
        onset = np.asarray(onset_envelope, dtype=float)
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        return reject("invalid neural or acoustic evidence")
    if (beats.ndim != 1 or anchors.ndim != 1 or onset.ndim != 1
            or beats.size < max(32, meter * 8) or anchors.size < 8
            or not np.isfinite(beats).all() or not np.isfinite(anchors).all()
            or not np.isfinite(onset).all() or np.any(onset < 0)
            or not np.isfinite(duration) or duration < 12
            or sample_rate <= 0 or hop_length <= 0
            or np.any(np.diff(beats) <= 0) or np.any(np.diff(anchors) <= 0)
            or beats[0] < 0 or beats[-1] > duration + .02):
        return reject("invalid or insufficient timing evidence")

    intervals = np.diff(beats)
    median = float(np.median(intervals))
    if not .24 <= median <= 1.5:
        return reject("unsupported neural tempo")
    # Reject pulse discontinuities before anything can affect cue planning.
    if (float(np.max(intervals)) > 1.35 * median
            or float(np.min(intervals)) < .70 * median):
        return reject("unstable neural beat intervals")
    if (beats[-1] - beats[0]) < duration * .75:
        return reject("neural beat coverage too short")
    bpm = 60.0 / float(np.median((beats[8:] - beats[:-8]) / 8))
    if not np.isfinite(bpm):
        return reject("invalid neural bpm")

    # Query a +/-1 onset-frame neighborhood to tolerate timestamp rounding.
    frames = np.rint(beats * sample_rate / hop_length).astype(int)
    if np.any(frames < 0) or np.any(frames >= len(onset)):
        return reject("neural grid outside onset envelope")
    sampled = np.maximum.reduce([
        onset[np.clip(frames + offset, 0, len(onset) - 1)]
        for offset in (-1, 0, 1)
    ])
    # Reduce large transients' leverage. Score phase medians in multiple
    # independent windows to reject songs whose accent phase drifts.
    accents = np.log1p(sampled)
    block_count = 4
    blocks = np.array_split(np.arange(len(beats)), block_count)
    votes = []
    margins = []
    for block in blocks:
        scores = np.asarray([
            np.median(accents[block[block % meter == phase]])
            if np.any(block % meter == phase) else 0.
            for phase in range(meter)
        ])
        order = np.argsort(scores)[::-1]
        best, runner = float(scores[order[0]]), float(scores[order[1]])
        if best <= 1e-7:
            return reject("insufficient onset accent evidence")
        margin = (best - runner) / best
        votes.append(int(order[0]))
        margins.append(float(margin))
    if len(set(votes)) != 1:
        return reject("acoustic bar phase disagrees between song regions")
    phase = votes[0]
    margin = min(margins)
    if margin < .22:
        return reject("ambiguous acoustic bar accent", phase, margin)
    proposed = beats[phase::meter]
    if proposed.size < 8:
        return reject("insufficient resolved bars", phase, margin)

    # Independent confirmation: at least 90% of deterministic bar landmarks
    # inside neural coverage must be within 70 ms of the same musical phase.
    # The deterministic anchors are imperfect, but agreement plus independent
    # acoustic margin is a safer opt-in than either in isolation.
    applicable = anchors[(anchors >= proposed[0]) & (anchors <= proposed[-1])]
    if applicable.size < 8:
        return reject("insufficient independent bar landmarks", phase, margin)
    indices = np.searchsorted(proposed, applicable)
    left = proposed[np.clip(indices - 1, 0, len(proposed) - 1)]
    right = proposed[np.clip(indices, 0, len(proposed) - 1)]
    error = np.minimum(np.abs(applicable - left), np.abs(applicable - right))
    agreement = float(np.mean(error <= min(.07, .15 * median)))
    if agreement < .90:
        return reject("independent bar landmarks disagree", phase, margin, agreement)

    return BarPhaseResolution(
        True, "independent acoustic phase and bar landmarks agree",
        meter, phase, margin, agreement,
        tuple(float(x) for x in beats),
        tuple(float(x) for x in proposed),
        float(bpm),
    )
