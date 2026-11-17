"""Feature-flagged, beat-only preview: useful neural timing without bar authority.

This module NEVER changes TrackAnalysis, downbeat/cue/phrase data, signed plans,
or model registry decisions. Its observations are for preview/display only.
"""
from __future__ import annotations

import math
import os
from typing import Any, Mapping

import numpy as np

FLAG = "DJAI_BEAT_ONLY_PREVIEW"


def preview_enabled() -> bool:
    return os.environ.get(FLAG, "0").strip().lower() in {"1", "true", "yes", "on"}


def beat_preview_from_evidence(
    evidence: Mapping[str, Any],
    *,
    duration_seconds: float,
) -> dict[str, Any]:
    """Return safe-to-display neural pulses (not safe-to-mix bar positions).

    This quality check is mathematical, NOT a musical correctness or render
    safety claim. Exposes rejected whole-grid neural beats independently.
    """
    unavailable = {
        "available": False,
        "source": "beat_this",
        "scope": "preview_only",
        "used_for_planning": False,
        "downbeats_authoritative": False,
        "reason": "",
        "beats_seconds": [],
        "bpm": None,
        "confidence": None,
    }
    if evidence.get("available") is not True:
        return {**unavailable, "reason": "neural beat prediction unavailable"}
    try:
        raw = evidence.get("beats_seconds")
        beats = np.asarray(raw, dtype=float)
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        return {**unavailable, "reason": "invalid neural beat preview data"}
    if (beats.ndim != 1 or beats.size < 8 or beats.size > 100000
            or not np.isfinite(beats).all() or not math.isfinite(duration)
            or duration < 4.0 or np.any(beats < 0)
            or np.any(beats > duration + 0.04)
            or np.any(np.diff(beats) <= 0)):
        return {**unavailable, "reason": "invalid or unordered neural beat timestamps"}
    intervals = np.diff(beats)
    median = float(np.median(intervals))
    # Quality criterion only: plausible music tempo and no pronounced
    # half/double-time hops or long interruptions in this preview v1.
    if (not 0.18 <= median <= 2.0
            or float(np.percentile(intervals, 95)) > 1.7 * median
            or float(np.percentile(intervals, 5)) < 0.55 * median):
        return {**unavailable, "reason": "unstable neural beat interval; preview abstained"}
    stride = min(8, max(1, beats.size // 4))
    wide = (beats[stride:] - beats[:-stride]) / stride
    bpm = 60.0 / float(np.median(wide))
    if not math.isfinite(bpm) or bpm <= 0:
        return {**unavailable, "reason": "invalid neural beat tempo"}
    try:
        confidence = float(evidence.get("confidence", 0))
    except (TypeError, ValueError, OverflowError):
        confidence = 0.0
    if not math.isfinite(confidence):
        confidence = 0.0
    return {
        **unavailable,
        "available": True,
        "reason": "neural pulses available for beat-only preview; bar safety not assessed",
        "beats_seconds": [round(float(x), 6) for x in beats],
        "bpm": round(bpm, 4),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "downbeats_authoritative": False,
        "used_for_planning": False,
    }
