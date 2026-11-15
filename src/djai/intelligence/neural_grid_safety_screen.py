"""Research-only, fail-closed counterfactual neural beat-grid safety screen.

This module cannot authorize live adoption. The screen uses same-level BPM
agreement and fallback-beat agreement, not model self-confidence as proof of
accuracy. Independently annotated musical beat/downbeat truth is still required.
"""
from __future__ import annotations

import math
from typing import Any

MAX_BPM_RELATIVE_DIFFERENCE = 0.02
MIN_FALLBACK_BEAT_AGREEMENT = 0.85
MIN_FALLBACK_BEATS = 16
MIN_FALLBACK_DOWNBEATS = 2


def screen_full_grid_candidate(
    *,
    baseline_bpm: float,
    neural_bpm: float | None,
    fallback_beat_count: int,
    fallback_downbeat_count: int,
    neural_beat_count: int,
    neural_downbeat_count: int,
    fallback_beat_agreement: float | None,
    neural_available: bool,
    existing_gate_would_pass: bool,
) -> dict[str, Any]:
    """Return a conservative *hypothetical* decision; never change DJ output."""
    reasons: list[str] = []
    valid_bpm = (
        isinstance(neural_bpm, (float, int))
        and math.isfinite(float(neural_bpm)) and 40 <= neural_bpm <= 240
        and isinstance(baseline_bpm, (float, int))
        and math.isfinite(float(baseline_bpm)) and 40 <= baseline_bpm <= 240
    )
    bpm_relative_difference = (
        abs(neural_bpm - baseline_bpm) / baseline_bpm if valid_bpm else None
    )
    if not neural_available:
        reasons.append("neural_evidence_unavailable")
    if not existing_gate_would_pass:
        reasons.append("existing_grid_gate_did_not_pass")
    if not valid_bpm:
        reasons.append("bpm_evidence_invalid_or_outside_range")
    elif bpm_relative_difference > MAX_BPM_RELATIVE_DIFFERENCE:
        reasons.append("same_level_bpm_disagreement_over_2_percent")
    if fallback_beat_count < MIN_FALLBACK_BEATS:
        reasons.append("insufficient_fallback_beats")
    if fallback_downbeat_count < MIN_FALLBACK_DOWNBEATS:
        reasons.append("insufficient_fallback_downbeats")
    if neural_beat_count < MIN_FALLBACK_BEATS:
        reasons.append("insufficient_neural_beats")
    if neural_downbeat_count < MIN_FALLBACK_DOWNBEATS:
        reasons.append("insufficient_neural_downbeats")
    if (
        not isinstance(fallback_beat_agreement, (float, int))
        or not math.isfinite(float(fallback_beat_agreement))
        or not 0 <= fallback_beat_agreement <= 1
    ):
        reasons.append("fallback_beat_agreement_unavailable")
    elif fallback_beat_agreement < MIN_FALLBACK_BEAT_AGREEMENT:
        reasons.append("fallback_beat_agreement_below_85_percent")
    return {
        "mode": "counterfactual_observation_only",
        "would_pass_provisional_screen": not reasons,
        "screen_reasons": reasons,
        "bpm_relative_difference": (
            round(bpm_relative_difference, 6) if bpm_relative_difference is not None else None
        ),
        "limits": {
            "max_relative_bpm_difference": MAX_BPM_RELATIVE_DIFFERENCE,
            "minimum_fallback_beat_agreement": MIN_FALLBACK_BEAT_AGREEMENT,
            "minimum_fallback_beats": MIN_FALLBACK_BEATS,
            "minimum_fallback_downbeats": MIN_FALLBACK_DOWNBEATS,
        },
        "approved_for_live_adoption": False,
        "independent_beat_alignment_verified": False,
        "independent_downbeat_accuracy_verified": False,
        "changes_dj_decisions": False,
        "note": (
            "Passing the provisional screen is not authorization for live use; "
            "the deterministic grid may itself be wrong, and downbeat phase is unverified."
        ),
    }
