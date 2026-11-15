"""Read-only validation of tempo arbitration against independently verified BPMs.

Reference categories are explicit: user-reported approximate tempos are useful
diagnostics but are NOT counted as verified validation examples.
"""
from __future__ import annotations

import math
from statistics import mean


VERIFIED_LABEL = "independently_verified"
APPROXIMATE_LABEL = "approximate"


def validate_references(references: list[dict]) -> list[dict]:
    """Reject ambiguous/invalid references before expensive audio inference."""
    if not isinstance(references, list):
        raise ValueError("References must be a JSON array")
    seen = set()
    validated = []
    for i, entry in enumerate(references):
        if not isinstance(entry, dict):
            raise ValueError(f"Reference {i}: expected an object")
        name = entry.get("filename")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Reference {i}: missing filename")
        if name in seen:
            raise ValueError(f"Duplicate reference filename: {name}")
        seen.add(name)
        bpm = entry.get("reference_bpm")
        if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"Reference {name}: BPM must be positive and finite")
        quality = entry.get("reference_quality", APPROXIMATE_LABEL)
        if quality not in (VERIFIED_LABEL, APPROXIMATE_LABEL):
            raise ValueError(f"Reference {name}: unknown reference_quality: {quality}")
        if quality == VERIFIED_LABEL and not str(entry.get("verification_method", "")).strip():
            raise ValueError(f"Reference {name}: independently verified tempo requires verification_method")
        validated.append({**entry, "reference_quality": quality})
    return validated


def prospective_impact(row: dict, *, accuracy_tolerance_bpm: float = 2.0) -> dict:
    """Evaluate a hypothetical review candidate, never adopt or modify it.

    Reference-error comparisons use straight BPM, not octave folded error.
    An approximate reference cannot establish a verified false correction.
    """
    output = {
        "reference_quality": row.get("reference_quality", APPROXIMATE_LABEL),
        "action_evaluated": "none",
        "hypothetical_adoption": False,
        "baseline_error_bpm": None,
        "candidate_error_bpm": None,
        "error_delta_bpm": None,
        "would_improve": None,
        "would_worsen": None,
        "would_introduce_error_over_tolerance": None,
    }
    comparison = row.get("comparison") or {}
    deterministic = comparison.get("deterministic")
    arbitration = row.get("arbitration") or {}
    if not deterministic or row.get("status") not in ("ok", "partial"):
        return output
    output["baseline_error_bpm"] = deterministic["absolute_error_bpm"]
    decision = arbitration.get("decision")
    if decision in ("retain", "abstain"):
        output["action_evaluated"] = decision
        return output
    suggested = arbitration.get("suggested_bpm")
    if decision != "review_candidate" or not isinstance(suggested, (int, float)) or isinstance(suggested, bool) or not math.isfinite(suggested) or suggested <= 0:
        return output
    if not isinstance(row.get("reference_bpm"), (float, int)):
        return output
    reference = row["reference_bpm"]
    candidate_error = abs(suggested - reference)
    baseline_error = deterministic["absolute_error_bpm"]
    output.update({
        "action_evaluated": "review_candidate",
        "hypothetical_adoption": True,
        "candidate_error_bpm": round(candidate_error, 4),
        "error_delta_bpm": round(candidate_error - baseline_error, 4),
        "would_improve": candidate_error < baseline_error - 0.01,
        "would_worsen": candidate_error > baseline_error + 0.01,
        "would_introduce_error_over_tolerance": (
            baseline_error <= accuracy_tolerance_bpm and candidate_error > accuracy_tolerance_bpm
        ),
    })
    return output


def validation_summary(rows: list[dict]) -> dict:
    """Separate verified holdout metrics from approximate-reference diagnostics."""
    verified = [r for r in rows if r.get("reference_quality") == VERIFIED_LABEL]
    approximate = [r for r in rows if r.get("reference_quality") != VERIFIED_LABEL]
    observed = [r for r in verified if r.get("prospective_impact", {}).get("baseline_error_bpm") is not None]
    simulated = [r["prospective_impact"] for r in verified if r.get("prospective_impact", {}).get("hypothetical_adoption")]
    false_corrections = sum(bool(item["would_introduce_error_over_tolerance"]) for item in simulated)
    return {
        "verified_reference_count": len(verified),
        "approximate_reference_count": len(approximate),
        "verified_evaluated_count": len(observed),
        "verified_mean_baseline_absolute_error_bpm": round(mean(
            r["prospective_impact"]["baseline_error_bpm"] for r in observed
        ), 4) if observed else None,
        "review_candidates_simulated": len(simulated),
        "candidate_improvements": sum(bool(r["would_improve"]) for r in simulated),
        "candidate_regressions": sum(bool(r["would_worsen"]) for r in simulated),
        "false_corrections_over_2_bpm": false_corrections,
        "false_correction_rate": round(false_corrections / len(simulated), 4) if simulated else None,
        "live_adoptions": 0,
        "ready_for_live_adoption": False,
        "note": "Metrics require verified BPM labels and remain observational. No live correction is performed.",
    }
