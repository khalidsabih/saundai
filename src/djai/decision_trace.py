"""Opt-in, read-only transition candidate evidence snapshots.

Snapshots describe candidates presented to sequence selection, not the entire
search tree or every candidate rendered by the separate professional solver.
They never influence candidate scoring or execution.
"""
from __future__ import annotations

from typing import Any, Iterable


def candidate_snapshot(plan: Any) -> dict[str, Any]:
    meta = getattr(plan, "metadata", {}) or {}
    return {
        "signature": str(getattr(plan, "execution_signature", "")),
        "technique": getattr(plan, "technique", None),
        "style": getattr(plan, "style", None),
        "exit_time_a": getattr(plan, "exit_time_a", None),
        "entry_time_b": getattr(plan, "entry_time_b", None),
        "transition_bars": getattr(plan, "transition_bars", None),
        "target_bpm": getattr(plan, "target_bpm", None),
        "render_rate_a": getattr(plan, "render_rate_a", None),
        "render_rate_b": getattr(plan, "render_rate_b", None),
        "beatmatched": getattr(plan, "beatmatched", None),
        "score": getattr(plan, "score", None),
        "cue_confidence": getattr(plan, "cue_confidence", None),
        "fallback_level": getattr(plan, "fallback_level", None),
        "selection_utility": meta.get("selection_utility"),
        "transition_router": meta.get("transition_router"),
        "transition_ranker_authority": meta.get("transition_ranker_authority"),
        "dj_move_opportunity": meta.get("dj_move_opportunity"),
    }


def candidate_pool_trace(plans: Iterable[Any], *, limit: int = 48) -> dict[str, Any]:
    items = list(plans)
    return {
        "schema": "saund-candidate-decision-trace-v1",
        "scope": "sequence_analytical_candidates_before_professional_rehearsal",
        "total_candidates": len(items),
        "retained_candidates": min(len(items), max(0, limit)),
        "truncated": len(items) > max(0, limit),
        "candidates": [candidate_snapshot(plan) for plan in items[:max(0, limit)]],
        "limitations": [
            "Does not contain all internally generated or rehearsed candidates.",
            "Does not prove any candidate's musical quality or causal influence.",
            "Phase integrity of the rejected professional selection is captured separately if available.",
        ],
    }


def rejected_phase_trace(plan: Any, phase_integrity: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "saund-rejected-phase-integrity-v1",
        "candidate": candidate_snapshot(plan),
        "phase_integrity": dict(phase_integrity),
        "reason": "bounded continuous phase correction could not hold lock",
    }
