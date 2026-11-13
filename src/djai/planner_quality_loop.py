from __future__ import annotations

"""Quality-loop integration plumbing for set planning.

This module adapts quality-loop configuration into Professional Brain compute
settings and owns measurement-run bookkeeping. It does not choose transitions,
change acceptance gates, or alter deterministic audio safety.
"""

import os
from typing import Sequence

from .brain.move_generator import CompleteMoveDecision
from .brain.professional import ProfessionalBrainConfig
from .candidate_renderer import QualityLoopConfig
from .measurement_store import finalize_measurement_run, measurement_log_offset
from .set_models import TransitionEdge


def professional_brain_config_for_transition(
    quality_loop: QualityLoopConfig,
    preferred_complete_decision: CompleteMoveDecision | None,
) -> ProfessionalBrainConfig:
    """Apply adaptive compute budgeting without changing Professional Brain gates."""

    thinking_budget = (
        preferred_complete_decision.thinking_budget
        if preferred_complete_decision is not None
        else None
    )
    return ProfessionalBrainConfig.from_environment(
        finalists=(
            int(os.environ["DJAI_BRAIN_FINALISTS"])
            if os.environ.get("DJAI_BRAIN_FINALISTS", "").isdigit()
            else max(6, quality_loop.top_k * 2)
        ),
        adaptive_pilot_finalists=(
            thinking_budget.professional_pilot_finalists
            if thinking_budget is not None
            else None
        ),
        adaptive_rehearsal=(
            thinking_budget.professional_adaptive_rehearsal
            if thinking_budget is not None
            else None
        ),
        measurement_sample_rate=quality_loop.sample_rate,
        measurement_context_seconds=quality_loop.context_seconds,
        measurement_minimum=quality_loop.minimum_measured_score,
    )


def measurement_run_start_offset(
    quality_loop: QualityLoopConfig | None,
) -> int:
    return measurement_log_offset(
        quality_loop.log_path
        if quality_loop is not None and quality_loop.enabled
        else None
    )


def finalize_quality_measurement_run(
    quality_loop: QualityLoopConfig | None,
    *,
    start_offset: int,
    transitions: Sequence[TransitionEdge],
) -> None:
    if (
        quality_loop is None
        or not quality_loop.enabled
        or not quality_loop.log_path
    ):
        return

    finalize_measurement_run(
        quality_loop.log_path,
        start_offset=start_offset,
        accepted_signatures=[
            edge.plan.execution_signature
            for edge in transitions
            if edge.plan.measurement is not None
        ],
        accepted_plans=[
            edge.plan
            for edge in transitions
            if edge.plan.measurement is not None
        ],
    )
