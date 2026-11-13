from __future__ import annotations

"""Sequence transition-policy context plumbing.

This module builds the existing policy context and optionally applies the
bounded-advisory policy. It does not own policy scoring, safety authority,
rehearsal, selection, fallback, or execution.
"""

from typing import Callable

from .brain.crowd_state import CrowdDirective
from .brain.transition_policy import TransitionPolicyContext
from .journey import JourneyMode
from .models import TransitionPlan


def apply_sequence_transition_policy(
    plans: list[TransitionPlan],
    *,
    journey: JourneyMode,
    journey_progress: float,
    tempo_policy: str,
    current_effective_bpm: float,
    current_rate: float,
    stems_available: bool,
    authority: str,
    crowd_directive: CrowdDirective | None,
    apply_bounded_policy: Callable[..., object],
) -> TransitionPolicyContext:
    context = TransitionPolicyContext(
        journey=journey,
        journey_progress=journey_progress,
        tempo_policy=tempo_policy,
        effective_bpm_a=current_effective_bpm,
        render_rate_a=current_rate,
        stems_available=stems_available,
    )
    if authority == "bounded_advisory":
        apply_bounded_policy(
            plans,
            context=context,
            crowd_directive=crowd_directive,
        )
    return context
