from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import TransitionPlan

TRANSITION_PROPOSAL_VERSION = "6.0"


@dataclass(frozen=True, slots=True)
class TransitionProposal:
    """Joint musical proposal compiled into an executable TransitionProgram.

    A proposal describes what the brain wants to do. It is deliberately
    separate from the immutable DSP IR, which describes exactly how to render
    the accepted proposal.
    """

    version: str
    track_a: str
    track_b: str
    exit_time_a: float
    entry_time_b: float
    transition_family: str
    intent: str
    technique: str
    duration_bars: int
    duration_seconds: float
    target_bpm: float
    tempo_strategy: str
    source_mode: str
    energy_shape: str
    set_intent: str | None = None
    choreography: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _energy_shape(plan: TransitionPlan) -> str:
    if plan.intent in {"energy_lift", "breakdown_to_drop"}:
        return "rise"
    if plan.intent == "energy_release":
        return "release"
    if plan.technique in {"quick_cut", "backspin", "echo_out_drop_in"}:
        return "reset"
    return "hold"


def transition_proposal_from_plan(
    plan: TransitionPlan,
    *,
    set_intent: str | None = None,
) -> TransitionProposal:
    router = dict(plan.metadata.get("transition_router", {}))
    program = dict(plan.performance_program or {})
    choreography = {
        key: value
        for key, value in dict(program.get("annotations", {})).items()
        if key in {"choreography", "incoming_alignment", "section_pair"}
    }
    return TransitionProposal(
        version=TRANSITION_PROPOSAL_VERSION,
        track_a=plan.track_a,
        track_b=plan.track_b,
        exit_time_a=round(float(plan.exit_time_a), 6),
        entry_time_b=round(float(plan.entry_time_b), 6),
        transition_family=str(router.get("family", "unclassified")),
        intent=plan.intent,
        technique=plan.technique,
        duration_bars=int(plan.transition_bars),
        duration_seconds=round(float(plan.transition_seconds), 6),
        target_bpm=round(float(plan.target_bpm), 4),
        tempo_strategy=plan.tempo_strategy,
        source_mode=str(program.get("source_mode", "master")),
        energy_shape=_energy_shape(plan),
        set_intent=set_intent,
        choreography=choreography,
    )


def attach_transition_proposal(
    plan: TransitionPlan,
    *,
    set_intent: str | None = None,
) -> TransitionPlan:
    plan.metadata = {
        **plan.metadata,
        "transition_proposal": transition_proposal_from_plan(
            plan,
            set_intent=set_intent,
        ).to_dict(),
    }
    return plan
