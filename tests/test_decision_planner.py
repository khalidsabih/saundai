from __future__ import annotations

from djai.decision import execution_signature, validate_execution_plan
from djai.planner import plan_transition, plan_transition_alternatives


def test_pair_plan_is_compiled_and_signed(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
    )
    validate_execution_plan(plan)
    assert plan.compiled_execution is True
    assert plan.execution_signature == execution_signature(plan)
    assert plan.render_rate_a == 1.0
    assert plan.source_duration_a > 0.0
    assert plan.intent
    assert plan.technique_assessments


def test_creative_plan_prefers_a_creative_intent(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="creative",
    )
    assert plan.intent in {
        "energy_lift",
        "breakdown_to_drop",
        "dramatic_switch",
        "vocal_handoff",
    }
    assert plan.style == "creative"


def test_alternatives_are_executable(track_pair):
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
    )
    assert [plan.style for plan in plans] == ["safe", "creative", "emergency"]
    for plan in plans:
        validate_execution_plan(plan)


def test_emergency_cut_advances_only_the_audible_incoming_tail(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="cut",
        style="emergency",
    )
    validate_execution_plan(plan)
    assert plan.transition_mode == "emergency_cut"
    assert plan.source_duration_b > 0.0
    assert plan.source_duration_b < plan.transition_seconds * plan.render_rate_b
    assert plan.source_end_b > plan.entry_time_b
