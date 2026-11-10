from __future__ import annotations

import copy

from djai.brain.style_policy import style_plan_adjustment, style_profile
from djai.brain.technique_policy import apply_technique_policy
from djai.brain.transition_router import route_transition
from djai.models import GenreEstimate
from djai.planner import plan_transition_alternatives


def _set_genre(track, label: str) -> None:
    track.genre = GenreEstimate(label=label, confidence=0.95, probabilities={label: 0.95})


def _plan_with(plan, *, technique: str, bars: int, mode: str | None = None):
    item = copy.deepcopy(plan)
    item.technique = technique
    item.transition_bars = bars
    if mode is not None:
        item.transition_mode = mode
    return item


def test_house_prefers_long_bass_handoff_over_quick_reset(track_pair):
    a, b = track_pair
    _set_genre(a, "house")
    _set_genre(b, "deep_house")
    base = plan_transition_alternatives(a, b, force=True)[0]
    profile = style_profile(a, b)

    blend = _plan_with(base, technique="eq_bass_swap", bars=16)
    reset = _plan_with(base, technique="quick_cut", bars=1, mode="emergency_cut")

    blend_adjustment, _ = style_plan_adjustment(blend, profile)
    reset_adjustment, _ = style_plan_adjustment(reset, profile)

    assert profile.name == "house"
    assert blend_adjustment > reset_adjustment


def test_hip_hop_open_format_prefers_reset_over_long_transparent_blend(track_pair):
    a, b = track_pair
    _set_genre(a, "hip_hop")
    _set_genre(b, "pop")
    base = plan_transition_alternatives(a, b, force=True)[0]
    profile = style_profile(a, b)

    reset = _plan_with(base, technique="quick_cut", bars=1, mode="emergency_cut")
    blend = _plan_with(base, technique="equal_power_blend", bars=16)

    reset_adjustment, _ = style_plan_adjustment(reset, profile)
    blend_adjustment, _ = style_plan_adjustment(blend, profile)

    assert profile.name == "hip_hop_open_format"
    assert reset_adjustment > blend_adjustment


def test_dnb_rewards_double_drop_and_shorter_phrase_decision(track_pair):
    a, b = track_pair
    _set_genre(a, "drum_and_bass")
    _set_genre(b, "drum_and_bass")
    base = plan_transition_alternatives(a, b, force=True)[0]
    profile = style_profile(a, b)

    double_drop = _plan_with(base, technique="double_drop", bars=4)
    long_blend = _plan_with(base, technique="equal_power_blend", bars=16)

    dd_adjustment, _ = style_plan_adjustment(double_drop, profile)
    blend_adjustment, _ = style_plan_adjustment(long_blend, profile)

    assert profile.name == "drum_and_bass"
    assert dd_adjustment > blend_adjustment


def test_pop_penalizes_long_vocal_overlap(track_pair):
    a, b = track_pair
    _set_genre(a, "pop")
    _set_genre(b, "pop")
    base = plan_transition_alternatives(a, b, force=True)[0]
    profile = style_profile(a, b)
    plan = _plan_with(base, technique="filter_build", bars=8)
    plan.compatibility.vocal_overlap = 0.05

    adjustment, reasons = style_plan_adjustment(plan, profile)

    assert profile.name == "pop"
    assert adjustment < 0.06
    assert any("lead-vocal coexistence" in reason for reason in reasons)


def test_trance_progressive_rewards_long_blend(track_pair):
    a, b = track_pair
    _set_genre(a, "trance")
    _set_genre(b, "progressive_trance")
    base = plan_transition_alternatives(a, b, force=True)[0]
    profile = style_profile(a, b)

    long_blend = _plan_with(base, technique="eq_bass_swap", bars=16)
    short_reset = _plan_with(base, technique="quick_cut", bars=1, mode="emergency_cut")

    long_adjustment, _ = style_plan_adjustment(long_blend, profile)
    short_adjustment, _ = style_plan_adjustment(short_reset, profile)

    assert profile.name == "trance_progressive"
    assert long_adjustment > short_adjustment


def test_style_policy_is_attached_by_technique_policy(track_pair):
    a, b = track_pair
    _set_genre(a, "house")
    _set_genre(b, "house")
    plans = plan_transition_alternatives(a, b, force=True)[:4]
    route = route_transition(a, b)

    scored = apply_technique_policy(a, b, route, plans)

    assert scored
    for plan in scored:
        policy = plan.metadata["style_policy"]
        assert policy["name"] == "house"
        assert policy["hard_safety_may_override"] is True
        assert abs(float(policy["utility_adjustment"])) <= 0.10


def test_style_policy_does_not_remove_existing_hard_failure(track_pair):
    a, b = track_pair
    _set_genre(a, "hip_hop")
    _set_genre(b, "hip_hop")
    plan = plan_transition_alternatives(a, b, force=True)[0]
    plan.measurement = copy.deepcopy(plan.measurement) if plan.measurement else None
    plan.requires_confirmation = True
    plan.warning = "synthetic hard safety condition"

    profile = style_profile(a, b)
    adjustment, _ = style_plan_adjustment(
        _plan_with(plan, technique="quick_cut", bars=1, mode="emergency_cut"),
        profile,
    )

    assert adjustment > -0.10
    assert plan.requires_confirmation is True
    assert plan.warning == "synthetic hard safety condition"
