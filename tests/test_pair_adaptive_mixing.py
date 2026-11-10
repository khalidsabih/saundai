from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from djai.brain.pair_adaptive_mixing import (
    design_pair_automation,
    pair_adaptive_status,
    pair_adaptive_variants,
)
from djai.models import TransitionMeasurement
from djai.performance.program import transition_program_from_dict
from djai.planner import plan_transition
from djai.rehearsal import headphones as headphones_module
from djai.rehearsal.headphones import AIHeadphonesConfig, listen_to_complete_moves
from djai.brain.dj_move import build_dj_state
from djai.brain.move_generator import CompleteMoveConfig, generate_complete_moves
from djai.brain.set_state import build_set_state
from djai.set_planner import build_transition_graph
from djai.version import MEASUREMENT_VERSION


def _curve_value(points, position: float) -> float:
    ordered = sorted(points, key=lambda point: point.position)
    if position <= ordered[0].position:
        return ordered[0].value
    for left, right in zip(ordered, ordered[1:]):
        if left.position <= position <= right.position:
            span = max(1e-9, right.position - left.position)
            frac = (position - left.position) / span
            return left.value + frac * (right.value - left.value)
    return ordered[-1].value


def test_pair_automation_preserves_bass_and_vocal_ownership_gaps(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")

    automation = design_pair_automation(track_a, track_b, plan, variant="balanced")

    assert automation.bass_in_start > automation.bass_out_end
    assert automation.vocal_in_start > automation.vocal_out_end
    assert automation.low_overlap_gap >= 0.02
    assert automation.vocal_overlap_gap >= 0.03
    assert automation.drums_in_start < automation.bass_in_start


def test_pair_variants_compile_distinct_immutable_programs(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")

    variants = pair_adaptive_variants(track_a, track_b, plan, maximum_variants=3)

    assert len(variants) == 3
    assert {item.metadata["pair_adaptive_mix"]["variant"] for item in variants} == {
        "conservative",
        "balanced",
        "assertive",
    }
    assert len({item.execution_signature for item in variants}) == 3
    assert len({item.performance_program_fingerprint for item in variants}) == 3

    for item in variants:
        program = transition_program_from_dict(item.performance_program)
        adaptive = item.metadata["pair_adaptive_mix"]
        assert program.annotations["renderer_contract"] == "execute_pair_adaptive_program_ir_exactly"
        assert program.annotations["single_bass_owner"] is True
        assert program.annotations["avoid_dual_lead_vocals"] is True
        assert program.annotations["pair_adaptive_variant"] == adaptive["variant"]

        a_low = program.curves["deck_a_low_gain"]
        b_low = program.curves["deck_b_low_gain"]
        midpoint = (adaptive["bass_out_end"] + adaptive["bass_in_start"]) / 2.0
        assert _curve_value(a_low, midpoint) < 0.05
        assert _curve_value(b_low, midpoint) < 0.05


def test_pair_design_changes_with_pair_conflict(track_pair):
    track_a, track_b = track_pair
    clean = plan_transition(track_a, track_b, force=True, style="safe")
    risky = copy.deepcopy(clean)
    risky.compatibility.vocal_overlap = 0.15
    risky.compatibility.bass_conflict = 0.18
    risky.compatibility.harmonic = 0.35
    risky.metadata = {
        **risky.metadata,
        "professional_mix": {
            **dict(risky.metadata.get("professional_mix", {}) or {}),
            "vocal_conflict": 0.88,
            "bass_conflict": 0.82,
            "melodic_conflict": 0.74,
            "rhythm_compatibility": 0.68,
            "phrase_compatibility": 0.72,
        },
    }

    clean_design = design_pair_automation(track_a, track_b, clean, variant="balanced")
    risky_design = design_pair_automation(track_a, track_b, risky, variant="balanced")

    assert risky_design.low_overlap_gap > clean_design.low_overlap_gap
    assert risky_design.vocal_overlap_gap > clean_design.vocal_overlap_gap
    assert risky_design.vocal_in_start > clean_design.vocal_in_start


def test_structural_reset_does_not_generate_fake_adaptive_blends(track_pair):
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    plan.technique = "quick_cut"
    plan.transition_mode = "emergency_cut"
    plan.metadata = {
        **plan.metadata,
        "professional_mix": {
            **dict(plan.metadata.get("professional_mix", {}) or {}),
            "structural_reset": True,
        },
    }

    variants = pair_adaptive_variants(track_a, track_b, plan)

    assert len(variants) == 1
    assert "pair_adaptive_mix" not in variants[0].metadata


def test_headphones_auditions_variants_and_keeps_real_audio_selected_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tracks = []
    for index, bpm in enumerate((122.0, 124.0, 126.0)):
        path = tmp_path / f"pair-adaptive-{index}.wav"
        path.write_bytes(f"fixture-{index}".encode())
        from conftest import analysis

        tracks.append(analysis(path, bpm=bpm, camelot=f"{8 + index}A", energy=0.5 + 0.08 * index))

    graph = build_transition_graph(tracks, workers=1)
    set_state = build_set_state([tracks[0]])
    dj_state = build_dj_state(tracks[0], source_cursor=0.0, set_state=set_state)
    candidates = generate_complete_moves(
        current_index=0,
        candidate_indices=[1, 2],
        analyses=tracks,
        graph=graph,
        dj_state=dj_state,
        set_state=set_state,
        journey="balanced",
        progress=0.2,
        config=CompleteMoveConfig(mode="move_authoritative"),
    )
    preferred = candidates[0]
    observed_variant_sets: list[list[str]] = []

    def fake_measure(track_a, track_b, plans, *, config, context=None, render_context=None):
        del track_a, track_b, config, render_context
        variants = [
            str(dict(plan.metadata.get("pair_adaptive_mix", {}) or {}).get("variant", "legacy"))
            for plan in plans
        ]
        observed_variant_sets.append(variants)
        chosen = copy.deepcopy(
            next((plan for plan in plans if dict(plan.metadata.get("pair_adaptive_mix", {}) or {}).get("variant") == "assertive"), plans[0])
        )
        chosen.measurement = TransitionMeasurement(
            version=MEASUREMENT_VERSION,
            sample_rate=44_100,
            duration_seconds=12.0,
            overall=0.90,
            musical_overall=0.90,
        )
        chosen.measured_score = 0.90
        chosen.measurement_passed = True
        chosen.quality_checked = True
        return SimpleNamespace(selected=chosen, evaluated=[copy.deepcopy(plan) for plan in plans])

    def fake_judge(plan, **kwargs):
        del kwargs
        return SimpleNamespace(
            overall=0.92,
            accepted=True,
            feasible=True,
            naturalness=0.90,
            uncertainty=0.08,
            critic_scores={"technical": 0.92, "rhythm": 0.90, "phrasing": 0.91},
            failure_domains=[],
        )

    monkeypatch.setattr(headphones_module, "measure_and_select_transition", fake_measure)
    monkeypatch.setattr(headphones_module, "judge_transition", fake_judge)

    decision = listen_to_complete_moves(
        preferred=preferred,
        candidates=candidates,
        analyses=tracks,
        set_state=set_state,
        journey="balanced",
        progress=0.2,
        config=AIHeadphonesConfig(
            minimum_renders=1,
            maximum_renders=1,
            serious_pool=4,
        ),
    )

    assert decision.authoritative is True
    assert observed_variant_sets
    assert observed_variant_sets[0] == ["conservative", "balanced", "assertive"]
    assert decision.winner.plan.metadata["pair_adaptive_mix"]["variant"] == "assertive"
    assert decision.winner.plan.metadata["pair_adaptive_rehearsal"]["variant_count"] == 3
    assert decision.winner.plan.metadata["pair_adaptive_rehearsal"]["selected_by_real_audio_quality_gate"] is True


def test_pair_adaptive_runtime_status_is_explicit():
    status = pair_adaptive_status()

    assert status["commit_boundary_only"] is True
    assert status["graph_search_expanded"] is False
    assert status["hard_safety_authoritative"] is True
    assert "bass_ownership_timing" in status["controls"]
