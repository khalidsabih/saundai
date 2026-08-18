from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

from conftest import analysis
from djai.brain.professional import ProfessionalBrainConfig
from djai.brain.technique_policy import duration_candidates
from djai.brain.transition_router import route_family_sequence, route_transition
from djai.brain import unified_transition as unified
from djai.judge.professional import judge_transition
from djai.planner import rank_transition_candidates
from djai.planning_context import build_pair_evidence_matrix
from djai.set_models import TransitionEdge
from djai.set_planner import optimize_order



def test_unified_acceptance_floor_survives_stale_42_env(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_BRAIN_MINIMUM_SCORE", "0.50")
    monkeypatch.delenv("DJAI_UNIFIED_MINIMUM_SCORE", raising=False)
    professional = ProfessionalBrainConfig.from_environment()
    config = unified.UnifiedTransitionConfig.from_professional(professional)
    assert professional.minimum_judge_score == 0.50
    assert config.minimum_judge_score == 0.56

def test_phrase_derived_duration_can_generate_exact_18_bar_program(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.64)
    b = analysis(tmp_path / "b.wav", bpm=127.0, camelot="9A", energy=0.67)
    for cue in a.exit_candidates:
        cue.available_bars = 18
        cue.phrase_strength = 0.93
        cue.boundary_strength = 0.88
    for cue in b.entry_candidates:
        cue.available_bars = 18
        cue.phrase_strength = 0.91
        cue.boundary_strength = 0.86

    matrix = build_pair_evidence_matrix(a, b, scoring_model=None)
    route = route_transition(a, b)
    durations = duration_candidates(route, matrix)
    assert 18 in durations

    plans = rank_transition_candidates(
        a,
        b,
        styles=("safe",),
        force=True,
        requested_bars=18,
        exact_bars=True,
        evidence_matrix=matrix,
        transition_route=route,
        limit_per_style=4,
    )
    assert plans
    assert all(plan.transition_bars == 18 for plan in plans)


def test_direct_blend_policy_removes_random_emergency_effects(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.64)
    b = analysis(tmp_path / "b.wav", bpm=127.0, camelot="9A", energy=0.67)
    route = route_transition(a, b)
    assert route.family == "direct_blend"
    plans = rank_transition_candidates(
        a,
        b,
        styles=("safe", "creative", "emergency"),
        force=True,
        transition_route=route,
        limit_per_style=5,
    )
    assert plans
    techniques = {plan.technique for plan in plans}
    assert "backspin" not in techniques
    assert "quick_cut" not in techniques



def test_complex_effect_must_earn_intervention_cost(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.64)
    b = analysis(tmp_path / "b.wav", bpm=127.0, camelot="9A", energy=0.67)
    base = rank_transition_candidates(a, b, styles=("safe",), force=True, limit_per_style=1)[0]
    clean = copy.deepcopy(base)
    flashy = copy.deepcopy(base)
    clean.technique = "equal_power_blend"
    flashy.technique = "backspin"
    clean.style = "safe"
    flashy.style = "safe"
    clean_result = judge_transition(clean, track_b=b, rejection_path=tmp_path / "none.jsonl", minimum_score=0.0)
    flashy_result = judge_transition(flashy, track_b=b, rejection_path=tmp_path / "none.jsonl", minimum_score=0.0)
    assert clean_result.overall > flashy_result.overall
    assert any("intervention cost" in reason for reason in flashy_result.reasons)

def test_route_retries_switch_musical_family(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.64)
    b = analysis(tmp_path / "b.wav", bpm=127.0, camelot="9A", energy=0.67)
    routes = route_family_sequence(a, b)
    assert routes[0].family == "direct_blend"
    assert len(routes) >= 2
    assert routes[1].family != routes[0].family


def test_review_needed_returns_replan_and_never_selected_plan(monkeypatch, tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.64)
    b = analysis(tmp_path / "b.wav", bpm=127.0, camelot="9A", energy=0.67)
    matrix = build_pair_evidence_matrix(a, b, scoring_model=None)
    plan = rank_transition_candidates(a, b, force=True, evidence_matrix=matrix, limit_per_style=1)[0]
    primary = route_transition(a, b)

    monkeypatch.setattr(unified, "route_family_sequence", lambda *_: (primary,))
    monkeypatch.setattr(
        unified,
        "_candidate_pool",
        lambda *args, **kwargs: ([copy.deepcopy(plan)], (16, 8, 4)),
    )
    monkeypatch.setattr(
        unified,
        "rehearse_candidate_pool",
        lambda *args, **kwargs: SimpleNamespace(
            selected=copy.deepcopy(plan),
            rehearsed=[copy.deepcopy(plan)],
            judgements=[object()],
        ),
    )
    monkeypatch.setattr(
        unified,
        "_judge_records",
        lambda *_: [
            {
                "execution_signature": plan.execution_signature,
                "technique": plan.technique,
                "intent": plan.intent,
                "bars": 16,
                "accepted": False,
                "feasible": True,
                "overall": 0.54,
                "uncertainty": 0.28,
                "intervention_cost": 0.05,
                "route_fit": 0.8,
                "transition_family": primary.family,
            }
        ],
    )

    result = unified.resolve_transition(
        a,
        b,
        evidence_matrix=matrix,
        professional_config=ProfessionalBrainConfig(finalists=3, retry_rounds=1),
    )
    assert result.accepted is False
    assert result.selected is None
    assert result.action in {"replan_edge", "replan_track"}
    assert any("ineligible for the final plan" in reason for reason in result.reasons)


def test_blocked_directed_edge_changes_set_order(tmp_path: Path) -> None:
    tracks = [
        analysis(tmp_path / "a.wav", bpm=124.0, camelot="8A", energy=0.55),
        analysis(tmp_path / "b.wav", bpm=125.0, camelot="8A", energy=0.60),
        analysis(tmp_path / "c.wav", bpm=126.0, camelot="9A", energy=0.65),
    ]
    graph: dict[tuple[int, int], TransitionEdge] = {}
    for i, left in enumerate(tracks):
        for j, right in enumerate(tracks):
            if i == j:
                continue
            plan = rank_transition_candidates(left, right, force=True, limit_per_style=1)[0]
            objective = 1.0
            plan.score = 0.20
            plan.metadata["transition_router"] = {"route_fit": 0.5, "bridge_preference": 0.0}
            if (i, j) == (0, 1):
                objective = 10.0
                plan.score = 0.99
            elif (i, j) == (1, 2):
                objective = 9.0
                plan.score = 0.98
            elif (i, j) == (0, 2):
                objective = 8.0
                plan.score = 0.45
            graph[(i, j)] = TransitionEdge(
                from_index=i,
                to_index=j,
                selected_style=plan.style,
                objective_score=objective,
                quality="good",
                plan=plan,
            )

    unblocked = optimize_order(tracks, graph, start_index=0)
    assert unblocked[:2] == [0, 1]

    blocked = optimize_order(tracks, graph, start_index=0, blocked_pairs={(0, 1)})
    assert blocked[:2] == [0, 2]
    assert (0, 1) not in set(zip(blocked, blocked[1:]))
