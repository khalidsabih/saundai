from __future__ import annotations

from pathlib import Path

from djai.brain.candidate_generator import candidate_distance, diversify_transition_candidates
from djai.brain.memory import append_rejection, rejection_penalty
from djai.brain.set_policy import SetPolicyConfig, build_receding_horizon_order
from djai.brain.set_state import build_set_state, next_track_context_score
from djai.judge.professional import judge_transition
from djai.models import TransitionMeasurement
from djai.performance.program import compile_transition_program
from djai.planner import plan_transition_alternatives


def _healthy_measurement() -> TransitionMeasurement:
    return TransitionMeasurement(
        version="test",
        sample_rate=22050,
        duration_seconds=12.0,
        loudness_continuity=0.9,
        spectral_continuity=0.88,
        bass_cleanliness=0.86,
        phase_safety=0.94,
        artifact_safety=0.96,
        silence_safety=0.95,
        overall=0.90,
        rhythm_coherence=0.91,
        transient_clarity=0.89,
        groove_continuity=0.90,
        harmonic_continuity=0.78,
        energy_intent_fit=0.84,
        drop_payoff=0.82,
        musical_overall=0.88,
    )


def test_candidate_diversity_separates_dj_moves(track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")
    selected = diversify_transition_candidates(plans, count=len(plans), diversity_weight=0.5)
    assert len(selected) == len(plans)
    assert len({plan.technique for plan in selected}) >= 2
    assert max(candidate_distance(selected[0], item) for item in selected[1:]) > 0.2


def test_rejection_memory_penalizes_exact_repeat(track_pair, tmp_path: Path):
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True)[0]
    path = tmp_path / "rejected.jsonl"
    append_rejection(plan, reasons=["bad timing"], path=path)
    penalty, record = rejection_penalty(plan, path=path)
    assert penalty == 1.0
    assert record is not None
    assert "bad timing" in record.reasons


def test_professional_judge_never_overrides_hard_failure(track_pair, tmp_path: Path):
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True)[0]
    plan.measurement = _healthy_measurement()
    plan.measurement.hard_failures.append("severe clipping")
    result = judge_transition(plan, track_b=b, rejection_path=tmp_path / "none.jsonl")
    assert result.feasible is False
    assert result.accepted is False
    assert "severe clipping" in result.hard_failures


def test_professional_judge_accepts_strong_unrejected_candidate(track_pair, tmp_path: Path):
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True)[0]
    plan.measurement = _healthy_measurement()
    plan.measured_score = 0.90
    state = build_set_state([a])
    result = judge_transition(
        plan,
        track_b=b,
        set_state=state,
        rejection_path=tmp_path / "none.jsonl",
        minimum_score=0.40,
    )
    assert result.feasible is True
    assert result.accepted is True
    assert plan.metadata["professional_judge_score"] == result.overall


def test_performance_program_is_explicit_and_fingerprinted(track_pair):
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True)[0]
    program = compile_transition_program(plan)
    payload = program.to_dict()
    assert payload["intent"] == plan.intent
    assert payload["technique"] == plan.technique
    assert "deck_a_gain" in payload["curves"]
    assert len(program.fingerprint) == 24


def test_set_state_scores_current_context(track_pair):
    a, b = track_pair
    state = build_set_state([a])
    score, parts = next_track_context_score(state, b, journey="build", progress=0.5)
    assert 0.0 <= score <= 1.0
    assert {"energy", "tension", "groove", "vocal_density", "brightness", "novelty"} <= set(parts)
