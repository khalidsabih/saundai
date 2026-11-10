from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai.planner import plan_transition_alternatives
from djai.training.teacher_consensus import build_consensus_file, consensus_for_scenario
from djai.training.teacher_dataset import (
    append_teacher_judgment,
    scenarios_from_rehearsal_evidence,
    teacher_candidate,
    teacher_prompt_payload,
    transition_preference_scenario,
    validate_teacher_judgment,
    write_scenarios,
)
from djai.training.teacher_governance import teacher_data_status


def test_teacher_candidate_hides_saund_aggregate_scores(track_pair):
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True)[0]
    plan.analytical_score = 0.91
    plan.measured_score = 0.88
    plan.measurement_passed = True
    plan.metadata = {
        **plan.metadata,
        "professional_judge_score": 0.93,
        "selection_utility": 0.95,
        "state_move_taste_shadow": {"score": 0.9},
    }

    candidate = teacher_candidate(plan)

    assert "analytical_score" not in candidate
    assert "measured_score" not in candidate
    assert "measurement_passed" not in candidate
    assert "professional_judge_score" not in candidate
    assert "selection_utility" not in candidate
    assert "state_move_taste_shadow" not in candidate


def test_teacher_scenario_contains_no_local_track_paths(track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)
    scenario = transition_preference_scenario(plans[0], plans[1], context={"journey": "balanced"})
    payload = scenario.to_dict()
    serialized = json.dumps(payload)

    assert payload["lane"] == "research"
    assert payload["raw_audio_included"] is False
    assert payload["local_paths_included"] is False
    assert str(a.path) not in serialized
    assert str(b.path) not in serialized


def test_teacher_prompt_explicitly_asks_for_independent_musical_judgment(track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)
    scenario = transition_preference_scenario(plans[0], plans[1])

    payload = teacher_prompt_payload(scenario)

    assert "not SAUND's existing scores" in payload["role"]
    assert payload["task"] == "transition_preference"
    assert len(payload["candidates"]) == 2


def test_teacher_judgment_is_machine_research_supervision_not_human_truth(track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)
    scenario = transition_preference_scenario(plans[0], plans[1])

    judgment = validate_teacher_judgment(
        scenario,
        {
            "choice": "a",
            "confidence": 0.82,
            "reason_codes": ["phrasing", "bass"],
            "rationale": "A preserves the phrase and bass handoff more cleanly.",
        },
        provider="example-provider",
        model="teacher-model",
        model_version="2026-11",
        temperature=0.2,
        seed="7",
    )

    assert judgment.label_source == "llm_teacher_research"
    assert judgment.human_preference is False
    assert judgment.production_authority is False
    assert judgment.confidence == pytest.approx(0.82)


def test_invalid_teacher_preference_choice_is_rejected(track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)
    scenario = transition_preference_scenario(plans[0], plans[1])

    with pytest.raises(ValueError):
        validate_teacher_judgment(
            scenario,
            {"choice": "candidate-c", "confidence": 0.9},
            provider="provider",
            model="model",
        )


def test_consensus_requires_agreement_and_confidence():
    rows = [
        {
            "scenario_id": "teacher-scenario-1",
            "task": "transition_preference",
            "choice": "a",
            "confidence": 0.8,
            "reason_codes": ["phrasing"],
        },
        {
            "scenario_id": "teacher-scenario-1",
            "task": "transition_preference",
            "choice": "a",
            "confidence": 0.7,
            "reason_codes": ["bass"],
        },
        {
            "scenario_id": "teacher-scenario-1",
            "task": "transition_preference",
            "choice": "b",
            "confidence": 0.6,
            "reason_codes": ["energy"],
        },
    ]

    result = consensus_for_scenario(rows)

    assert result.decisive is False
    assert result.consensus_choice is None

    result = consensus_for_scenario(rows, minimum_agreement=0.66)
    assert result.decisive is True
    assert result.consensus_choice == "a"
    assert result.human_preference is False
    assert result.production_authority is False


def test_rehearsal_evidence_generates_score_blind_pair_scenarios(tmp_path: Path, track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)[:3]
    evidence = tmp_path / "evidence.jsonl"
    with evidence.open("w", encoding="utf-8") as handle:
        for index, plan in enumerate(plans):
            candidate = {
                **teacher_candidate(plan),
                "analytical_score": 0.99 - index * 0.1,
                "professional_judge_score": 0.98 - index * 0.1,
                "track_a": str(a.path),
                "track_b": str(b.path),
            }
            handle.write(
                json.dumps(
                    {
                        "event_id": "event-1",
                        "candidate": candidate,
                        "selected": index == 0,
                        "context": {"mode": "automatic"},
                    }
                )
                + "\n"
            )

    scenarios = scenarios_from_rehearsal_evidence(evidence, maximum_pairs_per_event=2)

    assert len(scenarios) == 2
    for scenario in scenarios:
        serialized = json.dumps(scenario.to_dict())
        assert "analytical_score" not in serialized
        assert "professional_judge_score" not in serialized
        assert str(a.path) not in serialized
        assert str(b.path) not in serialized
        assert scenario.context["source_event_id"] == "event-1"


def test_teacher_files_and_status_round_trip(tmp_path: Path, track_pair):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)
    scenario = transition_preference_scenario(plans[0], plans[1])
    root = tmp_path
    scenarios_path = root / "datasets/generated/teacher/scenarios.jsonl"
    judgments_path = root / "datasets/generated/teacher/judgments.jsonl"
    consensus_path = root / "datasets/generated/teacher/consensus.jsonl"

    write_scenarios(scenarios_path, [scenario])
    for provider, model, choice in (
        ("p1", "m1", "a"),
        ("p2", "m2", "a"),
    ):
        judgment = validate_teacher_judgment(
            scenario,
            {"choice": choice, "confidence": 0.9, "reason_codes": ["phrasing"]},
            provider=provider,
            model=model,
        )
        append_teacher_judgment(judgments_path, judgment)

    build_consensus_file(judgments_path, consensus_path)
    status = teacher_data_status(root)

    assert status["scenario_count"] == 1
    assert status["judgment_count"] == 2
    assert status["consensus_count"] == 1
    assert status["decisive_consensus_count"] == 1
    assert status["production_authority"] is False
    assert status["policy"]["human_validation_required_for_production_authority"] is True
