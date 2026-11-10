from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai.training.teacher_consensus import build_consensus_file, consensus_for_scenario
from djai.training.teacher_quality import (
    balance_scenarios,
    build_hard_negative_queue,
    build_human_review_queue,
    build_teacher_calibration,
    calibrated_confidence,
    deduplicate_judgments,
    human_review_item,
)


def _judgment(
    scenario_id: str,
    *,
    provider: str,
    model: str,
    choice: str = "a",
    confidence: float = 0.8,
    reasons: tuple[str, ...] = ("phrasing",),
    seed: str = "1",
):
    return {
        "scenario_id": scenario_id,
        "task": "transition_preference",
        "teacher_provider": provider,
        "teacher_model": model,
        "teacher_model_version": "v1",
        "prompt_version": "1.0",
        "choice": choice,
        "confidence": confidence,
        "reason_codes": list(reasons),
        "seed": seed,
    }


def test_duplicate_teacher_judgments_are_removed_before_consensus():
    row = _judgment("scenario-1", provider="p1", model="m1")
    unique, duplicates = deduplicate_judgments([row, dict(row)])

    assert len(unique) == 1
    assert duplicates == 1


def test_consensus_requires_provider_and_model_diversity():
    rows = [
        _judgment("scenario-1", provider="p1", model="m1", seed="1"),
        _judgment("scenario-1", provider="p1", model="m2", seed="2"),
        _judgment("scenario-1", provider="p1", model="m3", seed="3"),
    ]

    result = consensus_for_scenario(rows)

    assert result.decisive is False
    assert "insufficient_provider_diversity" in result.quality_flags
    assert result.distinct_models == 3

    rows[1] = _judgment("scenario-1", provider="p2", model="m2", seed="2")
    rows[2] = _judgment("scenario-1", provider="p3", model="m3", seed="3")
    result = consensus_for_scenario(rows)

    assert result.decisive is True
    assert result.distinct_providers == 3
    assert result.distinct_models == 3
    assert result.distinct_families == 3


def test_default_confidence_is_shrunk_until_calibrated():
    row = _judgment("scenario-1", provider="p1", model="m1", confidence=0.90)
    assert calibrated_confidence(row) == pytest.approx(0.84)


def test_reference_calibration_penalizes_overconfident_teacher():
    rows = [
        _judgment("s1", provider="p1", model="m1", choice="a", confidence=0.95),
        _judgment("s2", provider="p1", model="m1", choice="a", confidence=0.95),
        _judgment("s3", provider="p1", model="m1", choice="a", confidence=0.95),
        _judgment("s4", provider="p1", model="m1", choice="a", confidence=0.95),
    ]
    calibration = build_teacher_calibration(
        rows,
        {"s1": "a", "s2": "b", "s3": "b", "s4": "b"},
    )

    profile = calibration["p1:m1:v1"]
    assert profile["accuracy"] == pytest.approx(0.25)
    assert profile["confidence_bias"] > 0.60
    assert calibrated_confidence(rows[0], calibration) < 0.60


def test_high_disagreement_is_routed_to_human_review():
    rows = [
        _judgment("scenario-1", provider="p1", model="m1", choice="a"),
        _judgment("scenario-1", provider="p2", model="m2", choice="b"),
        _judgment("scenario-1", provider="p3", model="m3", choice="neither"),
    ]
    consensus = consensus_for_scenario(rows)
    item = human_review_item(consensus.to_dict())

    assert consensus.decisive is False
    assert consensus.disagreement_entropy > 0.9
    assert item.priority >= 0.6
    assert "teacher_consensus_uncertain" in item.reasons
    assert "high_teacher_disagreement" in item.reasons


def test_balancing_caps_repeated_scenario_strata():
    scenarios = [
        {
            "scenario_id": f"s-{index}",
            "task": "transition_preference",
            "context": {"journey": "balanced"},
            "candidates": [
                {"style": "safe", "technique": "eq_bass_swap"},
                {"style": "safe", "technique": "filter_build"},
            ],
        }
        for index in range(5)
    ]

    selected = balance_scenarios(scenarios, maximum_per_stratum=2)

    assert len(selected) == 2
    assert all("data_quality_stratum" in row for row in selected)


def test_consensus_file_marks_duplicates_and_calibrated_confidence(tmp_path: Path):
    judgments = tmp_path / "judgments.jsonl"
    output = tmp_path / "consensus.jsonl"
    rows = [
        _judgment("scenario-1", provider="p1", model="m1", seed="1"),
        _judgment("scenario-1", provider="p2", model="m2", seed="2"),
        _judgment("scenario-1", provider="p3", model="m3", seed="3"),
    ]
    rows.append(dict(rows[0]))
    with judgments.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    build_consensus_file(judgments, output)
    result = json.loads(output.read_text(encoding="utf-8").strip())

    assert result["decisive"] is True
    assert result["raw_teacher_count"] == 4
    assert result["teacher_count"] == 3
    assert result["duplicate_count"] == 1
    assert result["calibrated_mean_confidence"] < result["mean_confidence"]


def test_review_and_hard_negative_queues_are_research_only(tmp_path: Path):
    consensus_path = tmp_path / "consensus.jsonl"
    review_path = tmp_path / "review.jsonl"
    hardneg_path = tmp_path / "hardneg.jsonl"
    consensus = {
        "scenario_id": "scenario-1",
        "task": "transition_preference",
        "decisive": True,
        "consensus_choice": "neither",
        "agreement": 1.0,
        "mean_confidence": 0.8,
        "calibrated_mean_confidence": 0.75,
        "distinct_providers": 2,
        "distinct_models": 2,
        "distinct_families": 2,
        "disagreement_entropy": 0.0,
        "reason_codes": ["beats", "vocals"],
    }
    consensus_path.write_text(json.dumps(consensus) + "\n", encoding="utf-8")

    build_human_review_queue(consensus_path, review_path, minimum_priority=0.01)
    build_hard_negative_queue(consensus_path, hardneg_path)

    review = json.loads(review_path.read_text(encoding="utf-8").strip())
    hardneg = json.loads(hardneg_path.read_text(encoding="utf-8").strip())

    assert review["hard_negative_candidate"] is True
    assert hardneg["label_source"] == "llm_teacher_hard_negative_research"
    assert hardneg["human_preference"] is False
    assert hardneg["production_authority"] is False
