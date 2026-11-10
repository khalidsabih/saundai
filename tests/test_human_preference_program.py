from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai.planner import plan_transition_alternatives
from djai.training.human_preference import (
    create_human_review_session,
    evaluator_id,
    human_preference_status,
    record_human_review_vote,
)
from djai.training.taste_evidence import capture_rehearsal_evidence


def _prepare_review_workspace(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)[:2]
    event_id = "event-review-1"

    scenarios_dir = tmp_path / "datasets/generated/teacher"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    scenario_id = "teacher-scenario-review-1"
    scenario = {
        "scenario_id": scenario_id,
        "task": "transition_preference",
        "context": {"source_event_id": event_id},
        "candidates": [
            {"execution_signature": plans[0].execution_signature},
            {"execution_signature": plans[1].execution_signature},
        ],
    }
    (scenarios_dir / "scenarios.jsonl").write_text(
        json.dumps(scenario) + "\n",
        encoding="utf-8",
    )
    review = {
        "scenario_id": scenario_id,
        "priority": 0.91,
        "reasons": ["high_teacher_disagreement"],
        "task": "transition_preference",
        "consensus_choice": None,
        "agreement": 0.5,
        "calibrated_confidence": 0.58,
        "disagreement_entropy": 0.9,
        "distinct_providers": 3,
        "distinct_models": 3,
        "hard_negative_candidate": False,
    }
    (scenarios_dir / "human-review-queue.jsonl").write_text(
        json.dumps(review) + "\n",
        encoding="utf-8",
    )

    cache = tmp_path / "output/human-review-cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{event_id}.json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "local_only": True,
                "training_export_allowed": False,
                "candidates": [
                    {
                        "execution_signature": plans[0].execution_signature,
                        "plan": plans[0].to_dict(),
                    },
                    {
                        "execution_signature": plans[1].execution_signature,
                        "plan": plans[1].to_dict(),
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_render(plan, output, **kwargs):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"review-audio")
        return path

    monkeypatch.setattr("djai.training.human_preference.render_transition", fake_render)
    return plans, scenario_id


def test_evaluator_identity_is_pseudonymous_and_cohort_specific():
    general = evaluator_id("alice@example.com", "general_listener")
    pro = evaluator_id("alice@example.com", "professional_dj")

    assert "alice" not in general
    assert general != pro
    assert general.startswith("human-general_listener-")
    assert pro.startswith("human-professional_dj-")


def test_active_learning_session_replays_exact_queued_transitions_blind(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    plans, scenario_id = _prepare_review_workspace(tmp_path, track_pair, monkeypatch)

    session = create_human_review_session(
        tmp_path,
        evaluator="expert-one",
        cohort="professional_dj",
    )

    assert session["teacher_scenario_id"] == scenario_id
    assert session["blind"] is True
    assert session["candidate_order_randomized"] is True
    assert session["queue_priority"] == pytest.approx(0.91)
    signatures = {item["execution_signature"] for item in session["candidates"]}
    assert signatures == {plans[0].execution_signature, plans[1].execution_signature}
    assert all(Path(item["audio_path"]).exists() for item in session["candidates"])


def test_human_vote_records_cohort_confidence_reasons_and_ground_truth_source(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    _, scenario_id = _prepare_review_workspace(tmp_path, track_pair, monkeypatch)
    session = create_human_review_session(
        tmp_path,
        evaluator="dj-27",
        cohort="experienced_dj",
    )

    output = record_human_review_vote(
        tmp_path,
        session_id=session["session_id"],
        choice="a",
        reasons=["phrasing", "bass"],
        confidence=0.88,
        experience_years=7.5,
    )

    row = json.loads(output.read_text(encoding="utf-8").strip())
    context = row["context"]
    assert row["choice"] == "a"
    assert row["listener"].startswith("human-experienced_dj-")
    assert context["label_source"] == "human_blind_preference"
    assert context["human_preference"] is True
    assert context["teacher_scenario_id"] == scenario_id
    assert context["evaluator_cohort"] == "experienced_dj"
    assert context["human_confidence"] == pytest.approx(0.88)
    assert context["experience_years"] == pytest.approx(7.5)
    assert set(context["reasons"]) == {"phrasing", "bass"}
    assert context["blind"] is True


def test_same_evaluator_cannot_review_same_active_learning_scenario_twice(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    _prepare_review_workspace(tmp_path, track_pair, monkeypatch)
    session = create_human_review_session(
        tmp_path,
        evaluator="same-reviewer",
        cohort="professional_dj",
    )
    record_human_review_vote(
        tmp_path,
        session_id=session["session_id"],
        choice="b",
        reasons=[],
        confidence=0.7,
    )

    with pytest.raises(ValueError, match="No reviewable"):
        create_human_review_session(
            tmp_path,
            evaluator="same-reviewer",
            cohort="professional_dj",
        )


def test_human_preference_status_reports_queue_and_cohorts(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    _prepare_review_workspace(tmp_path, track_pair, monkeypatch)
    session = create_human_review_session(
        tmp_path,
        evaluator="pro-reviewer",
        cohort="professional_dj",
    )
    record_human_review_vote(
        tmp_path,
        session_id=session["session_id"],
        choice="tie",
        reasons=["energy"],
        confidence=0.65,
        experience_years=12,
    )

    status = human_preference_status(tmp_path)

    assert status["queue_total"] == 1
    assert status["queue_reviewable"] == 1
    assert status["reviewed_scenarios"] == 1
    assert status["active_learning_votes"] == 1
    assert status["cohorts"]["professional_dj"] == 1
    assert status["policy"]["human_labels_are_ground_truth_anchor"] is True
    assert status["policy"]["human_labels_bypass_hard_safety"] is False


def test_runtime_evidence_creates_local_only_exact_replay_cache(
    tmp_path: Path,
    track_pair,
    monkeypatch: pytest.MonkeyPatch,
):
    a, b = track_pair
    plans = plan_transition_alternatives(a, b, force=True)[:2]
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))

    evidence_path = capture_rehearsal_evidence(
        plans,
        selected_signature=plans[0].execution_signature,
        context={"mode": "test"},
    )

    assert evidence_path is not None and evidence_path.exists()
    evidence_rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    event_id = evidence_rows[0]["event_id"]
    cache_path = tmp_path / "output/human-review-cache" / f"{event_id}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    assert cache["local_only"] is True
    assert cache["training_export_allowed"] is False
    assert len(cache["candidates"]) == 2
    assert str(a.path) not in evidence_path.read_text(encoding="utf-8")
    assert str(b.path) not in evidence_path.read_text(encoding="utf-8")
