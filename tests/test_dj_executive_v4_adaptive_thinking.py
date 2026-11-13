from __future__ import annotations

import hashlib
from pathlib import Path

from conftest import analysis
from djai.brain.executive import DJ_EXECUTIVE_VERSION, executive_status
from djai.brain.thinking_budget import ADAPTIVE_THINKING_VERSION, adaptive_thinking_status
from djai.rehearsal.headphones import AIHeadphonesDecision
from djai.set_planner import make_set_plan

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracks(tmp_path: Path, count: int = 5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = ("8A", "9A", "10A", "7A", "8A", "9A")
    tracks = []
    for index in range(count):
        path = tmp_path / f"executive-v4-{index}.wav"
        path.write_bytes(f"executive-v4-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=123.0 + 1.1 * index,
                camelot=keys[index],
                energy=0.42 + 0.08 * index,
            )
        )
    return tracks


def _configure(monkeypatch, *, easy: bool | None, headphones: bool = False) -> None:
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "authoritative")
    monkeypatch.setenv("DJAI_MOVE_LOOKAHEAD", "3")
    monkeypatch.setenv("DJAI_LAZY_OPENER_CANDIDATES", "1")
    monkeypatch.setenv("DJAI_HEADPHONES", "1" if headphones else "0")
    if easy is True:
        monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "1")
        monkeypatch.setenv("DJAI_THINKING_PREFLIGHT_EASY_THRESHOLD", "1.0")
        monkeypatch.setenv("DJAI_THINKING_EASY_THRESHOLD", "0.99")
        monkeypatch.setenv("DJAI_THINKING_BRITTLE_THRESHOLD", "1.0")
    elif easy is False:
        monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "1")
        monkeypatch.setenv("DJAI_THINKING_PREFLIGHT_EASY_THRESHOLD", "0.0")
        monkeypatch.setenv("DJAI_THINKING_EASY_THRESHOLD", "0.0")
        monkeypatch.setenv("DJAI_THINKING_BRITTLE_THRESHOLD", "0.01")
    else:
        monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")


def _plan(tmp_path: Path, monkeypatch, *, easy: bool | None, headphones: bool = False):
    _configure(monkeypatch, easy=easy, headphones=headphones)
    return make_set_plan(
        _tracks(tmp_path),
        source=f"executive-v4-{easy}",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )


def test_v4_status_exposes_brittleness_and_adaptive_thinking(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "1")
    status = executive_status()
    assert status["version"] == DJ_EXECUTIVE_VERSION == "2.5.0"
    assert status["commit_authority"] == "dj_executive_v5"
    assert status["adaptive_thinking"]["version"] == ADAPTIVE_THINKING_VERSION == "1.0"
    assert status["adaptive_thinking"]["enabled"] is True
    assert status["brittleness_controls_compute_only"] is True
    assert status["hard_safety_budget_never_reduced"] is True
    assert "assess_brittleness" in status["observed_lifecycle"]
    assert "allocate_thinking" in status["observed_lifecycle"]


def test_v4_easy_moves_use_shallow_search_and_small_rehearsal_budget(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, easy=True)
    route = plan.metadata["complete_move_planner"]["latest_route"]
    assert route and route["decisions"]
    budgets = [item["thinking_budget"] for item in route["decisions"]]
    assert all(item["brittleness"]["level"] == "easy" for item in budgets)
    assert all(item["lookahead_initial"] == 2 for item in budgets)
    assert all(item["lookahead_target"] == 2 for item in budgets)
    assert all(item["lookahead_used"] == 2 for item in budgets)
    assert all(item["headphones_minimum_renders"] == 1 for item in budgets)
    assert all(item["headphones_maximum_renders"] == 2 for item in budgets)
    assert all(item["professional_pilot_finalists"] == 1 for item in budgets)
    assert all(item["professional_adaptive_rehearsal"] is True for item in budgets)
    report = plan.metadata["dj_executive"]
    assert report["utilization"]["thinking_budget_levels"]["easy"] == len(plan.transitions)
    assert report["musical_behavior_changes"] is True


def test_v4_brittle_moves_escalate_to_deep_search_and_full_professional_rehearsal(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, easy=False)
    route = plan.metadata["complete_move_planner"]["latest_route"]
    assert route and route["decisions"]
    budgets = [item["thinking_budget"] for item in route["decisions"]]
    assert all(item["brittleness"]["level"] == "brittle" for item in budgets)
    assert budgets[0]["preflight"]["easy_candidate_context"] is False
    assert budgets[0]["lookahead_initial"] == 3
    assert budgets[0]["lookahead_target"] == 3
    assert budgets[0]["lookahead_used"] == 3
    assert budgets[0]["lookahead_escalated"] is False
    assert budgets[0]["headphones_maximum_renders"] >= 5
    assert budgets[0]["professional_adaptive_rehearsal"] is False
    assert plan.metadata["complete_move_planner"]["thinking_budget_deep_search_escalations"] == 0


def test_v4_default_preflight_sends_normal_context_directly_to_v3_depth(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "authoritative")
    monkeypatch.setenv("DJAI_MOVE_LOOKAHEAD", "3")
    monkeypatch.setenv("DJAI_LAZY_OPENER_CANDIDATES", "1")
    monkeypatch.setenv("DJAI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "1")
    monkeypatch.setenv("DJAI_THINKING_PREFLIGHT_EASY_THRESHOLD", "0.10")
    monkeypatch.setenv("DJAI_THINKING_EASY_THRESHOLD", "0.30")
    monkeypatch.setenv("DJAI_THINKING_BRITTLE_THRESHOLD", "0.62")
    plan = make_set_plan(
        _tracks(tmp_path),
        source="executive-v4-default-preflight",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    budgets = [
        item["thinking_budget"]
        for item in plan.metadata["complete_move_planner"]["latest_route"]["decisions"]
    ]
    assert budgets
    assert all(item["preflight"]["easy_candidate_context"] is False for item in budgets)
    assert all(item["lookahead_initial"] == 3 for item in budgets)
    assert all(item["lookahead_used"] == 3 for item in budgets)
    assert all(item["lookahead_escalated"] is False for item in budgets)
    assert plan.metadata["complete_move_planner"]["thinking_budget_deep_search_escalations"] == 0


def test_v4_adaptive_thinking_can_be_disabled_to_preserve_v3_budgets(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, easy=None)
    route = plan.metadata["complete_move_planner"]["latest_route"]
    assert route and route["decisions"]
    for decision in route["decisions"]:
        budget = decision["thinking_budget"]
        assert budget["enabled"] is False
        assert budget["brittleness"]["level"] == "normal"
        assert budget["lookahead_initial"] == 3
        assert budget["lookahead_target"] == 3
        assert budget["lookahead_used"] == 3
        assert budget["headphones_minimum_renders"] == 2
        assert budget["headphones_maximum_renders"] == 4
    assert plan.metadata["dj_executive"]["musical_behavior_changes"] is False


def test_v4_easy_budget_is_applied_to_ai_headphones(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_listen(*, preferred, candidates, config, **kwargs):
        del kwargs
        captured.append(config)
        return AIHeadphonesDecision(
            winner=preferred,
            preferred_before_listening=preferred,
            serious_candidates=len(candidates),
            render_budget=0,
            authoritative=False,
            reason="v4 budget capture",
        )

    _configure(monkeypatch, easy=True, headphones=True)
    monkeypatch.setattr("djai.set_planner.listen_to_complete_moves", fake_listen)
    make_set_plan(
        _tracks(tmp_path),
        source="executive-v4-headphones-easy",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    assert captured
    assert all(item.minimum_renders == 1 for item in captured)
    assert all(item.maximum_renders == 2 for item in captured)
    assert all(item.serious_pool <= 8 for item in captured)


def test_v4_brittle_budget_expands_ai_headphones_without_touching_safety(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_listen(*, preferred, candidates, config, **kwargs):
        del kwargs
        captured.append(config)
        return AIHeadphonesDecision(
            winner=preferred,
            preferred_before_listening=preferred,
            serious_candidates=len(candidates),
            render_budget=0,
            authoritative=False,
            reason="v4 brittle budget capture",
        )

    _configure(monkeypatch, easy=False, headphones=True)
    monkeypatch.setattr("djai.set_planner.listen_to_complete_moves", fake_listen)
    plan = make_set_plan(
        _tracks(tmp_path),
        source="executive-v4-headphones-brittle",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    assert captured
    assert captured[0].maximum_renders >= 5
    assert captured[0].minimum_renders >= 2
    assert all(item["hard_safety_unchanged"] is True for item in [d["thinking_budget"] for d in plan.metadata["complete_move_planner"]["latest_route"]["decisions"]])


def test_v4_transactions_record_the_budget_that_preceded_commit(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, easy=True)
    report = plan.metadata["dj_executive"]
    assert report["transactions"]
    for transaction, decision in zip(report["transactions"], report["decisions"]):
        assert "brittleness_assessed" in transaction["phase_history"]
        assert "thinking_budget_allocated" in transaction["phase_history"]
        assert transaction["thinking_budget"] == decision["evidence"]["thinking_budget"]
        assert transaction["thinking_budget"]["hard_safety_unchanged"] is True
        assert transaction["validation_passed"] is True


def test_v4_patch_preserves_rc1_frontend_transport() -> None:
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert '"/api/tracks/upload"' in api
    assert '"/api/tracks/upload-intents"' in api
    assert "EventSource" in api
    assert _sha256(ROOT / "frontend" / "next.config.ts") == "6e2f3d1a80e62e07e8757df97e11d5e450e8f78f705a6d7d995c6b9b28db556d"
    assert _sha256(ROOT / "compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"


def test_v4_budget_is_applied_to_professional_brain_without_lowering_thresholds(monkeypatch) -> None:
    from types import SimpleNamespace

    from djai.brain.thinking_budget import BrittlenessAssessment, build_thinking_budget
    from djai.candidate_renderer import QualityLoopConfig
    from djai.set_planner import _professional_brain_config_for_transition

    monkeypatch.delenv("DJAI_BRAIN_FINALISTS", raising=False)
    loop = QualityLoopConfig(enabled=True, top_k=4, sample_rate=22050, minimum_measured_score=0.42)

    easy_assessment = BrittlenessAssessment(
        version="1.0",
        score=0.20,
        level="easy",
        candidate_margin=0.20,
    )
    easy_budget = build_thinking_budget(
        assessment=easy_assessment,
        configured_lookahead=3,
        headphones_serious_pool=12,
        headphones_minimum_renders=2,
        headphones_maximum_renders=4,
        professional_pilot_finalists=2,
    )
    easy_cfg = _professional_brain_config_for_transition(
        loop, SimpleNamespace(thinking_budget=easy_budget)
    )
    assert easy_cfg.adaptive_rehearsal is True
    assert easy_cfg.adaptive_pilot_finalists == 1
    assert easy_cfg.minimum_judge_score == 0.56
    assert easy_cfg.measurement_minimum == 0.42

    brittle_assessment = BrittlenessAssessment(
        version="1.0",
        score=0.80,
        level="brittle",
        candidate_margin=0.01,
    )
    brittle_budget = build_thinking_budget(
        assessment=brittle_assessment,
        configured_lookahead=3,
        headphones_serious_pool=12,
        headphones_minimum_renders=2,
        headphones_maximum_renders=4,
        professional_pilot_finalists=2,
    )
    brittle_cfg = _professional_brain_config_for_transition(
        loop, SimpleNamespace(thinking_budget=brittle_budget)
    )
    assert brittle_cfg.adaptive_rehearsal is False
    assert brittle_cfg.finalists >= 6
    assert brittle_cfg.minimum_judge_score == easy_cfg.minimum_judge_score
    assert brittle_cfg.measurement_minimum == easy_cfg.measurement_minimum
