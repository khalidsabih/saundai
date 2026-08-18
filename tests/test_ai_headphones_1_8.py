from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from conftest import analysis
from djai.brain.dj_move import build_dj_state
from djai.brain.move_generator import CompleteMoveConfig, generate_complete_moves
from djai.brain.set_state import build_set_state
from djai.models import TransitionMeasurement
from djai.rehearsal import headphones as headphones_module
from djai.rehearsal.headphones import (
    AI_HEADPHONES_VERSION,
    AIHeadphonesConfig,
    AIHeadphonesDecision,
    ai_headphones_status,
    listen_to_complete_moves,
)
from djai.set_planner import build_transition_graph, make_set_plan
from djai.version import MEASUREMENT_VERSION


def _tracks(tmp_path: Path, count: int = 3):
    tracks = []
    keys = ("8A", "9A", "10A", "7A")
    for index in range(count):
        path = tmp_path / f"headphones-{index}.wav"
        path.write_bytes(f"fixture-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=122.0 + 2.0 * index,
                camelot=keys[index],
                energy=0.42 + 0.12 * index,
            )
        )
    return tracks


def _candidates(tmp_path: Path):
    tracks = _tracks(tmp_path, 3)
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
        journey="build",
        progress=0.0,
        config=CompleteMoveConfig(mode="move_authoritative"),
    )
    assert candidates
    assert len({item.next_index for item in candidates}) >= 2
    return tracks, set_state, candidates


def _judgement(score: float, *, accepted: bool = True, feasible: bool = True):
    return SimpleNamespace(
        overall=score,
        accepted=accepted,
        feasible=feasible,
        naturalness=min(1.0, score + 0.02),
        uncertainty=max(0.0, 1.0 - score),
        critic_scores={"technical": score, "rhythm": score, "phrasing": score},
        failure_domains=[] if feasible else ["technical"],
    )


def test_headphones_renders_real_finalists_and_can_change_the_commit_winner(tmp_path: Path, monkeypatch) -> None:
    tracks, set_state, candidates = _candidates(tmp_path)
    preferred = candidates[0]
    alternate = next(item for item in candidates if item.next_index != preferred.next_index)
    contexts: list[dict] = []

    def fake_measure(track_a, track_b, plans, *, config, context=None, render_context=None):
        del track_a, render_context
        contexts.append(dict(context or {}))
        plan = copy.deepcopy(plans[0])
        rendered = 0.94 if track_b.path == tracks[alternate.next_index].path else 0.64
        plan.measurement = TransitionMeasurement(
            version=MEASUREMENT_VERSION,
            sample_rate=config.sample_rate,
            duration_seconds=12.0,
            overall=rendered,
            musical_overall=rendered,
        )
        plan.measured_score = rendered
        plan.measurement_passed = True
        plan.quality_checked = True
        plan.metadata = {
            **plan.metadata,
            "rehearsal_cache_hit": False,
            "rendered_transition_embedding_captured": True,
        }
        return SimpleNamespace(selected=plan)

    def fake_judge(plan, **kwargs):
        del kwargs
        return _judgement(0.95 if plan.track_b == tracks[alternate.next_index].path else 0.66)

    monkeypatch.setattr(headphones_module, "measure_and_select_transition", fake_measure)
    monkeypatch.setattr(headphones_module, "judge_transition", fake_judge)
    decision = listen_to_complete_moves(
        preferred=preferred,
        candidates=candidates,
        analyses=tracks,
        set_state=set_state,
        journey="build",
        progress=0.25,
        listener="test-listener",
        config=AIHeadphonesConfig(
            serious_pool=12,
            minimum_renders=2,
            maximum_renders=4,
            ambiguous_margin=1.0,
            moderate_margin=1.0,
        ),
    )
    assert decision.authoritative is True
    assert decision.winner.next_index == alternate.next_index
    assert decision.winner.plan.measurement is not None
    assert len(decision.evaluations) >= 2
    assert contexts
    assert all(item["mode"] == "ai_headphones" for item in contexts)
    assert all(item["capture_render_embeddings"] is True for item in contexts)
    report = decision.to_dict()
    assert report["winner"]["changed_after_listening"] is True
    assert report["hard_audio_safety_authoritative"] is True


def test_headphones_render_failure_never_gains_authority(tmp_path: Path, monkeypatch) -> None:
    tracks, set_state, candidates = _candidates(tmp_path)
    preferred = candidates[0]

    def fail_measure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(headphones_module, "measure_and_select_transition", fail_measure)
    decision = listen_to_complete_moves(
        preferred=preferred,
        candidates=candidates,
        analyses=tracks,
        set_state=set_state,
        journey="balanced",
        progress=0.0,
        config=AIHeadphonesConfig(minimum_renders=2, maximum_renders=2),
    )
    assert decision.authoritative is False
    assert decision.winner.plan.execution_signature == preferred.plan.execution_signature
    assert decision.evaluations
    assert all(item.judgement.feasible is False for item in decision.evaluations)
    assert all(item.plan.measurement and item.plan.measurement.hard_failures for item in decision.evaluations)


def test_headphones_render_budget_expands_when_cheap_ranking_is_ambiguous(tmp_path: Path) -> None:
    _, _, candidates = _candidates(tmp_path)
    base = candidates[0]
    close = [replace(item, selection_score=0.80 - 0.005 * index) for index, item in enumerate(candidates[:4])]
    if len(close) < 4:
        close = close + [replace(base, selection_score=0.78 - 0.005 * index) for index in range(4 - len(close))]
    cfg = AIHeadphonesConfig(minimum_renders=2, maximum_renders=4, ambiguous_margin=0.02, moderate_margin=0.08)
    assert headphones_module._render_budget(close, cfg.normalized()) == 4
    separated = [replace(item, selection_score=max(0.0, 0.95 - 0.20 * index)) for index, item in enumerate(close)]
    assert headphones_module._render_budget(separated, cfg.normalized()) == 2


def test_route_scouts_openers_cheaply_then_listens_only_on_committed_route(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path, 4)
    calls: list[int] = []

    def fake_listen(*, preferred, candidates, **kwargs):
        del kwargs
        calls.append(preferred.current_index)
        return AIHeadphonesDecision(
            winner=preferred,
            preferred_before_listening=preferred,
            serious_candidates=len(candidates),
            render_budget=min(2, len(candidates)),
            authoritative=True,
            reason="test private rehearsal",
        )

    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_HEADPHONES", "1")
    monkeypatch.setattr("djai.set_planner.listen_to_complete_moves", fake_listen)
    plan = make_set_plan(
        tracks,
        source="headphones-route",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    assert len(calls) == len(plan.transitions)
    report = plan.metadata["complete_move_planner"]
    assert report["version"] == AI_HEADPHONES_VERSION
    assert report["ai_headphones"]["active_for_current_mode"] is True
    assert report["ai_headphones_authoritative_decisions"] == len(plan.transitions)
    assert all("ai_headphones" in item for item in report["latest_route"]["decisions"])



def test_headphones_expands_private_render_budget_when_rendered_judgement_is_uncertain(tmp_path: Path, monkeypatch) -> None:
    tracks, set_state, candidates = _candidates(tmp_path)
    assert len(candidates) >= 4
    ranked = [replace(item, selection_score=max(0.0, 0.98 - 0.16 * index)) for index, item in enumerate(candidates)]
    preferred = ranked[0]

    def fake_measure(track_a, track_b, plans, *, config, context=None, render_context=None):
        del track_a, track_b, context, render_context
        plan = copy.deepcopy(plans[0])
        plan.measurement = TransitionMeasurement(
            version=MEASUREMENT_VERSION,
            sample_rate=config.sample_rate,
            duration_seconds=10.0,
            overall=0.76,
            musical_overall=0.76,
        )
        plan.measured_score = 0.76
        plan.measurement_passed = True
        plan.quality_checked = True
        return SimpleNamespace(selected=plan)

    def uncertain_judge(plan, **kwargs):
        del plan, kwargs
        result = _judgement(0.74)
        result.uncertainty = 0.60
        return result

    monkeypatch.setattr(headphones_module, "measure_and_select_transition", fake_measure)
    monkeypatch.setattr(headphones_module, "judge_transition", uncertain_judge)
    decision = listen_to_complete_moves(
        preferred=preferred,
        candidates=ranked,
        analyses=tracks,
        set_state=set_state,
        journey="balanced",
        progress=0.2,
        config=AIHeadphonesConfig(
            minimum_renders=2,
            maximum_renders=4,
            ambiguous_margin=0.01,
            moderate_margin=0.02,
            uncertainty_expand_threshold=0.28,
        ),
    )
    assert decision.authoritative is True
    assert len(decision.evaluations) == 4
    assert decision.render_budget == 4

def test_headphones_runtime_status_is_honest(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_HEADPHONES", "1")
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    status = ai_headphones_status(planner_mode="move_authoritative")
    assert status["version"] == AI_HEADPHONES_VERSION
    assert status["active_for_current_mode"] is True
    assert status["commit_boundary_only"] is True
    assert status["lookahead_rendering"] is False
    assert status["render_budget"] == {"minimum": 2, "maximum": 4}
    assert status["uses_real_audio_renderer"] is True
    assert status["hard_audio_safety_authoritative"] is True
