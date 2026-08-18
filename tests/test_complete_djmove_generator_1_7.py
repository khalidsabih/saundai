from __future__ import annotations

from pathlib import Path

from conftest import analysis
from djai.brain.dj_move import build_dj_state
from djai.brain.move_generator import (
    COMPLETE_MOVE_VERSION,
    CompleteMoveConfig,
    complete_move_status,
    generate_complete_moves,
)
from djai.brain.set_state import build_set_state
from djai.set_planner import build_complete_move_route, build_transition_graph, make_set_plan


def _tracks(tmp_path: Path, count: int = 4):
    bpms = (122.0, 124.0, 126.0, 128.0)
    keys = ("8A", "9A", "10A", "9A")
    tracks = []
    for index in range(count):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(f"track-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=bpms[index],
                camelot=keys[index],
                energy=0.44 + 0.10 * index,
            )
        )
    return tracks


def test_complete_move_generator_jointly_binds_track_timing_and_technique(tmp_path: Path) -> None:
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
        config=CompleteMoveConfig(mode="move_shadow"),
    )

    assert candidates
    assert {candidate.next_index for candidate in candidates}.issuperset({1, 2})
    winner = candidates[0]
    assert winner.move.next_track == tracks[winner.next_index].path
    assert winner.move.exit_bar >= dj_state.source_bar + 8
    assert winner.move.entry_bar >= 0
    assert winner.move.transition_bars > 0
    assert winner.move.tempo_strategy
    assert winner.move.phase_strategy in {"phase_nudge", "beat_grid_locked"}
    assert winner.move.stem_strategy
    assert winner.move.technique
    assert winner.plan.execution_signature
    assert 0.0 <= winner.selection_score <= 1.0


def test_complete_move_generator_rejects_moves_behind_playhead(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path, 2)
    graph = build_transition_graph(tracks, workers=1)
    set_state = build_set_state([tracks[0]])
    # Fixture graph exits live around bars 75/105. At bar ~130 they are no
    # longer executable complete actions and must not be returned.
    dj_state = build_dj_state(tracks[0], source_cursor=260.0, set_state=set_state)
    candidates = generate_complete_moves(
        current_index=0,
        candidate_indices=[1],
        analyses=tracks,
        graph=graph,
        dj_state=dj_state,
        set_state=set_state,
        journey="balanced",
        progress=0.5,
        config=CompleteMoveConfig(mode="move_shadow", minimum_advance_bars=8),
    )
    assert candidates == []


def test_complete_move_route_respects_fixed_opener_and_closer(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path, 4)
    graph = build_transition_graph(tracks, workers=1)
    route = build_complete_move_route(
        tracks,
        graph,
        start_index=1,
        end_index=3,
        journey="build",
        config=CompleteMoveConfig(mode="move_shadow"),
    )
    assert route.completed is True
    assert route.order[0] == 1
    assert route.order[-1] == 3
    assert sorted(route.order) == [0, 1, 2, 3]
    assert len(route.decisions) == 3
    assert all(decision.winner.move.current_track == tracks[decision.current_index].path for decision in route.decisions)


def test_complete_move_modes_are_explicit_and_safe(monkeypatch) -> None:
    for mode in ("legacy", "move_shadow", "move_authoritative"):
        monkeypatch.setenv("DJAI_PLANNER_MODE", mode)
        cfg = CompleteMoveConfig.from_environment()
        assert cfg.mode == mode
        status = complete_move_status()
        assert status["version"] == COMPLETE_MOVE_VERSION
        assert status["planner_mode"] == mode
        assert 2 <= status["lookahead_moves"] <= 3
        assert status["lazy_candidate_retrieval"] is True
        assert status["lazy_candidate_width"] >= 3
        assert status["hard_audio_safety_authoritative"] is True


def test_shadow_mode_records_new_brain_without_taking_route_authority(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path, 3)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_shadow")
    plan = make_set_plan(
        tracks,
        source="test-shadow",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    report = plan.metadata["complete_move_planner"]
    assert report["version"] == COMPLETE_MOVE_VERSION
    assert report["mode"] == "move_shadow"
    assert report["shadow_only"] is True
    assert report["route_authoritative"] is False
    assert report["latest_route"]["completed"] is True
    assert len(report["proposed_order"]) == len(tracks)
    assert report["final_order"] == plan.order


def test_authoritative_mode_uses_complete_move_route_but_keeps_audio_safety(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path, 3)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    plan = make_set_plan(
        tracks,
        source="test-authoritative",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    report = plan.metadata["complete_move_planner"]
    assert report["mode"] == "move_authoritative"
    assert report["route_authoritative"] is True
    assert report["proposed_order"] == plan.order
    assert report["audio_safety_compiler_authoritative"] is True
    assert report["generated_transition_signature_matches"] == len(plan.transitions)
    assert all(edge.plan.metadata.get("complete_move_preference", {}).get("match") == 1.0 for edge in plan.transitions)
    assert "complete_move_route" in plan.algorithm
