from __future__ import annotations

from pathlib import Path

from conftest import analysis
from djai.brain.candidate_retrieval import retrieve_candidate_tracks
from djai.brain.set_state import build_set_state
from djai.planning_context import PlanningSession
from djai.set_planner import build_transition_graph, make_set_plan


def _tracks(tmp_path: Path, count: int = 8):
    keys = ("8A", "9A", "10A", "7A", "8A", "9A", "10A", "7A", "8A", "9A")
    tracks = []
    for index in range(count):
        path = tmp_path / f"lazy-{index}.wav"
        path.write_bytes(f"lazy-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=122.0 + (index % 5) * 1.5,
                camelot=keys[index],
                energy=0.34 + 0.07 * (index % 7),
            )
        )
    return tracks


def test_candidate_retrieval_shortlists_without_pair_graph(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path, 8)
    state = build_set_state([tracks[0]])
    result = retrieve_candidate_tracks(
        current_index=0,
        candidate_indices=list(range(1, 8)),
        analyses=tracks,
        set_state=state,
        journey="build",
        progress=0.25,
        width=3,
    )
    assert result.total_available == 7
    assert len(result.selected) == 3
    assert len(set(result.selected)) == 3
    assert all(index in range(1, 8) for index in result.selected)
    assert all(0.0 <= item.score <= 1.0 for item in result.ranked)


def test_transition_graph_can_materialize_only_requested_pairs(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path, 5)
    session = PlanningSession(scoring_model=None)
    requested = {(0, 1), (0, 2), (2, 4)}
    graph = build_transition_graph(
        tracks,
        workers=1,
        session=session,
        pair_indices=requested,
    )
    assert set(graph) == requested
    assert session.graph_mode == "lazy"
    assert session.graph_pair_count == len(requested)


def test_authoritative_route_uses_sparse_graph_and_lookahead(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path, 8)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_LAZY_CANDIDATE_WIDTH", "3")
    monkeypatch.setenv("DJAI_LAZY_EXPANSION_WIDTH", "2")
    monkeypatch.setenv("DJAI_LAZY_OPENER_CANDIDATES", "1")
    monkeypatch.setenv("DJAI_MOVE_LOOKAHEAD", "2")
    monkeypatch.setenv("DJAI_MOVE_SEARCH_BRANCH_WIDTH", "2")

    plan = make_set_plan(
        tracks,
        source="lazy-authoritative",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )
    report = plan.metadata["complete_move_planner"]
    possible = len(tracks) * (len(tracks) - 1)
    assert report["mode"] == "move_authoritative"
    assert report["route_authoritative"] is True
    assert report["sparse_graph_authority"] is True
    assert report["materialized_graph_pairs"] < possible
    assert plan.metadata["performance"]["planning"]["graph_mode"] == "lazy"
    decisions = report["latest_route"]["decisions"]
    assert decisions
    assert any(item["lookahead_depth"] >= 2 for item in decisions[:-1])
    assert all("retrieval" in item for item in decisions)
