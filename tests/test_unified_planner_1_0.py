from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from conftest import analysis
from djai.brain.unified_planner import UnifiedPlannerConfig, build_unified_ai_order


@dataclass
class Edge:
    objective_score: float
    plan: object | None = None


def test_unified_planner_builds_complete_order(tmp_path: Path) -> None:
    tracks = [
        analysis(tmp_path / "a.wav", bpm=122.0, energy=0.36),
        analysis(tmp_path / "b.wav", bpm=124.0, energy=0.52),
        analysis(tmp_path / "c.wav", bpm=126.0, energy=0.70),
        analysis(tmp_path / "d.wav", bpm=128.0, energy=0.84),
    ]
    graph = {
        (left, right): Edge(0.82 - 0.08 * abs((right - left) - 1))
        for left in range(len(tracks))
        for right in range(len(tracks))
        if left != right
    }
    result = build_unified_ai_order(
        graph,
        tracks,
        journey="build",
        start=0,
        config=UnifiedPlannerConfig(beam_width=6, candidate_width=4),
    )
    assert result.order[0] == 0
    assert set(result.order) == {0, 1, 2, 3}
    assert len(result.trace) == 3
    assert 0.0 <= result.story_score <= 1.0


def test_unified_planner_respects_blocked_edges(tmp_path: Path) -> None:
    tracks = [
        analysis(tmp_path / "a.wav", bpm=122.0, energy=0.4),
        analysis(tmp_path / "b.wav", bpm=124.0, energy=0.5),
        analysis(tmp_path / "c.wav", bpm=126.0, energy=0.6),
    ]
    graph = {(i, j): Edge(0.8) for i in range(3) for j in range(3) if i != j}
    result = build_unified_ai_order(graph, tracks, start=0, blocked_pairs={(0, 1)})
    assert result.order[1] == 2
