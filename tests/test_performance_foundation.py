from __future__ import annotations

from pathlib import Path

from djai import cache, planning_context
from djai.cache import AnalysisBatchStats, analyze_tracks, fingerprint_file
from djai.planner import plan_transition_alternatives
from djai.planning_context import PlanningSession
from djai.set_planner import build_transition_graph, make_set_plan
from djai.version import CACHE_SCHEMA_VERSION

from conftest import analysis


def _tracks(tmp_path: Path, count: int = 4):
    values = []
    camelots = ("8A", "9A", "10A", "9A", "8A", "10A")
    for index in range(count):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(f"track-{index}".encode())
        values.append(
            analysis(
                path,
                bpm=122.0 + index * 1.5,
                camelot=camelots[index],
                energy=0.44 + index * 0.08,
            )
        )
    return values


def _edge_signature(graph):
    return {
        key: (
            edge.selected_style,
            edge.objective_score,
            edge.quality,
            edge.plan.execution_signature,
            edge.plan.exit_time_a,
            edge.plan.entry_time_b,
            edge.plan.technique,
            edge.plan.intent,
            edge.plan.transition_bars,
            tuple(plan.execution_signature for plan in edge.alternatives),
        )
        for key, edge in graph.items()
    }


def test_alternatives_score_each_unique_cue_pair_once(monkeypatch, track_pair):
    track_a, track_b = track_pair
    original = planning_context.score_transition
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planning_context, "score_transition", counted)
    plans = plan_transition_alternatives(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
    )

    expected = len(track_a.exit_candidates) * len(track_b.entry_candidates)
    assert calls["count"] == expected
    assert [plan.style for plan in plans] == ["safe", "creative", "emergency"]
    assert all(plan.metadata["compatibility_matrix_size"] == expected for plan in plans)


def test_sequence_compilation_reuses_graph_evidence(monkeypatch, tmp_path: Path):
    tracks = _tracks(tmp_path, 4)
    original = planning_context.score_transition
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(planning_context, "score_transition", counted)
    plan = make_set_plan(
        tracks,
        source="test",
        journey="build",
        minimum_body_seconds=10.0,
        minimum_body_fraction=0.10,
        graph_workers=1,
    )

    expected = sum(
        len(track_a.exit_candidates) * len(track_b.entry_candidates)
        for from_index, track_a in enumerate(tracks)
        for to_index, track_b in enumerate(tracks)
        if from_index != to_index
    )
    assert calls["count"] == expected
    planning_stats = plan.metadata["performance"]["planning"]
    assert planning_stats["compatibility_evaluations"] == expected
    assert all(edge.plan.metadata.get("sequence_evidence_reused") for edge in plan.transitions)


def test_parallel_graph_matches_serial(tmp_path: Path):
    tracks = _tracks(tmp_path, 4)
    serial_session = PlanningSession()
    parallel_session = PlanningSession()

    serial = build_transition_graph(
        tracks,
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        workers=1,
        session=serial_session,
    )
    parallel = build_transition_graph(
        tracks,
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        workers=2,
        session=parallel_session,
    )

    assert _edge_signature(parallel) == _edge_signature(serial)
    assert parallel_session.graph_workers == 2
    assert parallel_session.compatibility_evaluations == serial_session.compatibility_evaluations


def test_parallel_cached_analysis_preserves_input_order(tmp_path: Path):
    cache_root = tmp_path / "cache"
    paths: list[Path] = []
    expected = []
    for index, bpm in enumerate((121.0, 123.0, 125.0)):
        path = tmp_path / f"audio-{index}.wav"
        path.write_bytes(f"audio-{index}".encode())
        item = analysis(path, bpm=bpm, energy=0.45 + index * 0.05)
        item.analysis_version = cache.ANALYSIS_VERSION
        fingerprint = fingerprint_file(path)
        cache_path = cache_root / f"{fingerprint}.json"
        cache._atomic_json_write(
            cache_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "analysis_version": cache.ANALYSIS_VERSION,
                "fingerprint": fingerprint,
                "path": str(path.resolve()),
                "analysis": item.to_dict(include_beats=True),
            },
        )
        paths.append(path)
        expected.append(item)

    serial_stats = AnalysisBatchStats()
    parallel_stats = AnalysisBatchStats()
    serial = analyze_tracks(
        paths,
        cache_dir=cache_root,
        workers=1,
        stats=serial_stats,
    )
    parallel = analyze_tracks(
        paths,
        cache_dir=cache_root,
        workers=2,
        stats=parallel_stats,
    )

    assert [item.path for item in parallel] == [item.path for item in serial]
    assert [item.bpm for item in parallel] == [item.bpm for item in expected]
    assert parallel_stats.cache_hits == 3
    assert parallel_stats.workers == 2
    assert parallel_stats.parallel is True


def test_parallel_full_set_plan_matches_serial(tmp_path: Path):
    tracks = _tracks(tmp_path, 4)

    serial = make_set_plan(
        tracks,
        source="serial",
        tempo_policy="safe",
        preferred_style="auto",
        journey="build",
        minimum_body_seconds=10.0,
        minimum_body_fraction=0.10,
        graph_workers=1,
    )
    parallel = make_set_plan(
        tracks,
        source="parallel",
        tempo_policy="safe",
        preferred_style="auto",
        journey="build",
        minimum_body_seconds=10.0,
        minimum_body_fraction=0.10,
        graph_workers=2,
    )

    assert parallel.order == serial.order
    assert [edge.selected_style for edge in parallel.transitions] == [
        edge.selected_style for edge in serial.transitions
    ]
    assert [edge.plan.execution_signature for edge in parallel.transitions] == [
        edge.plan.execution_signature for edge in serial.transitions
    ]
    assert [edge.objective_score for edge in parallel.transitions] == [
        edge.objective_score for edge in serial.transitions
    ]
    assert parallel.estimated_duration_seconds == serial.estimated_duration_seconds


def test_worker_policy_keeps_small_graphs_serial_and_honors_override(monkeypatch):
    from djai.parallel import resolve_worker_selection

    monkeypatch.delenv("DJAI_GRAPH_WORKERS", raising=False)
    automatic = resolve_worker_selection("graph", 20)
    explicit = resolve_worker_selection("graph", 20, requested=2)

    assert automatic.workers == 1
    assert automatic.parallel is False
    assert explicit.workers == 2
    assert explicit.parallel is True


def test_graph_build_remains_analytical(monkeypatch, tmp_path: Path):
    import djai.set_planner as set_planner

    tracks = _tracks(tmp_path, 4)
    calls = {"count": 0}

    def forbidden_measurement(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("Graph construction must not render or measure candidates.")

    monkeypatch.setattr(set_planner, "measure_and_select_transition", forbidden_measurement)
    graph = build_transition_graph(
        tracks,
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        workers=1,
        session=PlanningSession(),
    )

    assert len(graph) == len(tracks) * (len(tracks) - 1)
    assert calls["count"] == 0
