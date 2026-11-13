from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import analysis, candidate
from djai.brain import dj_move
from djai.intelligence import embeddings
from djai.planning_context import PlanningSession, build_pair_evidence_matrix
import djai.set_planner as set_planner


def _many_cues(track, *, count: int = 12) -> None:
    local = track.key
    track.entry_candidates = [
        candidate(
            "entry",
            float(index * 8),
            index * 4,
            "intro" if index < 2 else ("groove" if index < 8 else "drop"),
            bpm=track.bpm,
            local_key=local,
            energy=min(0.95, track.energy + 0.01 * index),
            score=0.92 - 0.015 * index,
            vocal=0.05 + 0.02 * (index % 3),
        )
        for index in range(count)
    ]
    track.exit_candidates = [
        candidate(
            "exit",
            float(80 + index * 10),
            40 + index * 5,
            "groove" if index < 5 else ("breakdown" if index < 9 else "outro"),
            bpm=track.bpm,
            local_key=local,
            energy=max(0.15, track.energy - 0.01 * index),
            score=0.94 - 0.012 * index,
            vocal=0.06 + 0.025 * (index % 4),
        )
        for index in range(count)
    ]


def test_large_crate_graph_uses_bounded_scout_cues(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_LARGE_SET_THRESHOLD", "8")
    monkeypatch.setenv("DJAI_GRAPH_SCOUT_CUES", "5")
    tracks = []
    for index in range(8):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(b"x")
        track = analysis(path, bpm=122.0 + index, energy=0.45 + index * 0.03)
        _many_cues(track)
        tracks.append(track)

    session = PlanningSession(scoring_model=None)
    graph = set_planner.build_transition_graph(tracks, workers=1, session=session)

    assert len(graph) == 8 * 7
    assert session.graph_mode == "scout"
    assert session.graph_scout_cues == 5
    assert session.graph_pair_count == 56
    assert all(len(matrix.exit_candidates) <= 5 for matrix in session.pair_evidence.values())
    assert all(len(matrix.entry_candidates) <= 5 for matrix in session.pair_evidence.values())
    assert all(edge.plan.metadata["dj_move_opportunity"]["graph_cached"] is True for edge in graph.values())


def test_final_sequence_restores_full_cue_matrix_after_scout(tmp_path: Path) -> None:
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    a = analysis(a_path, bpm=124.0, energy=0.56)
    b = analysis(b_path, bpm=126.0, energy=0.62)
    _many_cues(a)
    _many_cues(b)

    scout = build_pair_evidence_matrix(
        a,
        b,
        exit_candidates=a.exit_candidates[:3],
        entry_candidates=b.entry_candidates[:3],
        scoring_model=None,
    )
    session = PlanningSession(scoring_model=None)
    session.store((0, 1), scout)

    compiled = set_planner._compile_sequence_edges(
        [0, 1],
        [a, b],
        {},
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        minimum_body_seconds=30.0,
        minimum_body_fraction=0.35,
        session=session,
        quality_loop=None,
        automatic_delivery=False,
    )

    assert len(compiled) == 1
    restored = session.get((0, 1))
    assert restored is not None
    assert len(restored.exit_candidates) > 3
    assert len(restored.entry_candidates) > 3


def test_large_crate_limits_receding_horizon_opener_search(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_LARGE_SET_THRESHOLD", "8")
    monkeypatch.setenv("DJAI_LARGE_SET_POLICY_STARTS", "3")
    tracks = []
    for index in range(8):
        path = tmp_path / f"order-{index}.wav"
        path.write_bytes(b"x")
        tracks.append(analysis(path, bpm=120.0 + index, energy=0.42 + index * 0.04))
    graph = set_planner.build_transition_graph(tracks, workers=1, session=PlanningSession(scoring_model=None))

    calls: list[int] = []

    def fake_policy(start, graph, analyses, *, journey, end=None, config=None):
        calls.append(start)
        order = [start] + [idx for idx in range(len(analyses)) if idx != start and idx != end]
        if end is not None:
            order.append(end)
        return order, []

    monkeypatch.setattr(set_planner, "build_receding_horizon_order", fake_policy)
    monkeypatch.setattr(set_planner, "build_unified_ai_order", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip")))
    monkeypatch.setattr(set_planner, "_local_improve", lambda order, *args, **kwargs: order)

    order = set_planner.optimize_order(tracks, graph)

    assert sorted(order) == list(range(8))
    assert 1 <= len(calls) <= 3


def test_musical_timeline_is_cached_per_analysis_state(tmp_path: Path) -> None:
    path = tmp_path / "timeline.wav"
    path.write_bytes(b"x")
    track = analysis(path, bpm=126.0, energy=0.61)

    first = dj_move.build_musical_timeline(track)
    second = dj_move.build_musical_timeline(track)

    assert first is second


def test_track_embedding_uses_bounded_distributed_windows(monkeypatch) -> None:
    calls: list[int] = []

    def fake_embed_array(y, sr, *, backend=None, model_name=None, dimension=embeddings.DEFAULT_DIMENSION):
        calls.append(len(y))
        vector = np.zeros(dimension, dtype=np.float32)
        vector[len(calls) % dimension] = 1.0
        return embeddings.EmbeddingResult(
            model="fake",
            dimension=dimension,
            vector=tuple(float(value) for value in vector),
            backend="mert",
            fingerprint=f"fake-{len(calls)}",
        )

    monkeypatch.setenv("DJAI_TRACK_EMBED_WINDOW_SECONDS", "24")
    monkeypatch.setenv("DJAI_TRACK_EMBED_WINDOWS", "3")
    monkeypatch.setattr(embeddings, "embed_array", fake_embed_array)

    sr = 100
    audio = np.zeros(sr * 300, dtype=np.float32)
    result = embeddings.embed_track_array(audio, sr, backend="mert", dimension=12)

    assert result.backend == "mert"
    assert len(calls) == 3
    assert all(size <= 24 * sr + 1 for size in calls)
    assert sum(calls) <= 72 * sr + 3


def test_crate_aware_topk_budget_scales_and_retains_recovery(monkeypatch) -> None:
    from djai.brain.crate_topk import crate_topk
    from djai.brain.move_generator import CompleteMoveConfig

    monkeypatch.setenv("DJAI_CRATE_AWARE_TOPK", "1")
    small = crate_topk(4)
    medium = crate_topk(13)
    large = crate_topk(75)
    assert small.transition_pilot == 5
    assert medium.route_initial == 4
    assert medium.transition_pilot == 3
    assert medium.transition_max == 10
    assert medium.route_expansions == (6, 10)
    assert large.route_initial == 3
    assert large.transition_pilot == 2

    base = CompleteMoveConfig(mode="move_authoritative")
    initial = set_planner._large_crate_complete_move_config(base, 13)
    tiers = set_planner._progressive_sparse_configs(
        base, 13, experimental_retrieval=False
    )
    assert initial.lazy_candidate_width == 4
    assert [set_planner._large_crate_complete_move_config(tier, 13).lazy_candidate_width for tier in tiers] == [6, 10]
    assert len(set(set_planner._route_search_signature(tier, 13) for tier in tiers)) == len(tiers)


def test_crate_topk_opt_out_keeps_historical_large_crate_caps(monkeypatch) -> None:
    from djai.brain.move_generator import CompleteMoveConfig

    monkeypatch.setenv("DJAI_CRATE_AWARE_TOPK", "0")
    config = set_planner._large_crate_complete_move_config(
        CompleteMoveConfig(mode="move_authoritative"), 13
    )
    assert config.lazy_candidate_width == 6
    assert config.search_branch_width == 2
    assert config.opener_candidates == 2


def test_unauthorized_model_does_not_change_retrieval_ranking(tmp_path: Path, monkeypatch) -> None:
    from djai.brain import candidate_retrieval
    from djai.brain.set_state import SetState

    monkeypatch.setenv("DJAI_CRATE_AWARE_TOPK", "1")
    tracks = []
    for i in range(4):
        p = tmp_path / f"rank-{i}.wav"
        p.write_bytes(b"x")
        tracks.append(analysis(p, bpm=122.0 + i, energy=0.5))

    class Prediction:
        available = True
        score = 1.0
        authority = 0.0
        confidence = 1.0

    calls = []

    def fake_predict(a, b, *, set_state=None):
        calls.append(b)
        return Prediction()

    monkeypatch.setattr(candidate_retrieval, "next_track_prediction", fake_predict)
    state = SetState()
    learned = candidate_retrieval.retrieve_candidate_tracks(
        current_index=0, candidate_indices=[1, 2, 3],
        analyses=tracks, set_state=state, journey="balanced", progress=0.5,
    )
    monkeypatch.setenv("DJAI_CRATE_AWARE_TOPK", "0")
    deterministic = candidate_retrieval.retrieve_candidate_tracks(
        current_index=0, candidate_indices=[1, 2, 3],
        analyses=tracks, set_state=state, journey="balanced", progress=0.5,
    )
    assert len(calls) == 3
    assert learned.selected == deterministic.selected
    assert all(row.components["learned_weight"] == 0.0 for row in learned.ranked)

