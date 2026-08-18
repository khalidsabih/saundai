from __future__ import annotations

from pathlib import Path

import pytest
from types import SimpleNamespace

from djai.feature_store import FeatureStore

from djai.evaluation.release_benchmark import (
    BenchmarkStore,
    BenchmarkThresholds,
    benchmark_runtime_status,
    benchmark_suite_catalog,
    compare_benchmarks,
    extract_release_benchmark,
)


def _edge(
    score: float = 0.82,
    *,
    fallback: int = 0,
    emergency: bool = False,
    hard: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    protected: bool = False,
):
    measurement = SimpleNamespace(hard_failures=list(hard), warnings=list(warnings))
    plan = SimpleNamespace(
        measured_score=score,
        score=score,
        quality_checked=True,
        measurement_passed=not hard,
        measurement=measurement,
        fallback_level=fallback,
        transition_mode="emergency_cut" if emergency else "beatmatched_blend",
        technique="blend",
        metadata={"dj_move": {"protected_payoff_violation": protected}},
    )
    return SimpleNamespace(
        plan=plan,
        objective_score=score,
        quality="poor" if hard else "good",
        warnings=list(warnings),
    )


def _plan(
    *,
    plan_id: str = "benchmark-plan",
    track_count: int = 7,
    edges=None,
    total: float = 120.0,
    planning: float = 12.0,
    render: float = 70.0,
    memory: float = 900.0,
):
    transitions = list(edges or [_edge() for _ in range(max(0, track_count - 1))])
    return SimpleNamespace(
        plan_id=plan_id,
        order=list(range(track_count)),
        tracks=[SimpleNamespace(path=f"/private/music/{index}.wav", title=f"Secret {index}") for index in range(track_count)],
        transitions=transitions,
        metadata={
            "journey_arc": {"curve_fit_score": 0.84},
            "performance": {
                "latency": {
                    "analysis_seconds": 20.0,
                    "planning_seconds": planning,
                    "render_seconds": render,
                    "total_delivery_seconds": total,
                    "process_peak_rss_mb": memory,
                }
            },
            "complete_move_planner": {
                "materialized_graph_pairs": 14,
                "possible_graph_pairs": track_count * max(0, track_count - 1),
                "ai_headphones_rendered_candidates": 9,
                "run_count": 2,
                "latest_route": {
                    "decisions": [
                        {"candidates": [{}, {}, {}]},
                        {"candidates": [{}, {}]},
                    ]
                },
            },
        },
    )


def test_release_benchmark_extracts_real_plan_metrics_without_music_paths() -> None:
    plan = _plan(
        edges=[
            _edge(warnings=("vocal clash risk",)),
            _edge(warnings=("Low-frequency buildup is 8.0 dB",)),
            _edge(protected=True),
            _edge(),
            _edge(),
            _edge(),
        ]
    )
    snapshot = extract_release_benchmark(plan, suite="small")
    payload = snapshot.to_dict()
    assert snapshot.track_count == 7
    assert snapshot.materialized_graph_pairs == 14
    assert snapshot.possible_graph_pairs == 42
    assert snapshot.sparse_graph_ratio == pytest.approx(14 / 42, abs=1e-6)
    assert snapshot.headphones_rendered_candidates == 9
    assert snapshot.candidate_count == 5
    assert snapshot.replans == 1
    assert snapshot.vocal_collision_count == 1
    assert snapshot.bass_collision_count == 1
    assert snapshot.protected_payoff_violation_count == 1
    serialized = str(payload)
    assert "/private/music" not in serialized
    assert "Secret 0" not in serialized


def test_challenger_with_hard_failure_can_never_beat_champion() -> None:
    champion = extract_release_benchmark(_plan(plan_id="champion"), suite="small")
    challenger = extract_release_benchmark(
        _plan(plan_id="fast-but-unsafe", edges=[_edge(score=0.99, hard=("clipping",))] + [_edge(score=0.99) for _ in range(5)], total=60.0),
        suite="small",
    )
    result = compare_benchmarks(champion, challenger)
    assert not result.passed
    assert any("production-safe" in blocker or "hard rendered-audio" in blocker for blocker in result.blockers)


def test_release_gate_blocks_material_performance_regression() -> None:
    champion = extract_release_benchmark(_plan(plan_id="champion", total=100.0), suite="small")
    challenger = extract_release_benchmark(_plan(plan_id="slow", total=130.0), suite="small")
    result = compare_benchmarks(
        champion,
        challenger,
        thresholds=BenchmarkThresholds(maximum_total_time_regression_fraction=0.15),
    )
    assert not result.passed
    assert any("total regressed" in blocker for blocker in result.blockers)



def test_release_gate_blocks_different_audio_crates() -> None:
    champion = extract_release_benchmark(
        _plan(plan_id="champion"), suite="small", crate_fingerprint="a" * 64
    )
    challenger = extract_release_benchmark(
        _plan(plan_id="challenger"), suite="small", crate_fingerprint="b" * 64
    )
    result = compare_benchmarks(champion, challenger)
    assert not result.passed
    assert any("crate fingerprint mismatch" in blocker for blocker in result.blockers)

def test_benchmark_store_records_and_promotes_metrics_only(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path / "bench.sqlite3")
    snapshot = extract_release_benchmark(_plan(plan_id="first"), suite="small")
    run_id = store.record(snapshot)
    store.promote("small", run_id)
    champion = store.champion("small")
    assert champion is not None
    assert champion[0] == run_id
    status = store.status()
    assert status["run_count"] == 1
    assert status["champion_count"] == 1
    raw = (tmp_path / "bench.sqlite3").read_bytes()
    assert b"/private/music" not in raw
    assert b"Secret 0" not in raw


def test_benchmark_runtime_status_is_honest_before_first_run(tmp_path: Path) -> None:
    status = benchmark_runtime_status(tmp_path)
    assert status["version"] == "1.9.0"
    assert status["available"] is False
    assert status["run_count"] == 0
    assert status["privacy"] == "metrics_only_no_track_paths"
    assert status["suite_catalog"]["small"]["track_count"] == 7
    assert benchmark_suite_catalog()["stress"]["track_count"] == 100


def test_runtime_status_does_not_expose_champion_plan_ids(tmp_path: Path) -> None:
    db = tmp_path / "benchmarks" / "benchmark-results.sqlite3"
    store = BenchmarkStore(db)
    run_id = store.record(extract_release_benchmark(_plan(plan_id="private-plan-id"), suite="small"))
    store.promote("small", run_id)
    status = benchmark_runtime_status(tmp_path)
    assert status["available"] is True
    assert status["champion_count"] == 1
    assert "plan_id" not in status["champions"][0]


def test_cached_source_fingerprint_never_hashes_stale_source(tmp_path: Path) -> None:
    track = tmp_path / "track.wav"
    track.write_bytes(b"audio-v1")
    store = FeatureStore(tmp_path / "features")
    assert store.cached_source_fingerprint(track) is None
    store.register_source_fingerprint(track, "a" * 64)
    assert store.cached_source_fingerprint(track) == "a" * 64

    track.write_bytes(b"audio-v2-longer")
    assert store.cached_source_fingerprint(track) is None
