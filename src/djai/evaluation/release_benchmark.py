from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..set_models import SetPlan
from ..version import DJAI_VERSION
from .full_set import evaluate_full_set

RELEASE_BENCHMARK_VERSION = "1.9.0"
DEFAULT_BENCHMARK_DB = "benchmarks/benchmark-results.sqlite3"
_ALLOWED_HUMAN_RESULTS = {None, "champion", "challenger", "tie", "neither"}


BENCHMARK_SUITES: dict[str, dict[str, Any]] = {
    "small": {"track_count": 7, "kind": "scale", "purpose": "fast release smoke benchmark"},
    "standard": {"track_count": 14, "kind": "scale", "purpose": "default release comparison"},
    "large": {"track_count": 30, "kind": "scale", "purpose": "large-crate planning and render benchmark"},
    "stress": {"track_count": 100, "kind": "scale", "purpose": "sparse-search and resource stress benchmark"},
    "vocal-heavy": {"track_count": None, "kind": "specialist", "purpose": "vocal collision and handoff benchmark"},
    "mixed-bpm": {"track_count": None, "kind": "specialist", "purpose": "tempo trajectory and bridge benchmark"},
    "dnb": {"track_count": None, "kind": "specialist", "purpose": "drum-and-bass phrase and energy benchmark"},
    "house": {"track_count": None, "kind": "specialist", "purpose": "house groove and long-blend benchmark"},
    "techno": {"track_count": None, "kind": "specialist", "purpose": "techno structure and tension benchmark"},
    "pop-electronic": {"track_count": None, "kind": "specialist", "purpose": "vocal/pop structure benchmark"},
    "difficult-bridges": {"track_count": None, "kind": "specialist", "purpose": "genre/energy bridge benchmark"},
}


def benchmark_suite_catalog() -> dict[str, dict[str, Any]]:
    return {name: dict(value) for name, value in BENCHMARK_SUITES.items()}


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    maximum_quality_regression: float = 0.015
    maximum_analysis_time_regression_fraction: float = 0.20
    maximum_total_time_regression_fraction: float = 0.15
    maximum_planning_time_regression_fraction: float = 0.20
    maximum_render_time_regression_fraction: float = 0.15
    maximum_memory_regression_fraction: float = 0.20
    maximum_fallback_increase: int = 0
    require_performance_telemetry: bool = True
    require_human_preference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseBenchmarkSnapshot:
    benchmark_version: str
    saund_version: str
    suite: str
    plan_id: str
    crate_fingerprint: str | None
    captured_at_utc: str
    track_count: int
    transition_count: int
    full_set_overall: float
    transition_quality: float
    worst_transition: float
    story_fit: float
    quality_gate_pass_rate: float
    production_safe: bool
    fallback_count: int
    emergency_count: int
    hard_failure_count: int
    phrase_violation_count: int
    protected_payoff_violation_count: int
    vocal_collision_count: int
    bass_collision_count: int
    tempo_abnormality_count: int
    analysis_seconds: float | None
    planning_seconds: float | None
    render_seconds: float | None
    total_seconds: float | None
    process_peak_rss_mb: float | None
    materialized_graph_pairs: int
    possible_graph_pairs: int
    sparse_graph_ratio: float
    candidate_count: int
    headphones_rendered_candidates: int
    replans: int
    human_blind_preference: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    benchmark_version: str
    suite: str
    champion_plan_id: str
    challenger_plan_id: str
    passed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    deltas: dict[str, float | int | None]
    thresholds: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        payload["warnings"] = list(self.warnings)
        return payload


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_messages(plan: SetPlan, needles: tuple[str, ...]) -> int:
    count = 0
    for edge in plan.transitions:
        messages: list[str] = list(edge.warnings)
        measurement = edge.plan.measurement
        if measurement is not None:
            messages.extend(measurement.hard_failures)
            messages.extend(measurement.warnings)
        messages.extend(str(value) for value in edge.plan.metadata.get("warnings", []) if isinstance(value, str))
        haystack = "\n".join(messages).lower()
        if any(needle in haystack for needle in needles):
            count += 1
    return count


def _protected_payoff_violations(plan: SetPlan) -> int:
    total = 0
    for edge in plan.transitions:
        meta = _mapping(edge.plan.metadata.get("dj_move"))
        if bool(meta.get("protected_payoff_violation", False)):
            total += 1
    return total


def _candidate_count(complete: Mapping[str, Any]) -> int:
    latest = _mapping(complete.get("latest_route"))
    decisions = latest.get("decisions")
    if not isinstance(decisions, list):
        return 0
    count = 0
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        candidates = decision.get("candidates")
        if isinstance(candidates, list):
            count += len(candidates)
    return count


def combine_crate_fingerprints(fingerprints: list[str] | tuple[str, ...]) -> str | None:
    normalized = sorted(str(value).strip().lower() for value in fingerprints if str(value).strip())
    if not normalized:
        return None
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_release_benchmark(
    plan: SetPlan,
    *,
    suite: str = "ad-hoc",
    crate_fingerprint: str | None = None,
    stage_timings: Mapping[str, Any] | None = None,
    process_peak_rss_mb: float | None = None,
    human_blind_preference: str | None = None,
    notes: str | None = None,
) -> ReleaseBenchmarkSnapshot:
    if human_blind_preference not in _ALLOWED_HUMAN_RESULTS:
        raise ValueError(
            "human_blind_preference must be one of: champion, challenger, tie, neither"
        )
    full = evaluate_full_set(plan)
    performance = _mapping(plan.metadata.get("performance"))
    latency = _mapping(performance.get("latency"))
    supplied = _mapping(stage_timings)
    latency.update(supplied)
    complete = _mapping(plan.metadata.get("complete_move_planner"))
    existing_release = _mapping(plan.metadata.get("release_benchmark"))
    resolved_crate_fingerprint = (
        str(crate_fingerprint).strip().lower()
        if crate_fingerprint
        else str(existing_release.get("crate_fingerprint") or "").strip().lower() or None
    )
    possible = int(complete.get("possible_graph_pairs", len(plan.tracks) * max(0, len(plan.tracks) - 1)) or 0)
    materialized = int(complete.get("materialized_graph_pairs", plan.metadata.get("directed_edge_count", 0)) or 0)
    ratio = (materialized / possible) if possible > 0 else 0.0

    fallback_count = sum(int(edge.plan.fallback_level > 0) for edge in plan.transitions)
    emergency_count = sum(int(edge.plan.transition_mode == "emergency_cut") for edge in plan.transitions)
    hard_failures = 0
    for edge in plan.transitions:
        if edge.plan.measurement is not None:
            hard_failures += len(edge.plan.measurement.hard_failures)

    return ReleaseBenchmarkSnapshot(
        benchmark_version=RELEASE_BENCHMARK_VERSION,
        saund_version=DJAI_VERSION,
        suite=str(suite or "ad-hoc"),
        plan_id=plan.plan_id,
        crate_fingerprint=resolved_crate_fingerprint,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        track_count=len(plan.order),
        transition_count=len(plan.transitions),
        full_set_overall=float(full.overall),
        transition_quality=float(full.transition_quality),
        worst_transition=float(full.worst_transition),
        story_fit=float(full.story_fit),
        quality_gate_pass_rate=float(full.quality_gate_pass_rate),
        production_safe=bool(full.production_safe),
        fallback_count=fallback_count,
        emergency_count=emergency_count,
        hard_failure_count=hard_failures,
        phrase_violation_count=_count_messages(plan, ("phrase", "phrasing")),
        protected_payoff_violation_count=_protected_payoff_violations(plan),
        vocal_collision_count=_count_messages(plan, ("vocal clash", "vocal collision", "vocal overlap")),
        bass_collision_count=_count_messages(plan, ("bass collision", "low-frequency buildup", "low frequency buildup")),
        tempo_abnormality_count=_count_messages(plan, ("tempo artifact", "tempo abnormal", "forced tempo")),
        analysis_seconds=_number(latency.get("analysis_seconds")),
        planning_seconds=_number(latency.get("planning_seconds")),
        render_seconds=_number(latency.get("render_seconds")),
        total_seconds=_number(latency.get("total_delivery_seconds") or latency.get("total_seconds")),
        process_peak_rss_mb=_number(process_peak_rss_mb if process_peak_rss_mb is not None else latency.get("process_peak_rss_mb")),
        materialized_graph_pairs=materialized,
        possible_graph_pairs=possible,
        sparse_graph_ratio=round(ratio, 6),
        candidate_count=int(complete.get("complete_move_generated_candidates", 0) or 0) or _candidate_count(complete),
        headphones_rendered_candidates=int(complete.get("ai_headphones_rendered_candidates", 0) or 0),
        replans=max(0, int(complete.get("run_count", 1) or 1) - 1),
        human_blind_preference=human_blind_preference,
        notes=(str(notes).strip() or None) if notes is not None else None,
    )


def _relative_delta(champion: float | None, challenger: float | None) -> float | None:
    if champion is None or challenger is None or champion <= 1e-9:
        return None
    return (challenger - champion) / champion


def compare_benchmarks(
    champion: ReleaseBenchmarkSnapshot,
    challenger: ReleaseBenchmarkSnapshot,
    *,
    thresholds: BenchmarkThresholds | None = None,
) -> BenchmarkComparison:
    cfg = thresholds or BenchmarkThresholds()
    blockers: list[str] = []
    warnings: list[str] = []
    if champion.suite != challenger.suite:
        blockers.append(f"suite mismatch: {champion.suite} != {challenger.suite}")
    if champion.track_count != challenger.track_count:
        blockers.append(
            f"track-count mismatch: {champion.track_count} != {challenger.track_count}"
        )
    if champion.crate_fingerprint and challenger.crate_fingerprint:
        if champion.crate_fingerprint != challenger.crate_fingerprint:
            blockers.append("crate fingerprint mismatch: champion and challenger are not the same audio crate")
    else:
        warnings.append("crate identity could not be cryptographically verified for both runs")
    if not challenger.production_safe:
        blockers.append("challenger is not production-safe")
    if challenger.hard_failure_count > 0:
        blockers.append(f"challenger has {challenger.hard_failure_count} hard rendered-audio failures")
    if challenger.protected_payoff_violation_count > champion.protected_payoff_violation_count:
        blockers.append("protected-payoff violations increased")
    for label, left, right in (
        ("phrase violations", champion.phrase_violation_count, challenger.phrase_violation_count),
        ("vocal collisions", champion.vocal_collision_count, challenger.vocal_collision_count),
        ("bass collisions", champion.bass_collision_count, challenger.bass_collision_count),
        ("tempo abnormalities", champion.tempo_abnormality_count, challenger.tempo_abnormality_count),
    ):
        if right > left:
            blockers.append(f"{label} increased: {left} -> {right}")
    if challenger.fallback_count > champion.fallback_count + cfg.maximum_fallback_increase:
        blockers.append("fallback count regressed")

    quality_delta = challenger.full_set_overall - champion.full_set_overall
    if quality_delta < -abs(cfg.maximum_quality_regression):
        blockers.append(
            f"full-set quality regressed by {abs(quality_delta):.3f}, beyond allowed {cfg.maximum_quality_regression:.3f}"
        )

    timing_specs = (
        ("analysis", champion.analysis_seconds, challenger.analysis_seconds, cfg.maximum_analysis_time_regression_fraction),
        ("total", champion.total_seconds, challenger.total_seconds, cfg.maximum_total_time_regression_fraction),
        ("planning", champion.planning_seconds, challenger.planning_seconds, cfg.maximum_planning_time_regression_fraction),
        ("render", champion.render_seconds, challenger.render_seconds, cfg.maximum_render_time_regression_fraction),
        ("memory", champion.process_peak_rss_mb, challenger.process_peak_rss_mb, cfg.maximum_memory_regression_fraction),
    )
    timing_deltas: dict[str, float | None] = {}
    for label, left, right, limit in timing_specs:
        delta = _relative_delta(left, right)
        timing_deltas[label] = delta
        if delta is None:
            message = f"{label} regression could not be evaluated because one run lacks telemetry"
            if cfg.require_performance_telemetry and label in {"analysis", "total", "planning", "render"}:
                blockers.append(message)
            else:
                warnings.append(message)
        elif delta > limit:
            blockers.append(f"{label} regressed by {delta:.1%}, beyond allowed {limit:.1%}")

    if cfg.require_human_preference:
        if challenger.human_blind_preference is None:
            blockers.append("human blind preference is required but missing")
        elif challenger.human_blind_preference not in {"challenger", "tie"}:
            blockers.append(
                f"human blind preference did not support challenger ({challenger.human_blind_preference})"
            )

    deltas: dict[str, float | int | None] = {
        "quality": round(quality_delta, 6),
        "transition_quality": round(challenger.transition_quality - champion.transition_quality, 6),
        "worst_transition": round(challenger.worst_transition - champion.worst_transition, 6),
        "fallback_count": challenger.fallback_count - champion.fallback_count,
        "hard_failure_count": challenger.hard_failure_count - champion.hard_failure_count,
        "protected_payoff_violation_count": challenger.protected_payoff_violation_count - champion.protected_payoff_violation_count,
        "analysis_time_fraction": timing_deltas["analysis"],
        "total_time_fraction": timing_deltas["total"],
        "planning_time_fraction": timing_deltas["planning"],
        "render_time_fraction": timing_deltas["render"],
        "memory_fraction": timing_deltas["memory"],
        "phrase_violation_count": challenger.phrase_violation_count - champion.phrase_violation_count,
        "vocal_collision_count": challenger.vocal_collision_count - champion.vocal_collision_count,
        "bass_collision_count": challenger.bass_collision_count - champion.bass_collision_count,
        "tempo_abnormality_count": challenger.tempo_abnormality_count - champion.tempo_abnormality_count,
        "candidate_count": challenger.candidate_count - champion.candidate_count,
        "headphones_rendered_candidates": challenger.headphones_rendered_candidates - champion.headphones_rendered_candidates,
        "replans": challenger.replans - champion.replans,
    }
    return BenchmarkComparison(
        benchmark_version=RELEASE_BENCHMARK_VERSION,
        suite=challenger.suite,
        champion_plan_id=champion.plan_id,
        challenger_plan_id=challenger.plan_id,
        passed=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        deltas=deltas,
        thresholds=cfg.to_dict(),
    )


class BenchmarkStore:
    """SQLite-backed release benchmark history and named champion registry."""

    def __init__(self, path: str | Path = DEFAULT_BENCHMARK_DB) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suite TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    saund_version TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_benchmark_runs_suite
                    ON benchmark_runs(suite, run_id DESC);
                CREATE TABLE IF NOT EXISTS benchmark_champions (
                    suite TEXT PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    promoted_at_utc TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES benchmark_runs(run_id)
                );
                """
            )

    def record(self, snapshot: ReleaseBenchmarkSnapshot) -> int:
        payload = json.dumps(snapshot.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO benchmark_runs(suite, plan_id, saund_version, captured_at_utc, payload_json) VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot.suite,
                    snapshot.plan_id,
                    snapshot.saund_version,
                    snapshot.captured_at_utc,
                    payload,
                ),
            )
            return int(cursor.lastrowid)

    def get(self, run_id: int) -> ReleaseBenchmarkSnapshot | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload_json FROM benchmark_runs WHERE run_id = ?", (int(run_id),)
            ).fetchone()
        if row is None:
            return None
        return ReleaseBenchmarkSnapshot(**json.loads(str(row["payload_json"])))

    def latest(self, suite: str) -> tuple[int, ReleaseBenchmarkSnapshot] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT run_id, payload_json FROM benchmark_runs WHERE suite = ? ORDER BY run_id DESC LIMIT 1",
                (suite,),
            ).fetchone()
        if row is None:
            return None
        return int(row["run_id"]), ReleaseBenchmarkSnapshot(**json.loads(str(row["payload_json"])))

    def champion(self, suite: str) -> tuple[int, ReleaseBenchmarkSnapshot] | None:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT r.run_id, r.payload_json
                FROM benchmark_champions c
                JOIN benchmark_runs r ON r.run_id = c.run_id
                WHERE c.suite = ?
                """,
                (suite,),
            ).fetchone()
        if row is None:
            return None
        return int(row["run_id"]), ReleaseBenchmarkSnapshot(**json.loads(str(row["payload_json"])))

    def promote(self, suite: str, run_id: int) -> None:
        run = self.get(run_id)
        if run is None:
            raise KeyError(f"Unknown benchmark run {run_id}")
        if run.suite != suite:
            raise ValueError(f"Benchmark run {run_id} belongs to suite {run.suite}, not {suite}")
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO benchmark_champions(suite, run_id, promoted_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(suite) DO UPDATE SET
                    run_id = excluded.run_id,
                    promoted_at_utc = excluded.promoted_at_utc
                """,
                (suite, int(run_id), datetime.now(timezone.utc).isoformat()),
            )

    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            run_count = int(db.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0])
            rows = db.execute(
                """
                SELECT c.suite, c.run_id, r.saund_version, r.plan_id, r.payload_json
                FROM benchmark_champions c
                JOIN benchmark_runs r ON r.run_id = c.run_id
                ORDER BY c.suite
                """
            ).fetchall()
        champions = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            champions.append(
                {
                    "suite": str(row["suite"]),
                    "run_id": int(row["run_id"]),
                    "saund_version": str(row["saund_version"]),
                    "plan_id": str(row["plan_id"]),
                    "full_set_overall": float(payload.get("full_set_overall", 0.0)),
                    "production_safe": bool(payload.get("production_safe", False)),
                    "track_count": int(payload.get("track_count", 0)),
                }
            )
        return {
            "version": RELEASE_BENCHMARK_VERSION,
            "database": str(self.path),
            "run_count": run_count,
            "champion_count": len(champions),
            "champions": champions,
            "privacy": "metrics_only_no_track_paths",
        }


def benchmark_runtime_status(
    workspace: str | Path = ".",
    *,
    database: str | Path = DEFAULT_BENCHMARK_DB,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    path = Path(database).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        return {
            "version": RELEASE_BENCHMARK_VERSION,
            "available": False,
            "run_count": 0,
            "champion_count": 0,
            "champions": [],
            "suite_catalog": benchmark_suite_catalog(),
            "privacy": "metrics_only_no_track_paths",
            "promotion_gate": "champion_comparison_required_after_first_manual_champion",
        }
    status = BenchmarkStore(path).status()
    status["available"] = True
    status["champions"] = [
        {key: value for key, value in item.items() if key != "plan_id"}
        for item in status.get("champions", [])
        if isinstance(item, dict)
    ]
    status["suite_catalog"] = benchmark_suite_catalog()
    status["promotion_gate"] = "champion_comparison_required_after_first_manual_champion"
    # API callers need capability/status, not a host-specific absolute path.
    status.pop("database", None)
    return status
