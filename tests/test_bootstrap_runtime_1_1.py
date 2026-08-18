from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from conftest import analysis
from djai.intelligence.bootstrap_runtime import (
    BootstrapRuntimeConfig,
    clear_bootstrap_runtime_cache,
    cue_candidate_priors,
    rank_duration_values,
    sequence_pair_prediction,
)
from djai.intelligence.hybrid_next_track import score_hybrid_next_track
from djai.planner import rank_transition_candidates
from djai.planning_context import build_pair_evidence_matrix


def _artist_key(value: str) -> str:
    text = " ".join(value.casefold().split())
    return "artist-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _write_bundle(root: Path) -> Path:
    model_root = root / "config" / "bootstrap-models" / "corpus-test"
    model_root.mkdir(parents=True)
    cue = {
        "schema_version": "bootstrap-shadow/1.2",
        "model_id": "cue_point_prior",
        "corpus_id": "corpus-test",
        "research_shadow": True,
        "promotable_to_live": False,
        "feature_spaces": ["external-cue-structural-v3-ranking"],
        "feature_names": ["candidate_time_seconds"],
        "feature_mean": {"candidate_time_seconds": 0.0},
        "feature_scale": {"candidate_time_seconds": 1.0},
        "weights": [1.0],
        "bias": 0.0,
        "metrics": {"test": {"top1_accuracy": 1.0}},
    }
    duration = {
        "schema_version": "bootstrap-shadow/1.2",
        "model_id": "transition_duration_structure",
        "corpus_id": "corpus-test",
        "research_shadow": True,
        "promotable_to_live": False,
        "feature_spaces": ["external-transition-duration-v4-ranking"],
        "feature_names": ["duration_seconds"],
        "feature_mean": {"duration_seconds": 0.0},
        "feature_scale": {"duration_seconds": 1.0},
        "weights": [1.0],
        "bias": 0.0,
        "metrics": {"test": {"top1_accuracy": 1.0}},
    }
    adjacency = {
        "schema_version": "next-track-adjacency-shadow/1.3",
        "model_id": "next_track_adjacency_prior",
        "corpus_id": "corpus-test",
        "research_shadow": True,
        "promotable_to_live": False,
        "top_edges": [],
        "runtime_index": {"path": "sequence-runtime.sqlite"},
        "evaluation": {"metrics": {"test": {"model": {"hit_at_10": 0.10}}}},
    }
    generalizer = {
        "schema_version": "next-track-sequence-generalizer-shadow/1.1",
        "model_id": "next_track_sequence_generalizer",
        "corpus_id": "corpus-test",
        "research_shadow": True,
        "promotable_to_live": False,
        "top_artist_targets": [],
        "evaluation": {"metrics": {"test": {"model": {"hit_at_10": 0.07}}}},
    }
    for name, payload in (
        ("cue_point_prior.json", cue),
        ("transition_duration_structure.json", duration),
        ("next_track_adjacency_prior.json", adjacency),
        ("next_track_sequence_generalizer.json", generalizer),
    ):
        (model_root / name).write_text(json.dumps(payload), encoding="utf-8")
    (model_root / "training-report.json").write_text(
        json.dumps({"corpus_id": "corpus-test", "shadow_only": True}), encoding="utf-8"
    )

    db = sqlite3.connect(model_root / "sequence-runtime.sqlite")
    db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID")
    db.execute(
        "CREATE TABLE adjacency(source_track TEXT NOT NULL,target_track TEXT NOT NULL,count INTEGER NOT NULL,outgoing_count INTEGER NOT NULL,PRIMARY KEY(source_track,target_track)) WITHOUT ROWID"
    )
    db.execute(
        "CREATE TABLE artist_track(source_artist_key TEXT NOT NULL,target_track TEXT NOT NULL,count INTEGER NOT NULL,outgoing_count INTEGER NOT NULL,PRIMARY KEY(source_artist_key,target_track)) WITHOUT ROWID"
    )
    db.execute(
        "CREATE TABLE artist_pair(source_artist_key TEXT NOT NULL,target_artist_key TEXT NOT NULL,count INTEGER NOT NULL,outgoing_count INTEGER NOT NULL,PRIMARY KEY(source_artist_key,target_artist_key)) WITHOUT ROWID"
    )
    db.execute(
        "INSERT INTO artist_pair VALUES (?,?,?,?)",
        (_artist_key("Artist A"), _artist_key("Artist B"), 40, 100),
    )
    db.commit()
    db.close()
    return model_root


def _configure(monkeypatch, tmp_path: Path, *, mode: str = "advisory") -> Path:
    model_root = _write_bundle(tmp_path)
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_BOOTSTRAP_MODEL_DIR", str(model_root))
    monkeypatch.setenv("DJAI_BOOTSTRAP_RUNTIME_MODE", mode)
    monkeypatch.setenv("SAUND_ENV", "development")
    clear_bootstrap_runtime_cache()
    return model_root


def test_cue_prior_is_a_bounded_live_ranking_signal(tmp_path: Path, monkeypatch) -> None:
    _configure(monkeypatch, tmp_path)
    track = analysis(tmp_path / "track.wav", bpm=124.0)
    priors = cue_candidate_priors(track, track.entry_candidates)
    early = priors[next(key for key in priors if key[2] == 8.0)]
    late = priors[next(key for key in priors if key[2] == 64.0)]

    assert late.score > early.score
    assert late.effective_weight > 0.0
    assert late.mode == "advisory"


def test_production_forces_research_bootstrap_back_to_shadow(tmp_path: Path, monkeypatch) -> None:
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("SAUND_ENV", "production")
    monkeypatch.delenv("SAUND_BOOTSTRAP_PRODUCTION_APPROVED", raising=False)
    clear_bootstrap_runtime_cache()
    cfg = BootstrapRuntimeConfig.from_environment()
    track = analysis(tmp_path / "track.wav", bpm=124.0)
    priors = cue_candidate_priors(track, track.entry_candidates)

    assert cfg.mode == "shadow"
    assert cfg.production_guarded is True
    assert all(value.effective_weight == 0.0 for value in priors.values())


def test_duration_prior_reorders_only_existing_valid_candidates(tmp_path: Path, monkeypatch) -> None:
    _configure(monkeypatch, tmp_path)
    monkeypatch.setenv("DJAI_BOOTSTRAP_DURATION_WEIGHT", "0.55")
    a = analysis(tmp_path / "a.wav", bpm=124.0)
    b = analysis(tmp_path / "b.wav", bpm=124.0)

    ordered, payload = rank_duration_values(a, b, (4, 8, 16))

    assert set(ordered) == {4, 8, 16}
    assert ordered[0] == 16
    assert payload["changed_order"] is True
    assert payload["effective_weight"] > 0.0


def test_sequence_artist_prior_feeds_hybrid_brain_when_identity_resolves(tmp_path: Path, monkeypatch) -> None:
    _configure(monkeypatch, tmp_path)
    a_path = tmp_path / "Artist A - First.wav"
    b_path = tmp_path / "Artist B - Second.wav"
    a = analysis(a_path, bpm=124.0)
    b = analysis(b_path, bpm=125.0)

    prior = sequence_pair_prediction(a, b)
    hybrid = score_hybrid_next_track(a, b, edge_score=0.5)

    assert prior.available is True
    assert prior.score > 0.5
    assert prior.effective_weight > 0.0
    assert hybrid.components["bootstrap_sequence_weight"] > 0.0
    assert hybrid.learned["bootstrap_next_track"]["available"] is True


def test_cue_prior_reaches_live_transition_planner_without_bypassing_feasibility(tmp_path: Path, monkeypatch) -> None:
    _configure(monkeypatch, tmp_path)
    a = analysis(tmp_path / "a.wav", bpm=124.0)
    b = analysis(tmp_path / "b.wav", bpm=124.0)
    matrix = build_pair_evidence_matrix(a, b, scoring_model=None)

    plans = rank_transition_candidates(
        a,
        b,
        styles=("safe",),
        force=True,
        evidence_matrix=matrix,
        limit_per_style=6,
    )

    assert plans
    learned = [
        plan.metadata.get("bootstrap_learning", {}).get("cue_point_prior", {})
        for plan in plans
    ]
    assert any(item.get("available") for item in learned)
    assert any(float(item.get("effective_weight", 0.0)) > 0.0 for item in learned)
    assert all("heuristic_selection_utility" in plan.metadata for plan in plans)
    # The learned prior only ranks plans that survived the planner's existing
    # feasibility/safety construction; it never creates an invalid plan.
    assert all(plan.transition_seconds > 0.0 for plan in plans)
