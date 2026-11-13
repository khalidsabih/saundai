from __future__ import annotations

import importlib
import json
import os
import sys
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from djai import __version__
from djai.version import MUSICAL_TIMELINE_SCHEMA_VERSION
from djai.confirmation import create_confirmation_token, verify_confirmation_token


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_consistent() -> None:
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    version_file = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    frontend_version = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))["version"]
    assert __version__ == project_version == version_file == frontend_version == "2.0.0-rc.1"


def test_confirmation_token_is_bound_to_payload(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_CONFIRMATION_SECRET", "unit-test-secret")
    payload = {"plan_id": "abc", "order": [0, 1, 2], "risk": True}
    token = create_confirmation_token(payload)
    assert verify_confirmation_token(payload, token)
    assert not verify_confirmation_token({**payload, "order": [0, 2, 1]}, token)


def test_api_health_and_no_cache_job_headers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_CONFIRMATION_SECRET", "api-test-secret")
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)
    assert api.stem_manager is not api.manager

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["version"] == "2.0.0-rc.1"
    assert health.json()["features"]["account_foundation"] is True
    assert health.json()["features"]["authenticated_build_gate"] is True
    assert health.json()["features"]["account_resource_ownership"] is True
    assert health.json()["features"]["studio_account_gate"] is True
    assert health.json()["features"]["immutable_execution_plans"] is True
    assert health.json()["features"]["content_bound_execution_assets"] is True
    assert health.json()["features"]["executable_transition_ir"] is True
    assert health.json()["features"]["generic_dsp_execution_engine"] is True
    assert health.json()["features"]["joint_transition_proposals"] is True
    assert health.json()["features"]["musical_world_model_6"] is True
    assert health.json()["features"]["hierarchical_set_director"] is True
    assert health.json()["features"]["dj_performance_memory"] is True
    assert health.json()["features"]["multi_critic_rendered_judge"] is True
    assert health.json()["features"]["critic_guided_surgical_replanning"] is True
    assert health.json()["features"]["content_addressed_rehearsal_cache"] is True
    assert health.json()["features"]["release_readiness_gate"] is True
    assert health.json()["features"]["exact_set_rehearsal_context"] is True
    assert health.json()["features"]["final_transition_verification"] is True
    assert health.json()["features"]["atomic_render_publication"] is True
    assert health.json()["features"]["parallel_analysis"] is True
    assert health.json()["features"]["parallel_graph_planning"] is True
    assert health.json()["features"]["pair_evidence_reuse"] is True
    assert health.json()["features"]["render_measure_quality_gate"] is True
    assert health.json()["features"]["automatic_delivery"] is True
    assert health.json()["features"]["automatic_fallback"] is True
    assert health.json()["features"]["live_http_streaming"] is True
    assert health.json()["features"]["listener_personalization"] is True
    assert health.json()["features"]["automatic_performance_layers"] is True
    assert health.json()["features"]["layer_intelligence_v1"] is True
    assert health.json()["features"]["incoming_vocal_over_instrumental"] is True
    assert health.json()["features"]["dj_executive_v3"] is True
    assert health.json()["features"]["dj_executive_v4"] is True
    assert health.json()["features"]["dj_executive_v5"] is True
    assert health.json()["features"]["adaptive_thinking_v1"] is True
    assert health.json()["features"]["prepared_recovery_v1"] is True
    assert health.json()["features"]["djmove_intent_v1"] is True
    assert health.json()["features"]["unified_transition_intelligence"] is True
    assert health.json()["features"]["stem_ownership_choreography"] is True
    assert health.json()["features"]["review_needed_replans"] is True
    assert health.json()["features"]["simple_pro_shared_brain"] is True
    assert health.json()["features"]["duration_search"] is True
    assert health.json()["features"]["neural_model_registry"] is True
    assert health.json()["features"]["neural_transition_policy_heads"] is True
    assert health.json()["features"]["hybrid_next_track_model_1"] is True
    assert health.json()["features"]["unified_ai_planner_1"] is True
    assert health.json()["features"]["full_set_benchmarks_1"] is True
    assert health.json()["features"]["consent_gated_personalization"] is True
    assert health.json()["features"]["model_governance_1"] is True
    assert health.json()["features"]["taste_learning_loop_1_1_1"] is True
    assert health.json()["features"]["runtime_candidate_evidence"] is True
    assert health.json()["features"]["global_taste_shadow_training"] is True
    assert health.json()["features"]["rendered_dj_judge_shadow_training"] is True
    assert health.json()["features"]["dj_deck_tempo_handoff_1_2"] is True
    assert health.json()["features"]["beat_phase_nudge_1_2"] is True
    assert health.json()["features"]["stem_vocal_exchange_1_2"] is True
    assert health.json()["features"]["human_dj_move_brain_1_4"] is True
    assert health.json()["features"]["adaptive_musical_timeline_1_4"] is True
    assert health.json()["features"]["protected_musical_commitments_1_4"] is True
    assert health.json()["features"]["explicit_wait_actions_1_4"] is True
    assert health.json()["features"]["move_aware_track_ordering_1_4"] is True
    assert health.json()["features"]["large_crate_planning_1_5_1"] is True
    assert health.json()["features"]["job_wide_progress_1_5_2"] is True
    assert health.json()["features"]["engineering_integrity_1_5_3"] is True
    assert health.json()["features"]["canonical_musical_timeline_1_6"] is True
    assert health.json()["features"]["single_phrase_section_authority_1_6"] is True
    assert health.json()["features"]["complete_djmove_generator_1_7"] is True
    assert health.json()["features"]["lazy_crate_search_1_7_1"] is True
    assert health.json()["features"]["structural_deck_operations_1_7_2"] is True
    assert health.json()["features"]["ai_headphones_1_8"] is True
    assert health.json()["features"]["state_move_taste_learning_1_8_1"] is True
    assert health.json()["features"]["crate_memory_1_8_2"] is True
    assert health.json()["features"]["release_benchmarking_1_9"] is True
    assert health.json()["features"]["benchmark_champion_gate_1_9"] is True
    assert health.json()["features"]["release_candidate_certification_2_0"] is True
    assert health.json()["features"]["docker_authority_passthrough_2_0"] is True
    assert health.json()["features"]["persisted_schema_preflight_2_0"] is True
    assert health.json()["features"]["cooperative_job_cancellation_1_5_3"] is True
    assert health.json()["build"]["version_consistent"] is True
    runtime = client.get("/api/runtime/status")
    assert runtime.status_code == 200
    assert runtime.json()["schema_version"] == "2.0.0"
    assert runtime.json()["release_candidate"]["release_channel"] == "release-candidate"
    assert runtime.json()["release_candidate"]["authority"]["planner_mode"] == "move_authoritative"
    assert runtime.json()["resources"]["job_workers"] >= 1
    brain = client.get("/api/brain/status")
    assert brain.status_code == 200
    assert brain.json()["taste_learning"]["policy"]["hard_audio_safety_authoritative"] is True
    assert brain.json()["deck_performance"]["tempo_ownership_transfer"] is True
    assert brain.json()["deck_performance"]["hard_audio_safety_authoritative"] is True
    assert brain.json()["dj_move_brain"]["enabled"] is True
    assert brain.json()["dj_move_brain"]["planning_unit"] == "DJState -> DJMove -> DJState"
    assert brain.json()["dj_move_brain"]["explicit_wait_actions"] is True
    assert brain.json()["musical_timeline"]["schema_version"] == MUSICAL_TIMELINE_SCHEMA_VERSION
    assert brain.json()["musical_timeline"]["canonical_phrase_authority"] is True
    assert brain.json()["dj_move_brain"]["hard_audio_safety_authoritative"] is True
    executive = brain.json()["dj_executive"]
    assert executive["version"] == "2.5.0"
    assert executive["commit_authority"] == "dj_executive_v5"
    assert executive["djmove_intent_version"] == "1.0"
    assert executive["djmove_intent_required_for_commit"] is True
    assert executive["flexible_prepare_binding"] is True
    assert executive["adaptive_thinking"]["version"] == "1.0"
    assert executive["adaptive_thinking"]["enabled"] is True
    assert executive["prepared_recovery_version"] == "1.0"
    assert executive["prepared_recovery_capable"] is True
    assert executive["brittleness_controls_compute_only"] is True
    assert executive["hard_safety_budget_never_reduced"] is True
    crate_memory = brain.json()["crate_memory"]
    assert crate_memory["version"] == "1.8.2"
    assert crate_memory["hard_audio_safety_authoritative"] is True
    release_benchmarks = brain.json()["release_benchmarks"]
    assert release_benchmarks["version"] == "1.9.0"
    assert release_benchmarks["privacy"] == "metrics_only_no_track_paths"
    benchmark_status = client.get("/api/benchmarks/status")
    assert benchmark_status.status_code == 200
    assert benchmark_status.json()["suite_catalog"]["standard"]["track_count"] == 14
    release_status = client.get("/api/release/readiness")
    assert release_status.status_code == 200
    assert release_status.json()["version"] == "2.0.0-rc.1"
    assert release_status.json()["operational_ready"] is True
    headphones = brain.json()["dj_move_brain"]["complete_move_generator"]["ai_headphones"]
    assert headphones["version"] == "1.8.0"
    assert headphones["commit_boundary_only"] is True
    assert headphones["uses_real_audio_renderer"] is True
    assert brain.json()["large_crate_planning"]["graph_strategy"] == "bounded_scout_then_full_final_edges"
    assert brain.json()["large_crate_planning"]["full_cue_pool_restored_for_final_edges"] is True
    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.json()["schema_version"] == "5.0"

    registered = client.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "api-owner@example.com", "password": "a-long-api-test-password", "country": "Germany", "primary_use": "personal"},
    )
    owner_user_id = registered.json()["user"]["id"]
    job = api.manager.submit("test", lambda context: {"ok": True}, owner_user_id=owner_user_id)
    response = client.get(f"/api/jobs/{job.job_id}")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    assert response.headers["pragma"] == "no-cache"


def test_api_rejects_paths_outside_workspace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)
    response = client.get("/api/tracks", params={"folder": "../"})
    assert response.status_code == 400


def test_api_rejects_measurement_log_outside_workspace(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "audio").mkdir()
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)
    registered = client.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "planner@example.com", "password": "a-long-planner-password", "country": "Germany", "primary_use": "personal"},
    )
    assert registered.status_code == 201
    response = client.post(
        "/api/set/plan",
        json={"folder": "audio", "measurement_log": "../outside.jsonl"},
    )
    assert response.status_code == 400


def test_cli_public_brand_and_compatibility_alias(monkeypatch) -> None:
    from djai.cli import build_parser

    monkeypatch.setattr(sys, "argv", ["saund"])
    preferred = build_parser()
    assert preferred.prog == "saund"
    assert "SAUND.AI" in (preferred.description or "")

    monkeypatch.setattr(sys, "argv", ["djai"])
    compatibility = build_parser()
    assert compatibility.prog == "djai"
