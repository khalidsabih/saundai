from __future__ import annotations

import importlib
import time
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import analysis
from djai.decision import validate_execution_plan
from djai.fallback import build_last_resort_plan
from djai.journey import curve_fit_score, target_energy_curve
from djai.set_models import SetPlan, TransitionEdge


def automatic_plan(tmp_path: Path) -> SetPlan:
    audio_root = tmp_path / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    a_path = audio_root / "a.wav"
    b_path = audio_root / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    tracks = [
        analysis(a_path, bpm=122.0, energy=0.42),
        analysis(b_path, bpm=138.0, energy=0.82),
    ]
    transition = build_last_resort_plan(
        tracks[0],
        tracks[1],
        preferred_exit_time=180.0,
        reason="test fallback",
    )
    edge = TransitionEdge(
        from_index=0,
        to_index=1,
        selected_style="emergency",
        objective_score=transition.score,
        quality="risky",
        plan=transition,
        alternatives=[transition],
        warnings=["test fallback"],
        reasons=list(transition.rationale),
    )
    return SetPlan(
        plan_id="automatic-test-plan",
        tracks=tracks,
        order=[0, 1],
        transitions=[edge],
        baseline_order=[0, 1],
        baseline_score=transition.score,
        optimized_score=transition.score,
        estimated_duration_seconds=300.0,
        source="test",
        automatic_delivery=True,
        listener="web",
        metadata={"journey": "balanced", "fallback_transition_count": 1},
    )




def _register_account(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "automatic-test@example.com", "password": "automatic-test-password", "country": "Germany", "primary_use": "personal"},
    )
    assert response.status_code == 201

def test_last_resort_plan_is_playable_and_preserves_body(tmp_path: Path) -> None:
    plan = automatic_plan(tmp_path)
    transition = plan.transitions[0].plan
    validate_execution_plan(transition)
    assert transition.exit_time_a >= 175.0
    assert transition.fallback_level == 5
    assert transition.auto_approved is True
    assert transition.requires_confirmation is False
    assert plan.has_risks is True
    assert plan.requires_confirmation is False


def test_target_energy_modes_are_distinct(tmp_path: Path) -> None:
    warmup = target_energy_curve("warmup", 8)
    peak = target_energy_curve("peak", 8)
    sunset = target_energy_curve("sunset", 8)
    assert warmup[0] < warmup[-1]
    assert peak[2] > warmup[2]
    assert sunset[0] > sunset[-1]

    paths = []
    tracks = []
    for index, energy in enumerate(warmup):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(str(index).encode())
        paths.append(path)
        tracks.append(analysis(path, bpm=122 + index, energy=energy))
    assert curve_fit_score(list(range(8)), tracks, journey="warmup") > 0.98


def test_automatic_api_feedback_and_live_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "off")
    import sys

    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    plan = automatic_plan(tmp_path)
    api.FEATURE_STORE.register_source_fingerprint(plan.tracks[0].path, "a" * 64)
    api.FEATURE_STORE.register_source_fingerprint(plan.tracks[1].path, "b" * 64)

    monkeypatch.setattr(api, "analyze_tracks", lambda *args, **kwargs: plan.tracks)
    monkeypatch.setattr(api, "enrich_analyses_with_cached_stems", lambda values: 0)
    monkeypatch.setattr(api, "make_set_plan", lambda *args, **kwargs: plan)

    def fake_render(plan_value, output, **kwargs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF-test")
        report = output.with_suffix(".json")
        report.write_text("{}\n", encoding="utf-8")
        return output, report

    monkeypatch.setattr(api, "render_set", fake_render)
    monkeypatch.setattr(api, "iter_mp3_stream", lambda *args, **kwargs: iter([b"ID3-test"]))
    client = TestClient(api.app)
    _register_account(client)

    response = client.post(
        "/api/sets/automatic",
        json={"track_paths": [plan.tracks[0].path, plan.tracks[1].path], "journey": "peak"},
    )
    assert response.status_code == 202
    job_id = response.json()["job"]["job_id"]
    for _ in range(100):
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert record["status"] == "completed", record.get("error")
    result = record["result"]
    assert result["audio_url"].startswith("/api/media/")
    assert result["plan"]["automatic_delivery"] is True
    assert result["plan"]["requires_confirmation"] is False
    benchmark = result["plan"]["metadata"]["release_benchmark"]
    assert benchmark["benchmark_version"] == "1.9.0"
    assert len(benchmark["crate_fingerprint"]) == 64
    assert benchmark["total_seconds"] is not None
    assert benchmark["render_seconds"] is not None
    persisted = api.load_set_plan(tmp_path / result["plan_path"])
    assert persisted.metadata["release_benchmark"]["total_seconds"] is not None
    assert persisted.metadata["performance"]["latency"]["render_seconds"] >= 0.0

    manifest = client.get("/api/live/manifest", params={"plan_path": result["plan_path"]})
    assert manifest.status_code == 200
    assert manifest.json()["plan_id"] == plan.plan_id
    assert len(manifest.json()["timeline"]) == 3

    stream = client.get("/api/live/stream", params={"plan_path": result["plan_path"]})
    assert stream.status_code == 200
    assert stream.content == b"ID3-test"
    assert stream.headers["content-type"].startswith("audio/mpeg")

    feedback = client.post(
        "/api/feedback",
        json={
            "event": "set_like",
            "listener": "web-test",
            "plan_path": result["plan_path"],
            "journey": "peak",
        },
    )
    assert feedback.status_code == 200
    assert feedback.json()["listener_profile"] is None
    assert feedback.json()["personalization_applied"] is False
    assert feedback.json()["training_eligible"] is False


def test_automatic_api_background_stems_do_not_block_analysis(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "background")
    import sys

    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    plan = automatic_plan(tmp_path)
    events: list[str] = []

    monkeypatch.setattr(api, "demucs_available", lambda: True)
    monkeypatch.setattr(api, "find_cached_stems", lambda path: {"vocals": "cached.wav"})

    def fake_prepare(paths, **kwargs):
        events.append("prepare")
        raise AssertionError("background stem mode must never block Create Set")

    def fake_analyze(*args, **kwargs):
        events.append("analyze")
        return plan.tracks

    monkeypatch.setattr(api, "prepare_stems_for_tracks", fake_prepare)
    monkeypatch.setattr(api, "analyze_tracks", fake_analyze)
    monkeypatch.setattr(api, "enrich_analyses_with_cached_stems", lambda values: 2)
    monkeypatch.setattr(api, "enrich_analyses_with_performance_layers", lambda values: 2)
    monkeypatch.setattr(api, "make_set_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(api, "maybe_refit_scoring_model", lambda **kwargs: None)
    monkeypatch.setattr(api, "append_feedback_event", lambda **kwargs: None)

    def fake_render(plan_value, output, **kwargs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF-test")
        report = output.with_suffix(".json")
        report.write_text("{}\n", encoding="utf-8")
        return output, report

    monkeypatch.setattr(api, "render_set", fake_render)
    client = TestClient(api.app)
    _register_account(client)
    response = client.post(
        "/api/sets/automatic",
        json={"track_paths": [plan.tracks[0].path, plan.tracks[1].path]},
    )
    assert response.status_code == 202
    job_id = response.json()["job"]["job_id"]
    record = None
    for _ in range(100):
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert record is not None
    assert record["status"] == "completed", record.get("error")
    assert events == ["analyze"]
    metadata = record["result"]["plan"]["metadata"]["automatic_performance_layers"]
    assert metadata["mode"] == "background"
    assert metadata["demucs_available"] is True
    assert metadata["prepared_tracks"] == 2
    assert metadata["blocking_wait"] is False
    assert metadata["background_nonblocking"] is True
    assert metadata["one_master_input"] is True


def test_automatic_api_wait_stems_still_synchronizes_before_analysis(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "wait")
    import sys

    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    plan = automatic_plan(tmp_path)
    events: list[str] = []

    monkeypatch.setattr(api, "demucs_available", lambda: True)
    monkeypatch.setattr(api, "find_cached_stems", lambda path: {})

    def fake_prepare(paths, **kwargs):
        events.append("prepare")
        return {str(path): {"vocals": "cached.wav"} for path in paths}

    def fake_analyze(*args, **kwargs):
        events.append("analyze")
        return plan.tracks

    monkeypatch.setattr(api, "prepare_stems_for_tracks", fake_prepare)
    monkeypatch.setattr(api, "analyze_tracks", fake_analyze)
    monkeypatch.setattr(api, "enrich_analyses_with_cached_stems", lambda values: 2)
    monkeypatch.setattr(api, "enrich_analyses_with_performance_layers", lambda values: 2)
    monkeypatch.setattr(api, "make_set_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(api, "maybe_refit_scoring_model", lambda **kwargs: None)
    monkeypatch.setattr(api, "append_feedback_event", lambda **kwargs: None)

    def fake_render(plan_value, output, **kwargs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"RIFF-test")
        report = output.with_suffix(".json")
        report.write_text("{}\n", encoding="utf-8")
        return output, report

    monkeypatch.setattr(api, "render_set", fake_render)
    client = TestClient(api.app)
    _register_account(client)
    response = client.post(
        "/api/sets/automatic",
        json={"track_paths": [plan.tracks[0].path, plan.tracks[1].path]},
    )
    assert response.status_code == 202
    job_id = response.json()["job"]["job_id"]
    record = None
    for _ in range(100):
        record = client.get(f"/api/jobs/{job_id}").json()
        if record["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert record is not None
    assert record["status"] == "completed", record.get("error")
    assert events[:2] == ["prepare", "analyze"]
    metadata = record["result"]["plan"]["metadata"]["automatic_performance_layers"]
    assert metadata["mode"] == "wait"
    assert metadata["blocking_wait"] is True
    assert metadata["background_nonblocking"] is False


def test_automatic_fallback_defaults_to_audible_filtered_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    a_path = tmp_path / "a-rescue.wav"
    b_path = tmp_path / "b-rescue.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    a = analysis(a_path, bpm=124.0, energy=0.44)
    b = analysis(b_path, bpm=142.0, energy=0.79)
    plan = build_last_resort_plan(a, b, preferred_exit_time=160.0)
    validate_execution_plan(plan)
    assert plan.technique == "filter_build"
    assert plan.transition_bars >= 2
    assert plan.transition_seconds >= 4.0
    assert plan.source_duration_b >= plan.transition_seconds - 1e-6
    assert plan.metadata["rescue_transition_kind"] == "filtered_overlap"
    assert plan.metadata["continuous_mix_required"] is True


def test_automatic_fallback_allows_hard_cut_only_when_explicitly_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_ALLOW_HARD_CUTS", "1")
    a_path = tmp_path / "a-cut.wav"
    b_path = tmp_path / "b-cut.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    a = analysis(a_path, bpm=124.0, energy=0.44)
    b = analysis(b_path, bpm=142.0, energy=0.79)
    plan = build_last_resort_plan(a, b, preferred_exit_time=160.0)
    validate_execution_plan(plan)
    assert plan.technique == "quick_cut"
    assert plan.metadata["rescue_transition_kind"] == "hard_cut"
    assert plan.metadata["continuous_mix_required"] is False
