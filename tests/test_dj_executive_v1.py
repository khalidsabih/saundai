from __future__ import annotations

import hashlib
from pathlib import Path

from conftest import analysis
from djai.brain.executive import (
    DJ_EXECUTIVE_VERSION,
    build_executive_observation_safely,
    executive_status,
)
from djai.set_planner import make_set_plan

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracks(tmp_path: Path, count: int = 3):
    tracks = []
    for index in range(count):
        path = tmp_path / f"executive-{index}.wav"
        path.write_bytes(f"executive-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=122.0 + 2.0 * index,
                camelot=("8A", "9A", "10A")[index],
                energy=0.46 + 0.10 * index,
            )
        )
    return tracks


def test_dj_executive_observes_authoritative_plan_without_taking_authority(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "observe")
    monkeypatch.setenv("DJAI_AI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")

    plan = make_set_plan(
        tracks,
        source="dj-executive-observe",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )

    report = plan.metadata["dj_executive"]
    assert report["version"] == DJ_EXECUTIVE_VERSION
    assert report["mode"] == "observe"
    assert report["enabled"] is True
    assert report["authoritative"] is False
    assert report["musical_behavior_changes"] is False
    assert report["commit_authority"] == "existing_rc1_pipeline"
    assert report["decision_count"] == len(plan.transitions)
    assert report["lifecycle"] == [
        "observe", "propose", "assess_brittleness", "allocate_thinking",
        "search", "audition", "prepare", "prepare_backup", "validate", "commit"
    ]
    assert report["utilization"]["final_order"] == plan.order
    assert report["utilization"]["complete_move_decisions_correlated"] == len(plan.transitions)

    for position, (edge, decision) in enumerate(zip(plan.transitions, report["decisions"])):
        assert decision["snapshot"]["sequence_position"] == position
        assert decision["snapshot"]["current_index"] == edge.from_index
        assert decision["snapshot"]["next_index"] == edge.to_index
        assert "layer_state" in decision["snapshot"]
        assert "projected_layer_state" in decision["snapshot"]
        assert decision["technique"] == edge.plan.technique
        assert decision["evidence"]["compiled_execution_signature"] == edge.plan.execution_signature
        assert decision["observer_authoritative"] is False
        assert decision["decision_id"].startswith("decision-")
        assert decision["move_id"].startswith("move-")
        assert decision["alternatives"]
        assert decision["alternatives"][0]["selected"] is True



def test_observe_mode_preserves_order_and_execution_signatures(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_AI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")

    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "off")
    baseline = make_set_plan(tracks, source="executive-equivalence-off", journey="build", graph_workers=1, automatic_delivery=False)

    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "observe")
    observed = make_set_plan(tracks, source="executive-equivalence-observe", journey="build", graph_workers=1, automatic_delivery=False)

    assert observed.order == baseline.order
    assert [edge.plan.execution_signature for edge in observed.transitions] == [
        edge.plan.execution_signature for edge in baseline.transitions
    ]
    assert [edge.plan.technique for edge in observed.transitions] == [edge.plan.technique for edge in baseline.transitions]
    assert observed.metadata["dj_executive"]["musical_behavior_changes"] is False

def test_dj_executive_decision_ids_are_deterministic_for_same_final_plan(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "observe")
    monkeypatch.setenv("DJAI_AI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")

    first = make_set_plan(tracks, source="executive-first", journey="balanced", graph_workers=1, automatic_delivery=False)
    second = make_set_plan(tracks, source="executive-second", journey="balanced", graph_workers=1, automatic_delivery=False)

    first_ids = [(item["decision_id"], item["move_id"]) for item in first.metadata["dj_executive"]["decisions"]]
    second_ids = [(item["decision_id"], item["move_id"]) for item in second.metadata["dj_executive"]["decisions"]]
    assert first.order == second.order
    assert [edge.plan.execution_signature for edge in first.transitions] == [edge.plan.execution_signature for edge in second.transitions]
    assert first_ids == second_ids


def test_dj_executive_can_be_disabled_without_changing_planning(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_AI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "off")
    plan = make_set_plan(tracks, source="executive-off", journey="balanced", graph_workers=1, automatic_delivery=False)
    report = plan.metadata["dj_executive"]
    assert report["enabled"] is False
    assert report["decision_count"] == 0
    assert report["authoritative"] is False


def test_dj_executive_observation_failure_is_nonfatal(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "observe")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")
    payload = build_executive_observation_safely(
        analyses=None,
        order=None,
        transitions=[object()],
        decisions=(),
        journey="balanced",
    )
    assert payload["enabled"] is True
    assert payload["authoritative"] is False
    assert payload["musical_behavior_changes"] is False
    assert payload["error"]


def test_dj_executive_observation_mode_remains_available(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "observe")
    status = executive_status()
    assert status["version"] == DJ_EXECUTIVE_VERSION
    assert status["mode"] == "observe"
    assert status["authoritative"] is False
    assert status["single_commit_authority"] is False
    assert status["telemetry_failure_is_nonfatal"] is True


def test_dj_executive_patch_preserves_rc1_frontend_transport() -> None:
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert '"/api/tracks/upload"' in api
    assert '"/api/tracks/upload-intents"' in api
    assert "EventSource" in api
    next_config = (ROOT / "frontend" / "next.config.ts").read_text(encoding="utf-8")
    assert 'output: "standalone"' in next_config
    assert 'source: "/api/:path*"' in next_config
    assert 'destination: `${internalApi}/api/:path*`' in next_config
    assert 'X-Content-Type-Options' in next_config
    assert _sha256(ROOT / "compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"
