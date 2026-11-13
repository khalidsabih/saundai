from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from conftest import analysis
from djai.brain.executive import DJ_EXECUTIVE_VERSION, executive_status
from djai.brain.move_intent import (
    DJ_MOVE_INTENT_VERSION,
    DJMoveIntent,
    IntentBindingError,
    bind_prepared_transition_to_intent,
)
from djai.music.timeline import build_musical_timeline
from djai.set_planner import make_set_plan

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracks(tmp_path: Path, count: int = 4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracks = []
    camelot = ("8A", "9A", "10A", "7A")
    for index in range(count):
        path = tmp_path / f"executive-v3-{index}.wav"
        path.write_bytes(f"executive-v3-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=122.0 + 1.5 * index,
                camelot=camelot[index],
                energy=0.44 + 0.10 * index,
            )
        )
    return tracks


def _plan(tmp_path: Path, monkeypatch, *, mode: str, tracks=None):
    tracks = list(tracks) if tracks is not None else _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_AI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", mode)
    return make_set_plan(
        tracks,
        source=f"executive-v3-{mode}",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )


def test_v3_status_exposes_djmove_intent_and_flexible_prepare(monkeypatch) -> None:
    monkeypatch.delenv("DJAI_EXECUTIVE_MODE", raising=False)
    status = executive_status()
    assert status["version"] == DJ_EXECUTIVE_VERSION == "2.5.0"
    assert status["commit_authority"] == "dj_executive_v5"
    assert status["djmove_intent_version"] == DJ_MOVE_INTENT_VERSION == "1.0"
    assert status["djmove_intent_required_for_commit"] is True
    assert status["flexible_prepare_binding"] is True
    assert status["exact_proposal_signature_required"] is False
    assert status["same_target_safe_fallback_binding"] is True


def test_v3_complete_move_decisions_carry_phrase_bounded_intents(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    route = plan.metadata["complete_move_planner"]["latest_route"]
    assert route is not None
    assert route["decisions"]
    for decision in route["decisions"]:
        intent = decision["move_intent"]
        assert intent["intent_id"].startswith("intent-")
        assert intent["current_index"] == decision["current_index"]
        assert intent["next_index"] == decision["winner"]["next_index"]
        assert intent["acceptable_exit_window"]["start_bar"] <= intent["preferred_exit_bar"] <= intent["acceptable_exit_window"]["end_bar"]
        assert intent["acceptable_entry_window"]["start_bar"] <= intent["preferred_entry_bar"] <= intent["acceptable_entry_window"]["end_bar"]
        assert intent["preferred_transition_bars"] in intent["acceptable_transition_bars"]
        assert decision["winner"]["move"]["technique"] in intent["allowed_techniques"]
        assert intent["fallback_policy"] == "same_target_compiled_safe_fallback"


def test_v3_transactions_bind_intent_before_commit(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    report = plan.metadata["dj_executive"]
    assert report["transaction_count"] == len(plan.transitions)
    assert report["utilization"]["djmove_intents"] == len(plan.transitions)
    assert (
        report["utilization"]["exact_intent_bindings"]
        + report["utilization"]["flexible_intent_bindings"]
        + report["utilization"]["fallback_intent_bindings"]
        == len(plan.transitions)
    )
    for transaction, decision in zip(report["transactions"], report["decisions"]):
        assert transaction["intent_id"].startswith("intent-")
        assert transaction["move_intent"]["intent_id"] == transaction["intent_id"]
        assert transaction["intent_binding"]["intent_id"] == transaction["intent_id"]
        assert transaction["intent_binding_status"] in {"exact", "flexible", "fallback"}
        assert "intent_created" in transaction["phase_history"]
        assert "intent_bound" in transaction["phase_history"]
        assert any(check.startswith("djmove_intent_bound:") for check in transaction["validation_checks"])
        assert decision["move_intent"]["intent_id"] == transaction["intent_id"]
        assert decision["commit_authority"] == "dj_executive_v5"


def test_v3_flexible_binding_accepts_changed_execution_inside_intent_window(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    edge = plan.transitions[0]
    transaction = plan.metadata["dj_executive"]["transactions"][0]
    intent = DJMoveIntent.from_dict(transaction["move_intent"])
    prepared = copy.deepcopy(edge.plan)
    timeline = build_musical_timeline(plan.tracks[edge.from_index])
    window = intent.acceptable_exit_window
    alternate_bar = window.start_bar if window.start_bar != intent.preferred_exit_bar else min(window.end_bar, window.start_bar + 1)
    prepared.exit_time_a = timeline.bar_start_seconds(alternate_bar)
    prepared.execution_signature = "synthetic-different-signature"
    binding = bind_prepared_transition_to_intent(
        intent=intent,
        plan=prepared,
        track_a=plan.tracks[edge.from_index],
        track_b=plan.tracks[edge.to_index],
    )
    assert binding.status == "flexible"
    assert binding.semantic_intent_preserved is True
    assert binding.exact_proposal_preserved is False
    assert binding.fallback_used is False
    assert binding.within_exit_window is True


def test_v3_same_target_outside_envelope_is_explicit_fallback(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    edge = plan.transitions[0]
    transaction = plan.metadata["dj_executive"]["transactions"][0]
    intent = DJMoveIntent.from_dict(transaction["move_intent"])
    prepared = copy.deepcopy(edge.plan)
    timeline = build_musical_timeline(plan.tracks[edge.from_index])
    prepared.exit_time_a = timeline.bar_start_seconds(intent.acceptable_exit_window.end_bar + 4)
    prepared.execution_signature = "synthetic-safe-fallback"
    binding = bind_prepared_transition_to_intent(
        intent=intent,
        plan=prepared,
        track_a=plan.tracks[edge.from_index],
        track_b=plan.tracks[edge.to_index],
    )
    assert binding.status == "fallback"
    assert binding.fallback_used is True
    assert binding.semantic_intent_preserved is False
    assert binding.within_exit_window is False


def test_v3_intent_cannot_bind_to_different_target_track(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    edge = plan.transitions[0]
    transaction = plan.metadata["dj_executive"]["transactions"][0]
    intent = DJMoveIntent.from_dict(transaction["move_intent"])
    prepared = copy.deepcopy(edge.plan)
    prepared.track_b = "wrong-target.wav"
    with pytest.raises(IntentBindingError, match="target track"):
        bind_prepared_transition_to_intent(
            intent=intent,
            plan=prepared,
            track_a=plan.tracks[edge.from_index],
            track_b=plan.tracks[edge.to_index],
        )


def test_v3_authoritative_intent_layer_preserves_music_vs_observe(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path / "shared")
    observed = _plan(tmp_path, monkeypatch, mode="observe", tracks=tracks)
    authoritative = _plan(tmp_path, monkeypatch, mode="authoritative", tracks=tracks)
    assert authoritative.order == observed.order
    assert [edge.plan.technique for edge in authoritative.transitions] == [edge.plan.technique for edge in observed.transitions]
    assert [edge.plan.execution_signature for edge in authoritative.transitions] == [edge.plan.execution_signature for edge in observed.transitions]
    assert authoritative.metadata["dj_executive"]["musical_behavior_changes"] is True


def test_v3_patch_preserves_rc1_frontend_transport() -> None:
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert '"/api/tracks/upload"' in api
    assert '"/api/tracks/upload-intents"' in api
    assert "EventSource" in api
    assert _sha256(ROOT / "frontend" / "next.config.ts") == "6e2f3d1a80e62e07e8757df97e11d5e450e8f78f705a6d7d995c6b9b28db556d"
    assert _sha256(ROOT / "compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"
