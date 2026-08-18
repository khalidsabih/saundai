from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from conftest import analysis
from djai.brain.executive import DJ_EXECUTIVE_VERSION, ExecutiveCommitError, executive_status
from djai.set_planner import make_set_plan

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracks(tmp_path: Path, count: int = 4):
    tracks = []
    tmp_path.mkdir(parents=True, exist_ok=True)
    camelot = ("8A", "9A", "10A", "7A")
    for index in range(count):
        path = tmp_path / f"executive-v2-{index}.wav"
        path.write_bytes(f"executive-v2-{index}".encode())
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
        source=f"executive-v2-{mode}",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )


def test_dj_executive_v2_defaults_to_single_commit_authority(monkeypatch) -> None:
    monkeypatch.delenv("DJAI_EXECUTIVE_MODE", raising=False)
    status = executive_status()
    assert status["version"] == DJ_EXECUTIVE_VERSION == "2.5.0"
    assert status["mode"] == "authoritative"
    assert status["authoritative"] is True
    assert status["single_commit_authority"] is True
    assert status["commit_authority"] == "dj_executive_v5"
    assert status["projected_search_state_is_non_authoritative"] is True
    assert status["propose_prepare_commit"] is True
    assert status["exact_compiled_program_required_for_commit"] is True
    assert status["commit_validation_fail_closed"] is True
    assert status["telemetry_failure_is_nonfatal"] is False
    assert status["brittleness_controls_compute_only"] is True
    assert status["hard_safety_budget_never_reduced"] is True


def test_v2_authoritative_commit_preserves_music_vs_observe(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path / "shared")
    observed = _plan(tmp_path, monkeypatch, mode="observe", tracks=tracks)
    authoritative = _plan(tmp_path, monkeypatch, mode="authoritative", tracks=tracks)

    assert authoritative.order == observed.order
    assert [edge.plan.technique for edge in authoritative.transitions] == [
        edge.plan.technique for edge in observed.transitions
    ]
    assert [edge.plan.execution_signature for edge in authoritative.transitions] == [
        edge.plan.execution_signature for edge in observed.transitions
    ]

    report = authoritative.metadata["dj_executive"]
    assert report["authoritative"] is True
    assert report["single_commit_authority"] is True
    assert report["commit_authority"] == "dj_executive_v5"
    assert report["musical_behavior_changes"] is True
    assert report["transaction_count"] == len(authoritative.transitions)
    assert report["utilization"]["executive_commits"] == len(authoritative.transitions)


def test_v2_transactions_are_exact_program_commits_and_chain_state(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    report = plan.metadata["dj_executive"]
    transactions = report["transactions"]
    decisions = report["decisions"]

    assert transactions
    assert report["initial_authoritative_state"]["current_track"] == plan.tracks[plan.order[0]].path
    assert report["final_authoritative_state"]["current_track"] == plan.tracks[plan.order[-1]].path

    for position, (edge, transaction, decision) in enumerate(zip(plan.transitions, transactions, decisions)):
        assert transaction["status"] == "committed"
        assert transaction["sequence_position"] == position
        assert transaction["current_index"] == edge.from_index
        assert transaction["next_index"] == edge.to_index
        assert transaction["prepared_execution_signature"] == edge.plan.execution_signature
        assert transaction["prepared_exact_program"] is True
        assert transaction["validation_passed"] is True
        assert transaction["phase_history"] == [
            "proposed", "brittleness_assessed", "thinking_budget_allocated",
            "intent_created", "prepared", "intent_bound", "validated", "committed"
        ]
        assert transaction["thinking_budget"]
        assert "execution_contract_valid" in transaction["validation_checks"]
        assert "performance_program_fingerprint" in transaction["validation_checks"]
        assert transaction["rejection_reason"] is None
        assert decision["transaction_id"] == transaction["transaction_id"]
        assert decision["transaction_status"] == "committed"
        assert decision["commit_authority"] == "dj_executive_v5"
        assert decision["observer_authoritative"] is True
        assert decision["evidence"]["compiled_execution_signature"] == edge.plan.execution_signature
        if position:
            assert transaction["state_before"] == transactions[position - 1]["state_after"]


def test_v2_layer_state_is_part_of_authoritative_commit_ledger(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch, mode="authoritative")
    report = plan.metadata["dj_executive"]
    for transaction in report["transactions"]:
        assert "layer_state" in transaction["state_before"]
        assert "layer_state" in transaction["state_after"]
        assert transaction["state_after"]["layer_state"]["vocal_owner"] in {"A", "B", "mixed", "none"}
        assert transaction["state_after"]["layer_state"]["instrumental_owner"] in {"A", "B", "mixed", "none"}


def test_v2_prepare_failure_cannot_commit_or_advance_state(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path / "good", monkeypatch, mode="observe")
    broken_transitions = copy.deepcopy(plan.transitions)
    broken_transitions[0].plan.execution_signature = "tampered"

    from djai.brain.executive import build_executive_authoritative_commit

    with pytest.raises(ExecutiveCommitError, match="invalid"):
        build_executive_authoritative_commit(
            analyses=plan.tracks,
            order=plan.order,
            transitions=broken_transitions,
            decisions=(),
            journey="build",
        )


def test_v2_hard_audio_failure_cannot_commit(tmp_path: Path, monkeypatch) -> None:
    plan = _plan(tmp_path / "good", monkeypatch, mode="observe")
    broken_transitions = copy.deepcopy(plan.transitions)
    first = broken_transitions[0].plan
    if first.measurement is None:
        from djai.models import TransitionMeasurement

        first.measurement = TransitionMeasurement(
            version="test",
            sample_rate=44100,
            duration_seconds=max(0.1, first.transition_seconds),
        )
    first.measurement.hard_failures = ["synthetic_clip_failure"]

    from djai.brain.executive import build_executive_authoritative_commit

    with pytest.raises(ExecutiveCommitError, match="hard rendered-audio failures"):
        build_executive_authoritative_commit(
            analyses=plan.tracks,
            order=plan.order,
            transitions=broken_transitions,
            decisions=(),
            journey="build",
        )


def test_v2_patch_preserves_rc1_frontend_transport() -> None:
    assert _sha256(ROOT / "frontend" / "lib" / "api.ts") == "b5255b01733771d460cb4d82bde744df021ae4d6c72b526a85344ccefc7f20bd"
    assert _sha256(ROOT / "frontend" / "next.config.ts") == "6e2f3d1a80e62e07e8757df97e11d5e450e8f78f705a6d7d995c6b9b28db556d"
    assert _sha256(ROOT / "compose.yaml") == "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"
