from __future__ import annotations

from types import SimpleNamespace

from conftest import analysis
from djai.brain.executive import DJ_EXECUTIVE_VERSION, executive_status
from djai.brain.recovery import (
    PREPARED_RECOVERY_VERSION,
    build_prepared_recovery_plan,
    select_prepared_backup,
)
from djai.set_planner import make_set_plan


def _tracks(tmp_path, count: int = 5):
    keys = ("8A", "9A", "10A", "7A", "8A")
    tracks = []
    for index in range(count):
        path = tmp_path / f"executive-v5-{index}.wav"
        path.write_bytes(f"executive-v5-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=123.0 + 1.2 * index,
                camelot=keys[index],
                energy=0.42 + 0.09 * index,
            )
        )
    return tracks


def _plan(tmp_path, monkeypatch, *, mode: str = "authoritative"):
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", mode)
    monkeypatch.setenv("DJAI_HEADPHONES", "0")
    monkeypatch.setenv("DJAI_ADAPTIVE_THINKING", "0")
    return make_set_plan(
        _tracks(tmp_path),
        source=f"executive-v5-{mode}",
        journey="build",
        graph_workers=1,
        automatic_delivery=False,
    )


def test_v5_status_exposes_prepared_recovery(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_EXECUTIVE_MODE", "authoritative")
    status = executive_status()
    assert status["version"] == DJ_EXECUTIVE_VERSION == "2.5.0"
    assert status["commit_authority"] == "dj_executive_v5"
    assert status["prepared_recovery_version"] == PREPARED_RECOVERY_VERSION == "1.0"
    assert status["prepared_recovery_capable"] is True
    assert status["backup_requirement_by_brittleness"]["normal"] == 1
    assert status["maximum_backups_by_brittleness"]["brittle"] == 2
    assert status["same_target_recovery_only"] is True
    assert status["alternate_target_recovery_requires_plan_repair"] is True
    assert status["recovery_readiness_levels"] == ["prepared", "auditioned", "hot"]
    assert "prepare_backup" in status["observed_lifecycle"]


def test_v5_transactions_carry_recovery_receipts_without_changing_primary(tmp_path, monkeypatch) -> None:
    observed = _plan(tmp_path / "observed", monkeypatch, mode="observe")
    authoritative = _plan(tmp_path / "authoritative", monkeypatch, mode="authoritative")
    # Different fixture paths mean signatures may differ, but the musical route
    # and technique sequence must remain the same between modes.
    assert authoritative.order == observed.order
    assert [edge.plan.technique for edge in authoritative.transitions] == [
        edge.plan.technique for edge in observed.transitions
    ]
    report = authoritative.metadata["dj_executive"]
    assert report["transaction_count"] == len(authoritative.transitions)
    for transaction, decision, edge in zip(report["transactions"], report["decisions"], authoritative.transitions):
        recovery = transaction["prepared_recovery"]
        assert recovery["version"] == PREPARED_RECOVERY_VERSION
        assert recovery["primary_execution_signature"] == edge.plan.execution_signature
        assert recovery["next_index"] == edge.to_index
        assert decision["prepared_recovery"] == recovery
        assert transaction["phase_history"][-1] == "committed"
        assert transaction["phase_history"][-3] in {"backup_prepared", "backup_unavailable"}


def test_v5_recovery_prefers_hot_then_auditioned_and_keeps_route_changes_advisory(monkeypatch) -> None:
    monkeypatch.setattr("djai.brain.recovery.validate_execution_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "djai.brain.recovery.bind_prepared_transition_to_intent",
        lambda **kwargs: SimpleNamespace(status="flexible", semantic_intent_preserved=True),
    )
    track_a = SimpleNamespace(path="a.wav")
    track_b = SimpleNamespace(path="b.wav")
    intent = SimpleNamespace(intent_id="intent-test")

    def plan(signature, *, track_b_path="b.wav", quality=False, passed=False, hard=()):
        return SimpleNamespace(
            track_a="a.wav",
            track_b=track_b_path,
            execution_signature=signature,
            technique="eq_bass_swap",
            compiled_execution=True,
            performance_program_fingerprint=f"fp-{signature}",
            quality_checked=quality,
            measurement_passed=passed,
            measurement=SimpleNamespace(hard_failures=tuple(hard)) if hard or quality else None,
            fallback_level=0,
        )

    def candidate(signature, score, *, next_index=1, **kwargs):
        return SimpleNamespace(
            next_index=next_index,
            selection_score=score,
            plan=plan(signature, **kwargs),
            move=SimpleNamespace(technique="eq_bass_swap", rationale=("safe alternate",)),
        )

    recovery = build_prepared_recovery_plan(
        current_index=0,
        next_index=1,
        track_a=track_a,
        track_b=track_b,
        intent=intent,
        primary_plan=plan("primary"),
        candidates=(
            candidate("prepared", 0.99),
            candidate("auditioned", 0.90),
            candidate("hot", 0.80, quality=True, passed=True),
            candidate("unsafe", 1.00, quality=True, passed=False, hard=("clip",)),
            candidate("alternate-target", 0.95, next_index=2, track_b_path="c.wav"),
        ),
        headphones_report={"rendered": [{"execution_signature": "auditioned"}]},
        required_backups=1,
        maximum_backups=3,
    )
    assert recovery.recovery_ready is True
    assert [item.readiness for item in recovery.prepared_backups] == ["hot", "auditioned", "prepared"]
    assert "unsafe" not in [item.execution_signature for item in recovery.prepared_backups]
    assert recovery.alternate_target_advisories[0]["status"] == "advisory_only_route_change_required"
    assert select_prepared_backup(recovery).execution_signature == "hot"


def test_v5_recovery_requires_no_new_render_for_receipt(tmp_path, monkeypatch) -> None:
    plan = _plan(tmp_path, monkeypatch)
    report = plan.metadata["dj_executive"]
    assert report["utilization"]["prepared_recovery_transactions"] == len(plan.transitions)
    # V5 only classifies evidence already generated by the current brain. It
    # does not increase Headphones render counts merely to create a receipt.
    assert report["utilization"]["ai_headphones_rendered_candidates"] == 0
