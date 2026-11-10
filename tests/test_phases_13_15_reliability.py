from __future__ import annotations

from pathlib import Path

from conftest import analysis
from djai.brain.runtime_control import PerformanceHorizonController
from djai.brain.timing import SetTimingIntent, build_timing_state, timing_candidate_adjustment
from djai.performance.reliability import (
    RecoveryAction,
    SessionJournal,
    VenueWatchdog,
    WatchdogObservation,
    decide_asset_recovery,
)
from djai.performance.safety import SafetyGovernor, SafetyTelemetry
from djai.planner import plan_transition_alternatives
from djai.set_models import SetPlan, TransitionEdge


def _plan(tmp_path: Path) -> SetPlan:
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    a = analysis(a_path, bpm=124.0, energy=0.70)
    b = analysis(b_path, bpm=126.0, energy=0.42)
    plans = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")
    primary = plans[0]
    alternatives = plans[1:3]
    edge = TransitionEdge(
        from_index=0,
        to_index=1,
        selected_style=primary.style,
        objective_score=primary.score,
        quality="good",
        plan=primary,
        alternatives=alternatives,
    )
    return SetPlan(
        plan_id="phase-13-15",
        tracks=[a, b],
        order=[0, 1],
        transitions=[edge],
        baseline_order=[0, 1],
        baseline_score=0.0,
        optimized_score=0.0,
        estimated_duration_seconds=580.0,
        source="test",
    )


def test_phase13_timing_progress_uses_event_clock(tmp_path: Path) -> None:
    tracks = []
    for index in range(3):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(b"x")
        tracks.append(analysis(path, bpm=122.0 + index, energy=0.4 + index * 0.1))

    intent = SetTimingIntent(
        target_duration_seconds=1800.0,
        peak_fraction=0.65,
        closing_window_seconds=300.0,
        ending_energy=0.35,
    )
    state = build_timing_state(
        tracks,
        intent=intent,
        actual_elapsed_seconds=900.0,
        projected_total_seconds=1790.0,
    )

    assert state.progress == 0.5
    assert state.planned_peak_seconds == 1170.0
    assert state.remaining_seconds == 900.0
    assert state.in_closing_window is False


def test_phase13_closing_window_biases_ending_energy(tmp_path: Path) -> None:
    low_path = tmp_path / "low.wav"
    high_path = tmp_path / "high.wav"
    low_path.write_bytes(b"l")
    high_path.write_bytes(b"h")
    low = analysis(low_path, bpm=122.0, energy=0.30)
    high = analysis(high_path, bpm=124.0, energy=0.90)
    state = build_timing_state(
        [],
        intent=SetTimingIntent(
            target_duration_seconds=1800.0,
            closing_window_seconds=300.0,
            ending_energy=0.30,
        ),
        actual_elapsed_seconds=1740.0,
        projected_total_seconds=1810.0,
    )

    low_adjustment, _ = timing_candidate_adjustment(state, low)
    high_adjustment, _ = timing_candidate_adjustment(state, high)

    assert state.in_closing_window is True
    assert state.urgency > 0.0
    assert low_adjustment > high_adjustment


def test_phase14_horizon_exposes_prepared_backup(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    controller = PerformanceHorizonController(plan, continuous_replanning=True)
    snapshot = controller.snapshot()

    if plan.transitions[0].alternatives:
        assert snapshot["hot_backup_ready"] or snapshot["emergency_ready"]
    assert snapshot["now_replannable"] is False
    assert snapshot["next_replannable"] is False
    assert snapshot["future_replannable"] is True


def test_phase14_underrun_requests_backup_when_available(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    governor = SafetyGovernor(plan, sample_rate=44_100)
    decision = governor.evaluate_telemetry(
        SafetyTelemetry(audio_underruns=1, buffer_seconds=0.05),
        backup_available=True,
    )

    assert decision.action == "use_backup"
    assert "underrun" in decision.reason


def test_phase15_watchdog_is_non_musical_and_fail_closed() -> None:
    watchdog = VenueWatchdog()
    decision = watchdog.evaluate(
        WatchdogObservation(
            monotonic_seconds=100.0,
            heartbeat_age_seconds=4.0,
            audio_progress_age_seconds=0.2,
            fallback_ready=True,
        )
    )
    assert decision.action == RecoveryAction.PROMOTE_HOT_BACKUP.value

    stopped = watchdog.evaluate(
        WatchdogObservation(
            monotonic_seconds=101.0,
            audio_progress_age_seconds=2.0,
            fallback_ready=False,
        )
    )
    assert stopped.action == RecoveryAction.SAFE_STOP.value


def test_phase15_session_journal_round_trip(tmp_path: Path) -> None:
    journal = SessionJournal(tmp_path / "runtime" / "session.json")
    journal.write(
        plan_id="plan-1",
        commit_digest="abc",
        schedule_generation=3,
        current_track=4,
        next_track=7,
        source_cursor=91.25,
        set_time=1200.5,
        edge_position=4,
        prepared_backup_signatures=("backup-a", "backup-b"),
    )
    payload = journal.read()

    assert payload is not None
    assert payload["plan_id"] == "plan-1"
    assert payload["schedule_generation"] == 3
    assert payload["recovery_semantics"] == "safe_boundary_only"
    assert payload["prepared_backup_signatures"] == ["backup-a", "backup-b"]


def test_phase15_future_asset_failure_replans_without_touching_now(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    edge = plan.transitions[0]

    decision = decide_asset_recovery(
        failed_track=9,
        current_track=0,
        next_edge=edge,
    )
    assert decision.action == RecoveryAction.REPLAN_FUTURE.value
