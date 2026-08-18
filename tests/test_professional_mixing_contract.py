from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from conftest import analysis
from djai.brain.professional_mixing import apply_professional_mixing_policy
from djai.brain.transition_router import route_transition
from djai.models import GenreEstimate
from djai.performance.engine import _loop_prefix_window
from djai.live import _render_live_hold_loop
from djai.performance.executor import DeterministicSetExecutor, ExecutorSafetyHold
from djai.performance.mix_contract import evaluate_professional_mix_contract
from djai.planner import plan_transition
from djai.set_models import SetPlan, TransitionEdge


def test_close_pair_gets_continuous_professional_mix_contract(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")

    profile = dict(plan.metadata.get("professional_mix", {}) or {})
    contract = evaluate_professional_mix_contract(plan)

    assert profile["family"] in {"full_blend", "structural_bridge"}
    assert profile["continuous_mix_required"] is True
    assert profile["reset_is_last_resort"] is True
    assert contract.applicable is True
    assert contract.passed is True
    assert contract.dual_deck_overlap_fraction >= 0.22
    assert contract.incoming_established_position is not None
    assert contract.outgoing_release_position is not None
    assert contract.incoming_established_position <= contract.outgoing_release_position + 0.08


def test_reset_effect_is_rejected_when_pair_does_not_justify_reset(track_pair) -> None:
    track_a, track_b = track_pair
    base = plan_transition(track_a, track_b, force=True, style="safe")
    reset = copy.deepcopy(base)
    reset.technique = "quick_cut"
    reset.style = "creative"
    reset.transition_mode = "emergency_cut"
    reset.alternative_id = "adversarial-reset"

    annotated = apply_professional_mixing_policy(
        track_a,
        track_b,
        [reset],
        route=route_transition(track_a, track_b),
    )[0]
    contract = evaluate_professional_mix_contract(annotated)

    assert annotated.metadata["professional_mix"]["structural_reset"] is True
    assert annotated.metadata["professional_mix"]["reset_justified"] is False
    assert contract.passed is False
    assert any("not explicitly justified" in item for item in contract.hard_failures)


def test_distant_pair_prefers_bridge_over_forced_reset(tmp_path: Path) -> None:
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    track_a = analysis(a_path, bpm=122.0, camelot="8A", energy=0.48)
    track_b = analysis(b_path, bpm=168.0, camelot="10A", energy=0.88)
    track_a.genre = GenreEstimate(label="deep_house", confidence=0.94, probabilities={"deep_house": 0.94})
    track_b.genre = GenreEstimate(label="drum_and_bass", confidence=0.96, probabilities={"drum_and_bass": 0.96})

    plan = plan_transition(track_a, track_b, force=True, style="creative", tempo_policy="cut")
    profile = dict(plan.metadata["professional_mix"])

    assert profile["bridge_preferred"] is True
    assert profile["professional_brain_may_reject_pair"] is True
    contract = evaluate_professional_mix_contract(plan)
    if profile["structural_reset"]:
        assert profile["reset_justified"] is False
        assert contract.passed is False
    else:
        assert profile["family"] == "structural_bridge"
        assert profile["continuous_mix_required"] is True


def test_incoming_runway_loop_fills_prefix_and_lands_on_committed_source() -> None:
    sample_rate = 1_000
    source_frames = 1_000
    target_frames = 4_000
    t = np.arange(source_frames, dtype=np.float32) / sample_rate
    source = 0.2 * np.sin(2.0 * np.pi * 7.0 * t)
    padded = np.zeros((2, target_frames), dtype=np.float32)
    padded[:, :source_frames] = source

    result = _loop_prefix_window(
        padded,
        sample_rate=sample_rate,
        bpm=120.0,
        beats=1.0,
        hold_fraction=0.84,
        source_fraction=source_frames / target_frames,
    )

    assert result.shape == padded.shape
    assert float(np.max(np.abs(result[:, : target_frames - source_frames]))) > 0.05
    assert float(np.max(np.abs(result[:, -source_frames:]))) > 0.05
    assert np.allclose(result[:, -200:], padded[:, source_frames - 200 : source_frames], atol=1e-6)


def test_executor_hold_is_authoritative_for_audible_transition(track_pair) -> None:
    track_a, track_b = track_pair
    transition = plan_transition(track_a, track_b, force=True, style="safe")
    edge = TransitionEdge(
        from_index=0,
        to_index=1,
        selected_style="safe",
        objective_score=transition.score,
        quality=transition.quality,
        plan=transition,
        alternatives=[],
    )
    set_plan = SetPlan(
        plan_id="hold-test",
        tracks=[track_a, track_b],
        order=[0, 1],
        transitions=[edge],
        baseline_order=[0, 1],
        baseline_score=0.5,
        optimized_score=0.6,
        estimated_duration_seconds=500.0,
        source="test",
        automatic_delivery=True,
    )
    runtime = DeterministicSetExecutor(set_plan, sample_rate=44_100)
    runtime.arm()
    initial = runtime.begin_load_track()
    runtime.complete_load_track(initial, loaded_sample_rate=44_100)
    body = runtime.begin_body(edge_position=0, source_cursor=0.0)
    runtime.complete_body(
        body,
        source_cursor_after=transition.exit_time_a,
        set_time=transition.exit_time_a,
        frames=1,
    )

    hold = runtime.update_telemetry(cpu_ratio=1.25, prefetch_ready=False)
    assert hold["action"] == "hold"

    load = runtime.begin_load_track(edge_position=0)
    runtime.complete_load_track(load, loaded_sample_rate=44_100, set_time=transition.exit_time_a)
    with pytest.raises(ExecutorSafetyHold):
        runtime.begin_transition(edge_position=0)

    allow = runtime.update_telemetry(cpu_ratio=0.20, prefetch_ready=True)
    assert allow["action"] == "allow"
    command = runtime.begin_transition(edge_position=0)
    assert command.kind == "transition"


def test_master_only_structural_bridge_rejects_severe_musical_collision(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    profile = dict(plan.metadata.get("professional_mix", {}) or {})
    profile.update(
        {
            "family": "structural_bridge",
            "continuous_mix_required": True,
            "structural_reset": False,
            "reset_justified": False,
            "vocal_conflict": 0.82,
            "bass_conflict": 0.76,
            "melodic_conflict": 0.79,
        }
    )
    plan.metadata = {**plan.metadata, "professional_mix": profile}
    program = dict(plan.performance_program or {})
    program["source_mode"] = "master"
    curves = dict(program.get("curves", {}) or {})
    curves.pop("deck_a_low_gain", None)
    curves.pop("deck_b_low_gain", None)
    program["curves"] = curves
    plan.performance_program = program

    contract = evaluate_professional_mix_contract(plan)

    assert contract.passed is False
    assert any("separated sources or replan" in item for item in contract.hard_failures)
    assert any("low-frequency ownership" in item for item in contract.hard_failures)


def test_live_hold_loop_keeps_outgoing_audio_alive_without_silence() -> None:
    sample_rate = 2_000
    seconds = 12.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    mono = 0.18 * np.sin(2.0 * np.pi * 5.0 * t)
    audio = np.vstack([mono, mono]).astype(np.float32)

    hold = _render_live_hold_loop(
        audio,
        sample_rate=sample_rate,
        exit_time=10.0,
        rate=1.0,
        target_bpm=120.0,
        time_signature=4,
        bars=4,
    )

    assert hold.shape == (2, 16_000)
    assert float(np.max(np.abs(hold))) > 0.10
    assert float(np.mean(np.abs(hold))) > 0.05
