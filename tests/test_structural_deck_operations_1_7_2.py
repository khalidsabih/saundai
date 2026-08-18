from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from conftest import analysis
from djai.brain.move_generator import CompleteMoveConfig, generate_complete_moves
from djai.brain.dj_move import build_dj_state
from djai.brain.set_state import build_set_state
from djai.performance.engine import execute_transition_program
from djai.performance.operations import ProgramDeckOperation, structural_deck_runtime_status
from djai.performance.program import (
    PERFORMANCE_PROGRAM_VERSION,
    CurvePoint,
    TransitionProgram,
    attach_performance_program,
    compile_transition_program,
)
from djai.planner import plan_transition_alternatives
from djai.set_planner import build_transition_graph, make_set_plan
from djai.set_renderer import _render_pre_transition_body


def _constant(value: float) -> list[CurvePoint]:
    return [CurvePoint(0.0, value), CurvePoint(1.0, value)]


def _tracks(tmp_path: Path):
    tracks = []
    for index, bpm in enumerate((122.0, 124.0, 126.0)):
        path = tmp_path / f"structural-{index}.wav"
        path.write_bytes(f"structural-{index}".encode())
        tracks.append(analysis(path, bpm=bpm, camelot=("8A", "9A", "10A")[index], energy=0.48 + 0.08 * index))
    return tracks


def test_true_wait_is_part_of_the_complete_move_ir(tmp_path: Path) -> None:
    tracks = _tracks(tmp_path)
    graph = build_transition_graph(tracks, workers=1)
    set_state = build_set_state([tracks[0]])
    dj_state = build_dj_state(tracks[0], source_cursor=0.0, set_state=set_state)
    candidates = generate_complete_moves(
        current_index=0,
        candidate_indices=[1, 2],
        analyses=tracks,
        graph=graph,
        dj_state=dj_state,
        set_state=set_state,
        journey="balanced",
        progress=0.0,
        config=CompleteMoveConfig(mode="move_authoritative"),
    )
    assert candidates
    wait_ops = [op for op in candidates[0].move.deck_operations if op.kind == "WAIT"]
    assert wait_ops
    assert wait_ops[0].phase == "pre_transition"
    assert wait_ops[0].parameters["bars"] == candidates[0].move.wait_bars
    assert candidates[0].plan.performance_program["deck_operations"]
    assert candidates[0].plan.performance_program_fingerprint


def test_authoritative_set_plan_signs_selected_structural_operations(tmp_path: Path, monkeypatch) -> None:
    tracks = _tracks(tmp_path)
    monkeypatch.setenv("DJAI_PLANNER_MODE", "move_authoritative")
    plan = make_set_plan(tracks, source="structural-authoritative", graph_workers=1, automatic_delivery=False)
    assert plan.transitions
    for edge in plan.transitions:
        operations = edge.plan.performance_program.get("deck_operations", [])
        assert operations
        assert any(item["kind"] == "WAIT" for item in operations)
        assert edge.plan.metadata["structural_operation_authority"]["deterministic_renderer_authoritative"] is True


def test_pre_transition_loop_and_jump_change_consumed_body_without_guessing(track_pair) -> None:
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")[0]
    plan = replace(plan, exit_time_a=8.0)
    loop = ProgramDeckOperation(
        kind="LOOP_OUTGOING",
        phase="pre_transition",
        deck="a",
        parameters={
            "source_start_seconds": 2.0,
            "source_end_seconds": 4.0,
            "repeats": 2,
        },
        reason="test loop",
    )
    plan.metadata = {**plan.metadata, "deck_operations": [loop.to_dict()]}
    attach_performance_program(plan)
    sr = 1000
    t = np.arange(sr * 10, dtype=np.float32) / sr
    audio = np.vstack([np.sin(2 * np.pi * 7 * t), np.sin(2 * np.pi * 9 * t)]).astype(np.float32)
    rendered, events = _render_pre_transition_body(
        audio,
        sample_rate=sr,
        source_cursor=0.0,
        exit_time=8.0,
        rate=1.0,
        gain=1.0,
        plan=plan,
    )
    assert rendered.shape[-1] == sr * 10  # 8 seconds + one extra 2-second loop pass
    assert events[0]["kind"] == "LOOP_OUTGOING"

    jump = ProgramDeckOperation(
        kind="SKIP_BREAKDOWN",
        phase="pre_transition",
        deck="a",
        parameters={"trigger_source_seconds": 2.0, "target_source_seconds": 4.0},
        reason="test forward skip",
    )
    plan.metadata = {**plan.metadata, "deck_operations": [jump.to_dict()]}
    attach_performance_program(plan)
    rendered, events = _render_pre_transition_body(
        audio,
        sample_rate=sr,
        source_cursor=0.0,
        exit_time=8.0,
        rate=1.0,
        gain=1.0,
        plan=plan,
    )
    assert rendered.shape[-1] == sr * 6
    assert events[0]["kind"] == "SKIP_BREAKDOWN"


def test_incoming_intro_extension_executes_inside_signed_transition_program() -> None:
    sr = 4000
    length = sr * 4
    a = np.zeros((2, length), dtype=np.float32)
    # Non-periodic ramp makes a loop operation observably different.
    ramp = np.linspace(-0.8, 0.8, length, dtype=np.float32)
    b = np.vstack([ramp, ramp[::-1]]).astype(np.float32)
    baseline = TransitionProgram(
        version=PERFORMANCE_PROGRAM_VERSION,
        intent="smooth_continuation",
        technique="equal_power_blend",
        transition_bars=8,
        transition_seconds=4.0,
        target_bpm=120.0,
        tempo_strategy="native_blend",
        source_mode="master",
        curves={"deck_a_gain": _constant(0.0), "deck_b_gain": _constant(1.0)},
    )
    extended = TransitionProgram(
        version=PERFORMANCE_PROGRAM_VERSION,
        intent=baseline.intent,
        technique=baseline.technique,
        transition_bars=baseline.transition_bars,
        transition_seconds=baseline.transition_seconds,
        target_bpm=baseline.target_bpm,
        tempo_strategy=baseline.tempo_strategy,
        source_mode="master",
        curves=baseline.curves,
        deck_operations=[
            ProgramDeckOperation(
                kind="EXTEND_INTRO",
                phase="transition",
                deck="b",
                parameters={
                    "source_start_seconds": 0.0,
                    "source_end_seconds": 1.0,
                    "repeats": 2,
                    "loop_beats": 2.0,
                    "hold_fraction": 0.55,
                },
                reason="test incoming intro hold",
            )
        ],
    )
    plain = execute_transition_program(baseline, {"deck_a": a, "deck_b": b}, sample_rate=sr, bpm=120.0).audio
    looped = execute_transition_program(extended, {"deck_a": a, "deck_b": b}, sample_rate=sr, bpm=120.0).audio
    assert not np.allclose(plain[:, : length // 2], looped[:, : length // 2])
    assert looped.shape == plain.shape


def test_hold_stem_is_compiled_into_stem_automation(track_pair) -> None:
    a, b = track_pair
    plan = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")[0]
    # Force the generic stem-aware compiler path with placeholder paths; this
    # test inspects the immutable curve IR and never opens these files.
    plan.technique = "stem_choreography"
    plan.stem_paths_a = {name: f"a-{name}.wav" for name in ("drums", "bass", "vocals", "other")}
    plan.stem_paths_b = {name: f"b-{name}.wav" for name in ("drums", "bass", "vocals", "other")}
    hold = ProgramDeckOperation(
        kind="HOLD_STEM",
        phase="transition",
        deck="b",
        parameters={"stem": "drums", "start_position": 0.10, "end_position": 0.45, "minimum_gain": 0.65},
        reason="test stem hold",
    )
    plan.metadata = {**plan.metadata, "deck_operations": [hold.to_dict()]}
    program = compile_transition_program(plan)
    held = [point.value for point in program.stem_curves_b["drums"] if 0.10 <= point.position <= 0.45]
    assert held
    assert min(held) >= 0.65


def test_structural_operation_status_is_honest() -> None:
    status = structural_deck_runtime_status()
    assert status["deterministic_ir"] is True
    assert "WAIT" in status["supported_operations"]
    assert "SKIP_BREAKDOWN" in status["supported_operations"]
    assert status["learned_code_manipulates_samples"] is False
