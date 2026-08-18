from __future__ import annotations

from pathlib import Path

from djai.decision import validate_execution_plan
from djai.set_planner import load_set_plan, make_set_plan, save_set_plan
from djai.set_renderer import prepare_render_sequence

from conftest import analysis


def test_set_plan_compiles_continuous_execution(tmp_path: Path):
    tracks = []
    for index, bpm in enumerate((122.0, 124.0, 126.0, 128.0)):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(f"track-{index}".encode())
        tracks.append(
            analysis(
                path,
                bpm=bpm,
                camelot=("8A", "9A", "10A", "9A")[index],
                energy=0.45 + index * 0.10,
            )
        )

    plan = make_set_plan(
        tracks,
        source="test",
        tempo_policy="safe",
        preferred_style="auto",
        journey="build",
        minimum_body_seconds=10.0,
        minimum_body_fraction=0.10,
    )
    assert len(plan.transitions) == len(tracks) - 1
    previous_rate = 1.0
    previous_bpm = plan.tracks[plan.order[0]].bpm
    for edge in plan.transitions:
        validate_execution_plan(edge.plan)
        assert edge.plan.metadata.get("dj_move", {}).get("version") == "1.4"
        assert edge.plan.metadata.get("dj_state_before_move", {}).get("version") == "1.4"
        assert abs(edge.plan.render_rate_a - previous_rate) < 1e-6
        assert abs(edge.plan.effective_bpm_a - previous_bpm) < 0.02
        previous_rate = edge.plan.post_transition_rate_b
        previous_bpm = edge.plan.post_transition_bpm_b or (plan.tracks[edge.to_index].bpm * previous_rate)

    states, edge_states = prepare_render_sequence(plan)
    assert len(states) == len(tracks)
    assert len(edge_states) == len(tracks) - 1

    output = save_set_plan(plan, tmp_path / "plan.json")
    restored = load_set_plan(output)
    assert restored.order == plan.order
    assert [edge.plan.execution_signature for edge in restored.transitions] == [
        edge.plan.execution_signature for edge in plan.transitions
    ]
