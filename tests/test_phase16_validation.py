from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from djai import phase16_validation


def _fake_plan(
    *,
    hard_failures: list[str] | None = None,
    fallback_level: int = 0,
    production_safe: bool = True,
):
    measurement = SimpleNamespace(
        hard_failures=list(hard_failures or []),
        to_dict=lambda: {"hard_failures": list(hard_failures or [])},
    )
    transition_plan = SimpleNamespace(
        measurement=measurement,
        measured_score=0.91,
        fallback_level=fallback_level,
    )
    edge = SimpleNamespace(
        from_index=0,
        to_index=1,
        plan=transition_plan,
    )
    metadata = {
        "journey": "build",
        "quality_loop": {
            "checked_transitions": 1,
            "passed_transitions": 0 if hard_failures else 1,
        },
        "unified_transition_intelligence": {"replan_count": 1},
        "timing": {
            "state": {
                "target_duration_seconds": 3600.0,
                "projected_total_seconds": 3580.0,
                "duration_error_seconds": -20.0,
                "planned_peak_seconds": 2340.0,
                "closing_window_seconds": 600.0,
                "ending_energy_target": 0.45,
            }
        },
        "full_set_benchmark": {
            "score": 0.9,
            "production_safe": production_safe and not hard_failures,
        },
    }
    return SimpleNamespace(
        plan_id="phase16-test",
        tracks=[SimpleNamespace(), SimpleNamespace()],
        transitions=[edge],
        estimated_duration_seconds=3580.0,
        warnings=[],
        requires_confirmation=False,
        metadata=metadata,
    )


def test_summarize_real_set_plan_ready() -> None:
    summary = phase16_validation.summarize_real_set_plan(_fake_plan())

    assert summary["planning_ready"] is True
    assert summary["hard_failure_count"] == 0
    assert summary["quality_pass_rate"] == 1.0
    assert summary["fallback_transition_count"] == 0
    assert summary["replan_count"] == 1
    assert summary["timing"]["duration_error_seconds"] == -20.0


def test_summarize_real_set_plan_blocks_hard_failure() -> None:
    summary = phase16_validation.summarize_real_set_plan(
        _fake_plan(hard_failures=["clipping"], fallback_level=1)
    )

    assert summary["planning_ready"] is False
    assert summary["hard_failure_count"] == 1
    assert summary["fallback_transition_count"] == 1
    assert summary["quality_pass_rate"] == 0.0


def test_phase16_plan_can_pass_quality_but_fail_set_safety() -> None:
    summary = phase16_validation.summarize_real_set_plan(
        _fake_plan(production_safe=False)
    )

    assert summary["planning_ready"] is True
    assert summary["quality_pass_rate"] == 1.0
    assert summary["hard_failure_count"] == 0
    assert summary["production_safe"] is False


def test_validate_real_music_library_writes_phase_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    music = tmp_path / "music"
    music.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    planned = _fake_plan()

    monkeypatch.setattr(
        phase16_validation,
        "plan_folder",
        lambda *args, **kwargs: planned,
    )

    def fake_save(plan, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(phase16_validation, "save_set_plan", fake_save)

    report = phase16_validation.validate_real_music_library(
        workspace,
        music,
        journeys=("build",),
        output_dir="output/phase16",
        render=False,
        target_duration_seconds=3600.0,
    )

    assert report["phase"] == 16
    assert report["summary"]["phase16_passed"] is True
    assert report["authority"]["learned_model_changes_live_decisions"] is False
    assert report["runs"][0]["journey"] == "build"

    report_path = workspace / "output" / "phase16" / "phase16-report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["all_rehearsal_ready"] is True
