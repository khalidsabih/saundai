from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from djai.jobs import JobManager


ROOT = Path(__file__).resolve().parents[1]


def _wait(manager: JobManager, job_id: str) -> None:
    for _ in range(300):
        if manager.get(job_id).status in {"completed", "failed"}:
            return
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_job_wide_progress_is_monotonic_and_carries_real_stage_counts(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", workers=1)

    def task(context):
        context.update_absolute("starting", 0.01, "Preparing 14 tracks", current=0, total=14)
        analysis = context.ranged_progress_callback(0.08, 0.32)
        analysis("analysis", 0, 14, "analysis workers")
        analysis("analysis", 7, 14, "half analyzed")
        analysis("analysis", 14, 14, "analysis complete")

        planning = context.ranged_progress_callback(
            0.32,
            0.80,
            stage_ranges={
                "graph": (0.32, 0.50),
                "order-replan": (0.50, 0.56),
                "measure": (0.56, 0.80),
            },
        )
        planning("graph", 0, 182, "scouting")
        planning("graph", 91, 182, "half scouted")
        planning("graph", 182, 182, "graph complete")
        planning("order-replan", 0, 8, "choosing route")
        planning("order-replan", 8, 8, "route chosen")
        planning("measure", 0, 13, "rehearsing")
        planning("measure", 13, 13, "rehearsal complete")

        render = context.ranged_progress_callback(0.80, 0.985)
        render("render", 0, 14, "rendering")
        render("render", 7, 14, "half rendered")
        render("render", 14, 14, "render complete")
        context.update_absolute("finalizing", 0.99, "saving", current=1, total=1)
        return {"ok": True}

    record = manager.submit("automatic-set", task)
    _wait(manager, record.job_id)
    completed = manager.get(record.job_id)
    assert completed.status == "completed"
    assert completed.progress == 1.0

    payload = json.loads((tmp_path / "jobs" / record.job_id / "job.json").read_text(encoding="utf-8"))
    running_events = [event for event in payload["events"] if event["status"] == "running"]
    positive = [event for event in running_events if event["progress"] > 0]
    assert positive
    assert positive[0]["progress"] == 0.01
    assert [event["progress"] for event in positive] == sorted(event["progress"] for event in positive)

    graph_half = next(event for event in running_events if event["message"] == "half scouted")
    assert graph_half["progress"] == pytest.approx(0.41)
    assert graph_half["stage_current"] == 91
    assert graph_half["stage_total"] == 182

    measure_done = next(event for event in running_events if event["message"] == "rehearsal complete")
    assert measure_done["progress"] == pytest.approx(0.80)
    assert measure_done["stage_current"] == 13
    assert measure_done["stage_total"] == 13


def test_create_progress_contract_is_backend_driven_not_timer_faked() -> None:
    api = (ROOT / "src/djai/api.py").read_text(encoding="utf-8")
    jobs = (ROOT / "src/djai/jobs.py").read_text(encoding="utf-8")
    component = (ROOT / "frontend/components/JobProgress.tsx").read_text(encoding="utf-8")
    frontend_api = (ROOT / "frontend/lib/api.ts").read_text(encoding="utf-8")

    automatic = api.split('@app.post("/api/sets/automatic"', 1)[1].split('@app.post("/api/set/render"', 1)[0]
    assert 'context.update_absolute(' in automatic
    assert 'context.ranged_progress_callback(0.08, 0.32)' in automatic
    assert '"graph": (0.32, 0.50)' in automatic
    assert '"measure": (0.56, 0.80)' in automatic
    assert 'context.ranged_progress_callback(0.80, 0.985)' in automatic
    assert 'setInterval' not in component
    assert 'setTimeout' not in component
    assert 'stage_current' in jobs and 'stage_total' in jobs
    assert 'stage_current: update.stage_current' in frontend_api
    assert 'stageCount' in component
