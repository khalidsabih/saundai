from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from djai.build_info import build_identity
from djai.jobs import JobManager, JobRecord
from djai.resources import resolve_resource_budget


ROOT = Path(__file__).resolve().parents[1]


def _wait_terminal(manager: JobManager, job_id: str, timeout: float = 3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = manager.get(job_id)
        if record.status in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


def test_version_is_authoritative_and_sync_check_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync-version.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2.0.0-rc.1" in result.stdout


def test_build_identity_reports_consistent_source(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(ROOT))
    build_identity.cache_clear()
    identity = build_identity()
    assert identity["version"] == "2.0.0-rc.1"
    assert identity["version_consistent"] is True
    assert identity["authoritative_version_source"] == "VERSION"
    assert len(identity["source_fingerprint"]) == 16
    build_identity.cache_clear()


def test_resource_budget_clamps_top_level_workers(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    budget = resolve_resource_budget(
        {
            "DJAI_CPU_BUDGET": "4",
            "DJAI_JOB_WORKERS": "12",
            "DJAI_STEM_JOB_WORKERS": "5",
        }
    )
    assert budget.logical_cpus == 8
    assert budget.cpu_budget == 4
    assert budget.job_workers == 3
    assert budget.stem_workers == 1
    assert budget.constrained is True


def test_job_cancellation_is_cooperative_and_persisted(tmp_path: Path) -> None:
    manager = JobManager(tmp_path / "jobs", workers=1, heartbeat_seconds=1)
    entered = threading.Event()

    def task(context):
        entered.set()
        for index in range(1000):
            context.update_absolute("work", min(0.9, index / 1000), "working", current=index, total=1000)
            time.sleep(0.002)
        return {"unexpected": True}

    submitted = manager.submit("integrity-test", task)
    assert entered.wait(1.0)
    requested = manager.cancel(submitted.job_id)
    assert requested.cancel_requested is True
    completed = _wait_terminal(manager, submitted.job_id)
    assert completed.status == "cancelled"
    payload = json.loads((tmp_path / "jobs" / submitted.job_id / "job.json").read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert payload["cancel_requested"] is True
    assert payload["cancellable"] is False


def test_job_diagnostics_distinguish_stall_from_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("DJAI_JOB_STALL_SECONDS", "15")
    now = datetime.now(timezone.utc)
    record = JobRecord(
        job_id="x",
        kind="test",
        status="running",
        started_at_utc=(now - timedelta(minutes=2)).isoformat(),
        heartbeat_at_utc=(now - timedelta(seconds=2)).isoformat(),
        last_progress_at_utc=(now - timedelta(seconds=45)).isoformat(),
        stage_started_at_utc=(now - timedelta(seconds=45)).isoformat(),
    )
    payload = record.to_dict()
    assert payload["heartbeat_age_seconds"] < 5
    assert payload["progress_age_seconds"] >= 44
    assert payload["stalled"] is True
    assert payload["cancellable"] is True


def test_engineering_integrity_source_contracts() -> None:
    api = (ROOT / "src" / "djai" / "api.py").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    frontend = (ROOT / "frontend" / "components" / "JobProgress.tsx").read_text(encoding="utf-8")
    release_gate = (ROOT / "scripts" / "release-gate.sh").read_text(encoding="utf-8")
    assert '"engineering_integrity_1_5_3": True' in api
    # Router extraction moves registration out of api.py. Check the real
    # FastAPI routing table and handler identity rather than decorator text.
    from djai.api import app, cancel_job, runtime_status

    def registered_endpoint(path: str, method: str):
        matches = [
            route
            for route in app.routes
            if getattr(route, "path", None) == path
            and method in getattr(route, "methods", set())
        ]
        assert len(matches) == 1, f"{method} {path} must have exactly one route"
        return matches[0].endpoint

    assert registered_endpoint("/api/runtime/status", "GET") is runtime_status
    assert registered_endpoint("/api/jobs/{job_id}/cancel", "POST") is cancel_job
    assert "SAUND_IMAGE_TAG:-local" in compose
    assert "saund-ai-engine:1.5.0" not in compose
    assert "onCancel" in frontend and "No stage progress" in frontend
    assert "sync-version.py --check" in release_gate
    assert "1.0.0 release gate" not in release_gate
