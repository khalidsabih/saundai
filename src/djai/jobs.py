from __future__ import annotations

import json
import os
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Task = Callable[["JobContext"], dict[str, Any]]
_TERMINAL = {"completed", "failed", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    point = now or datetime.now(timezone.utc)
    return max(0.0, (point - parsed).total_seconds())


class JobCancelled(RuntimeError):
    """Cooperative cancellation signal raised only at safe job boundaries."""


@dataclass(slots=True)
class JobRecord:
    job_id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    message: str = ""
    stage_current: int = 0
    stage_total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at_utc: str = field(default_factory=_utc_now)
    updated_at_utc: str = field(default_factory=_utc_now)
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    heartbeat_at_utc: str | None = None
    last_progress_at_utc: str | None = None
    stage_started_at_utc: str | None = None
    cancel_requested: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)
    owner_user_id: str | None = None
    resource_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        terminal = self.status in _TERMINAL
        started = _parse_utc(self.started_at_utc) or _parse_utc(self.created_at_utc)
        finished = _parse_utc(self.finished_at_utc) if terminal else now
        elapsed = max(0.0, (finished - started).total_seconds()) if started and finished else 0.0
        try:
            stall_after = max(15, int(os.environ.get("DJAI_JOB_STALL_SECONDS", "180")))
        except ValueError:
            stall_after = 180
        progress_age = _age_seconds(self.last_progress_at_utc or self.started_at_utc, now=now)
        heartbeat_age = _age_seconds(self.heartbeat_at_utc or self.updated_at_utc, now=now)
        stage_age = _age_seconds(self.stage_started_at_utc or self.started_at_utc, now=now)
        stalled = bool(
            self.status == "running"
            and not self.cancel_requested
            and progress_age is not None
            and progress_age >= stall_after
        )
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "stage_current": self.stage_current,
            "stage_total": self.stage_total,
            "result": self.result,
            "error": self.error,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "heartbeat_at_utc": self.heartbeat_at_utc,
            "last_progress_at_utc": self.last_progress_at_utc,
            "stage_started_at_utc": self.stage_started_at_utc,
            "elapsed_seconds": round(elapsed, 3),
            "stage_elapsed_seconds": round(stage_age or 0.0, 3),
            "heartbeat_age_seconds": round(heartbeat_age or 0.0, 3),
            "progress_age_seconds": round(progress_age or 0.0, 3),
            "stall_after_seconds": stall_after,
            "stalled": stalled,
            "cancel_requested": self.cancel_requested,
            "cancellable": self.status in {"queued", "running"} and not self.cancel_requested,
            "event_count": len(self.events),
        }


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str, output_dir: Path) -> None:
        self.manager = manager
        self.job_id = job_id
        self.output_dir = output_dir

    @property
    def cancelled(self) -> bool:
        try:
            return bool(self.manager.get(self.job_id).cancel_requested)
        except KeyError:
            return True

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled("Job cancelled by user request.")

    def update(self, stage: str, current: int, total: int, message: str = "") -> None:
        self.raise_if_cancelled()
        progress = 0.0 if total <= 0 else max(0.0, min(1.0, current / total))
        self.manager._update(
            self.job_id,
            status="running",
            stage=stage,
            progress=progress,
            message=message,
            stage_current=max(0, int(current)),
            stage_total=max(0, int(total)),
        )

    def update_absolute(
        self,
        stage: str,
        progress: float,
        message: str = "",
        *,
        current: int = 0,
        total: int = 0,
        monotonic: bool = True,
    ) -> None:
        """Publish real job-wide progress instead of resetting at each stage."""
        self.raise_if_cancelled()
        value = max(0.0, min(1.0, float(progress)))
        if monotonic:
            try:
                value = max(value, float(self.manager.get(self.job_id).progress))
            except KeyError:
                pass
        self.manager._update(
            self.job_id,
            status="running",
            stage=stage,
            progress=value,
            message=message,
            stage_current=max(0, int(current)),
            stage_total=max(0, int(total)),
        )

    def ranged_progress_callback(
        self,
        start: float,
        end: float,
        *,
        stage_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> Callable[[str, int, int, str], None]:
        """Map stage-local counters into a monotonic slice of the whole job."""
        lo = max(0.0, min(1.0, float(start)))
        hi = max(lo, min(1.0, float(end)))
        ranges = dict(stage_ranges or {})

        def callback(stage: str, current: int, total: int, message: str) -> None:
            self.raise_if_cancelled()
            stage_lo, stage_hi = ranges.get(stage, (lo, hi))
            stage_lo = max(lo, min(hi, float(stage_lo)))
            stage_hi = max(stage_lo, min(hi, float(stage_hi)))
            fraction = 0.0 if total <= 0 else max(0.0, min(1.0, current / total))
            mapped = stage_lo + (stage_hi - stage_lo) * fraction
            self.update_absolute(
                stage,
                mapped,
                message,
                current=current,
                total=total,
                monotonic=True,
            )

        return callback

    def progress_callback(self, stage: str, current: int, total: int, message: str) -> None:
        self.update(stage, current, total, message)


class JobManager:
    def __init__(
        self,
        root: str | Path,
        *,
        workers: int = 2,
        heartbeat_seconds: float | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, int(workers))
        if heartbeat_seconds is None:
            try:
                heartbeat_seconds = float(os.environ.get("DJAI_JOB_HEARTBEAT_SECONDS", "5"))
            except ValueError:
                heartbeat_seconds = 5.0
        self.heartbeat_seconds = max(1.0, float(heartbeat_seconds))
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="djai-job")
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.RLock()
        self._conditions: dict[str, threading.Condition] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for state_path in self.root.glob("*/job.json"):
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                record = JobRecord(
                    job_id=str(payload["job_id"]),
                    kind=str(payload.get("kind", "unknown")),
                    status=str(payload.get("status", "failed")),
                    progress=float(payload.get("progress", 0.0)),
                    stage=str(payload.get("stage", "recovered")),
                    message=str(payload.get("message", "")),
                    stage_current=int(payload.get("stage_current", 0) or 0),
                    stage_total=int(payload.get("stage_total", 0) or 0),
                    result=payload.get("result"),
                    error=payload.get("error"),
                    created_at_utc=str(payload.get("created_at_utc", _utc_now())),
                    updated_at_utc=str(payload.get("updated_at_utc", _utc_now())),
                    started_at_utc=payload.get("started_at_utc"),
                    finished_at_utc=payload.get("finished_at_utc"),
                    heartbeat_at_utc=payload.get("heartbeat_at_utc"),
                    last_progress_at_utc=payload.get("last_progress_at_utc"),
                    stage_started_at_utc=payload.get("stage_started_at_utc"),
                    cancel_requested=bool(payload.get("cancel_requested", False)),
                    events=list(payload.get("events", [])),
                    owner_user_id=(str(payload["owner_user_id"]) if payload.get("owner_user_id") else None),
                    resource_context=dict(payload.get("resource_context", {})),
                )
                if record.status in {"queued", "running", "cancelling"}:
                    record.status = "failed"
                    record.stage = "interrupted"
                    record.message = "Job was interrupted by a previous process shutdown."
                    record.error = record.message
                    record.finished_at_utc = _utc_now()
                self._jobs[record.job_id] = record
                self._conditions[record.job_id] = threading.Condition(self._lock)
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                continue

    def submit(
        self,
        kind: str,
        task: Task,
        *,
        owner_user_id: str | None = None,
        resource_context: dict[str, Any] | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        output_dir = self.root / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            owner_user_id=owner_user_id,
            resource_context=dict(resource_context or {}),
            last_progress_at_utc=_utc_now(),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._conditions[job_id] = threading.Condition(self._lock)
            self._event(record, "queued", "Job queued")
            self._persist(record)
        future = self._executor.submit(self._run, record, task, output_dir)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda _future, jid=job_id: self._forget_future(jid))
        return record

    def _forget_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _heartbeat_loop(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None or record.status in _TERMINAL:
                    return
                record.heartbeat_at_utc = _utc_now()
                record.updated_at_utc = record.heartbeat_at_utc
                self._persist(record)
                self._conditions[job_id].notify_all()

    def _run(self, record: JobRecord, task: Task, output_dir: Path) -> None:
        started = _utc_now()
        self._update(
            record.job_id,
            status="running",
            stage="starting",
            progress=0.0,
            message="Job started",
            started_at_utc=started,
            heartbeat_at_utc=started,
            last_progress_at_utc=started,
            stage_started_at_utc=started,
        )
        context = JobContext(self, record.job_id, output_dir)
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(record.job_id, heartbeat_stop),
            name=f"djai-heartbeat-{record.job_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            context.raise_if_cancelled()
            result = task(context)
            context.raise_if_cancelled()
        except JobCancelled as exc:
            self._update(
                record.job_id,
                status="cancelled",
                stage="cancelled",
                message=str(exc),
                error=None,
                finished_at_utc=_utc_now(),
            )
            return
        except BaseException as exc:
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._update(
                record.job_id,
                status="failed",
                stage="failed",
                progress=1.0,
                message=str(exc),
                error=detail[-12000:],
                finished_at_utc=_utc_now(),
            )
            return
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=0.2)
        self._update(
            record.job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            message="Job completed",
            result=result,
            finished_at_utc=_utc_now(),
        )

    def _event(self, record: JobRecord, stage: str, message: str) -> None:
        diagnostics = record.to_dict()
        next_sequence = max((int(event.get("sequence", -1)) for event in record.events), default=-1) + 1
        record.events.append(
            {
                "sequence": next_sequence,
                "time": _utc_now(),
                "status": record.status,
                "stage": stage,
                "progress": record.progress,
                "message": message,
                "stage_current": record.stage_current,
                "stage_total": record.stage_total,
                "cancel_requested": record.cancel_requested,
                "cancellable": diagnostics["cancellable"],
                "elapsed_seconds": diagnostics["elapsed_seconds"],
                "stage_elapsed_seconds": diagnostics["stage_elapsed_seconds"],
                "progress_age_seconds": diagnostics["progress_age_seconds"],
                "stalled": diagnostics["stalled"],
            }
        )

    def _persist(self, record: JobRecord) -> None:
        path = self.root / record.job_id / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict()
        payload["events"] = record.events[-500:]
        payload["owner_user_id"] = record.owner_user_id
        payload["resource_context"] = record.resource_context
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            previous_stage = record.stage
            previous_progress = record.progress
            previous_counts = (record.stage_current, record.stage_total)
            previous_message = record.message
            for key, value in changes.items():
                setattr(record, key, value)
            now = _utc_now()
            record.updated_at_utc = now
            if record.status == "running":
                record.heartbeat_at_utc = now
            if record.stage != previous_stage:
                record.stage_started_at_utc = now
            if (
                record.progress != previous_progress
                or (record.stage_current, record.stage_total) != previous_counts
                or record.stage != previous_stage
                or record.message != previous_message
            ):
                record.last_progress_at_utc = now
            self._event(record, record.stage, record.message)
            self._persist(record)
            self._conditions[job_id].notify_all()

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs[job_id]
            if record.status in _TERMINAL:
                return record
            record.cancel_requested = True
            record.updated_at_utc = _utc_now()
            record.message = "Cancellation requested; SAUND will stop at the next safe boundary."
            future = self._futures.get(job_id)
            if record.status == "queued" and future is not None and future.cancel():
                record.status = "cancelled"
                record.stage = "cancelled"
                record.finished_at_utc = record.updated_at_utc
                record.message = "Job cancelled before execution."
            self._event(record, record.stage, record.message)
            self._persist(record)
            self._conditions[job_id].notify_all()
            return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list_for_owner(
        self,
        owner_user_id: str,
        *,
        kinds: set[str] | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        with self._lock:
            records = [
                record
                for record in self._jobs.values()
                if record.owner_user_id == owner_user_id
                and (kinds is None or record.kind in kinds)
            ]
            records.sort(key=lambda item: (item.created_at_utc, item.job_id), reverse=True)
            return records[: max(1, int(limit))]

    def events_after(self, job_id: str, sequence: int) -> list[dict[str, Any]]:
        with self._lock:
            record = self.get(job_id)
            return [event for event in record.events if int(event["sequence"]) > sequence]

    def wait_for_events(self, job_id: str, sequence: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        with self._lock:
            events = self.events_after(job_id, sequence)
            if events:
                return events
            self._conditions[job_id].wait(timeout=timeout)
            return self.events_after(job_id, sequence)

    def status_summary(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            stalled = 0
            cancelling = 0
            for record in self._jobs.values():
                counts[record.status] = counts.get(record.status, 0) + 1
                payload = record.to_dict()
                stalled += int(bool(payload.get("stalled")))
                cancelling += int(record.cancel_requested and record.status == "running")
            return {
                "workers": self.workers,
                "heartbeat_seconds": self.heartbeat_seconds,
                "counts": counts,
                "active": counts.get("queued", 0) + counts.get("running", 0),
                "stalled": stalled,
                "cancellation_pending": cancelling,
            }
