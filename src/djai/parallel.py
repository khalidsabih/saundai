from __future__ import annotations

import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from typing import Literal

WorkerKind = Literal["analysis", "graph"]

_ANALYSIS_AUTO_LIMIT = 4
_GRAPH_AUTO_LIMIT = 8
_AUTO_THRESHOLDS: dict[WorkerKind, int] = {
    "analysis": 3,
    "graph": 24,
}
_ENV_NAMES: dict[WorkerKind, str] = {
    "analysis": "DJAI_ANALYSIS_WORKERS",
    "graph": "DJAI_GRAPH_WORKERS",
}


@dataclass(frozen=True, slots=True)
class WorkerSelection:
    kind: WorkerKind
    workers: int
    task_count: int
    explicit: bool
    parallel: bool


def prepare_process_environment() -> None:
    """Prevent numerical libraries from oversubscribing process-pool workers."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def worker_initializer() -> None:
    prepare_process_environment()


def process_context() -> mp.context.BaseContext:
    """Return a safe multiprocessing context for the active platform.

    Native macOS/Windows use ``spawn``. Linux prefers ``forkserver`` so graph
    workers do not inherit numerical-library thread/semaphore state created by
    earlier librosa analysis. An explicit environment value still wins.
    """
    available = mp.get_all_start_methods()
    requested = os.environ.get("DJAI_PROCESS_START_METHOD", "auto").strip().lower()
    if requested in {"", "auto"}:
        if sys.platform.startswith(("darwin", "win")):
            method = "spawn"
        elif "forkserver" in available:
            method = "forkserver"
        else:
            method = "spawn"
    else:
        method = requested
    if method not in available:
        method = "spawn" if "spawn" in available else available[0]
    return mp.get_context(method)


def _parse_worker_value(raw: str | None, *, source: str) -> int | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source} must be an integer >= 0") from exc
    if value < 0:
        raise ValueError(f"{source} must be >= 0")
    return value


def resolve_worker_selection(
    kind: WorkerKind,
    task_count: int,
    *,
    requested: int | None = None,
) -> WorkerSelection:
    """Resolve workers from an argument, environment, or conservative auto mode.

    ``0`` and ``None`` mean automatic, ``1`` forces serial execution, and
    ``2+`` requests that many workers (capped by the number of tasks).
    """
    if kind not in _ENV_NAMES:
        raise ValueError(f"Unknown worker kind: {kind}")
    task_count = max(0, int(task_count))
    if task_count <= 1:
        return WorkerSelection(kind, max(1, task_count), task_count, False, False)

    if requested is not None:
        if requested < 0:
            raise ValueError("workers must be >= 0")
        raw_value = int(requested)
        explicit = raw_value > 0
    else:
        parsed = _parse_worker_value(os.environ.get(_ENV_NAMES[kind]), source=_ENV_NAMES[kind])
        raw_value = 0 if parsed is None else parsed
        explicit = raw_value > 0

    cpu_count = max(1, os.cpu_count() or 1)
    # Top-level API jobs can run concurrently. Automatic inner process pools
    # share a conservative CPU budget instead of each assuming the whole host.
    try:
        concurrent_jobs = max(1, int(os.environ.get("SAUND_EFFECTIVE_JOB_WORKERS", os.environ.get("DJAI_JOB_WORKERS", "1"))))
    except ValueError:
        concurrent_jobs = 1
    try:
        configured_budget = int(os.environ.get("DJAI_CPU_BUDGET", "0"))
    except ValueError:
        configured_budget = 0
    host_budget = min(cpu_count, configured_budget) if configured_budget > 0 else cpu_count
    per_job_budget = max(1, host_budget // concurrent_jobs)
    if raw_value == 0:
        if kind == "analysis":
            workers = min(_ANALYSIS_AUTO_LIMIT, max(1, per_job_budget // 2 or 1))
        else:
            workers = min(_GRAPH_AUTO_LIMIT, max(1, per_job_budget - 1))
    else:
        workers = raw_value

    workers = max(1, min(int(workers), task_count))
    if not explicit and task_count < _AUTO_THRESHOLDS[kind]:
        workers = 1

    return WorkerSelection(
        kind=kind,
        workers=workers,
        task_count=task_count,
        explicit=explicit,
        parallel=workers > 1,
    )
