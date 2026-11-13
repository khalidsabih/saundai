from __future__ import annotations

"""Process-pool execution helpers for planner graph work.

This module owns only multiprocessing lifecycle. Musical planning, cache
persistence, progress reporting and result ordering remain with set_planner.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

from .parallel import prepare_process_environment, process_context

TaskT = TypeVar("TaskT")
ResultT = TypeVar("ResultT")


def iter_graph_worker_results(
    tasks: Sequence[TaskT],
    *,
    workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    worker: Callable[[TaskT], ResultT],
) -> Iterator[ResultT]:
    """Yield graph-worker results in completion order.

    Exceptions retain the existing fail-fast behavior: pending futures are
    cancelled before the original exception is re-raised.
    """
    if not tasks:
        return

    prepare_process_environment()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=process_context(),
        initializer=initializer,
        initargs=initargs,
    ) as executor:
        future_map = {
            executor.submit(worker, task): task
            for task in tasks
        }
        try:
            for future in as_completed(future_map):
                yield future.result()
        except BaseException:
            for future in future_map:
                future.cancel()
            raise
