from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _integer(source: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = source.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    return max(minimum, value)


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    logical_cpus: int
    cpu_budget: int
    requested_job_workers: int
    requested_stem_workers: int
    job_workers: int
    stem_workers: int
    per_job_inner_cpu_budget: int
    constrained: bool

    def to_dict(self) -> dict[str, int | bool | str]:
        return {
            "schema_version": "1.5.3",
            "logical_cpus": self.logical_cpus,
            "cpu_budget": self.cpu_budget,
            "requested_job_workers": self.requested_job_workers,
            "requested_stem_workers": self.requested_stem_workers,
            "job_workers": self.job_workers,
            "stem_workers": self.stem_workers,
            "per_job_inner_cpu_budget": self.per_job_inner_cpu_budget,
            "constrained": self.constrained,
            "policy": "single-host-cooperative-budget",
        }


def resolve_resource_budget(env: Mapping[str, str] | None = None) -> ResourceBudget:
    source = os.environ if env is None else env
    logical = max(1, os.cpu_count() or 1)
    configured_cpu = _integer(source, "DJAI_CPU_BUDGET", 0)
    cpu_budget = logical if configured_cpu <= 0 else max(1, min(logical, configured_cpu))

    requested_jobs = max(1, _integer(source, "DJAI_JOB_WORKERS", 2, 1))
    requested_stems = max(1, _integer(source, "DJAI_STEM_JOB_WORKERS", 1, 1))

    # Leave at least one logical CPU available to the process and its inner DSP
    # work whenever possible. The existing analysis/graph pools divide this same
    # budget again per active top-level job.
    max_job_workers = max(1, cpu_budget - 1) if cpu_budget > 1 else 1
    job_workers = min(requested_jobs, max_job_workers)
    remaining = max(1, cpu_budget - job_workers)
    stem_workers = min(requested_stems, remaining)
    per_job = max(1, cpu_budget // max(1, job_workers))

    return ResourceBudget(
        logical_cpus=logical,
        cpu_budget=cpu_budget,
        requested_job_workers=requested_jobs,
        requested_stem_workers=requested_stems,
        job_workers=job_workers,
        stem_workers=stem_workers,
        per_job_inner_cpu_budget=per_job,
        constrained=job_workers != requested_jobs or stem_workers != requested_stems or cpu_budget != logical,
    )
