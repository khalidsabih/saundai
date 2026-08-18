#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keep numerical/audio runtimes and multiprocessing tests in fresh interpreters.
# librosa/Numba and multiprocessing resource trackers can keep process state alive
# after assertions have completed when every concern shares one pytest process.
GROUPS = (
    (
        "core, API and product tests",
        [
            "tests",
            "--ignore=tests/test_performance_foundation.py",
            "--ignore=tests/test_analysis_local_sources.py",
            "--ignore=tests/test_audio_integration.py",
            "--ignore=tests/test_stem_truth.py",
        ],
    ),
    (
        "local source-analysis tests",
        ["tests/test_analysis_local_sources.py"],
    ),
    (
        "pair audio integration test",
        ["tests/test_audio_integration.py::test_pair_pipeline_renders_preview_and_full_mix"],
    ),
    (
        "set audio integration test",
        ["tests/test_audio_integration.py::test_set_pipeline_measures_final_junctions_only"],
    ),
    (
        "stem truth and stem-render tests",
        ["tests/test_stem_truth.py"],
    ),
    (
        "parallel performance and determinism tests",
        ["tests/test_performance_foundation.py"],
    ),
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    # Native regression runs use a reduced rehearsal fan-out. The Docker/API
    # production defaults remain 48 analytical ideas -> 8 private rehearsals.
    environment.setdefault("DJAI_BRAIN_ANALYTICAL_POOL", "24")
    environment.setdefault("DJAI_BRAIN_FINALISTS", "4")
    environment.setdefault("DJAI_BRAIN_RETRY_ROUNDS", "1")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        environment.setdefault(name, "1")
    return environment


def main() -> int:
    environment = _environment()
    total_groups = len(GROUPS)
    for index, (label, arguments) in enumerate(GROUPS, start=1):
        print(f"\n[{index}/{total_groups}] {label}", flush=True)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *arguments],
            cwd=ROOT,
            env=environment,
            check=False,
            close_fds=True,
        )
        if completed.returncode != 0:
            return completed.returncode
    print("\nSAUND.AI test groups passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
