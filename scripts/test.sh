#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

# Keep numerical/audio runtimes and process-pool tests in fresh containers or
# Python interpreters. The grouping is part of the test harness, not a reduced
# suite: every collected test still runs.
if command -v docker >/dev/null 2>&1 \
  && docker compose version >/dev/null 2>&1 \
  && docker info >/dev/null 2>&1; then
  docker compose run --rm test tests \
    --ignore=tests/test_performance_foundation.py \
    --ignore=tests/test_analysis_local_sources.py \
    --ignore=tests/test_audio_integration.py \
    --ignore=tests/test_stem_truth.py
  docker compose run --rm test tests/test_analysis_local_sources.py
  docker compose run --rm test \
    tests/test_audio_integration.py::test_pair_pipeline_renders_preview_and_full_mix
  docker compose run --rm test \
    tests/test_audio_integration.py::test_set_pipeline_measures_final_junctions_only
  docker compose run --rm test tests/test_stem_truth.py
  docker compose run --rm test tests/test_performance_foundation.py
  exit 0
fi

exec python scripts/run_tests.py
