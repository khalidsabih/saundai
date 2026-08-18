#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/benchmark.sh <audio-folder> [benchmark options...]" >&2
  exit 2
fi

if command -v docker >/dev/null 2>&1 \
  && docker compose version >/dev/null 2>&1 \
  && docker info >/dev/null 2>&1; then
  exec docker compose run --rm --no-deps --entrypoint python djai scripts/benchmark.py "$@"
fi

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/benchmark.py "$@"
