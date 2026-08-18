#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if command -v docker >/dev/null 2>&1 \
  && docker compose version >/dev/null 2>&1 \
  && docker info >/dev/null 2>&1; then
  exec docker compose run --rm --no-deps djai "$@"
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m djai "$@"
