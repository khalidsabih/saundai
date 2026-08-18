#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
./scripts/test.sh
rm -rf output/demo-pipeline .djai-cache/tracks

if command -v docker >/dev/null 2>&1 \
  && docker compose version >/dev/null 2>&1 \
  && docker info >/dev/null 2>&1; then
  docker compose run --rm --no-deps --entrypoint python djai scripts/generate_verification_pipeline.py
  exec docker compose run --rm --no-deps --entrypoint python djai scripts/verify_release.py
fi

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/generate_verification_pipeline.py
exec python scripts/verify_release.py
