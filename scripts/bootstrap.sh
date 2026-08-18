#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Desktop or run scripts/setup-native.sh." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is unavailable." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is installed but its engine is not running." >&2
  exit 1
fi

echo "Building SAUND.AI backend and test images..."
docker compose build api test djai

if [ "${DJAI_SKIP_FRONTEND_BUILD:-0}" != "1" ]; then
  echo "Building SAUND.AI frontend..."
  docker compose build frontend
else
  echo "Skipping frontend build because DJAI_SKIP_FRONTEND_BUILD=1."
fi

./scripts/test.sh
docker compose run --rm --no-deps djai --version

echo "SAUND.AI $(cat VERSION) is ready. Start it with ./scripts/start.sh"
