#!/usr/bin/env bash
set -euo pipefail

export DJAI_API_DOCKER_TARGET=stems
export DJAI_ENGINE_DOCKER_TARGET=stems
export DJAI_AUTO_STEMS=${DJAI_AUTO_STEMS:-background}

docker compose build api djai
docker compose up -d api frontend

echo "SAUND.AI started with automatic Demucs stem separation."
echo "Frontend: http://localhost:${DJAI_FRONTEND_PORT:-3000}"
echo "API docs: http://localhost:${DJAI_API_PORT:-8000}/docs"
