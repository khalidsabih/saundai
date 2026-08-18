#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
docker compose up -d api frontend
printf '%s\n' "SAUND.AI: http://localhost:${DJAI_FRONTEND_PORT:-3000}"
printf '%s\n' "SAUND.AI API docs: http://localhost:${DJAI_API_PORT:-8000}/docs"
