#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

printf '%s\n' 'SAUND.AI runtime identity'
printf '%s\n' '========================'
printf 'source VERSION: %s\n' "$(cat VERSION)"
printf '\nAPI runtime:\n'
if command -v curl >/dev/null 2>&1; then
  curl -fsS "${SAUND_API_STATUS_URL:-http://localhost:8000/api/runtime/status}" 2>/dev/null || printf '%s\n' 'API unavailable'
  printf '\n\nFrontend build:\n'
  curl -fsS "${SAUND_FRONTEND_BUILD_URL:-http://localhost:3000/build-info.json}" 2>/dev/null || printf '%s\n' 'Frontend build info unavailable'
else
  printf '%s\n' 'curl not available'
fi
printf '\n\nDocker images:\n'
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose images 2>/dev/null || true
  printf '\nDocker disk use:\n'
  docker system df 2>/dev/null || true
else
  printf '%s\n' 'Docker engine unavailable'
fi
