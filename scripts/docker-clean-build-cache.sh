#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  printf '%s\n' 'Docker engine is not available.' >&2
  exit 1
fi
printf '%s\n' 'Before:'
docker system df
printf '\n%s\n' 'Removing Docker build cache and unused images only.'
printf '%s\n' 'Persistent Docker volumes are NOT pruned.'
docker builder prune -af
docker image prune -af
printf '\n%s\n' 'After:'
docker system df
