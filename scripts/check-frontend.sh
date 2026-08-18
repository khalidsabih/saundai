#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required for the frontend syntax check." >&2
  exit 1
fi
NODE_PATH="$(npm root -g)${NODE_PATH:+:$NODE_PATH}" node scripts/check-frontend.js
