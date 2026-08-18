#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/../frontend"
npm install --package-lock-only --ignore-scripts --no-audit --no-fund
echo "frontend/package-lock.json updated. Commit it for fully repeatable npm installs."
