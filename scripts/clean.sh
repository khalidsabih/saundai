#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
rm -rf .pytest_cache build dist frontend/.next frontend/tsconfig.tsbuildinfo
find . -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find src tests scripts -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
if [ "${1:-}" = "--all" ]; then
  rm -rf .djai-cache output/* ratings/*.jsonl audio/uploads
  mkdir -p output ratings audio config
  : > output/.gitkeep
  : > ratings/.gitkeep
  : > audio/.gitkeep
  : > config/.gitkeep
fi
echo "SAUND.AI build/test artifacts removed."
