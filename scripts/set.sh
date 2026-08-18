#!/usr/bin/env sh
set -eu
folder=${1:-audio}
journey=${2:-balanced}
cd "$(dirname "$0")/.."
mkdir -p output/set
./scripts/saund.sh set "$folder" \
  --journey "$journey" \
  --transition-style auto \
  --tempo-policy safe \
  --min-play-seconds 30 \
  --min-play-fraction 0.35 \
  --output output/set/full-set.wav \
  --report output/set/full-set.json \
  --plan-output output/set/set-plan.json \
  --confirm-risky
