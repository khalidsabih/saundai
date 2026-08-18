#!/usr/bin/env sh
set -eu
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 TRACK_A TRACK_B [STYLE] [TEMPO_POLICY]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
a=$1
b=$2
style=${3:-safe}
policy=${4:-safe}
mkdir -p output/pair
./scripts/saund.sh transition "$a" "$b" --style "$style" --tempo-policy "$policy" --force --preview --output output/pair/preview.wav --report output/pair/preview.json
./scripts/saund.sh transition "$a" "$b" --style "$style" --tempo-policy "$policy" --force --output output/pair/full-mix.wav --report output/pair/full-mix.json
printf '%s\n' "Preview: output/pair/preview.wav" "Full mix: output/pair/full-mix.wav"
