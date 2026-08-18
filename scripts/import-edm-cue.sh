#!/usr/bin/env sh
set -eu
if [ "$#" -lt 1 ]; then
  echo "usage: $0 /path/to/edm-cue-annotations.{json,jsonl,csv} [output]" >&2
  exit 2
fi
OUTPUT="${2:-datasets/edm-cue/annotations.jsonl}"
exec "$(dirname "$0")/saund.sh" edm-cue-import "$1" -o "$OUTPUT"
