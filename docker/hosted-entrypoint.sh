#!/bin/sh
set -eu

workspace="${DJAI_WORKSPACE:-/workspace}"
mkdir -p "$workspace/audio" "$workspace/output" "$workspace/ratings" "$workspace/accounts" "$workspace/config"

if [ -d /opt/djai/default-config ]; then
  for source in /opt/djai/default-config/*.json; do
    [ -f "$source" ] || continue
    target="$workspace/config/$(basename "$source")"
    if [ ! -e "$target" ]; then
      cp "$source" "$target"
    fi
  done
fi

exec python -m djai "$@"
