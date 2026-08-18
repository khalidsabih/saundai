#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
rm -rf .djai-cache/tracks
mkdir -p .djai-cache/tracks
echo "SAUND.AI analysis cache cleared."
