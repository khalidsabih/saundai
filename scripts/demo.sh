#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
rm -rf output/demo-pipeline
./scripts/saund.sh demo-set --output-dir output/demo-pipeline --bars 32 --journey build --render
