#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
version=$(cat VERSION)
case "$version" in
  *rc*|*RC*) rc_build=1 ;;
  *) rc_build=0 ;;
esac

if [ "${SAUND_SKIP_RELEASE_GATE:-0}" != "1" ]; then
  ./scripts/release-gate.sh
else
  printf '%s\n' 'WARNING: release gate skipped; package is not certified for final public release.' >&2
fi
out=${1:-"dist/SAUND_AI_${version}_FULL_PROJECT.zip"}
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT INT TERM
mkdir -p "$(dirname "$out")" "$stage/SAUND_AI"
out_dir=$(cd "$(dirname "$out")" && pwd)
out_abs="$out_dir/$(basename "$out")"
# zip updates existing archives in place; remove an older release first so
# deleted or renamed files cannot survive as stale entries.
rm -f "$out_abs"
# Copy project while excluding runtime, secrets, caches, and build output.
tar \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='.venv' \
  --exclude='.cuevenv' \
  --exclude='.djai-cache' \
  --exclude='.pytest_cache' \
  --exclude='dist' \
  --exclude='build' \
  --exclude='*.egg-info' \
  --exclude='.coverage' \
  --exclude='.mypy_cache' \
  --exclude='frontend/node_modules' \
  --exclude='frontend/.next' \
  --exclude='frontend/tsconfig.tsbuildinfo' \
  --exclude='audio/*' \
  --exclude='output/*' \
  --exclude='ratings/*' \
  --exclude='benchmarks/*.sqlite3' \
  --exclude='benchmarks/*.sqlite3-*' \
  --exclude='accounts/*' \
  --exclude='datasets/downloads' \
  --exclude='datasets/external' \
  --exclude='datasets/bootstrap-corpora' \
  --exclude='config/bootstrap-models' \
  --exclude='config/*.json' \
  --exclude='config/neural/*' \
  --exclude='*.pyc' \
  --exclude='*.orig' \
  --exclude='*.rej' \
  --exclude='__pycache__' \
  -cf - . | tar -xf - -C "$stage/SAUND_AI"
mkdir -p "$stage/SAUND_AI/audio" "$stage/SAUND_AI/output" "$stage/SAUND_AI/ratings" "$stage/SAUND_AI/accounts" "$stage/SAUND_AI/config/neural"
: > "$stage/SAUND_AI/audio/.gitkeep"
: > "$stage/SAUND_AI/output/.gitkeep"
: > "$stage/SAUND_AI/ratings/.gitkeep"
: > "$stage/SAUND_AI/accounts/.gitkeep"
: > "$stage/SAUND_AI/config/.gitkeep"
: > "$stage/SAUND_AI/config/neural/.gitkeep"
# The default governance manifest is source configuration, not learned user state.
# Learned/scoring checkpoints remain excluded from release archives.
cp config/model-registry.json "$stage/SAUND_AI/config/model-registry.json"
if [ "$rc_build" = "1" ] && [ "${SAUND_SKIP_RELEASE_GATE:-0}" != "1" ]; then
  certification="dist/release-certification.json"
  [ -f "$certification" ] || {
    printf '%s\n' 'ERROR: RC package requires dist/release-certification.json from the release gate.' >&2
    exit 1
  }
  cp "$certification" "$stage/SAUND_AI/RELEASE_CERTIFICATION.json"
fi
(
  cd "$stage"
  zip -qr "$out_abs" SAUND_AI
)
echo "$out"
