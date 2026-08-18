#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

fail() {
  printf 'RELEASE GATE FAILED: %s\n' "$1" >&2
  exit 1
}

version=$(cat VERSION)
case "$version" in
  *rc*|*RC*) rc_build=1 ;;
  *) rc_build=0 ;;
esac
printf 'SAUND.AI %s release gate\n' "$version"
printf '%s\n' '============================'

python scripts/sync-version.py --check >/dev/null || fail 'VERSION metadata is not synchronized; run scripts/sync-version.py'
[ "${SAUND_HERO_INTRO_RIGHTS_CONFIRMED:-0}" = "1" ] || fail 'hero intro distribution rights must be confirmed: rerun with SAUND_HERO_INTRO_RIGHTS_CONFIRMED=1 only if you hold the required rights'
[ -f frontend/package-lock.json ] || fail 'frontend/package-lock.json is required; generate it with the pinned package.json and commit it before release'
if find . -type f \( -name '*.orig' -o -name '*.rej' \) -not -path './.git/*' | grep -q .; then
  fail 'patch debris (*.orig/*.rej) must be removed before release'
fi

printf '%s\n' '[1/15] Authoritative version contract'
python scripts/sync-version.py --check >/dev/null

printf '%s\n' '[2/15] Python compile check'
python -m compileall -q src

printf '%s\n' '[3/15] Frontend source syntax check'
./scripts/check-frontend.sh

printf '%s\n' '[4/15] Production training-data provenance audit'
./scripts/saund.sh dataset-audit --lane production >/dev/null

printf '%s\n' '[5/15] Production privacy + model governance audit'
./scripts/production-audit.py >/dev/null

printf '%s\n' '[6/15] Frontend clean dependency install'
(
  cd frontend
  npm ci --no-audit --no-fund --ignore-scripts
)

printf '%s\n' '[7/15] Frontend dependency audit'
(
  cd frontend
  npm audit --omit=dev --audit-level=high
)

printf '%s\n' '[8/15] Frontend typecheck + production build'
(
  cd frontend
  npm run typecheck
  npm run build
)

printf '%s\n' '[9/15] Frontend build identity'
python - <<'PY'
import json
from pathlib import Path
version = Path('VERSION').read_text(encoding='utf-8').strip()
payload = json.loads(Path('frontend/public/build-info.json').read_text(encoding='utf-8'))
if payload.get('version') != version:
    raise SystemExit(f"frontend build identity {payload.get('version')} != {version}")
PY

printf '%s\n' '[10/15] Playwright browser install'
(
  cd frontend
  npx playwright install chromium firefox webkit
)

printf '%s\n' '[11/15] Browser release smoke tests'
(
  cd frontend
  npm run e2e
)

printf '%s\n' '[12/15] Backend + product contract tests'
python -m pytest -q

printf '%s\n' '[13/15] Release benchmark champion gate'
if [ "$rc_build" = "1" ] || [ "${SAUND_REQUIRE_RELEASE_BENCHMARK:-0}" = "1" ]; then
  suite=${SAUND_RELEASE_BENCHMARK_SUITE:-standard}
  PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/release-benchmark.py compare \
    --suite "$suite" \
    --challenger latest \
    --require-version "$version" >/dev/null \
    || fail "current-version benchmark did not pass the $suite champion gate"
else
  printf '%s\n' 'Benchmark champion gate is advisory until SAUND_REQUIRE_RELEASE_BENCHMARK=1.'
fi

printf '%s\n' '[14/15] Compose configuration'
docker compose config >/dev/null

printf '%s\n' '[15/15] Release-candidate certification manifest'
if [ "$rc_build" = "1" ]; then
  suite=${SAUND_RELEASE_BENCHMARK_SUITE:-standard}
  PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/release-candidate.py certify \
    --suite "$suite" \
    --output dist/release-certification.json >/dev/null \
    || fail '2.0 RC certification manifest could not be issued'
else
  printf '%s\n' 'Stable build: RC certification manifest not required.'
fi

printf 'RELEASE GATE PASSED: SAUND.AI %s\n' "$version"
