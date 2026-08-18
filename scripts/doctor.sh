#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
echo "SAUND.AI doctor"
echo "Project: $(pwd)"
echo "Source version: $(cat VERSION)"
python scripts/sync-version.py --check 2>/dev/null || echo "Version metadata: MISMATCH"
if find . -type f \( -name "*.orig" -o -name "*.rej" \) -not -path "./.git/*" | grep -q .; then
  echo "Patch debris: present (*.orig/*.rej)"
else
  echo "Patch debris: clean"
fi
if command -v docker >/dev/null 2>&1; then
  echo "Docker: $(docker --version)"
  docker compose version 2>/dev/null || true
  if docker info >/dev/null 2>&1; then
    echo "Docker engine: running"
    docker system df 2>/dev/null | sed -n "1,8p" || true
  else
    echo "Docker engine: not running"
  fi
else
  echo "Docker: not found"
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import importlib
import sys
print("Python:", sys.version.split()[0])
for name in ("numpy", "scipy", "librosa", "soundfile", "fastapi", "pydantic", "httpx"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'available')}")
    except Exception as exc:
        print(f"{name}: missing ({exc})")
try:
    import djai
    print("SAUND.AI:", djai.__version__)
    print("Engine module:", djai.__file__)
except Exception as exc:
    print("Engine import failed:", exc)
PY
if [ -f benchmarks/benchmark-results.sqlite3 ]; then
  echo "Release benchmarks:"
  PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/release-benchmark.py status 2>/dev/null || echo "  benchmark store unreadable"
else
  echo "Release benchmarks: no local history yet"
fi
echo "Release candidate readiness:"
PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/release-candidate.py status 2>/dev/null \
  | python -c 'import json,sys; p=json.load(sys.stdin); print("  operational_ready:", p.get("operational_ready")); print("  planner_mode:", p.get("authority",{}).get("planner_mode")); print("  warnings:", p.get("warning_count")); print("  blockers:", p.get("blocker_count"))' \
  || echo "  readiness check failed"
command -v ffmpeg >/dev/null 2>&1 && ffmpeg -version 2>/dev/null | head -n 1 || echo "FFmpeg: not found"
command -v node >/dev/null 2>&1 && echo "Node: $(node --version)" || echo "Node: not found"
command -v npm >/dev/null 2>&1 && echo "npm: $(npm --version)" || echo "npm: not found"
