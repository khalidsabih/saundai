#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

choose_python() {
  if [ -n "${DJAI_PYTHON:-}" ]; then
    printf '%s\n' "$DJAI_PYTHON"
    return
  fi
  for candidate in /opt/homebrew/bin/python3.12 python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      "$candidate" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)' >/dev/null 2>&1 && {
        printf '%s\n' "$candidate"
        return
      }
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "SAUND.AI requires Python 3.11, 3.12, or 3.13." >&2
  echo "On Apple Silicon macOS, install Python 3.12 with: brew install python@3.12" >&2
  exit 1
fi

VERSION="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "Creating SAUND.AI native environment with $PYTHON_BIN (Python $VERSION)..."

if [ -d .venv ]; then
  VENV_VERSION="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  SOURCE_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "$VENV_VERSION" != "$SOURCE_VERSION" ]; then
    echo "Removing stale .venv created with Python ${VENV_VERSION:-unknown}."
    rm -rf .venv
  fi
fi

"$PYTHON_BIN" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -c constraints.txt -e '.[dev]'
python scripts/run_tests.py
python -m djai --version
printf '%s\n' "Native SAUND.AI environment is ready." "Activate later with: source .venv/bin/activate"
