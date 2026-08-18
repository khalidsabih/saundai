#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."

choose_python() {
  if [ -n "${DJAI_CUE_PYTHON:-}" ]; then
    printf '%s\n' "$DJAI_CUE_PYTHON"
    return
  fi
  for candidate in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "CUE-DETR reference mode requires Python 3.11 or 3.12." >&2
  echo "On Apple Silicon macOS: brew install python@3.12" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)'; then
  echo "CUE-DETR reference mode requires Python 3.11 or 3.12." >&2
  exit 1
fi

rm -rf .cuevenv
"$PYTHON_BIN" -m venv .cuevenv
. .cuevenv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[cue]'
python -m djai --version
printf '%s\n' \
  "CUE-DETR research/reference environment is ready." \
  "Activate with: source .cuevenv/bin/activate" \
  "First model fetch: DJAI_CUE_MODEL_ALLOW_DOWNLOAD=1 ./.cuevenv/bin/saund cue-predict audio/your-track.mp3"
