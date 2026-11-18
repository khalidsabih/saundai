#!/usr/bin/env bash
# M1 offline render gate. Runs focused checks and the existing rights-gated
# CURRENT-grid canary. Does not run the frozen 0.50 challenger or use test audio.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${SAUND_PYTHON:-python3}"
audio_root="${SAUND_AUDIO_ROOT:-}"
manifest="${SAUND_RENDER_MANIFEST:-}"
output_dir="${SAUND_OUTPUT_DIR:-}"

if [[ -z "$audio_root" || -z "$manifest" || -z "$output_dir" ]]; then
  printf '%s\n' "Set SAUND_AUDIO_ROOT, SAUND_RENDER_MANIFEST, and SAUND_OUTPUT_DIR." >&2
  exit 2
fi
if [[ "${SAUND_RENDER_RIGHTS_CONFIRMED:-}" != "yes" ]]; then
  printf '%s\n' "Refusing to render: set SAUND_RENDER_RIGHTS_CONFIRMED=yes only after reviewing local derived-audio rights." >&2
  exit 2
fi
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
  printf 'Refusing to overwrite existing output: %s\n' "$output_dir" >&2
  exit 2
fi
if [[ ! -f "$manifest" || ! -d "$audio_root" ]]; then
  printf '%s\n' "Manifest must exist and audio root must be a directory." >&2
  exit 2
fi

# No implicit model downloads or research-model promotion.
export DJAI_MODEL_ALLOW_DOWNLOAD=0
export DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0
export DJAI_NEURAL_BEATS=1
export DJAI_CUE_MODEL_ENABLED=0
export DJAI_EMBEDDINGS_ENABLED=0

"$python_bin" -m pytest -q \
  tests/test_m1_ballroom_transition_timing_shadow.py \
  tests/test_m1_real_transition_render_canary.py

"$python_bin" scripts/m1-real-transition-render-canary.py \
  --manifest "$manifest" \
  --audio-root "$audio_root" \
  --output-dir "$output_dir" \
  --confirm-render-rights

printf '\nPrivate canary results: %s/canary-report.json\n' "$output_dir"
printf '%s\n' "This tests current selected grids only. Challenger comparison and human listening are NOT certified."
