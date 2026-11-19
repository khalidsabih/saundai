# M1 independent beat-reference research (PR #30)

This is an **offline, research-only** check on private, rights-cleared development audio.
It does not change selected grids, model authority, thresholds, the signed plan, or production.
No audio or labels are committed.

## Independent evidence sources

1. **Essentia RhythmExtractor2013 (multifeature)** operates on each original
   source track at the algorithm's required 44.1 kHz rate. It returns candidate
   beat times independent of DJAI's selected beat grid.
2. **Beat This!** uses the already pinned local checkpoint through the existing
   DJAI advisory evidence interface, with model downloads disabled.
3. Events are compared one-to-one within **45 ms** on the actual signed
   overlap source windows. No arbitrary beat-index shifts, half-time or
   double-time realignment is allowed. Gaps, missing coverage or disagreements
   produce **inconclusive**.
4. Agreed candidate beats are projected through the signed phase-warp function
   into rendered time. Relative beat-phase drift is evaluated as an **event
   reference proxy**. This projection is *not* an independent measurement
   of the actual rendered signal's beat displacement.
5. A fresh signed-plan render must reproduce the private canary WAV before
   the reference comparison runs. Outputs include a manifest SHA256, reference
   coverage and uncertainties, and explicit
   `independent_rendered_phase_certified=false`.

The 25 ms production phase threshold and 45 ms correction envelope remain
unchanged. Even an apparent proxy pass is **not** certification. For full
acoustic certification, independently adjudicated beat identities and
source-aligned measurements of the *actual isolated rendered deck signals*
are still required. This script does not set a release gate to PASS.

Essentia's Python module is optional and must be installed in the **same
Python environment** used to run the script. On macOS see the official
Essentia installation guide: https://essentia.upf.edu/installing.html .
Confirm it is importable before running:

```bash
python -c "import essentia.standard as es; print('Essentia ready')"
```

## Run on the approved private audio

From the repository root, with the approved local checkpoint installed:

```bash
export SAUND_AUDIO_ROOT="$HOME/Downloads/lab/SAUND_AI/data/m1"
export SAUND_RENDER_MANIFEST="$SAUND_AUDIO_ROOT/manifest.json"
export DJAI_BEAT_THIS_CHECKPOINT="$HOME/Downloads/lab/SAUND_AI/.djai-cache/neural/beat-this-final0.ckpt"
export DJAI_BEAT_THIS_SHA256="8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
export DJAI_TEMPO_CONSISTENCY=1
export DJAI_MODEL_ALLOW_DOWNLOAD=0
export DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0

PYTHONPATH="$PWD/src" python scripts/m1-independent-beat-phase-verify.py \
  --manifest "$SAUND_RENDER_MANIFEST" \
  --audio-root "$SAUND_AUDIO_ROOT" \
  --canary-report /tmp/saund-m1-slew-fixed-17-001/canary-report.json \
  --rendered-wav /tmp/saund-m1-slew-fixed-17-001/approved-pair-01.wav \
  --phase-samples 17 \
  --output /tmp/saund-m1-independent-phase-001.json \
  --confirm-render-rights
```

Outputs are private; don't attach the underlying WAV/MP3 to PR comments.

## Validation

```bash
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_m1_independent_beat_phase_verify.py \
  tests/test_m1_rendered_phase_acoustic_check.py \
  tests/test_m1_phase_slew_rate.py \
  tests/test_m1_automatic_candidate_recovery.py
```

A low-agreement, ambiguous or absent reference result is expected to be
`inconclusive`, not a failure of the production planner. Any 25 ms
measurement of aligned *predicted* beats must remain labeled proxy only.
