# M0 local run evidence — 2026-11-16

**Scope:** User-supplied terminal transcript from the isolated `/tmp/saund-m0-baseline` worktree. The transcript is a local report, not an independently reproduced run by GitHub CI. No audio or private checkpoint is uploaded.

## Model registry observation

Command: `PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" -m djai models-status`

- Registry schema: `5.0`. Runtime registry was resolved from the user's main workspace, **outside** the M0 worktree; local configuration is therefore an uncontrolled dependency until explicitly frozen.
- 7 model entries present. `beat_this` advisory weight `0.34`, min confidence `0.66`; `music_encoder` advisory weight `0.35`, min confidence `0.50`.
- Five DJ learned heads (`transition_family_policy`, `duration_policy`, `stem_choreography_policy`, `rendered_dj_judge`, `next_track_policy`) are shadow, zero effective weight.
- Every entry reports `checkpoint_exists: false` and `checkpoint_verified: null`; this applies to the **registry's configured checkpoint path** and must **not** be interpreted as a comprehensive scan of Python/Torch caches or proof that neural inference cannot run.
- No model reported `production_approved: true`.

## End-to-end synthetic demo observation

Command: `PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" -m djai demo-set --render`

- All five named synthetic tracks reached the analyzed stage (20%–100%).
- Planner used a sparse-graph-to-broad-graph fallback and logged route repair after 1, 2 and 3 rejected transitions. Prefix reuse logged for two already-rehearsed transitions.
- Four preview audio files were reported along with the preview index.
- Render advanced to `[render] 100% set complete`; the tool printed locations for `set-plan.json`, `previews/index.json`, `full-set.wav`, `full-set.json`.
- User subsequently confirmed `full-set.wav` exists and is **49 MB** as displayed by `ls -lh`.
- SHA-256, user-reported via `shasum -a 256`: `5468e2440b1219f4c4767755dadef18afe55a57e633b9496c4a3cead3d1b8370`.
- Warnings: Librosa attempted tuning from empty frequency sets during synthetic audio analysis. This is a warning, not shown as a fatal error.
- **Unexpected behavior:** four concurrent Beat This! `small0.ckpt` downloads were initiated through the Torch Hub cache; each reported 8.06 MB. Thus demo startup was **not network-free** and altered cache state. This contradicts any blanket claim that regular analysis never silently downloads model weights. Investigate under M0/M1 without changing runtime behavior.
- Not yet examined: detailed contents of `full-set.json`, exact set duration, actual audio channels/rate, model artifact/hash state *after* download, reported failure semantics, whether real app API/frontend are running.

## Interpretation

**PASS for synthetic end-to-end command reaching an audio artifact.** This does **not** demonstrate deterministic run-to-run hashes, absence of model cache/network side effects, or real-music/genre quality. Output checksum is a **single-run identity**, not proof of reproducibility.

M0 gate remains **IN PROGRESS / NOT ACCEPTED** pending:
1. private runtime/registry/cache capture and acknowledging unexpected downloads;
2. artifact inspection of `full-set.json` and WAV metadata;
3. rights-cleared private real-music scenario manifest, frozen with split/track hashes;
4. optional app readiness if already running and owner acceptance.

No modification to live DJ engine, neural authority, cues or playback was made in this documentation update. Research PR #27 remains separate and draft.
