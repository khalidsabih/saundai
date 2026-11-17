# M1 — Ballroom independent beat and downbeat validation pilot

**Scope:** eight previously extracted Ballroom WAVs plus independent annotation timestamps. Ballroom is *evaluation only*, never training data. No model promotion, downloads or production inference changes. The prior Groove tempo-policy experiment stays shadow-only.

## Local inputs

- Private draft manifest: `/tmp/saund-m1-ballroom-pilot/ballroom-development-DRAFT.json`.
- WAV extraction root: inspect local draft manifest `items[*].audio` and find the actual prefix in `/tmp/saund-m1-ballroom-pilot`; never assume a root until verified.
- The user has verified the source archive MD5 `2872a3e52070bc342a4510a95e2fa0b8`; sample pilot contains eight exact-matched WAVs with independently labeled beat/downbeat times. Confirm actual manifest coverage before running.
- License for Ballroom audio is not the same as permissively licensed Groove. Intended local use, attribution, and noncommercial restrictions must be reviewed; `rights_confirmed=false` draft is not authorized to evaluate. Never set it to true automatically in repository scripts.

## Benchmark procedure

1. Review relevant terms for the intended local **noncommercial research evaluation**. The owner explicitly approves a separate private manifest by copying the draft with `rights_confirmed=true` only after review; do not modify the draft. Run `scripts/m1-rhythm-benchmark.py` against this manifest with the *verified* local WAV root.
2. Run deterministic baseline (without `--allow-neural`), then neural-enabled candidate (with `--allow-neural`, `DJAI_NEURAL_BEATS=1`, model downloads disabled). Use distinct output paths; neither records to ledger by default.
3. Compare independent **beat F1** at 70ms tolerance, **downbeat F1** at 100ms, and tempo MAE where reference BPM is available. Null/absent labels are **not** scored as zero. Review per-track failures; do not only interpret aggregate scores.
4. Verify neural evidence `beat_downbeat` `available`, `used`, `confidence` and `reason` on representative examples. The `--allow-neural` opt-in alone does not prove a neural beat grid was accepted.
5. **Grid/tempo consistency must be assessed separately**: compare `analysis.bpm` and robust median beat-grid tempo `60 / median(diff(beats_seconds))` and report disagreement. A shadow neural BPM candidate from Groove cannot be blindly paired with a deterministic grid. For a rigorous DJ safety claim, inspect bar lengths/cue quantization and real rendered transitions, not only event F1.

Do not treat an eight-track exploratory pilot as statistically sufficient for production; freeze a larger rights-reviewed independent Ballroom holdout without tuning on it. Consult original annotation conventions before interpreting downbeats. No audio or rights-approved private manifest is committed.
