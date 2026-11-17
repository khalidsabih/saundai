# M1 — Groove MIDI first real-audio deterministic tempo baseline (2026-11-17)

**Evidence:** owner-reported output from local SAUND worktree on the official Magenta Groove v1.0.0 dataset. The assistant has not inspected the private result JSON or run the user's local audio. No model weights or production inference changed.

## Source and data gates

- Official full original archive SHA-256 verified on owner's Mac: `21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9`.
- Publisher split metadata: 897 train, 124 validation, 129 test; total 1,150.
- Of 897 development records: 846 exact-matched WAVs, 51 rows with empty `audio_filename`, **zero expected WAV paths missing**.
- Official-split auditor reported 17 drummer/session groups spanning more than one split; do not claim group-disjoint results from official splits.
- Pilot filters WAV durations at least 8s; 157 shorter/invalid files encountered before first 12 eligible items selected. **157 is not the full-cohort exclusion count.**
- The 12 selected pilot IDs all belong to `drummer1/session1`, an unrepresentative smoke cohort.
- The owner created local rights-reviewed approved manifest `/tmp/saund-m1-groove/development-pilot-12-long-APPROVED.json`; do not commit private audio/manifests.

## Actual benchmark result

```json
{
  "result": "/tmp/saund-m1-groove/pilot-12-long-deterministic-results.json",
  "track_count": 12,
  "metric_coverage": {"tempo": 12, "beat": 0, "downbeat": 0},
  "metrics": {"tempo_mae_bpm": 35.290125, "beat_f1": null, "downbeat_f1": null},
  "ledger_recorded": false
}
```

**Interpretation:** pipeline analyzed 12/12 eligible WAVs and produced a high absolute tempo MAE. We have not yet examined per-track errors, octave/half-time ambiguity, predicted/reference BPM pairings, or neural comparison. This is not sufficient to pass M1, claim generalization, or change model authority.

## Next evidence needed

1. Inspect **private result JSON `per_track`** (ID and absolute error; predictions are not necessarily included) to rank failures. If desired, derive predicted BPM by rerunning inference or enhance evaluator to record paired prediction/reference BPM in a privacy-preserving report.
2. Freeze a more diverse deterministic development cohort stratified by style/drummer/tempo and WAV eligibility; do not tune on validation/test.
3. Compare absolute tempo MAE with an octave-tolerant diagnostic metric **without replacing the primary metric**. Separate wrong tempo from half/double tempo.
4. Fix the largest reproducible issue, rerun on exactly the same cohort, report improvement and regressions.
5. Do not promote Beat This! or any learned head based on this one 12-song pilot.

**Status:** M1 real-audio pilot measured; more representative baseline and error characterization pending.

## Owner's diagnostic BPM pairs (2026-11-17)

Reran deterministic `analyze_track` for eight worst errors, as reported by owner:

| Recording ID suffix | Reference BPM | Predicted BPM | Ratio |
|---|---:|---:|---:|
| 3 | 80 | 161.50 | 2.019 |
| 184 | 78 | 152.00 | 1.949 |
| 201 | 64 | 129.20 | 2.019 |
| 101 | 120 | 60.09 | 0.501 |
| 176 | 105 | 143.55 | 1.367 |
| 100 | 94 | 129.20 | 1.374 |
| 102 | 95 | 129.20 | 1.360 |
| 1 | 80 | 107.67 | 1.346 |

All eight recordings are from `drummer1/session1`. The first four indicate probable half/double-tempo ambiguity. The latter four cluster near 4:3 ratio but cannot yet be labeled as a specific meter/subdivision failure. The exact recurring prediction 129.20 BPM appears in three recordings, which warrants checking recording identity, durations and estimator/fallback diagnostics rather than assuming a root cause. **No weight, algorithm, or model authority changes justified by this pilot alone.**

## Identity check for repeated 129.20 BPM estimates

Owner confirmed the three recordings have different SHA-256 prefixes and durations, ruling out duplicate audio bytes as the explanation for the repeated deterministic estimate:

| ID | Reference BPM | Predicted BPM | Duration | SHA-256 prefix |
|---|---:|---:|---:|---|
| drummer1/session1/100 | 94 | 129.20 | 135.32 s | `6b1ea54caa94` |
| drummer1/session1/102 | 95 | 129.20 | 224.77 s | `036c4deed4db` |
| drummer1/session1/201 | 64 | 129.20 | 112.50 s | `c336210485cf` |

This supports investigating systematic estimator behavior, not a proven specific root cause. Next compare against neural evidence on the same 12-track cohort if an already installed checkpoint can run without downloads. A neural run must be treated as an experiment, not model promotion or definitive model-quality evidence.

## Same-cohort neural-enabled result (owner's Mac)

Owner ran the same approved 12-WAV development manifest using `DJAI_NEURAL_BEATS=1`, `--allow-neural`, with model downloads disabled. Output reported `tempo_mae_bpm=19.198308`, 12/12 tempo coverage, no beat/downbeat reference coverage, and no ledger recording. Previous deterministic MAE on the same manifest: `35.290125`. Absolute improvement `16.091817` BPM, relative MAE reduction **45.6%**.

**Caveats:** `allow-neural` and environment opt-in alone do not prove that Beat This! predictions were accepted or that no fallback occurred; backend provenance is still unverified. Single drummer/session and 12 tracks: promising comparison, **not generalization evidence**, a promotion gate, or M1 completion. Next inspect per-track delta and runtime neural acceptance/rejection diagnostics, then run a drummer/style-diverse development cohort with frozen IDs. Do not touch reserved test cohort or alter production model authority.

## Per-track deterministic versus neural-enabled comparisons

Owner-provided report comparison from identical 12-track Groove cohort: **5 improved, 6 unchanged, 1 regressed**, deterministic MAE `35.290125`, neural-enabled MAE `19.198308` (45.6% reduction). Absolute errors in BPM:

| ID suffix | Deterministic | Neural-enabled | Improvement |
|---|---:|---:|---:|
| 184 | 74.00 | 0.95 | +73.05 |
| 101 | 59.91 | 0.00 | +59.91 |
| 102 | 34.20 | 1.25 | +32.95 |
| 1 | 27.67 | 1.08 | +26.58 |
| 202 | 3.55 | 2.86 | +0.70 |
| 100 | 35.20 | 35.20 | 0 |
| 146 | 0.70 | 0.70 | 0 |
| 176 | 38.55 | 38.55 | 0 |
| 183 | 1.64 | 1.64 | 0 |
| 201 | 65.20 | 65.20 | 0 |
| 3 | 81.50 | 81.50 | 0 |
| 47 | 1.36 | 1.45 | -0.09 |

Rounding of printed rows means sums may not precisely reproduce full-precision MAE. The **unchanged six tracks** could indicate selective neural acceptance, deterministic fallback, identical predictions, or other behavior. Inspect implementation and instrumentation before attributing causality or promoting a checkpoint. The largest residual errors are suffixes 3, 201, 176, 100. This remains a single-session development smoke test, not a generalization claim.

## Confirmed neural confidence gate diagnostic (owner run)

The owner reran the six selected tracks with `DJAI_NEURAL_BEATS=1`, both download flags set to zero and embeddings disabled, and read `analysis.neural_evidence["beat_downbeat"]`. Installed Beat This! is `available=True` on all six. Results:

| ID suffix | Ref BPM | Final BPM | Neural BPM | Neural used | Confidence | Authority |
|---|---:|---:|---:|---|---:|---:|
| 3 | 80 | 161.50 | 81.0811 | false | 0.4943 | 0.0 |
| 201 | 64 | 129.20 | 65.2174 | false | 0.4872 | 0.0 |
| 176 | 105 | 143.55 | 103.4483 | false | 0.3759 | 0.0 |
| 100 | 94 | 129.20 | 93.75 | false | 0.6806 | 0.2314 |
| 184 | 78 | 78.95 | 78.9474 | true | 0.8996 | 0.3059 |
| 101 | 120 | 120.00 | 120.0 | true | 0.7314 | 0.2487 |

These four rejected neural BPM estimates are within ~1.6 BPM of ground truth, while deterministic selected BPMs err by ~35–81 BPM. This **confirms the gate, not model availability**, is the proximal reason those accurate predictions did not replace deterministic tempo on these tracks. It does **not** establish neural beat/downbeat grid quality, calibration on other songs, or whether all rejected neural predictions are beneficial. The implementation in `src/djai/intelligence/neural_ears.py` computes confidence from grid regularity, fallback agreement and downbeat support; requires authority>0, threshold default 0.70 and at least two downbeats before `used=True`. `src/djai/analysis.py` swaps scalar BPM and beat/downbeat grid together when `used=True`.

**Next design experiment:** evaluate neural tempo independently of full beat-grid acceptance; log candidate/reference tempo and reason, quantify tempo-only gains and beat/downbeat F1 on independent annotated Ballroom data; do not lower safety threshold or change production authority on a 12-song one-session pilot. An optional *shadow-only* tempo candidate score should not affect beat/downbeat positions, cue timing or model promotion.
