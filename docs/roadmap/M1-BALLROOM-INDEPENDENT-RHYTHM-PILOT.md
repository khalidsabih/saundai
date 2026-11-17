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

## First owner-run independent eight-track results (2026-11-17)

Owner explicitly entered `yes` after the Ballroom local rights-review prompt and generated a separate local `ballroom-development-APPROVED.json` manifest. All eight tracks had independently annotated beats and downbeats. Local audio root `/tmp/saund-m1-ballroom-pilot/audio`. No audio or private manifest is committed. Rights attestation is specific to owner's permitted local research evaluation; this does not establish general redistribution or training rights.

| Metric | Deterministic | Neural-enabled | Difference |
|---|---:|---:|---:|
| Beat F1 (70ms) | 0.753846 | 0.773073 | +0.019227 |
| Downbeat F1 (100ms) | 0.221154 | 0.532110 | +0.310956 |
| Tempo MAE | null | null | n/a (no reference BPM in manifest) |
| Beat annotation coverage | 8/8 | 8/8 | — |
| Downbeat annotation coverage | 8/8 | 8/8 | — |

No ledger entries recorded. Neural-enabled is not proof of neural acceptance for every track: inspect per-track neural evidence `used`, `confidence`, `reason` and compare event F1 per track. These exploratory scores fall short of proposed targets (beat F1 0.90, downbeat F1 0.85). Next inspect annotation alignment/convention and per-track failure clusters, and **tempo vs median beat-grid BPM consistency** without enabling the scalar shadow tempo selector. Do not promote model or change production on eight-track pilot.

## Per-track eight-track beat/downbeat diagnostic (owner result)

| Track | Deterministic beat F1 | Neural-enabled beat F1 | Deterministic downbeat F1 | Neural-enabled downbeat F1 |
|---|---:|---:|---:|---:|
| Albums-AnaBelen_Veneo-01 | 0.925 | 1.000 | 0.000 | 1.000 |
| Albums-AnaBelen_Veneo-02 | 0.680 | 0.680 | 0.000 | 0.000 |
| Albums-AnaBelen_Veneo-13 | 0.340 | 0.340 | 0.087 | 0.087 |
| Albums-Ballroom_Classics4-06 | 0.976 | 1.000 | 0.444 | 1.000 |
| Albums-Ballroom_Classics4-18 | 0.571 | 0.571 | 0.000 | 0.000 |
| Albums-Cafe_Paradiso-05 | 0.992 | 1.000 | 1.000 | 1.000 |
| Albums-Cafe_Paradiso-14 | 0.619 | 0.619 | 0.000 | 0.000 |
| Media-103511 | 0.959 | 1.000 | 0.000 | 1.000 |

Four neural-enabled tracks have beat F1=1.0 and downbeat F1=1.0. Four poor-scoring tracks have **exactly unchanged** beat/downbeat F1; possible neural grid rejection or identical grid, not yet confirmed. Next capture `neural_evidence["beat_downbeat"]` for all eight and inspect backend `available`, `used`, `confidence`, `authority`, `reason` and tempo/grid ratio without modifying production. If available but rejected, score neural *candidate* beat/downbeat timestamps against independent Ballroom labels in an offline-only tool, as was done with scalar BPM. Do not lower beat grid gate based only on this pilot.

## Full Ballroom neural-grid acceptance diagnosis (owner run)

The 8-WAV pilot has exactly four neural grids accepted and four rejected. In all eight, selected scalar BPM matches median tempo of the **selected** beat grid, which establishes internal consistency but not independent correctness. Four rejected tracks are exactly the four with no improvement in beat/downbeat F1.

| Track | Beat This! candidate BPM | Current selected BPM | Neural used | Confidence | Authority |
|---|---:|---:|---|---:|---:|
| Albums-Cafe_Paradiso-05 | 125.0000 | 125.00 | true | 0.9155 | 0.3113 |
| Albums-Cafe_Paradiso-14 | 176.4706 | 86.13 | false | 0.6673 | 0.2269 |
| Albums-Ballroom_Classics4-18 | 200.0000 | 99.38 | false | 0.6533 | 0.0 |
| Albums-AnaBelen_Veneo-13 | 130.4348 | 89.10 | false | 0.6731 | 0.2288 |
| Albums-AnaBelen_Veneo-01 | 96.7742 | 96.77 | true | 0.8890 | 0.3023 |
| Media-103511 | 100.0000 | 100.00 | true | 0.8947 | 0.3042 |
| Albums-AnaBelen_Veneo-02 | 100.0000 | 99.38 | false | 0.4347 | 0.0 |
| Albums-Ballroom_Classics4-06 | 130.4348 | 130.43 | true | 0.8836 | 0.3004 |

Next decisive validation: compare **candidate** neural beats/downbeats of four rejected grids with independent Ballroom annotations using existing one-to-one event matcher (70ms beat, 100ms downbeat). Do not swap the grids or alter production thresholds before independent scoring. Candidate BPM ratios alone cannot establish correct beat phase or downbeat position.

## Rejected Beat This! candidate beat-grid scoring against independent Ballroom labels

Owner independently scored all four rejected neural candidate event grids versus selected SAUND grids on the eight-track Ballroom local development pilot, using **70ms beat** and **100ms downbeat** one-to-one tolerances. This is a direct reference-timestamp comparison, not just a comparison of scalar BPM:

| Track | Selected beat F1 | Neural candidate beat F1 | Selected downbeat F1 | Neural candidate downbeat F1 |
|---|---:|---:|---:|---:|
| Albums-Cafe_Paradiso-14 | 0.619 | 0.988 | 0.000 | 1.000 |
| Albums-Ballroom_Classics4-18 | 0.571 | 1.000 | 0.000 | 1.000 |
| Albums-AnaBelen_Veneo-13 | 0.340 | 0.977 | 0.087 | 1.000 |
| Albums-AnaBelen_Veneo-02 | 0.680 | 0.875 | 0.000 | 0.880 |

**Findings:** All four rejected Beat This! event grids outperform the retained deterministic grids on both beat F1 and downbeat F1, including tracks where neural tempo estimates differ ~2x or ~1.46x from selected scalar BPM. Current gate was over-conservative on all four examples; it may still protect other tracks, and the 8-recording Ballroom sample is not enough to retune or loosen production safeguards. In four other tracks the neural grid was accepted and earlier per-track evaluation found beat F1 1.0 / downbeat F1 1.0, but the above is NOT a complete micro-aggregated all-neural score; it reports only four rejected candidates. No production or checkpoint promotion.

**Next:** create auditable, read-only candidate-versus-selected beat/downbeat F1 across a larger *non-overlapping* independently annotated Ballroom evaluation cohort, inspect length/genre/meter and neural-confidence failure cases, and preregister a gate challenger with deterministic fail-closed safeguards. Also audit internal BPM-to-beat-grid consistency before integrating any scalar-only tempo selection. Do not assume current Ballroom study permits model training or unrestricted redistribution.

## Larger-cohort candidate-grid shadow evaluator available

Added `scripts/m1-ballroom-grid-shadow.py` with focused tests `tests/test_m1_ballroom_grid_shadow.py`. The tool validates a rights-reviewed Ballroom manifest using the existing M1 benchmark validator, runs installed Beat This! in shadow mode with model downloads disabled, and evaluates selected vs neural candidate timestamp grids at the same 70ms/100ms event matching tolerances. Unavailable neural candidates are **excluded** from candidate score rather than treated as zero. Report includes acceptance/availability coverage and BPM-vs-median-grid BPM disagreement per track. Does not modify `analysis.py`, model registry or live behavior.

Run first on eight existing approved WAVs:

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_ballroom_grid_shadow.py
DJAI_NEURAL_BEATS=1 DJAI_MODEL_ALLOW_DOWNLOAD=0 DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0 \\
 "$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-grid-shadow.py \\
 --manifest /tmp/saund-m1-ballroom-pilot/ballroom-development-APPROVED.json \\
 --audio-root /tmp/saund-m1-ballroom-pilot/audio \\
 --output /tmp/saund-m1-ballroom-pilot/grid-shadow-8.json
```

Before expanding beyond eight, **inventory the user's actual local annotation files and original Ballroom archive/extraction layout** and establish a fresh disjoint track-ID manifest; do not guess paths or reconstruct annotations from predictions. Because Ballroom is evaluation-only and subject to intended-use/attribution constraints, any expanded manifest must be explicitly reviewed and approved; do not blanket-authorize additional audio. Freeze untouched final holdout separately from exploratory gate development.

## Auditable eight-track candidate-grid shadow evaluator execution (owner Mac)

After fetching PR branch at `430843c`, owner executed `tests/test_m1_ballroom_grid_shadow.py`: **3/3 tests passed**. The real-data `scripts/m1-ballroom-grid-shadow.py` run on the eight rights-approved local Ballroom development WAVs reported:

| Metric | SAUND selected grid | All Beat This! candidate grids |
|---|---:|---:|
| Beat F1 (70 ms) | **0.773073** | **0.983732** |
| Downbeat F1 (100 ms) | **0.532110** | **0.988593** |
| Tempo MAE | null (no reference BPM) | null |
| Neural candidate coverage | — | 8/8 |
| Accepted by current gate | — | 4/8 |
| Rejected by current gate | — | 4/8 |

`production_changed=false`, `model_promoted=false`. **This pilot exceeds proposed M1 beat/downbeat thresholds for raw candidate grids, not the production-selected grid**, and the sample of eight exploratory recordings is insufficient for representative performance or safety certification. Owner's inventory of `/tmp` found neither the specifically named Ballroom beat-annotation file examples nor `data1.tar.gz`, and `/tmp/saund-m1-ballroom-pilot` contains only its `audio` subdirectory. Next locate original annotation/archive files in the owner's actual acquisition directory rather than inventing paths or regenerating labels from SAUND. For a larger research-only set, freeze non-overlapping IDs and conduct separate applicable rights review; preserve a final untouched holdout.

## Full local corpus verified; reproducible 80-track draft importer

Owner inventory of original `B_1.0` reported **698 beat .beats files, 698 tempo .bpm files, 698 WAVs, exact basename intersection 698**. Sample `.beats` rows have two fields (seconds, beat-in-bar index), e.g. `0.3000000000 1` followed by positions 2–4; `1` marks downbeat. Example .bpm is `124`. Audio directories and WAV counts: ChaChaCha 111, Jive 60, Quickstep 82, Rumba-American 7, Rumba-International 51, Rumba-Misc 40, Samba 86, Tango 86, VienneseWaltz 65, Waltz 110 (plus empty `nada`). Source root is under user's private local `~/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0`.

New `scripts/m1-ballroom-manifest.py` scans original files, requires exact audio+annotations by basename, validates monotonic beat timestamps and separate positive BPM, derives downbeats only from **source beat-position labels**, excludes old 8 pilot IDs and <8s files, selects a seeded genre-round-robin sample (default 80), produces a private `rights_confirmed=false` *development draft*, and refuses to overwrite results. No audio copying, generated beat labels, source downloads, or training.

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_ballroom_manifest.py
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-manifest.py \\
  --source-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0" \\
  --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-development-APPROVED.json \\
  --count 80 \\
  --output /tmp/saund-m1-ballroom-pilot/ballroom-diverse-80-DRAFT.json
```

The previous 8-track research permission does **not** automatically authorize an expanded cohort. Owner must review expanded evaluation rights and explicitly attest locally before generating a *distinct* approved manifest with `rights_confirmed=true`; do not auto-approve in importer. After approval, benchmark with `--audio-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0/audio"` and `scripts/m1-ballroom-grid-shadow.py` using a **new** result path. These 80 are exploratory development only; freeze a separate never-inspected holdout and do not tune against it. Zero production behavior changed.

## Owner-run 80-track independent Ballroom development cohort (2026-11-17)

Owner confirmed the expanded 80-track local research evaluation was permitted and generated a distinct `ballroom-diverse-80-APPROVED.json`, separate from the original eight-track cohort. Source B_1.0 contains 698 matched original beat, BPM and WAV files; seeded importer selected 80 from 690 eligible recordings (8 explicitly excluded), with representation across 10 genre directories (ChaChaCha 9, Jive 9, Quickstep 8, Rumba-American 6, Rumba-International 8, Rumba-Misc 8, Samba 8, Tango 8, VienneseWaltz 8, Waltz 8). Private manifest and source audio remain local, not committed.

Executed `scripts/m1-ballroom-grid-shadow.py` with model downloads disabled against the locally approved independent beat/downbeat and BPM references. Output at `/tmp/saund-m1-ballroom-pilot/grid-shadow-80.json` (not committed):

| Metric | Selected SAUND grid | Beat This! raw candidate grid |
|---|---:|---:|
| Beat F1 (70 ms) | **0.792782** | **0.975817** |
| Downbeat F1 (100 ms) | **0.694564** | **0.978045** |
| Tempo MAE (absolute, no octave folding) | **22.547576 BPM** | **2.558026 BPM** |
| Neural candidate availability | — | **80 / 80** |
| Neural grid currently accepted | — | **50 / 80** |
| Neural grid currently rejected | — | **30 / 80** |

Raw Beat This! candidate beat/downbeat F1 exceeds proposed 0.90/0.85 M1 gates **on exploratory development data only**. Current selected integration does not. This result strongly suggests over-conservative gating in aggregate but cannot establish whether all 30 rejected candidate grids improve independently, and is not an untouched holdout or safe transition-render certification. `production_changed=false`, `model_promoted=false`. No automatic threshold relaxation or promotion. Next analyze 80-track per-track paired selected-vs-candidate event counts and failures by genre, confidence, authority, and acceptance outcome; keep separate final untouched holdout for gate selection validation.

## Pairwise diagnosis of 30 rejected grids in the 80-track Ballroom development sample

Owner's read-only analysis of `grid-shadow-80.json` showed **29/30 rejected raw Beat This! candidates improved per-track beat F1**, **1/30 reduced beat F1**, and **30/30 improved downbeat F1**. No ties. The one beat regression is `Media-104011` (selected 0.979 beat F1 vs candidate 0.950, while downbeat F1 improves 0.000 to 0.923). All selected and neural candidate grids across the complete 80-track cohort had absolute scalar-BPM vs median beat-grid BPM discrepancies <=2 BPM (0 above 2 in both). Selected tempo-vs-grid consistency by itself does not imply tempo accuracy; the candidate tempo MAE is lower (2.558 vs 22.548 BPM).

**Decision:** Evidence supports an *offline challenger* to the existing authority/confidence gate, not immediate production rule modification. Even a high mean F1 and 29/30 improvements do not prove absence of catastrophic mistakes outside Ballroom or of unsafe bar-phase changes in DJ transition rendering. Assess raw candidate scores, paired selected-vs-candidate regressions, and clinical-like fail-safes on an untouched holdout; include Groove regressions and transition continuity before promotion. Avoid selecting a production threshold from these 80 exploratory examples and then treating them as independent validation.

## Neural-confidence band diagnosis (owner-run exploratory 80-track Ballroom development cohort)

| Neural confidence | Tracks | Currently accepted | Candidate beat F1 vs selected | Candidate downbeat F1 vs selected | Mean per-track beat gain | Mean per-track downbeat gain |
|---|---:|---:|---|---|---:|---:|
| <0.50 | 9 | 0 | 8 improve, 1 regress | 9 improve | +0.3365 | +0.6143 |
| 0.50–0.70 | 21 | 0 | 21 improve, 0 regress | 21 improve, 0 regress | +0.5603 | +0.8567 |
| 0.70–0.85 | 21 | 21 | 0 change (already selected) | 0 change (already selected) | 0.0 | 0.0 |
| >=0.85 | 29 | 29 | 0 change (already selected) | 0 change (already selected) | 0.0 | 0.0 |

All 30 neural candidate rejections are in confidence <0.70; all 50 current neural acceptances are confidence >=0.70 on this data. This pattern is **descriptive**, not a verified implementation-wide hard threshold: multiple authority/confidence signals may influence acceptance. The zero F1 deltas in accepted cases are **tautological** because selected equals candidate, and must not be interpreted as poor neural quality. The 21/21 improvement in the 0.50–0.70 bin is substantial, but cannot justify lowering production cutoff directly: the dataset has already been inspected, the <0.50 band contains one beat regression, and other genres/source data could behave differently.

**Next experiment:** preregister a conservative *shadow-only* challenger (e.g. propose acceptance for `confidence >=0.50`, no production changes), compare track-paired selected vs counterfactual selected metrics across a **new disjoint validation cohort**. Keep independent test set untouched for once policy is frozen. Check downbeat phase/bar-grid consistency and rendered DJ transitions on any candidate to be promoted. Avoid reusing these 80 development tracks as validation of the chosen cutoff.

## Frozen confidence 0.50 challenger — fresh disjoint validation, shadow only

**Policy frozen from the exploratory 80-track development observations:** retain the current selected grid when Beat This! was already accepted; otherwise choose the *whole* Beat This! event grid and paired BPM only when it is available, confidence **>=0.50** and |candidate BPM - median candidate beat-grid BPM| **<=2.0 BPM**. Otherwise keep the currently selected grid. This policy is **not production code**. No grid/BPM hybrid, no authority threshold change, no training or auto-promotion. The validation report includes track-paired beat/downbeat regressions and counts, micro-F1, tempo MAE and raw-candidate comparison.

New `scripts/m1-ballroom-gate-challenger.py` reads private grid-shadow JSON and a **validation** (not development) rights-approved Ballroom manifest, checks manifest SHA256 and exact ID matching, evaluates the fixed policy and writes a new no-overwrite report. `scripts/m1-ballroom-manifest.py` now accepts repeatable `--exclude-manifest` flags and a `--split validation` for a distinct seeded genre-diverse cohort. Tests: `tests/test_m1_ballroom_gate_challenger.py`, `tests/test_m1_ballroom_manifest.py`.

Run on the owner's Mac from PR #28 branch, with source root under `~/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0`:

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_ballroom_gate_challenger.py tests/test_m1_ballroom_manifest.py
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-manifest.py \
  --source-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0" \
  --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-development-APPROVED.json \
  --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-diverse-80-APPROVED.json \
  --split validation --count 80 --seed 20261118 \
  --output /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-DRAFT.json
```

After a **new, explicit local rights review** of this validation cohort, create an approved copy manually with `rights_confirmed=true` (never within the importer), then run:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-grid-shadow.py \
  --manifest /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-APPROVED.json \
  --audio-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0/audio" \
  --output /tmp/saund-m1-ballroom-pilot/grid-shadow-validation-80.json
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-gate-challenger.py \
  --manifest /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-APPROVED.json \
  --shadow /tmp/saund-m1-ballroom-pilot/grid-shadow-validation-80.json \
  --output /tmp/saund-m1-ballroom-pilot/frozen-gate-validation-80.json
```

Do not tune after seeing validation results and call those results holdout. Reserve a later **test** split for final one-shot review. Success requires >=0.90 beat F1, >=0.85 downbeat F1, scrutiny of any >0.10 per-track beat regression and tempo/grid consistency, and separate DJ-render transition safety. This is a research workflow only; scripts do not certify production readiness.

## Pre-frozen 0.50 gate: fresh disjoint 80-track validation results (owner, 2026-11-17)

After owner explicitly reviewed and approved local 80-track Ballroom validation use, the user executed `scripts/m1-ballroom-grid-shadow.py` on `ballroom-validation-80-APPROVED.json` and `scripts/m1-ballroom-gate-challenger.py` with the **previously frozen** confidence >=0.50 threshold plus <=2 BPM candidate-grid consistency guard. The validation manifest had 80 unique Ballroom recordings disjoint from the original 8 + development 80. Inputs/outputs are private local files, not committed. Tests for the importer/challenger were 9/9 passed before this run.

| Metric | Existing selected grid | Frozen confidence-0.50 challenger | Raw neural candidate |
|---|---:|---:|---:|
| Beat F1 (70ms) | 0.850554 | **0.952999** | 0.977795 |
| Downbeat F1 (100ms) | 0.770713 | **0.946314** | 0.981265 |
| Tempo MAE (BPM) | 17.738065 | **3.506765** | 1.831609 |

Coverage: neural candidate available **80/80**; current gate accepts **60/80**, rejects **20/80**; frozen challenger **newly accepts 16** (total neural use **76/80**). Paired per-track regression counts **0 beat F1 decreased, 0 downbeat F1 decreased, 0 beat-F1 drops exceeding 0.10**. No production changes, no model promotion, no new model weights. The frozen challenger passes the proposed beat >=0.90/downbeat >=0.85 benchmarks **on the disjoint validation set**, not as a general production safety certification. The four currently rejected candidates remaining unselected under the challenger warrant review, but **do not tune** on validation data after observing outcomes. The raw candidate exceeds the challenger in aggregate, but it has not passed cross-domain transition-safety tests; do not promote unconditional neural use.

**Decision:** Freeze policy and these independent validation results; proceed to separately protected test cohort and practical DJ-render safety checks (downbeat phase stability, cue positions, bar-count quantization, first/last bar correctness, mix handoff and regressions on non-Ballroom music). Revalidate BPM/beat-grid coupling when swapping entire grid and BPM; never substitute the scalar tempo alone. Do not lower threshold further based on the validation result.

## Next safety stage: integrity audit on validation + sealed final test manifest

The previously frozen 0.50 challenger passed the independent 80-track validation: selected beat F1 **0.850554**, challenger **0.952999**, raw neural **0.977795**; selected downbeat F1 **0.770713**, challenger **0.946314**, raw **0.981265**; selected tempo MAE **17.738065 BPM**, challenger **3.506765**, raw **1.831609**. Challenger newly accepted **16** neural grids (total 76/80) with **zero beat or downbeat F1 regressions**. Threshold stays frozen, with no production behavior change.

Added `scripts/m1-ballroom-grid-integrity.py` plus `tests/test_m1_ballroom_grid_integrity.py`. This read-only validation audit performs its own existing Beat This! inference, selects an entire candidate grid + BPM with the frozen rule, then compares structural requirements for currently selected and challenger grids: nonempty increasing events, BPM-vs-median beat interval <=2 BPM, at least four beats/two downbeats, and each predicted downbeat within 100ms of a predicted beat. It produces private per-track failure reasons and counts; it **does not prove the underlying downbeats correct against annotations and does not render a DJ transition**. No alternate policy or threshold optimization is performed.

To audit the already approved 80-track **validation** set:

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_ballroom_grid_integrity.py
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-grid-integrity.py \
 --manifest /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-APPROVED.json \
 --audio-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0/audio" \
 --output /tmp/saund-m1-ballroom-pilot/grid-integrity-validation-80.json
```

Create a **sealed rights-unapproved, unseen final test draft** without evaluating it until validation integrity and cross-domain/rendered transition safety procedures are reviewed. The importer excludes all **168** previously used Ballroom tracks (8 pilot, 80 development, 80 validation). A new seed is used:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-manifest.py \
 --source-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0" \
 --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-development-APPROVED.json \
 --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-diverse-80-APPROVED.json \
 --exclude-manifest /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-APPROVED.json \
 --split test --count 80 --seed 20261119 \
 --output /tmp/saund-m1-ballroom-pilot/ballroom-final-test-80-DRAFT.json
```

**Do not inspect final test annotations/scores or approve final test use automatically.** The test set is only a final one-shot evaluation after the policy, acceptance criteria, license scope, cross-domain failure cases, and actual rendered-transition safety checks have been locked. The mere existence of a test manifest does not make it an authorized input. Ballroom remains evaluation-only. Future production promotion needs dedicated real mix render QC with bar and cue alignment, beat-grid swaps/tempo coupling and safe rollback.

## Validation grid integrity and final sealed test draft (owner, 2026-11-17)

Owner ran `scripts/m1-ballroom-grid-integrity.py` on the already-approved, 80-track *validation* cohort, obtaining **2 selected-grid structural integrity failures, 0 challenger-grid failures and 0 raw neural candidate failures**; frozen challenger newly accepted **16** Beat This! grids. This is structural validation only: `rendered_transition_safety_tested=false`, `production_changed=false`. The two selected-grid failure reasons and IDs were not included in the CLI summary and should be inspected from `/tmp/saund-m1-ballroom-pilot/grid-integrity-validation-80.json` before concluding the cases were repaired.

Owner also generated a separate unapproved final test draft `/tmp/saund-m1-ballroom-pilot/ballroom-final-test-80-DRAFT.json` with `--split test --count 80 --seed 20261119`, excluding **168** previously examined Ballroom recordings. Source had **530** eligible after exclusions; the draft selected **80** across 9 genres (9 each ChaChaCha/Jive/Quickstep/Rumba-International/Rumba-Misc/Samba/Tango/VienneseWaltz, 8 Waltz). `rights_confirmed=false` and `evaluation_only=true`; **no test inference or approval yet**. Keep test sealed until cross-domain checks and rendered transition criteria are reviewed. This final test set is disjoint by exact source track ID, not necessarily by performer or recording-family group.

## Identity and failure reason of both selected-grid integrity failures (owner Mac)

The two tracks flagged in the 80-recording validation integrity report are both **existing-gate rejected** and **frozen-challenger neural accepted**. Both existing selected grids fail only the `bpm_grid_mismatch` structural check; `offgrid_downbeats=0` in both. Both neural candidate/challenger grids pass with no failures:

| Source track | Genre | Selected beat-grid BPM | Absolute selected scalar-BPM vs grid discrepancy | Neural/challenger beat-grid BPM | Neural/challenger discrepancy | Selected failure | Challenger failure |
|---|---|---:|---:|---:|---:|---|---|
| `Media-104917` | Quickstep | 99.38415 | 3.97525 BPM | 200.0 | 0.0 BPM | `bpm_grid_mismatch` | none |
| `Albums-Chrisanne3-09` | VienneseWaltz | 117.45384 | 5.10674 BPM | 176.47059 | 0.00001 BPM | `bpm_grid_mismatch` | none |

This confirms the integrity audit’s original **2 selected failures → 0 challenger failures** on these two recordings, but it does **not** establish the musical correctness of faster grid interpretations or rendered DJ-transition stability. Independent annotated beat/downbeat F1 results remain from the earlier 80-track validation benchmark; do not conflate structural consistency with annotated accuracy. Next investigate selected scalar BPM vs actual median beat-grid BPM for these cases and implement separate offline transition/bar-cue quantization tests on **validation** tracks, while preserving unopened 80-recording test draft.

## Offline bar-aware transition-cue timing preflight (validation only)

Added `scripts/m1-ballroom-transition-timing-shadow.py` and six focused tests `tests/test_m1_ballroom_transition_timing_shadow.py`. Unlike the earlier simple grid integrity check, this script uses the **independently annotated number of beats per bar** from Ballroom `.beats` position 1 downbeat markers to avoid mistakenly imposing 4/4 on VienneseWaltz/Waltz. It runs the already-installed neural inference in shadow mode with downloads disabled, derives a hypothetical **4-bar transition cue window**, and checks: beat and downbeat ordering, >=5 downbeats, BPM vs beat-grid agreement <=2 BPM, downbeat-to-beat snap <=100ms, bar-duration plausibility (within 20% of median beat interval × annotated meter), cue-span plausibility (within 20% of 4 expected bars), and whether the hypothetical cue start is within 100ms of independent annotated downbeat. It evaluates current selected, frozen 0.50 challenger, and raw neural candidate on the existing **approved 80-track validation cohort only**. The stored final 80-track test draft remains sealed/unapproved. No production changes.

```sh
cd /tmp/saund-m1-evaluator
git fetch origin docs/m0-baseline-capability-audit-20261116
git checkout --detach FETCH_HEAD
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_ballroom_transition_timing_shadow.py
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-ballroom-transition-timing-shadow.py \
  --manifest /tmp/saund-m1-ballroom-pilot/ballroom-validation-80-APPROVED.json \
  --audio-root "$HOME/Downloads/lab/SAUND_AI/data/beat-benchmark/ballroom/B_1.0/audio" \
  --output /tmp/saund-m1-ballroom-pilot/transition-timing-validation-80.json
```

The report is deliberately labeled **`rendered_transitions_tested=false`, `cross_domain_tested=false`**. Hypothetical cue boundaries and reference downbeat proximity do not establish actual DJ render safety, any concrete SAUND cue policy, usable musical phrases, audible mix continuity, source rights beyond local authorized evaluation, or reliability outside Ballroom. Failing records must be diagnosed before a real mix-render canary can be designed and executed. **No final test inference until those checks and final evaluation criteria are frozen.**

## Offline bar-aware transition-timing validation result (owner, 2026-11-17)

Owner executed `scripts/m1-ballroom-transition-timing-shadow.py` on the already rights-approved 80-track Ballroom **validation** manifest. The summary was:

| Measure | Current selected grid | Frozen 0.50 challenger | Raw neural candidate |
|---|---:|---:|---:|
| Tracks failing stricter hypothetical bar/cue preflight | **22/80** | **12/80** | **13/80** |
| Tracks passing preflight | 58/80 | 68/80 | 67/80 |

Challenger's hypothetical four-bar cue start was within 100 ms of some independently annotated downbeat for **76/80** tracks (**95%**); the cue comparison was evaluable on all 80. Challenger selected a neural grid for **76/80** tracks. **No rendered transitions tested**, no production changes.

This is a broader hypothetical cue/bar integrity audit than the earlier basic beat-grid structure test, where the challenger had 0/80 failures; **do not confuse the two measures**. This audit imposes a reference-derived beats-per-bar assumption, <=20% deviation from median estimated bar duration, >=5 predicted downbeats, and a 4-bar cue span. It may flag meter interpretation differences, irregular intros/outros, truncated or missing downbeat events, or genuine rhythm failures. The 12 failures and four reference cue-start misses must be inspected per track and by failure reason **before interpreting them as correctable musical errors**. In particular, proximity of a *hypothetical* cue start to a labeled downbeat is not an actual DJ transition safety validation. Keep the final 80-track test dataset unapproved/unscored, preserve the frozen 0.50 challenger, then investigate false positives/real failures and plan source-rights-safe render tests on non-test data.

## Root-cause triage of 12 hypothetical bar/cue preflight flags (owner-reported)

From `transition-timing-validation-80.json`, all **12/80** frozen-challenger failures included `irregular_bar_duration`. Of these, **3/80** additionally had `cue_window_duration_mismatch` (`Media-104008`, `Media-105111`, `Media-104915`), and **all three** were on tracks for which the frozen challenger retained the **deterministic** grid. Nine other failures were on neural-selected tracks: `Media-105020` (4/4), `Albums-Latin_Jam2-15` (4/4), `Media-103616` (4/4), `Media-105810` (3/4), `Media-105209` (4/4), `Albums-Secret_Garden-05` (3/4), `Albums-Chrisanne1-14` (4/4), `Media-103501` (4/4), `Media-104005` (4/4). Three failures on deterministic tracks `Media-104008` (4/4), `Media-105111` (3/4), `Media-104915` (4/4).

Four challenger hypothetical cue starts were **not** within 100 ms of an independently annotated downbeat: `Media-104008` (Rumba-Misc, deterministic), `Media-103707` (Samba, **neural accepted**, even though it passed bar-duration preflight), `Media-105111` (VienneseWaltz, deterministic), `Media-104915` (Quickstep, deterministic). Note that a cue-start phase miss can occur without an irregular bar flag. Thus neither structural consistency nor bar-duration checks alone certify correct bar *phase*.

Next inspect per-bar downbeat intervals by position, especially intros/outros and missing predicted downbeats; compare predicted downbeats against independent annotations near the proposed cue start and assess meter-specific conventions. Distinguish legitimate gaps and variable recording boundaries from sustained bar-grid errors. Do not assume the strict `any bar outside 20% tolerance` preflight is an actual rendered-mix safety guarantee. Do not revise the frozen 0.50 confidence policy from these validation observations. No test split inspection or production changes.
