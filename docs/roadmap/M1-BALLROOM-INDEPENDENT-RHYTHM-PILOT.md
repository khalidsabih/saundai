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
