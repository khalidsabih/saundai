# M1 real-reference acquisition pilot — annotations first (2026-11-16)

## Status and decisions

- `scripts/m1-import-references.py` passed focused tests in the owner's Mac environment (18/18 total importer + evaluator + registry).
- Public [GiantSteps Tempo annotations](https://github.com/GiantSteps/giantsteps-tempo-dataset) and [Ballroom beat/bar annotations](https://github.com/CPJKU/BallroomAnnotations) are available independently of audio.
- GiantSteps `annotations_v2/tempo/*.LOFI.bpm` provides single numeric BPM; Ballroom `*.beats` provides seconds plus beat position, where `1` means downbeat. The [Ballroom annotation README](https://github.com/CPJKU/BallroomAnnotations/blob/master/README.md) says **audio is not included**, notes known exact and recording replicas, and advises version pinning.
- **No audio access, acquisition or use-rights clearance is assumed**. Do not automatically download copyrighted previews or 1.5 GB Ballroom archive; no scraped third-party copies.
- For M1, an honest step forward is acquiring small annotation repositories, recording their immutable commits and auditing actual reference counts before making any audio decision.

## One-time local metadata acquisition and audit

From the owner's Mac; this downloads **GitHub annotation repositories only**, no music:

```sh
mkdir -p /tmp/saund-m1-reference-sources
git clone --depth 1 https://github.com/GiantSteps/giantsteps-tempo-dataset.git \
  /tmp/saund-m1-reference-sources/giantsteps-tempo-dataset
git clone --depth 1 https://github.com/CPJKU/BallroomAnnotations.git \
  /tmp/saund-m1-reference-sources/BallroomAnnotations

cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-audit-annotations.py \
  --dataset giantsteps_tempo \
  --annotations /tmp/saund-m1-reference-sources/giantsteps-tempo-dataset \
  --output /tmp/saund-m1-giantsteps-annotations-audit.json

"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-audit-annotations.py \
  --dataset ballroom \
  --annotations /tmp/saund-m1-reference-sources/BallroomAnnotations \
  --output /tmp/saund-m1-ballroom-annotations-audit.json
```

The audit is stdlib-only, performs no model inference, hashes the complete exact annotation collection in stable order, counts available reference labels, stores the Git source commit if known, and outputs `audio_acquired: false`, `audio_use_rights_confirmed: false`, `benchmark_ready: false`. Zero-BPM GiantSteps labels are missing and are **not** scored as valid zero BPM. No per-recording audio names or absolute paths are uploaded.

If the target directories exist, do **not** blindly delete or overwrite them. Use different private paths and output JSON filenames.

## Next gate: rights-reviewed real audio

After audits, identify a legally usable sample of audio **exactly matching the annotations**, including duplicate recording-group and split protection. Do not substitute arbitrary SAUND crate tracks for annotation-backed recordings. The optional importer in `scripts/m1-import-references.py` matches local audio by canonical basename only; results are deliberately unapproved drafts. Only after rights, provenance and matching are verified can `scripts/m1-rhythm-benchmark.py` report real M1 results.

## Isolated focused test

```sh
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q \
 tests/test_m1_audit_annotations.py \
 tests/test_m1_import_references.py \
 tests/test_m1_rhythm_benchmark.py \
 tests/test_dataset_benchmark_registry.py
```

Four new audit-focused tests are checked in but **not yet executed in the owner's Mac**. Previously, 18 importer/evaluator/registry focused tests passed. Expected combined count: 22 (pending local run).

This is metadata acquisition preparation, **not** a measured musical benchmark, model improvement, live authority change or M0 signoff.


## Owner's actual annotation acquisition and audit results — 2026-11-16

The owner successfully cloned **annotations only** into private `/tmp/saund-m1-reference-sources/` and ran the audited tools from detached `/tmp/saund-m1-evaluator`, then ran all focused tests:

| Source | Pinned annotation commit | Collection hash SHA-256 | Verified coverage |
|---|---|---|---|
| GiantSteps Tempo v2 | `d51ab2422e76abacfaa86616a57054bc222ec9fd` | `0be3f2f15d8e9b9caa73c49b19db73f609677aeccc5587256da3a3138db0aec5` | 664 files; 661 BPM present, 3 missing |
| Ballroom bar-position annotations | `1db08914a8ae15edb01f104046e30bad88effe67` | `0fab8c8ac7229c1112b1b07f0a8b8fb36295802ccff293a5f12a164b62f2bd6a` | 698 files; 44,605 beats; 12,222 downbeats; 698 tracks with both |

Tests: `22 / 22 passed` (4 annotation audits + 7 importer + 6 evaluator + 5 registry tests).

Observed audit JSON flags for **both sources**: `audio_acquired=false`, `audio_use_rights_confirmed=false`, `benchmark_ready=false`. No audio or rights are inferred from these results. No music accuracy measurements or champions exist from this evidence.

**Next bounded gate:** obtain audio clips matching exact annotation IDs **with appropriate permitted-use rights**, protect known Ballroom recording replicas from split leakage, create owner-reviewed private development manifest, then run first actual SAUND M1 development benchmark. Do not rerun the 22 synthetic tests unless code changes.
