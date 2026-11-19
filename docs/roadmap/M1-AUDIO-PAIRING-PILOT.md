# M1 audio pairing pilot — integrity first, no silent music downloads

## Evidence and limits

The owner has already acquired and audited **annotations only**:
- GiantSteps Tempo v2: 664 files, 661 positive BPM references, pinned `d51ab2422e76abacfaa86616a57054bc222ec9fd`.
- CPJKU Ballroom: 698 beat/downbeat reference files, 44,605 beat events, 12,222 downbeat events, pinned `1db08914a8ae15edb01f104046e30bad88effe67`.
- 22/22 previous focused tests passed.

The [Ballroom source README](https://github.com/CPJKU/BallroomAnnotations/blob/master/README.md) points to the original audio archive `https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`, describes about **1.5 GB** of audio, and publishes **MD5 `2872a3e52070bc342a4510a95e2fa0b8`**. The [mirdata Ballroom source](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/ballroom.py) independently lists the same official URL and checksum. **These are access/integrity facts, not automatic commercial-use or training authorization.** The source README documents 4 exact duplicate pairs and 9 recording-replica pairs; all must be kept on a single split side.

GiantSteps annotations' ID/LOFI structure is available, but **audio source and permitted use are still unresolved**. Do not scrape Beatport or assume 30-second licensed previews may be repurposed for model training.

## Newly implemented read-only pairing audit

`scripts/m1-audio-readiness.py`:
- Reads the local original Ballroom `.tar.gz` **without extracting or modifying it**; before archive inspection verifies the published archive MD5.
- Alternatively scans an explicitly supplied local audio directory (Ballroom or GiantSteps), matching exact annotation IDs to audio basenames, case-insensitively and extension-independently.
- Reports matched, missing and ambiguous IDs and 13 known Ballroom replica pairs that must not cross development/test splits.
- Rejects tar path traversal, links and ambiguous matches. All reports set `audio_rights_confirmed=false` and `benchmark_ready=false` by design.
- Requires an existing local source. No network access or downloads; no rights attestation or benchmark score is inferred.

### Option A: local pre-existing audio (fastest, no download)

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-audio-readiness.py \
 --dataset ballroom \
 --annotations /tmp/saund-m1-reference-sources/BallroomAnnotations \
 --audio-root /path/to/legally-held/ballroom-audio \
 --output /tmp/saund-m1-ballroom-audio-audit.json
```

If no official/matching audio is present, **do not point the script to arbitrary SAUND music**; there is no valid connection from those filenames to Ballroom's annotated tracks.

### Option B: official original Ballroom archive, **only if separately authorized**

The owner may personally review the acquisition/use conditions and choose to download about 1.5GB, but the SAUND development process does **not** do so silently. With the archive on disk:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-audio-readiness.py \
 --dataset ballroom \
 --annotations /tmp/saund-m1-reference-sources/BallroomAnnotations \
 --archive /path/to/official/data1.tar.gz \
 --output /tmp/saund-m1-ballroom-archive-audit.json
```

The tool verifies the original archive's published MD5 before listing. **Do not extract/use the archive until license and intended research/commercial-use scope have been reviewed**. The checksum is for historical file identity, not cryptographic security.

### GiantSteps local audio only when appropriate

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-audio-readiness.py \
 --dataset giantsteps_tempo \
 --annotations /tmp/saund-m1-reference-sources/giantsteps-tempo-dataset \
 --audio-root /path/to/legally-held/giantsteps-clips \
 --output /tmp/saund-m1-giantsteps-audio-audit.json
```

## Subsequent evaluation (not yet authorized or executed)

1. Approve a small development-only set with exact recording IDs and proven matching annotation/audio identities.
2. Confirm the *intended* rights for running local analysis; preserve separate ML training rights review.
3. Produce reviewed manifest using `scripts/m1-import-references.py`, explicitly record rights after human review, and run `scripts/m1-rhythm-benchmark.py` with `DJAI_NEURAL_BEATS=0` and no weight downloads.
4. Record real metrics in a private ledger, not an M0 synthetic champion. Freeze test groups later.

## Regression test

```sh
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q \
 tests/test_m1_audio_readiness.py \
 tests/test_m1_audit_annotations.py \
 tests/test_m1_import_references.py \
 tests/test_m1_rhythm_benchmark.py \
 tests/test_dataset_benchmark_registry.py
```

Previous 22 focused tests passed. Five new read-only pairing tests are checked in, **Mac execution pending**.

**Gate status:** source URL/checksum verified; importer and reference annotations validated; **real matching audio, license clearance and measured model accuracy remain unavailable**.
