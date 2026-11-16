# M1 Ballroom official-audio research pilot

**Implementation committed; local test and actual audio benchmark still pending.** The goal is an initial, private, **development-only** evaluation on up to eight officially paired Ballroom recordings, not a held-out test or model training.

The [official BallroomAnnotations README](https://github.com/CPJKU/BallroomAnnotations/blob/master/README.md) points to the MTG Ballroom audio archive `https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz`; [mirdata's Ballroom implementation](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/ballroom.py) records its MD5 as `2872a3e52070bc342a4510a95e2fa0b8` and identifies a **CC BY-NC-SA 4.0** dataset license. The [MTG datasets page](https://www.upf.edu/web/mtg/datasets-download) describes public MTG datasets as freely available for **research purposes**; their [software/datasets page](https://www.upf.edu/en/web/mtg/software-datasets) directs users to review individual dataset terms and contact them for commercial licensing. The audio archive is reported around 1.5GB; downloading it can consume substantial time and disk space.

**License gate:** Attribution, noncommercial and share-alike obligations may apply. Verify the exact audio ownership/terms for your intended activity. This pilot is not cleared for redistributing audio, training a commercial model, or production deployment. The importer continues to write `rights_confirmed=false` and will not trigger the actual rhythm evaluator. No blanket rights assertion is made.

## Step 1 — optional owner-approved archive download

Only after you decide your **private, noncommercial research evaluation** is within the relevant terms:

```sh
mkdir -p /tmp/saund-m1-official-audio
curl -fL --retry 3 -C - \
  'https://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz' \
  -o /tmp/saund-m1-official-audio/data1.tar.gz
```

This downloads the **official** recording archive. It is opt-in; scripts never fetch music automatically. If the source fails, do not switch to unverified mirrors or scrape recordings. Do not run the archive extraction on arbitrary untrusted tarballs.

## Step 2 — inspect a tiny sample (no extraction)

```sh
cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-prepare-ballroom-pilot.py \
  --archive /tmp/saund-m1-official-audio/data1.tar.gz \
  --annotations /tmp/saund-m1-reference-sources/BallroomAnnotations \
  --destination /tmp/saund-m1-ballroom-pilot \
  --max-tracks 8 --inspect-only
```

The command first computes the original 1.5GB archive MD5 and **rejects a mismatch**, then safely inventories archive members and builds a deterministic, genre-diverse shortlist. It excludes the 13 documented Ballroom exact/recording-replica pairs and ambiguous duplicate basenames.

## Step 3 — optional local, bounded extraction into a NEW private directory

Only if you understand the license and **intend noncommercial private research**:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-prepare-ballroom-pilot.py \
  --archive /tmp/saund-m1-official-audio/data1.tar.gz \
  --annotations /tmp/saund-m1-reference-sources/BallroomAnnotations \
  --destination /tmp/saund-m1-ballroom-pilot \
  --max-tracks 8 --research-only-ack
```

This extracts **only selected WAVs** after checksum and path checks, never overwrites the destination, and generates:

- `/tmp/saund-m1-ballroom-pilot/audio/*.wav` (private local files)
- `/tmp/saund-m1-ballroom-pilot/ballroom-development-DRAFT.json` (exact independent beat/downbeat labels, `rights_confirmed=false`, `split=development`, file hashes)

The `--research-only-ack` switch is an acknowledgment of the proposed intended use, **not a grant of rights**.

## Step 4 — actual metrics after use-rights review

The existing evaluator deliberately rejects the draft manifest. **Only after verifying intended dataset/audio use** should you copy the draft to a new private reviewed manifest, set `rights_confirmed=true`, and record the decision. Do not change the original draft:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-rhythm-benchmark.py \
  --manifest /path/to/private/reviewed-ballroom-development.json \
  --audio-root /tmp/saund-m1-ballroom-pilot/audio \
  --output /tmp/saund-m1-ballroom-first-measurement.json
```

Use no `--record-ledger` at first. Independently inspect the measured track-level errors and metric coverage before recording baselines. The evaluator defaults to deterministic beat analysis with neural models and network weights disabled. **Do not call this a held-out benchmark**; this small pilot is development data only and should help identify pipeline problems, not demonstrate generalization.

## Tests

```sh
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_prepare_ballroom_pilot.py
```

Four tests are included for official archive checksum refusal, bounded selection and replica exclusion, default-deny rights flags, and private destination protection. These have not yet run on the user's Mac; no actual official audio was acquired through the assistant.

## Evidence ledger

- Reference audits: GiantSteps Tempo 661 / 664 BPM labels; Ballroom 44,605 beats and 12,222 downbeats across 698 tracks.
- Prior tooling: 27 / 27 focused tests passed on the owner's Mac.
- Pilot implementation: source committed; **new 4 focused tests pending**.
- No local music was accessed by the assistant; **no real rhythm accuracy result or champion has been recorded**.
