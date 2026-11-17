# M0 — read-only real-music crate inventory

Purpose: convert an existing local collection into a **draft**, hash-addressed, private scenario manifest without reading audio contents into GitHub or claiming genres/rights are verified.

**Use the local research repository, not production/live.** The published M0 tool is on draft PR #28 only; do not merge or switch a working branch. This step never edits the audio, updates grids/cues or plays music.

## 1. Extract the script into /tmp without changing an existing checkout

```sh
export DJAI_WORKSPACE="/Users/khalidsabih/Downloads/lab/SAUND_AI"
cd "$DJAI_WORKSPACE"
git fetch origin docs/m0-baseline-capability-audit-20261116
git show FETCH_HEAD:scripts/m0_freeze_crate.py > /tmp/saund-m0-freeze-crate.py
```

Use the explicit path to the private audio directory; the script recursively recognizes `.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, `.m4a`, `.aac`, `.ogg` and ignores symlinks. It hashes full audio bytes but doesn't decode them.

```sh
"$DJAI_WORKSPACE/.venv/bin/python" /tmp/saund-m0-freeze-crate.py inventory \
  --audio-root "$DJAI_WORKSPACE/audio" \
  --output /tmp/saund-m0-crate-draft.json
```

**Safety:** refuses to overwrite `/tmp/saund-m0-crate-draft.json` or create an output under the audio directory; if rerunning, choose `/tmp/saund-m0-crate-draft-v2.json`. This is a private manifest containing file names and local path hints, so do not commit it. A duplicate-content hash or inaccessible audio returns a nonzero status and must be resolved transparently.

## 2. Understand the output

The script reports count of tracks, failed scans, 12 unassigned scenarios and unreviewed rights. This is **not an error in music understanding**: human-confirmed genre, artist, provenance/rights and development/test assignment are intentionally blank to avoid false ground truth.

Each track holds a relative filename, immutable SHA-256 fingerprint, size, stable ID, null `genre`, `artist`, `recording_group`, `rights_status=unreviewed`, `split=unassigned`. Each fixed scenario `S01`–`S12` has empty `track_ids_in_order` and `status=unassigned`.

Assign the metadata based on the user's knowledge and permitted local research use, not guesses. The same artist/recording group must not appear across development and held-out test splits. Scenarios not covered must say `unavailable`, never `ready`.

## 3. Validate draft and later the fully reviewed manifest

```sh
"$DJAI_WORKSPACE/.venv/bin/python" /tmp/saund-m0-freeze-crate.py validate \
  --manifest /tmp/saund-m0-crate-draft.json
```

Strict mode catches missing rights clearance, missing genre/artist/recording groups, invalid IDs, unassigned scenarios, mixed splits and cross-split artist/recording leaks:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" /tmp/saund-m0-freeze-crate.py validate \
  --manifest /tmp/saund-m0-crate-draft.json --strict
```

A strict failure is **expected** until the owner reviews the actual music/usage rights and makes concrete scenario selections. Do not change `freeze_status` to frozen or mark M0 passed based on inventory counts.

## 4. Evidence and privacy

The user may share only the command's **aggregate terminal summary**, not the private manifest or audio itself. The team can then decide whether there are enough genre-diverse rights-cleared tracks to cover all 12 scenarios. We will not independently verify the rights of the user's songs from file names.

M0 closes only after the other local source/environment/cache observations, report audit, approved private crate and final owner signoff are recorded.
