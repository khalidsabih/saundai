# M1 — GiantSteps Tempo and Ballroom reference import

**Status:** importers and focused tests **verified on owner's Mac (2026-11-16): 18/18 passed**, dataset/audio acquisition and real musical metrics pending. No audio was downloaded, no training data was created, and no engine/model authority changed.

## Verified upstream formats

- [GiantSteps Tempo](https://github.com/GiantSteps/giantsteps-tempo-dataset): `annotations_v2/tempo/<id>.LOFI.bpm` is a single numeric BPM value (e.g. `1030011.LOFI.bpm` is `127.0`). The importer defaults to **v2** instead of silently combining the older `annotations/tempo/` labels. Invalid or missing/zero BPM values do not become fabricated rhythm references.
- [CPJKU BallroomAnnotations](https://github.com/CPJKU/BallroomAnnotations): `<track>.beats` lines are `beat_time_seconds beat_position_in_bar`, e.g. `0.8499773240 1`, where **position 1 indicates an annotated downbeat**. Source README says original audio is not included and notes four exact recording replicas and nine recording replicas; leakage-proof splits require de-duplicating/collapsing these recording groups (not just distinct file names). Silent/non-musical intros are unannotated, so matching must honor annotation coverage when reporting results.
- These GitHub repositories document annotation formats; they **do not automatically grant rights to audio** and user must review permitted uses. Avoid mixing annotation vintages or assuming tempo tags are downbeat ground truth.

## Import with locally obtained, rights-reviewed files

From the isolated branch checkout:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-import-references.py \
  --dataset giantsteps_tempo \
  --annotations /private/path/giantsteps-tempo-dataset \
  --audio-root /private/path/giantsteps-audio \
  --output /tmp/saund-m1-giantsteps-dev-draft.json

"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-import-references.py \
  --dataset ballroom \
  --annotations /private/path/BallroomAnnotations \
  --audio-root /private/path/ballroom-audio \
  --output /tmp/saund-m1-ballroom-dev-draft.json
```

Audio path joining is **exact basename matching, ignoring file extension and case**, and refuses multiple files with identical basenames; no fuzzy artist matching or fabricated references. GiantSteps v2 label `1030011.LOFI.bpm` expects an audio basename `1030011.LOFI`; Ballroom `Albums-AnaBelen_Veneo-01.beats` expects audio basename `Albums-AnaBelen_Veneo-01`, with optional subfolders. Manually rename/link in a private audio directory only after independently verifying any actual filename mismatch.

Both commands create a **new private draft manifest**, intentionally containing:
- `rights_confirmed: false` (must be reviewed explicitly);
- `split: development` provisional, **not a held-out split**;
- `bpm` for GiantSteps; `beats_seconds` and `downbeats_seconds` for Ballroom; missing labels set to `null`;
- `annotation_source` with source dataset, version and relative annotation path;
- no audio bytes, no automatically inferred genre, no inferred per-track BPM for Ballroom.

No output overwrite is allowed. If no annotations/audio match, import fails instead of creating an empty passing benchmark.

## Only after the owner reviews licenses and split leakage

Copy a draft manifest to a separately named private reviewed manifest, set `rights_confirmed: true` only where the intended research use is allowed, and confirm audio matching and independent labels. For the first baseline use only `development`, not `test` until recording/artist groups are separated.

Then run the existing evaluator (without ledger or promotion initially):

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-rhythm-benchmark.py \
  --manifest /private/path/reviewed-giantsteps-dev.json \
  --audio-root /private/path/giantsteps-audio \
  --output /tmp/saund-m1-giantsteps-first.json
```

This should report tempo MAE; it must show **beat/downbeat F1 = null**, because GiantSteps Tempo does not provide beat timestamps. For Ballroom, beat and downbeat F1 are evaluated, while tempo MAE remains null absent independent BPM labels.

To keep the evaluation honest, do **not** benchmark the same file with many small tweaks and call it a held-out test. Fix/iterate on `development`; freeze `test` after artist and recording-duplicate protection.

## Tests

```sh
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q \
  tests/test_m1_import_references.py \
  tests/test_m1_rhythm_benchmark.py \
  tests/test_dataset_benchmark_registry.py
```

Seven importer tests cover actual published formats, missing labels, unsafe or ambiguous matches, no-output overwrite and proper private manifest flags. Combined with the previously passing 11 tests, expected focused count is **18**, passed on owner's Mac, 2026-11-16.

**Next gate:** importer-focused tests pass, then source/audio acquisition and rights review; only then first automatic real-music metric. Dataset availability and the current 34-track private crate do not automatically authorize publication, commercial use or ML training.
