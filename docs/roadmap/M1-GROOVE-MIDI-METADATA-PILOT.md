# M1 Groove MIDI v1.0.0 — metadata-first permissive dataset lane

**Status:** initial source registry, importer and five fixture tests committed; Mac execution and official download audit pending. No trained models, no real prediction scores, no audio downloaded by the assistant.

## Official source and license

[Google Magenta Groove MIDI](https://magenta.tensorflow.org/datasets/groove) contains 1,150 human-drummed MIDI performances (~13.6 hours), with per-performance BPM, drummer, session, music style, time signature, beat/fill category and official train/validation/test split. The publisher lists it as **CC BY 4.0**, requiring attribution. This is not the same as a blanket assertion about third-party samples or downstream compositions. Cite Gillick et al., *Learning to Groove with Inverse Sequence Transformations*, ICML 2019.

Download options:
- **MIDI and metadata only, 3.11 MB:** `https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip`, official SHA-256 `651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e`.
- **Full MIDI, metadata, aligned drum WAV, 4.76 GB:** `https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip`, official SHA-256 `21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9`.

**Start with the MIDI-only archive**. Unlike Ballroom, this is a publisher-listed CC BY 4.0 dataset. Keep attribution and license review documented.

## First local step: 3 MB download and reproducible audit

```sh
mkdir -p /tmp/saund-m1-groove
curl -fL --retry 3 \
 'https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip' \
 -o /tmp/saund-m1-groove/groove-v1.0.0-midionly.zip

cd /tmp/saund-m1-evaluator
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-groove-metadata.py \
 --source /tmp/saund-m1-groove/groove-v1.0.0-midionly.zip \
 --output /tmp/saund-m1-groove/official-metadata-audit.json
```

The script checks the exact publisher SHA-256 and parses official `info.csv` directly inside the ZIP without extracting MIDI/audio. It reports row counts and split/style coverage and detects drummer/session group overlap across published splits. Does not mark source rights or music-analysis benchmark readiness automatically.

## Honest label boundaries

`bpm` is the set metronome tempo of an expressive drum performance, appropriate as a *tempo* reference. MIDI **drum note-on events are not identical to beat or downbeat ground truth**. This importer intentionally sets `beats_seconds=null` and `downbeats_seconds=null` in any generated M1 manifest. It cannot create a SAUND audio-analysis result from the 3 MB MIDI-only ZIP: original aligned audio must also be present and matched exactly.

The source's official `train` maps to SAUND `development`, `validation` maps to `validation`, and `test` remains `test`—never reassign a held-out test performance to training. Check the reported drummer/session overlap before claiming recording-artist-disjoint evaluation.

Once full aligned audio is separately acquired and its intended use reviewed, an unapproved manifest may be built explicitly with:

```sh
"$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-groove-metadata.py \
 --source /tmp/saund-m1-groove/groove-v1.0.0.zip \
 --audio-root /private/path/extracted-groove-WAV-root \
 --official-split train \
 --manifest-output /tmp/saund-m1-groove/development-DRAFT.json
```

The audio root must contain exact audio paths as listed in `info.csv`; a manifest remains `rights_confirmed=false`. The `scripts/m1-rhythm-benchmark.py` evaluator accepts source `groove_midi` and will compute **tempo MAE only** after explicit rights review, not beat or downbeat F1. Audio/model source alteration and production promotion remain prohibited.

## Focused local check

```sh
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q tests/test_m1_groove_metadata.py
```

There are five new tests; prior 27 focused tests already passed. Do not rerun unchanged old tests needlessly. Do not download 4.76 GB without a specific need.

**First gate:** verify the publisher's 3 MB ZIP integrity, metadata counts and official splits. Real audio analysis remains a separate gate.
