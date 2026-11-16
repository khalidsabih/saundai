# M1 rhythm benchmark — first automatic evaluator (research-only)

**Status:** source and focused tests committed; **Mac focused tests passed (2026-11-16): 11/11 across evaluator and registry**. Real dataset integration pending. This adds evaluation tooling to the isolated M0 research branch. It does not start M1 implementation changes, modify DJ analysis, touch Live or enable neural model authority.

## Data contract

Local private reference JSON (not committed) must have:
- `schema_version: saund-m1-rhythm-v1`
- `dataset_id: giantsteps_tempo` or `ballroom` (also allows private `local_34` with independently verified labels)
- `split: development`, `validation` or `test`
- `rights_confirmed: true` **only after explicit use-rights review**
- `items`: unique IDs, relative audio filenames, annotation source, reference BPM or beats or downbeats.

For event labels, `null` means **no independent ground truth** and must not be scored as F1=0. `[]` means a genuinely verified *zero-event* reference and is not a substitute for missing labels.

Example **schema only**; numbers and track are fictional fixture values:

```json
{
  "schema_version": "saund-m1-rhythm-v1",
  "dataset_id": "ballroom",
  "split": "development",
  "rights_confirmed": true,
  "items": [{
    "id": "fixture-001",
    "audio": "example.wav",
    "annotation_source": "verified independent Ballroom annotation",
    "bpm": 120.0,
    "beats_seconds": [0.5, 1.0, 1.5, 2.0],
    "downbeats_seconds": null
  }]
}
```

## Benchmark command

Once approved dataset audio and annotation manifest are acquired and put in an **isolated private directory**, run:

```sh
cd /tmp/saund-m0-dataset-registry
PYTHONPATH=src "$DJAI_WORKSPACE/.venv/bin/python" scripts/m1-rhythm-benchmark.py \
  --manifest /path/to/private/rhythm-dev.json \
  --audio-root /path/to/private/audio \
  --output /tmp/saund-m1-rhythm-dev-result.json
```

The CLI runs `djai.analysis.analyze_track` on each track with `DJAI_NEURAL_BEATS=0`, `DJAI_MODEL_ALLOW_DOWNLOAD=0`, `DJAI_BEAT_THIS_ALLOW_DOWNLOAD=0` and cue model off by default, using a separate Python subprocess for each track. This is **deterministic-path baseline**, not neural-model comparison. It reports:
- absolute BPM mean absolute error, no half/double-time substitution;
- one-to-one tolerance-matched beat F1 at 70 ms;
- one-to-one tolerance-matched downbeat F1 at 100 ms;
- track-level errors and explicit metric coverage.

No missing label is silently converted into a passing or failing score. Source audio is never committed. Output path uses exclusive creation and refuses overwrite. Invalid paths, rights status and annotations fail closed.

### Offline metric-path validation without audio analysis

The CLI accepts precomputed prediction JSON with:
```json
{
  "schema_version": "saund-m1-predictions-v1",
  "predictions": {
    "fixture-001": {
      "bpm": 122.0,
      "beats_seconds": [0.5, 1.02, 1.49, 2.05],
      "downbeats_seconds": []
    }
  }
}
```
All IDs must correspond to the reference manifest. This is for test harness verification; it **does not** prove real model accuracy.

### Ledger integration

Optional `--record-ledger --implementation deterministic-baseline --source-commit <sha>` invokes the existing `scripts/dataset-benchmark-registry.py record` command using only available numerical metrics. It requires a reviewed rights flag and explicit existing manifest and writes ledger data to local, gitignored SQLite. **No champion promotion occurs automatically.** Do not pass `--record-ledger` for invented fixture outputs or unapproved music.

## Local smoke gate

In the isolated worktree:
```sh
cd /tmp/saund-m0-dataset-registry
"$DJAI_WORKSPACE/.venv/bin/python" -m pytest -q \
  tests/test_m1_rhythm_benchmark.py tests/test_dataset_benchmark_registry.py
```

The first six focused tests check event matching, BPM MAE, missing downbeat truth, annotation validation and path safety. The existing five registry tests protect ledger cohort comparability and champion rules.

**Next:** verify datasets/rights, then import actual reference metadata and only after that run the first benchmark. Dataset research may be tracked in M0; real musical understanding improvement belongs to M1. No claims about data download, model improvement or real scores are made now.
