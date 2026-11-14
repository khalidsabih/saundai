# Raveform independent EDM reference preparation (offline only)

Raveform's extracted `structures/beats/*.beat.csv` files contain `time,downbeat,section`.
The `downbeat` column is a **beat position within the bar**: value `1` marks
a downbeat; all `time` values are beats. The 1,423 filename track identifiers
were matched to `tracks.jsonl` in the operator's November 2026 extraction.

## Stage reference annotations (does not require audio)

From the **SAUND_AI repository root**, first find the extraction:

```bash
find datasets -path '*/raveform/structures/beats' -type d
```

The extraction may be nested at `datasets/fma/datasets/raveform/extracted/raveform`
because the original commands were run while inside `datasets/fma`.
Use the **actual** path from `find`, not a guessed path.

```bash
python scripts/prepare_raveform_independent.py \
  --raveform-root datasets/fma/datasets/raveform/extracted/raveform \
  --output-dir datasets/raveform/benchmark
```

Expected with no verified audio: `annotated: 1423`, `verified_audio: 0`,
`manifest: null`. It produces `labels/<track_id>.beats`,
`labels/<track_id>.downbeats`, and `raveform-candidates.json`. It **does not**
produce an evaluation-ready manifest without matching audio. It also removes
a stale previously generated manifest from the same output directory during
a labels-only run. `raveform-candidates.json` lists track IDs, titles,
durations, annotation paths, and label counts to support **predeclared**
independent recording selection.

## Verified-audio handoff (only after lawful acquisition)

**Do not automatically download copyrighted audio from streaming platforms.**
Acquire isolated matching recordings through sources that permit your intended
use. Verify recording identity/version, timing alignment and no overlap with
prior datasets. Do not correct offsets by looking at benchmark predictions.

Place recordings under `datasets/raveform/audio/`. For every preselected,
verified recording, calculate:

```bash
shasum -a 256 datasets/raveform/audio/example.wav
```

Create a local JSON map outside Git:

```json
{
  "tracks": [
    {
      "id": "EXACT_RAVEFORM_TRACK_ID",
      "audio": "example.wav",
      "sha256": "64_LOWERCASE_HEX_DIGITS_FROM_SHASUM",
      "source": "Document lawful source and the exact recording/version",
      "recording_verified": true,
      "alignment_verified": true,
      "verification_notes": "Document exact-version check, alignment check, overlap exclusion"
    }
  ]
}
```

The boolean checks are **human attestations**, not automatic proof of
the recording's identity or timing alignment. SHA-256 checks integrity
of the supplied bytes, not their relationship to the annotation.
Do not set these flags true without checking.

```bash
python scripts/prepare_raveform_independent.py \
  --raveform-root datasets/fma/datasets/raveform/extracted/raveform \
  --output-dir datasets/raveform/benchmark \
  --audio-root datasets/raveform/audio \
  --verified-map datasets/raveform/verified-audio-map.json
```

This emits `datasets/raveform/benchmark/raveform-independent-manifest.json`,
using `split=independent` and explicit audio/reference label paths. The
independent evaluator then runs on that file, **after** the operator confirms
the selected recordings are genuinely independent:

```bash
python scripts/benchmark_independent_bar_start.py \
  --manifest datasets/raveform/benchmark/raveform-independent-manifest.json \
  --output output/beat-integrity/raveform-independent-comparison.json
```

Never tune frozen hypotheses based on this holdout. Labels are only consumed
by the benchmark after autonomous phase inference. Research scoring windows
are 70 ms beats / 100 ms downbeats; **25 ms cross-deck phase-integrity
production safety remains unchanged**. No DJ engine, cache, model or live
mixing modifications are authorized by this workflow.

## Tests

```bash
python -m pytest -q tests/test_prepare_raveform_independent.py tests/test_independent_bar_start_comparison.py
```

The annotation preparation uses Python's standard library only.
