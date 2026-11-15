# SAUND BPM validation protocol (observation-only)

This benchmark compares the original librosa tempo, Beat This! tempo, and
research-only arbitration results. It never changes playback or DJ decisions.

## Add reference tracks

Create a separate local JSON list rather than modifying or guessing BPM labels
for music without verification:

```json
[
  {
    "filename": "verified-song.wav",
    "reference_bpm": 123.0,
    "reference_quality": "independently_verified",
    "verification_method": "manually count 64 consecutive beats over a known time span; checked by a second annotator",
    "source": "manual listening and annotation"
  }
]
```

The `filename` must match the audio basename in `--audio-root`. To include
the existing two reference tracks, copy their entries into the same local JSON
file with `reference_quality: "approximate"`. Do not upgrade the existing
78 BPM and 123 BPM labels to verified without further work.

The `verification_method` is **self-reported metadata**, not an independent
audit by SAUND. Confirm the reference pulse's intended metrical level
(half/double BPM) manually and do not infer reference BPM from either estimator
under test. Avoid labeling confidence from `tempo_stability`.

Run the benchmark from the isolated repository worktree:

```sh
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" \
  -m djai.intelligence.tempo_benchmark \
  --audio-root "$DJAI_WORKSPACE/audio" \
  --references /absolute/path/to/verified-bpm-references.json \
  --output /tmp/saund-bpm-validation.json
```

## Interpret the output

- `comparison`: per-track error for baseline deterministic BPM and neural BPM.
- `arbitration`: coverage and retain/abstain/review-candidate recommendation.
- `prospective_impact`: comparison of a hypothetical review candidate against
  the baseline **without ever applying that candidate**.
- `validation`: metrics **only for explicitly independently verified labels**.
  Approximate references are excluded, not silently promoted.
- `false_correction_rate`: share of simulated review candidates (verified
  references only) that would move a baseline within 2 BPM to more than 2 BPM
  away from reference. It is `null` when there are zero simulated candidates.
  This is not a deployed failure rate.
- `ready_for_live_adoption`: always `false` in this research harness.

Keep genre, beat emphasis, sparse intros/breakdowns, and half-time/double-time
examples represented in the reference set. Deduplicate tracks and preserve an
untouched test set before tuning thresholds. Two approximate track labels
cannot support a general accuracy claim. Beat and downbeat placement are
not evaluated by this BPM-only benchmark.

## Build a reproducible 24-track annotation sheet (local only)

The following workflow lists local filenames without decoding tracks or
guessing BPM, genre, beat phase, or verification quality. It samples across
folders for diversity where possible; **folders are not a proxy for genre**.
Record actual genre, time signature / intended metrical level and difficult
breakdowns manually in the `notes` field. For a reliable holdout, freeze the
sample before inspecting model predictions.

```sh
cd /tmp/saund-brain-test
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" \
  -m djai.intelligence.tempo_reference_workflow prepare \
  --audio-root "$DJAI_WORKSPACE/audio" \
  --count 24 \
  --seed saund-bpm-v1 \
  --output /tmp/saund-bpm-annotation-sheet.csv
```

The CSV includes: `filename`, `reference_bpm`,
`reference_quality`, `verification_method`, `source`, `genre`, `notes`.
All fields except filename start blank. A user should set
`reference_quality=independently_verified` only after manually checking the
**intended** BPM against an external, credible measurement independent of
SAUND's estimators, recording exactly how it was checked. Mark a merely
estimated BPM `approximate` instead. Check whether a reported BPM corresponds
to half-time or double-time musical meter. A filename with duplicate basenames
in different folders is excluded because the benchmark's file lookup by
basename cannot disambiguate it. The CLI refuses to overwrite existing outputs.

Once annotated, export **only completed rows** to local JSON:

```sh
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" \
  -m djai.intelligence.tempo_reference_workflow export \
  --csv /tmp/saund-bpm-annotation-sheet.csv \
  --output /tmp/saund-bpm-verified-sample.json
```

A partially annotated row fails instead of being silently treated as
verified. Fully blank pending rows are counted and omitted; no BPM is made up.
The exported file is suitable as the benchmark's `--references` argument.
Do not commit raw audio, local paths, personal listening notes, or annotation
files unless you explicitly choose to share them. If fewer than 24 uniquely
named audio files exist, the sheet contains fewer than 24 tracks.

These procedures do not establish independent scientific validation by
themselves: the documented verification method is provided by the annotator.
To evaluate real-world reliability, keep a holdout subset untouched while
adjusting thresholds or models.

## 17-track diagnostic follow-up (November 2026)

The 17 supplied reference tempos are **approximate** and cannot support
claims about independent accuracy, beat phase or musical downbeat. The
observational report showed 17 arbitration abstentions, including seven
consistent Lane 8 Fingerprint onset windows and mixed ~122/~81 BPM onset
peaks in Cola. In addition, the fallback-free benchmark inference marked
the full-grid authority gate as passed for Toc Toc 4, despite strong
disagreement with its approximate BPM.

The benchmark calls `predict_beats(path)` **without fallback beats**;
the live analysis code instead calls it with deterministic beats and
downbeats. Consequently the historical `full_grid_adopted` benchmark
field records only a model-gate decision in that different call context,
**not observed live grid adoption**. Inspect
`full_grid_adopted_in_live_analysis` (always `null` unless a separate
live trace supplies evidence) and `grid_adoption_risk`. A flagged
risk requires reproduction with the actual fallback grid; never infer a
live replacement from benchmark-only output.

Strong onset estimates at ~2:3 ratios can be periodic subdivisions.
`possible_two_to_three_subdivision_windows` and associated indices
are descriptive, not proof of a true subdivision or corroboration of a
beat grid. Those windows continue to count as contradictions for the
strict arbitration gate; its thresholds remain unchanged.

When enough strong, conflict-free audio windows match the existing
**exact** deterministic metrical BPM (within the existing tolerance),
the research decision may now say `retain`, even if octave-related
alternatives are representable. This is not a tempo correction. Mere
half/double agreement with a *different* baseline BPM does not qualify,
and beat/downbeat alignment remains unvalidated.

All these results are observation-only. No changes are made to the
running app, its beat grids or transition cues.

## Reproduce the neural full-grid gate with fallback beats (read-only)

The original BPM benchmark invokes Beat This! without a fallback beat grid.
The actual `analyze_track` path first computes deterministic beats and
downbeats and passes both to `predict_beats`. This command repeats the
**pre-neural deterministic beat/downbeat extraction** and calls the neural
predictor with the same fallback argument names, without running the live
analyzer, replacing grids, or writing to the running app:

```sh
cd /tmp/saund-brain-test
export DJAI_WORKSPACE="/Users/khalidsabih/Downloads/lab/SAUND_AI"
export DJAI_BEAT_THIS_CHECKPOINT="$DJAI_WORKSPACE/.djai-cache/neural/beat-this-final0.ckpt"
PYTHONPATH="$PWD/src" "$DJAI_WORKSPACE/.venv/bin/python" \
  -m djai.intelligence.neural_grid_gate_audit \
  --audio-root "$DJAI_WORKSPACE/audio" \
  --filename "18. Lmoutchou - Toc Toc 4.mp3" \
  --filename "Camelphat_&_Elderbrook_‘Cola’.wav" \
  --output /tmp/saund-grid-gate-audit.json
```

Outputs include `neural_grid_gate_would_pass`,
`neural_fallback_beat_agreement`, baseline and neural BPM, and
`requires_human_review` for a passing grid gate with >2% BPM
disagreement. This is a **live-signature reproduction**, not direct
evidence of what an already-running server did. No beat-grid or
downbeat accuracy is established. The tool fails if filenames are
ambiguous, rather than choosing a duplicate song silently, and refuses
to overwrite a prior JSON output. It reads and decodes local audio but
never changes the original files.

Note: This checks the *existing* full-grid selection behavior; the
separate tempo-only adoption switch remains off by default.
