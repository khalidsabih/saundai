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
