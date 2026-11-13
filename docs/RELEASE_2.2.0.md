# DJAI 2.2.0 release notes

> **Historical status:** This document records an earlier implementation/release milestone. It is retained for provenance and must not be used as the current runtime/version contract. See [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md) for current architecture, authority and schema sources of truth.


DJAI 2.2.0 implements Phase 1 of the automatic-quality roadmap: the final transition is no longer selected only from formulas. DJAI renders a small analytical candidate set, measures the audio that was actually produced, rejects objective failures, and saves the best passing plan.

## New modules

- `candidate_renderer.py`
- `measurement.py`
- `measurement_store.py`

## Main behavior changes

- Pair planning uses top-K render-and-measure selection by default.
- Full-set graph construction remains analytical.
- Final path compilation measures top-K candidates at each selected junction.
- Both source tracks are decoded once per measured pair.
- Measurement output is attached to `TransitionPlan` and survives JSON round trips.
- Every candidate measurement can be appended to JSONL for future learning.
- Logging failures are reported in metadata but do not invalidate usable audio.
- API measurement-log paths are workspace constrained.

## Objective metrics

- loudness continuity;
- spectral continuity;
- low-frequency cleanliness;
- stereo phase safety;
- clipping and sample-boundary artifacts;
- unexpected silence.

## Not included yet

- automatic never-fail fallback ladder;
- stem-derived ground truth;
- automatic learning from measurement records;
- full-set target energy curves;
- production streaming.
