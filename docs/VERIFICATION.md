# DJAI 4.0.0 verification — Professional Brain Core

> **Historical status:** This document records an earlier implementation/release milestone. It is retained for provenance and must not be used as the current runtime/version contract. See [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md) for current architecture, authority and schema sources of truth.


## Scope

This report verifies the 4.0.0 source tree implementing the professional-DJ-brain architecture. It distinguishes executable architecture from learned artistic skill that still requires real expert data.

## Backend regression suite

Pytest collection: **67 tests**.

All 67 passed in isolated groups:

- 31/31 brain, judge, data, API, decision, measurement and set-policy tests.
- 25/25 automatic-product, compatibility, personalization, parallel/determinism, EDM-CUE adapter and frontend-contract tests.
- 8/8 local source analysis, stem truth, serialization/cache and model-cache tests.
- 1/1 real pair-audio integration render.
- 1/1 real full-set audio integration render.
- 1/1 release-packaging replacement/cleanliness test.

The heavy audio/multiprocessing groups are deliberately run in fresh processes. Test-mode professional-brain fan-out is 24 analytical candidates -> 4 rehearsals -> 1 retry round; production/API defaults remain 48 -> 8 -> up to 2 rounds.

## Professional-brain checks covered

The suite covers, among other existing contracts:

- diverse candidate selection;
- exact rejection-memory matching;
- hard technical failure cannot be overridden by learned/artistic score;
- accepted professional judgement on a strong candidate;
- immutable TransitionProgram fingerprinting/validation;
- multidimensional SetState context;
- production dataset provenance default-deny behavior;
- synthetic technical-failure generation;
- DJAI Arena deployment gate and technical-regression veto;
- cache/serialization invalidation after schema changes;
- deterministic/parallel planning invariants;
- pair and full-set actual audio rendering;
- stem-derived truth/render behavior;
- API/version contract.

## Frontend

- `scripts/check-frontend.sh`: **22 TypeScript/TSX files, 0 syntax errors**.
- Frontend contract pytest group: **4/4 passed** (included in the 25-test group above).
- Full Next.js `npm run typecheck && npm run build` could not be completed in this execution environment because Docker is unavailable and npm dependency installation did not complete within the available registry window. `scripts/bootstrap.sh` performs the real Docker frontend typecheck/build on the target machine before declaring the project ready.

No successful full Next.js build is claimed by this report.

## Packaging / runtime

- Runtime/API/project version: **4.0.0**.
- Python support: 3.11 <= Python < 3.14.
- Wheel build succeeded with all new `djai.brain`, `djai.judge`, `djai.perception`, `djai.performance`, `djai.training`, and `djai.evaluation` packages included.
- Production dataset registry audit currently reports no policy blockers for the two *definitions* of production sources (`djai-expert-owned` and `djai-synthetic-failures`). This does **not** mean expert data has already been collected; individual production records still require the documented rights checks.

## Release cleanliness requirements

The packaged archive excludes:

- `.env` and secrets;
- `.venv` / `.cuevenv`;
- Python caches / pytest caches;
- build/dist/egg-info artifacts;
- frontend `node_modules` / `.next`;
- local music in `audio/`;
- generated renders/jobs in `output/`;
- private ratings and measurement logs;
- locally fitted `config/*.json` brain/scoring models.

Empty `.gitkeep` directories are restored for runtime locations.

## CUE-DETR boundary

CUE-DETR remains an optional research/reference cue sensor. It is not required by the professional brain and has no authority to bypass beat/source/execution safety. 4.0 provides an isolated reference environment script so its older ML dependency stack does not destabilize the core product.

This verification does not claim that the published CUE-DETR checkpoint produces useful cues for all tracks or that its weights/data are cleared for the user's intended commercial use.

## Learned-taste boundary

A fresh DJAI 4.0.0 installation has the complete professional-brain control loop, but **does not ship fabricated elite-DJ taste**. Pairwise preference authority and expert-imitation authority are intentionally bounded until real human/expert outcomes are collected and win held-out Arena comparisons.

That is a release requirement, not a missing safety check: professional taste must be earned from data rather than represented by hard-coded confidence.
