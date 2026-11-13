# SAUND.AI modernization phases

Status: **Phases 1–3 complete; Phase 4 extraction and controlled three-track execution/PCM equivalence verified; benchmark exit evidence pending**. This plan governs engineering consolidation on top of
the existing Phase 13–15 implementation; it does not replace the product feature
roadmap in [ROADMAP.md](ROADMAP.md).

## Source of truth and invariants

- Product version: root `VERSION` (currently `2.0.0-rc.1`), synchronized with Python and frontend package metadata.
- Execution authority: deterministic audio feasibility, content-bound execution programs, and final rendered verification always outrank models or preferences.
- Live authority: NOW and committed NEXT are immutable; only FUTURE may be replanned, and stale/late work is discarded.
- Model authority: shadow/advisory/production are governed separately from model training and presence.
- Never change musical behavior merely to satisfy a stale literal, snapshot or hash assertion. First establish the intended contract.
- Each phase should be independently reviewable, with tests and a rollback boundary.

## Phase 1 — Establish a reliable baseline and migration map

**Scope:** categorize test failures (behavioral regression, outdated contract,
fixture defect, integration/environment); fix demonstrably incorrect fixtures;
add a focused CI lane for planner/Phase 13–15 regression checks; record canonical
runtime and source-contract ownership.

**Exit gate:** focused tests pass locally and in CI; remaining full-suite
failures are inventoried with explicit disposition rather than silently skipped.
Do not claim the full suite is green until it is.

## Phase 2 — Restore full-suite contract coherence

**Scope:** repair genuine timing, render, naturalness, and Headphones regressions;
replace brittle frontend source-text and transport-file hash assertions with
behavioral/API-contract checks; update historical schema/version expectations
only after verifying authoritative runtime behavior; align Phase 16's
`render=False` readiness semantics with the test's intended meaning.

**Exit gate:** supported Python unit/integration suite green; intentional
compatibility breaks called out in changelog; no hard-safety checks weakened.
Prioritize regression reproduction over changing assertions.

## Phase 3 — Canonical architecture and terminology

**Scope:** publish a current architecture document with the exact boundaries
among DJMove, TransitionPlan, immutable execution program, rehearsal result,
Executive commit, and live runtime state. Mark older 4.x/5.x verification and
release notes as historical. Create a source-of-truth table for schema, version,
model registry, safety, and planner modes.

**Exit gate:** documentation agrees with current APIs and execution code;
historical milestones remain available but cannot be mistaken for release truth.

## Phase 4 — Backend orchestration decomposition

**Scope:** extract graph-pair cache, sparse route search/configuration, sequence
compilation, plan assembly/reporting, and quality-loop integration from
`set_planner.py` into private cohesive modules. Decompose `api.py` by
router/feature, while preserving public imports and routes.

**Exit gate:** golden-plan/transition signature and exact-render regression
checks show equivalent behavior on fixed fixtures, except for separately
approved bug fixes. No unreviewed change in safety or model authority.

**Implementation checkpoint (2026-11-13):** The private planner modules now own
graph caching/workers, route search, complete-move lookahead and routing,
candidate/sequence compilation helpers, Professional Brain resolution plumbing,
final sequence safety/edge commit, reporting and quality-loop integration.
`api.py` retains endpoint implementations and compatibility imports while
four router modules own registration for all HTTP endpoints.

The operator verified the full supported Python suite green on
`phase-13-15-time-live-venue-hardening` after the router migration,
including repairs to source-text tests that incorrectly required decorators
to remain in `api.py`. The large-crate test also verifies the extracted
order optimizer respects injectable legacy planning hooks.

For exact operator commands and the fixed pre-extraction commit, see
[PHASE4_EQUIVALENCE_GATE.md](PHASE4_EQUIVALENCE_GATE.md).

**Phase 4 controlled-fixture equivalence passed (2026-11-13):** the operator
compared pre-extraction commit `cb731ea0` against the current branch with the
same three real tracks, analyzed evidence, scoring-model snapshot and automatic
route configuration. The track order, plan/execution signatures and program
fingerprints matched. With the existing measured quality loop enabled on both
branches, the final safety checks passed and decoded rendered PCM hashes
**MATCHED**. Both branches independently rejected the unrehearsed fixture for
the same hard 16.1 dB RMS transition-boundary defect; no safety limit was
weakened. The archived operator evidence is identified in
[PHASE4_EQUIVALENCE_GATE.md](PHASE4_EQUIVALENCE_GATE.md).

**Phase 4 controlled benchmark checkpoint (2026-11-13):** all 13-track
full-planner source hashes, analysis digests, scoring-model fingerprint, track
order, transition execution signatures, fallback count, and emergency count
**MATCH** between baseline and refactored code. Isolated cold-graph work
measured 19.95 s vs 20.07 s, while cold end-to-end planning measured
518.28 s vs 530.04 s and peak RSS 3.320 GB vs 3.442 GB. These modest
single-run deltas do not establish a statistically reliable regression or
speedup. See the evidence and constraints in
[PHASE4_EQUIVALENCE_GATE.md](PHASE4_EQUIVALENCE_GATE.md).

**Phase 4 refactor exit: fixture-scoped correctness and benchmark evidence
verified.** The operator's supported Python suite, three-track rehearsed
render PCM identity, and 13-track full-planner identity/performance checks
support proceeding to Phase 5 frontend consolidation. **Not a release or
music-quality approval:** 11 fallback transitions and 9 emergency transitions
out of 12 large-crate edges remain on both branches; investigate and validate
them with real music, rendered-audio safety, and listening in Phase 6.
Do not declare Phase 5 complete on the strength of these Python gates.

## Phase 5 — Frontend consolidation

**Scope:** delete only verified unused empty/legacy components; move
`globals.css` ownership into scoped feature styles; consolidate redundant
Create/Studio/Live state and API adapters; preserve accessibility and truthful
engine-state projections. Avoid using literal visual copy as a behavioral test.

**Exit gate:** typecheck, build, critical Playwright flows, keyboard/mobile
checks, upload, planning, review, audition and Live smoke tests pass.

## Phase 6 — Release evidence and real-music quality gate

**Scope:** unify CI jobs and branch protection requirements; benchmark cached
and cold 13-track crates with planner traces, timing, candidates, rehearsal,
safe fallback and resource metrics; record Phase 16 outputs. Validate venue
watchdog/live recovery separately with fault injection; maintain rights and
model-governance release checks.

**Exit gate:** required checks green, comparable benchmark evidence published,
no hard-audio-safety regression, real-music outcomes reviewed by listening
rather than synthetic scores alone.

## Execution policy

Finish and review each phase before treating the next as complete. Refactors
must be reversible; record baseline and after metrics for expensive planning.
Use `python -m pytest` from the supported virtualenv, and keep nightly/heavy
integration jobs separate from quick pull-request checks.
