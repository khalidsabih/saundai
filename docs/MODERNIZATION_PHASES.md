# SAUND.AI modernization phases

Status: **Phase 2 complete; Phase 3 in progress**. This plan governs engineering consolidation on top of
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
