# Phase 5 frontend consolidation — initial read-only audit

Status: **inventory checkpoint; no frontend behavior or CSS changed**.

## Verified ownership map

- `frontend/app/page.tsx` is a single client-side experience router. It owns the
  local `"create" | "studio"` view selection and wraps both views in
  `SessionProvider`. Create and Studio are **not** separate Next.js page routes
  under `frontend/app/create` and `frontend/app/studio`.
- `frontend/components/SimpleExperience.tsx` implements the Create entry
  experience, reusing `LivePlayer`, `BrainWorkspace`,
  `FinishedSetPlayer`, product shell, music crate, intent, build-progress,
  and account-dialog components.
- `frontend/components/ProWorkspace.tsx` owns Studio orchestration and
  integrates `TrackLibrary`, `SetBuilder`, `PlanView`,
  `TransitionInspector`, `RenderPanel`, `ControlDeck`,
  `StudioIntelligenceWorkbench`, job progress and accounts.
- Both experiences depend on `frontend/lib/session.tsx`, the common
  authenticated/user-preferences context.
- Both experiences call `frontend/lib/api.ts` (currently ~22 KB),
  containing fetch/error semantics and transport APIs. Preserve request
  behavior, response models, cancellation semantics, and current session
  authentication while deduplicating any repeated feature hooks.
- `frontend/app/layout.tsx` imports five global stylesheets in order:
  `globals.css` (~223 KB), `design-tokens.css` (~2.4 KB),
  `autonomous.css` (~62 KB), `studio-intelligence.css` (~22 KB),
  `product-system.css` (~27 KB). The import order is part of the
  current cascade and **must not be shuffled as an unreviewed cleanup**.
- `frontend/package.json` provides `npm run typecheck`,
  `npm run build`, `npm run e2e`. Playwright config has Chromium,
  Firefox, WebKit, and iPhone 13 projects. Playwright's `webServer`
  starts `npm run start`, which depends on a built Next.js application.

## Boundary and risk assessment

1. **First audit dependency/test references.** A component with no direct
   reference in one file is not necessarily unused; dynamic imports, story
   demos, tests, CSS selectors, and client/server boundaries matter. Never
   delete based on filenames alone.
2. **Prefer extracting shared API request and cancellable-job behavior only
   after behavior tests exist.** Both experiences import service functions
   directly; changing error propagation or cancellation can affect truthful
   engine-state reporting.
3. **Treat Create and Studio as different UI workflows over a shared
   backend authority.** Do not unify their state by bypassing the current
   `SessionProvider`, job lifecycle or authoritative plan status.
4. **Split CSS in slices by verified ownership, preserving selector specificity
   and import ordering.** Begin with an inventory of unique selectors and
   overlap across sheets; migrate one feature at a time with visual
   regression checks across desktop and mobile.
5. **Live controls are safety-sensitive.** Preserve NOW/committed NEXT
   immutability; UI events cannot bypass back-end Executive authority.
6. Phase 4's 13-track fixture showed 11 fallback edges and 9 emergency
   transitions on both versions; this remains a separate Phase 6
   music-quality/release blocker, not a frontend regression signal.

## Reproducible local baseline (before editing)

```bash
cd "$SAUND_MAIN/frontend"
npm ci
npm run typecheck
npm run build
npm run e2e
```

Note: `npm run e2e` requires the backend/service fixtures expected by the
tests; distinguish missing running services from frontend behavior failures.
If tests require separate backend startup or seeded data, record those exact
requirements before treating the result as a regression. Preserve the full
Playwright report and note any pre-existing failures. Include keyboard/mobile,
upload, planning, review, audition and Live flows in the exit gate.

## Initial incremental implementation sequence

1. Capture typecheck/build/E2E baseline on the current branch.
2. Inventory API call sites and job polling/cancellation across
   `SimpleExperience`, `ProWorkspace`, and `lib/api.ts`; add focused
   request-contract tests around the shared helper before extraction.
3. Inventory overlapping CSS selectors across the five imported sheets.
   Extract one low-risk feature slice without changing the cascade.
4. Audit truly unused components using import graphs **and** runtime/test
   verification, then remove in separate reversible commits.
5. Rerun all baseline checks after each slice; review keyboard, mobile
   and truthful Live state projections before Phase 5 sign-off.

**Phase 5 is in progress, not complete.** This audit does not establish
that tests were executed or that any component is dead code.

## First Playwright baseline and accessibility repair (2026-11-13)

The operator reported `npm run typecheck` and `npm run build` passing.
The frontend npm audit reported 3 advisories (2 high, 1 critical), not
yet triaged. A locally rebuilt frontend redirected its API to
`http://127.0.0.1:8000`.

First Playwright run showed missing mocked Create track selectors,
a Next.js route announcer collision with unscoped `role=alert`,
and Training Lab initial API fan-out requiring a stable fixture.
Focused fixture updates were made in `frontend/e2e/release.spec.ts`.

Second Playwright run: **17 passed / 19 failed** across Chromium, Firefox,
WebKit and mobile. The repeated Create/Brain/Live failure remained at the
track selector, even after targeting `role=listitem`. Inspection found
`CrateSelector` overrode native `<button>` semantics with
`role=listitem`, making it a non-button accessible control and removing
the expected button accessible name. The native button semantics have now
been restored and test locators returned to `role=button`; this is an
accessibility correction, not a new musical behavior.

Several mobile assertions expected deliberately hidden responsive
metadata: the opening system label, Training Lab header metadata and
readiness status topbar. Test assertions now check for truthful text
presence instead of incorrectly requiring these desktop-specific labels
to be visible on mobile. Core headings, readiness cards and interactive
workflow assertions remain.

**Third run pending:** rerun `npm run typecheck`, `npm run build`,
`npm run e2e` after pulling these commits. The current suite is **not
claimed green** until the operator reports results. Audit dependency
advisories separately from Playwright failures.

## Third Playwright run (2026-11-13)

Operator reports **28 passed / 8 failed** across Chromium, Firefox, WebKit
and mobile. The remaining eight are the same two named scenarios on
all four browser projects:

- `live performance makes NOW, NEXT, FUTURE and the Executive the product surface`
- `Brain is a first-class autonomous system view and Studio is separate`

The Brain test expected the obsolete heading `Complex underneath` whereas
the verified `BrainWorkspace.tsx` heading is
`See what SAUND actually knows.`. The heading assertion was updated;
the remaining architecture and state tests are untouched. The Live
scenario's error line was not included in the provided summary. Keep it
open and inspect one targeted Playwright failure before editing Live
assertions or runtime behavior. Repeat tests to verify all changes.

## Focused Chromium Live/Brain diagnostic (2026-11-13)

The operator ran two targeted Chromium scenarios: **Brain passed**,
**Live failed**. Live's failure was a Playwright strict-mode locator
collision: the `NEXT` label is displayed in both the Executive state and
the labelled Next/future horizon. The test asserted an unscoped
`getByText("NEXT", { exact: true })` and therefore matched two legitimate
nodes. The test now scopes NEXT to the named horizon region. The
live-telemetry, execution-authority and safety logic were not modified.

Run the targeted scenarios, then the complete four-project suite.
Neither a clean Live test nor an all-browser pass is claimed yet.

## Fourth full Playwright run (2026-11-13)

Operator-reported result: **35 passed / 1 failed** across the four projects,
with supported TypeScript check passing. The sole remaining failure was the
mobile Live test at the `Afterglow` next-track assertion. The unscoped
`getByText("Afterglow", { exact: true }).first()` selected a hidden compact
label in the responsive Live UI. The component also renders the authoritative
next-track title as `<h2>` inside the labelled `Next and future horizon`
section. The assertion was changed to target that heading within the
horizon, retaining a meaningful visible-mobile check rather than allowing
a hidden label to satisfy it.

**Verification remains pending** until the next complete Playwright run.

## Green browser baseline and first module extraction (2026-11-13)

Operator reports `npm run typecheck` passed and **36 / 36 Playwright
E2E tests passed in 37.3 s**, across Chromium, Firefox, WebKit and
mobile. The previously reported build also passed, but the latest
successful test command did not include a fresh production build.

The first implementation slice extracts the unchanged browser request
transport into `frontend/lib/api-transport.ts`. It preserves
`API_BASE` normalization, same-origin URL composition, `no-store`,
`credentials: include`, overriding caller `RequestInit`, JSON parsing
fallback, and HTTP error message/status/detail behavior. The public
`API_BASE` export, endpoint functions and `JobCancelledError` remain
available through `frontend/lib/api.ts`. No API route signatures,
session state, job policy, UI, live control, or CSS ownership changed.
This is a code-organization change, not a performance optimization.

**Post-extraction verification pending:** rerun
`npm run typecheck`, `SAUND_INTERNAL_API_URL=http://127.0.0.1:8000 npm run build`,
and `npm run e2e`. Do not mark the extraction verified until these
commands pass on the updated branch. The 3 npm dependency advisories
previously reported remain a separate triage item.

## Verified post-extraction gate (2026-11-13)

Operator pulled the API transport extraction and reported all of:
- `npm run typecheck` **passed**;
- `SAUND_INTERNAL_API_URL=http://127.0.0.1:8000 npm run build` **passed**;
- `npm run e2e` **36/36 passed in 37.4 s** across four browser projects.

The first code-organization slice is therefore verified within the
existing test coverage. Next candidate review identified `waitForJob`
(and its SSE/diagnostic polling/fallback transitions) as shared
between Create and Studio; the event lifecycle and cancellation/error
semantics are safety- and state-sensitive. Do **not** move or simplify
job progress handling until focused tests assert completed, failed,
cancelled, SSE disconnect, and HTTP polling fallback outcomes.
The same caution applies to Live telemetry. No such follow-on
job lifecycle extraction has been performed yet.

Known outstanding tasks include package audit advisories,
Next.js standalone/tracing-root warnings, CSS ownership inventory,
and the additional frontend modules. Phase 5 remains in progress.

## Second implementation slice: shared job lifecycle (2026-11-13)

After operator verification of the previous transport slice (**36/36 E2E,
production build and typecheck passing**), a small, reversible
organizational extraction moved the existing `getJob`, `cancelJob`,
`pollForJob`, `waitForJob`, and `JobCancelledError` implementations
from `lib/api.ts` into `lib/api-jobs.ts`. `lib/api.ts` re-exports the
same public names, so Create and Studio imports are unchanged.

The SSE event processing, 10-second diagnostic poll, malformed-event
handling, final authoritative GET, HTTP fallback on disconnect,
cancellation behavior, status and errors are intentionally unmodified
by this extraction. Live telemetry and executive control remain separate
and untouched.

The focused `npm run test:jobs` regression harness compiles the **real**
`api-jobs.ts` module in memory with TypeScript, supplies fake transport,
clock, and EventSource via a VM, and asserts:

- already completed / failed / cancelled initial states;
- polling completion and cancelled terminal state;
- SSE progress followed by authoritative final job GET;
- malformed SSE progress and disconnect fallback to polling;
- failed terminal SSE job;
- explicit POST cancellation;
- diagnostic polling and terminal completion.

**Verification gate OPEN:** after pulling the commits run:

```bash
cd "$SAUND_MAIN/frontend"
npm run typecheck
npm run test:jobs
SAUND_INTERNAL_API_URL=http://127.0.0.1:8000 npm run build
npm run e2e
```

These changes are not yet claimed passing on the operator's machine.
The new tests are focused code-level contracts and complement, but do not
replace, the Playwright browser suite or backend verification.

## Verified post-job-extraction gate (2026-11-13)

Operator pulled the shared job lifecycle extraction and reported:
- `npm run typecheck`: **passed**;
- `npm run test:jobs`: **9/9 passed**, no failures or skips;
- `SAUND_INTERNAL_API_URL=http://127.0.0.1:8000 npm run build`: **passed**;
- `npm run e2e`: **36/36 passed in 36.7 seconds**, across Chromium, Firefox, WebKit and mobile.

The `frontend/lib/api-jobs.ts` extraction is verified under these
focused code-level and browser regression gates. This does **not**
independently certify production audio quality, live safety, or every
job race/disconnect scenario; the existing backend and music-quality
gates remain authoritative.

Next.js continued reporting the external lockfile /
`outputFileTracingRoot` warning and the `next start` versus
`output: standalone` warning. Both are outstanding deployment/test
harness cleanup, distinct from the passing frontend regression suite.
Previously reported npm dependency advisories also remain open.

## Dependency security patch and standalone-server harness cleanup (2026-11-13)

The operator reported successfully rebasing and pushing the dedicated
frontend production dependency patch. The branch advanced to
`a4469d3` (security patch), following `ff1e921`.
GitHub's committed lockfile now contains Next.js **16.4.6**,
sharp **0.35.5**, and source-map-js **1.2.2**. The operator reported
the local audit and regression verification **PASSED**, but did not
paste their detailed post-upgrade test counts; do not present them as
independently observed logs.

The next configuration-only slice explicitly sets both
`outputFileTracingRoot` and `turbopack.root` to the frontend build
working directory to avoid inferring an unrelated external lockfile.
Playwright now uses the generated standalone production server,
rather than unsupported `next start` with `output: "standalone"`.
Because standalone output intentionally omits copied static assets,
`frontend/scripts/prepare-e2e-standalone.mjs` mirrors the same
`public` and `.next/static` directories the production Dockerfile
already copies. No UI, service API, job or audio control code changed.

**Infrastructure cleanup verification pending**. Rebuild before E2E
because the Playwright server now uses `.next/standalone/server.js`:

```bash
cd "$SAUND_MAIN"
git pull origin phase-13-15-time-live-venue-hardening
cd frontend
npm audit --omit=dev
npm run typecheck
npm run test:jobs
SAUND_INTERNAL_API_URL=http://127.0.0.1:8000 npm run build
npm run e2e
```

Review any Next.js diagnostics and compare the complete test results
before calling Phase 5.4 green.

## Phase 5.4 operator verification (2026-11-13)

The operator reports that the full requested Phase 5.4 verification sequence **passed** after pulling the standalone-server harness and explicit Next.js root configuration changes. The requested gate included production npm audit, TypeScript typecheck, 9 focused job lifecycle tests, optimized build, and 36 Playwright browser tests. Detailed command output and updated warning text were **not** supplied for this run, so exact test counts and warning disappearance are not independently confirmed from this checkpoint. Treat the infrastructure change as operator-verified, while retaining the next full command output for warning review. No app-side runtime or DJ audio policy change was made in this slice.
