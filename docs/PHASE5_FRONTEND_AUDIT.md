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
