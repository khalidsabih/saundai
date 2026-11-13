# SAUND Frontend Architecture

This frontend is a projection of SAUND's execution model, not a separate product fiction.

## Product grammar

The user-facing lifecycle is:

1. Understand
2. Explore
3. Rehearse
4. Decide
5. Perform

Backend concepts map to product concepts deliberately:

- analysis/perception -> Understand
- candidate retrieval and route search -> Explore
- rendered professional judging -> Rehearse
- Executive commitment -> Decide
- signed execution plan and deterministic runtime -> Perform

## UI truth rules

1. If the audio engine did not do it, the UI does not show it.
2. A lock is shown only for a backend-authoritative commitment.
3. Safety/readiness labels require authoritative state.
4. Progress comes from job/runtime events, never decorative timers.
5. Model authority must be shown explicitly when model output is surfaced.
6. Rejected and alternative routes are evidence, not executable state.
7. Red is reserved for actual failure or blocked unsafe state.

## Layers

### app/

Routing and global CSS composition only.

### components/product/

Persistent product chrome and navigation.

### components/ui/

Small reusable visual primitives with no SAUND business logic.

### components/intelligence/

Presentation of planning, rehearsal, authority, model, and decision state.

### components/music/

Crate, upload, track selection, readiness and analysis presentation.

### components/set/

Performance Ribbon, set summaries, transition inspection and review.

### lib/

API contracts, product types, session state and data adapters.

## Design tokens

`app/design-tokens.css` is the source of truth for color semantics, spacing,
radii, typography scale, motion timing and layout dimensions.

Product code should prefer semantic tokens:

- `--color-analysis`
- `--color-possibility`
- `--color-decision`
- `--color-success`
- `--color-danger`

Raw palette values should not be introduced in new components unless a new
semantic token is first justified.

## Migration policy

The existing frontend contains large historical global styles and several
feature-heavy components. Migration is incremental:

1. establish semantic tokens and product shell;
2. extract backend-driven intelligence components;
3. move Create onto reusable music/set primitives;
4. move Set Ready/Review onto the Performance Ribbon;
5. migrate Live;
6. migrate Brain and Lab;
7. remove legacy CSS only after no active surface depends on it.

Do not create CSS overrides indefinitely. Every migration should reduce the
number of owners for layout, state semantics, or product behavior.

## Accessibility

- keyboard navigation is required for every interactive surface;
- progress uses native ARIA progress semantics;
- color never carries meaning alone;
- focus-visible states are required;
- reduced-motion preferences must be respected;
- disabled product areas must not link to non-existent routes.

## Responsive behavior

Desktop is the full product. Tablet preserves Create, Sets, Live and Review with
drawers for dense intelligence. Mobile prioritizes Create, Sets, playback, Live
and basic Review. Brain and Studio may be reduced or vertically composed.

## Production definition

A frontend feature is production-ready only when:

- its state is grounded in a real API/runtime contract;
- loading, empty, failure and degraded states are designed;
- keyboard and responsive behavior are intentional;
- the component is not duplicating an existing layout/state owner;
- TypeScript/build checks pass;
- high-value flows have end-to-end coverage.
