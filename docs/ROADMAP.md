# SAUND.AI production roadmap

This is the production spine for SAUND.AI 1.0. The 1.0 codebase closes each architecture milestone while preserving a strict distinction between **implemented capability** and **earned model authority**. A learned model with insufficient rights, held-out, Arena, or governance evidence remains shadow/advisory even when the surrounding production path is complete.

```text
NOW
 │
 ├── RC11.6B  More sequence coverage                 COMPLETE (code)
 ├── RC11.6C  Cold-target intelligence               COMPLETE (code)
 ├── RC11.6D  Hybrid next-track model                COMPLETE
 │
 ├── RC11.7   Rendered DJ Judge                      COMPLETE (governed; weights may remain shadow)
 ├── RC11.8   Global Taste Ranker                    COMPLETE (bounded TransitionRanker)
 ├── RC11.9+  Transition policies                    COMPLETE (family/duration/stem policies)
 │
 ├── RC12     Unified AI planner                     COMPLETE
 ├── RC12.x   Full-set intelligence + benchmarks     COMPLETE
 │
 ├── RC13     Personalization + privacy + security   COMPLETE
 ├── RC13.x   Beta + model governance                COMPLETE
 │
 ├── RC14     Production candidate                   COMPLETE (release gate)
 │
 └── SAUND.AI 1.0                                   CODE COMPLETE
```

## What “complete” means

- **RC11.6B**: ListenBrainz sequence ingestion can use disk-backed state and deterministic overlap handling rather than unbounded per-user RAM.
- **RC11.6C**: cold targets are evaluated separately; unseen tracks can back off through known-artist evidence instead of being silently treated as ordinary known targets.
- **RC11.6D**: `hybrid_next_track` combines deterministic musical/set evidence with earned learned authority and has an absolute rendered hard-failure veto.
- **RC11.7**: rendered transition embeddings flow into the rendered DJ Judge; Training Lab exports pair-grouped A/B/Tie/Neither training rows.
- **RC11.8**: the existing TransitionRanker remains the global taste residual over candidates that already passed technical feasibility.
- **RC11.9+**: transition family, duration and stem choreography policies are registry governed; deterministic execution remains authoritative.
- **RC12**: the unified planner performs full-set beam search using hybrid next-track evidence and contributes candidate orders to the canonical set objective rather than bypassing it.
- **RC12.x**: every set gets full-set benchmark telemetry. Hard failures zero the benchmark and cannot win a champion/challenger comparison.
- **RC13**: account-scoped pseudonymous personalization, explicit personalization/training consent, export/delete, secure production cookies, security headers, credentialed-CORS wildcard rejection.
- **RC13.x**: production neural promotion requires rights review, held-out evidence, blind Arena readiness, governance approval and checkpoint pinning.
- **RC14**: the release gate checks product/version consistency, data provenance, model governance, frontend/backend tests, browser smoke tests and Compose configuration.

## Production evidence is separate from code completion

SAUND.AI 1.0 does not fabricate trained checkpoints or benchmark wins. External training data is production-default-deny. DJ-specific heads may remain shadow until real rights-cleared held-out and blind Arena evidence satisfies governance. Hard audio safety remains non-overridable at every authority level.


## Phase 13 — Time/Event-Aware Set Intelligence

Implemented on the current architecture as an optional event-timing layer. Set planning can now receive a target duration, peak fraction, closing window and ending-energy intent. Complete-move search derives journey progress from the performance clock when a target duration is present, while retaining historical track-position progress when it is absent.

Timing authority is bounded: it may bias route/move ranking for duration overshoot and closing-energy fit, but it cannot bypass transition feasibility, rendered-audio judgement, execution signatures or hard safety.

Set metadata records the normalized timing intent and final timing state, including elapsed/projected time, target duration, planned peak, remaining time, closing-window state, duration error and timing urgency.

## Phase 14 — Live Autonomous Performance Hardening

The existing NOW/NEXT/FUTURE architecture is hardened rather than replaced.

- NOW and NEXT remain immutable.
- FUTURE replans carry a schedule generation and a monotonic planning deadline.
- Late or stale work is discarded while the prepared schedule continues.
- Horizon telemetry exposes prepared hot-backup and emergency programs.
- Safety telemetry now treats audio underruns, slow prefetch and storage stalls as execution barriers in addition to CPU, buffer and deadline pressure.
- Executor status exposes the active safety telemetry and backup readiness.

The release invariant is that planning failure or lateness must not force advancement into unsafe material.

## Phase 15 — Venue / Production Reliability

Venue reliability is implemented as a separate non-musical layer.

- VenueWatchdog evaluates heartbeat, audio-progress, buffer and CPU state without choosing music.
- SessionJournal atomically records safe execution boundaries for restart recovery.
- Recovery semantics distinguish NOW, NEXT and FUTURE asset failures.
- Runtime journals intentionally promise safe-boundary recovery, not sample-exact restoration of arbitrary DSP state.
- Asset-health helpers classify missing, unreadable and empty files without giving them musical authority.

The deterministic Executor can optionally persist a safe-boundary journal through DJAI_SESSION_JOURNAL. External supervision can consume the same recovery semantics without sharing the DJ Executive's musical decision logic.
