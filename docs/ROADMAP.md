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
