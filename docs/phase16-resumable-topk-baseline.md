# Phase 16 real-crate baseline — resumable Top-K (2026-11-13)

This document records a **user-reported local benchmark**, not a run reproduced
by CI or by the code-change author. The 13 copyrighted local WAV files are not
in the repository.

## Reproduce on the frozen code

Reference main commit before subsequent model-ranking/diagnostic changes:
`c2b51330db1c0fa0262c238404ea2b71559fa2cd`

```bash
./scripts/saund.sh phase16-validate-real audio/uploads \
  --journey balanced --render \
  --output-dir output/phase6/resumable-topk-13
```

## Observed result

| Metric | Reported value |
| --- | ---: |
| Phase 16 passed | true |
| Production safe | true |
| Rehearsal ready | true |
| Tracks | 13 |
| Transitions and passed quality gates | 12 / 12 |
| Hard audio failures | 0 |
| Final emergency escapes | 0 |
| Replans | 4 |
| Fallback transitions | 11 / 12 |
| Structural resets | 11 / 12 |
| Overall score | 0.704416 |
| Mean measured transition score | 0.983308 |
| Story fit | 0.232619 |
| Transition diversity | 0.5 |
| Planning | 558.4 s |
| Route search | 241.688968 s |
| Compilation/rehearsal | 312.938695 s |
| Broad graph materialized | true |
| Complete sparse search | false |
| Best partial sparse route | 10 / 13 tracks |
| Model changed live decisions | false |

Reported final plan ID: `35e91f3aa8b011cb`.

**Release guard:** Do not improve fallback/structural reset metrics by lowering
Beat Integrity residual limits, skipping rendered-audio rehearsal, treating a
deterministic emergency escape as a normal blend, or promoting an unvalidated
model. Every modified version must be compared with this baseline on the same
crate, journey and validation configuration.

## Investigation target

The accepted baseline is technically safe but uses structural resets on 11 of
12 transitions. Opt-in phase-lock rejection tracing and Top-K ranking telemetry
are diagnostic only; neither implies that unrendered alternatives would have
passed the professional rehearsal.
