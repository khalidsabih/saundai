# Canonical architecture and source of truth

Status: Phase 3 canonical architecture baseline for SAUND.AI `2.0.0-rc.1`.

This document describes the current runtime and authority boundaries. Historical
4.x/5.x milestone documents remain useful as implementation history, but they
must not be treated as the current release contract when they disagree with the
sources listed here.

## Canonical planning and execution pipeline

```text
Track perception
      ↓
Canonical track analysis / musical timeline
      ↓
Set state + DJ state
      ↓
Candidate retrieval
      ↓
Complete DJMove generation
      ↓
Route / lookahead search
      ↓
Path-dependent TransitionPlan compilation
      ↓
Private rendered rehearsal (AI Headphones / professional quality loop)
      ↓
Hard safety + deterministic validation
      ↓
Executive commit
      ↓
Immutable performance program + execution signature
      ↓
Renderer / Live executor
```

Taste, Crate Memory, crowd state, learned models and personalization may influence
candidate ranking only through bounded interfaces. They do not bypass hard
technical failures or deterministic execution validation.

## Core contracts

### TrackAnalysis and MusicalTimeline

`TrackAnalysis` is the canonical perception result for a source track.
`MusicalTimeline` derives bar/phrase/song-form semantics used by DJ state,
move scoring and timing constraints.

Current musical timeline schema: `1.1.1`.

Authority: perception evidence informs planning; it is not an execution command.

### DJState

`DJState` is the planner's state snapshot for the current deck and set context.
It includes source cursor, bar/phrase position, effective/native BPM, render rate,
tempo ownership, current musical characteristics and pending commitments.

Current DJMove/DJState version: `1.4`.

Authority: state description only.

### DJMove

`DJMove` is the semantic musical action proposed by the Move Brain. It binds
what should happen: wait/transition intent, exit and entry phrases, technique,
tempo strategy, structural operations, payoff protection and projected state.

A DJMove is **not** the DSP program. It is a musical proposal that must still be
compiled and validated.

### CompleteMoveCandidate / CompleteMoveDecision

A complete move binds a DJMove to a concrete next-track candidate and an
executable TransitionPlan proposal. Receding-horizon search ranks these complete
moves and may use AI Headphones at the commit boundary.

Current complete-move version: `1.8.0`.

Planner modes:

- `legacy`
- `move_shadow`
- `move_authoritative`

The configured default is `move_authoritative` when no environment override is
present.

### TransitionPlan

`TransitionPlan` is the path-dependent concrete transition contract. It binds
source positions, durations, render rates, BPM state, technique, intent,
compatibility evidence, measurements, quality state, metadata and the attached
performance program.

Current plan/decision/planner/set-plan schema versions: `8.0.0`.

Authority: concrete compiled proposal; still subject to rehearsal, hard safety
and final execution validation.

### TransitionProgram

`TransitionProgram` is the authoritative executable intermediate
representation consumed by the DSP engine. It contains source buses, automation
curves, deterministic processors and deck operations.

Renderer code must not infer musical strategy from the TransitionPlan
`technique` label when a performance program is present.

Current performance-program version: `4.2`.

Its content fingerprint contributes to immutable execution identity.

### Execution signature

The execution signature is the immutable identity of the committed executable
transition. It binds the executable program and relevant transition state.

A model preference or later heuristic must never mutate the executable program
without producing a new signature and passing validation again.

### AI Headphones / rendered rehearsal

AI Headphones privately renders a bounded set of serious current-move finalists
using the production transition engine, then applies rendered measurements and
professional judgement.

Current AI Headphones version: `1.8.0`.

Headphones can become authoritative over the *musical finalist selection* only
when a rendered candidate passes its measurement and professional judgement
requirements. Hard audio safety remains authoritative afterwards.

Render/measurement failure never grants model authority.

### Executive

The Executive is the single supervisory commit authority. Specialist components
may propose, score, rehearse or request replans; the Executive owns route repair,
the replan loop and the final commit boundary.

Current Executive orchestrator version: `1.0.0`.

The Executive does not perform DSP.

Prepared execution and live execution share the same principle: deterministic
execution consumes committed programs; the brain does not improvise inside the
DSP engine.

### Live runtime state

Live execution protects three horizons:

- **NOW**: currently executing material; immutable.
- **NEXT**: committed next transition/track; immutable.
- **FUTURE**: eligible for replanning at safe boundaries.

Stale or late asynchronous replan work must be rejected by generation/version
checks. Resource pressure may trigger hold/backup behavior but cannot silently
advance into unprepared material.

The Live executor maintains source cursor and execution protocol state; it does
not select musical strategy.

## Authority hierarchy

From strongest to weakest:

1. Hard audio/integrity safety and executor protocol constraints.
2. Immutable committed performance program and execution signature.
3. Executive commit/replan authority.
4. Rendered rehearsal and professional judgement.
5. Complete-move / route-search decisions.
6. Heuristic, learned, taste, crowd and personalization ranking signals.

Lower layers may never override a failure reported by a higher layer.

## Version and schema source-of-truth table

| Contract | Current value | Canonical source |
| --- | --- | --- |
| Product version | `2.0.0-rc.1` | `src/djai/version.py:DJAI_VERSION` and root `VERSION` |
| Musical timeline schema | `1.1.1` | `src/djai/version.py:MUSICAL_TIMELINE_SCHEMA_VERSION` |
| Analysis/cache schema | `5.0.0` | `src/djai/version.py:CACHE_SCHEMA_VERSION` |
| Plan schema | `8.0.0` | `src/djai/version.py:PLAN_SCHEMA_VERSION` |
| Decision version | `8.0.0` | `src/djai/version.py:DECISION_VERSION` |
| Planner version | `8.0.0` | `src/djai/version.py:PLANNER_VERSION` |
| Set-plan version | `8.0.0` | `src/djai/version.py:SET_PLAN_VERSION` |
| Measurement version | `3.0.0` | `src/djai/version.py:MEASUREMENT_VERSION` |
| DJMove / DJState | `1.4` | `src/djai/brain/dj_move.py:DJ_MOVE_VERSION` |
| Complete move | `1.8.0` | `src/djai/brain/move_generator.py:COMPLETE_MOVE_VERSION` |
| AI Headphones | `1.8.0` | `src/djai/rehearsal/headphones.py:AI_HEADPHONES_VERSION` |
| Performance program | `4.2` | `src/djai/performance/program.py:PERFORMANCE_PROGRAM_VERSION` |
| Executive orchestrator | `1.0.0` | `src/djai/brain/orchestrator.py:EXECUTIVE_ORCHESTRATOR_VERSION` |

## Safety and model authority source of truth

- Hard audio safety: `src/djai/performance/safety.py`.
- Executor protocol and source-cursor authority: `src/djai/performance/executor.py`.
- Executable program compilation: `src/djai/performance/program.py`.
- Execution sealing/validation: `src/djai/decision.py`.
- Executive orchestration and component ownership:
  `src/djai/brain/orchestrator.py`.
- Complete-move planner modes and bounded authority:
  `src/djai/brain/move_generator.py`.
- Rendered finalist authority: `src/djai/rehearsal/headphones.py`.
- Automatic rescue hierarchy: `src/djai/fallback.py`.

## Historical documentation policy

Documents that describe older "SAUND 4.x", "5.x", or pre-8.0 planner/schema
milestones are historical evidence. They should remain available, but current
code, current schema constants and this document win when terminology or version
numbers conflict.

New documentation should link to this file for architectural terminology rather
than inventing another independent version/authority table.
