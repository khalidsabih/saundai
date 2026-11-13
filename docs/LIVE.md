# SAUND.AI live architecture

Current authority boundaries are defined in [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md).

The live audio path executes committed immutable programs while allowing bounded
planning work only beyond the committed horizon.

## NOW / NEXT / FUTURE

- **NOW** is executing and immutable.
- **NEXT** is committed and immutable.
- **FUTURE** may be replanned at safe boundaries.

FUTURE replans carry a schedule generation and deadline. Late or stale results
are discarded instead of mutating the active schedule.

## Prepared alternatives

The live manifest exposes prepared primary/backup/emergency transition metadata.
Backup and emergency programs remain deterministic recovery assets; they do not
grant a model permission to bypass execution safety.

## Runtime safety

Prefetch, buffer, deadline, storage and asset-health telemetry can block unsafe
advancement. If NEXT is not ready by its real audible deadline, Live may hold or
use prepared recovery behavior rather than advancing into unprepared material.

The executor owns source-cursor/protocol state. The Executive owns musical
replanning. Model latency and route search are never placed inside the audio
callback.
