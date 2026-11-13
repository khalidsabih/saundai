# Automatic delivery — current architecture

Current authority boundaries are defined in [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md).

## Goal

For valid decodable masters, the default listener still does no DJ engineering:

```text
upload full masters
  -> prepare/cache stems
  -> deterministic + neural perception
  -> Executive-owned route search proposes futures
  -> solve each directed relationship
  -> family/cue/duration/stem composition search
  -> privately render finalists
  -> judge actual audio
  -> accept OR block/replan
  -> render the frozen set
```

## Neural setup

The Docker `brain` target includes the runtime dependencies, but model weights are never silently fetched. After the image is built:

```bash
./scripts/djai.sh models-status
./scripts/djai.sh models-bootstrap
```

MERT is optional:

```bash
./scripts/djai.sh models-bootstrap --mert
```

Without a cached neural checkpoint DJAI remains functional through deterministic fallbacks.

## Source preparation

With `DJAI_AUTO_STEMS=background`, upload can start source separation asynchronously. Automatic set creation is the synchronization point. Clean supplied originals are optional and outrank AI separation. Separation failure leaves the stereo master usable.

## Unified transition solver

Every proposed A→B receives a primary route family and bounded fallback families. A trained/authorized family head may rerank that decision, but hard feasibility and private rehearsal remain downstream gates.

## Duration and choreography

The solver searches family-standard 4/8/16/32-bar lengths where appropriate and phrase-derived custom lengths when cue evidence supports them. A trained duration head may reorder only these valid candidates. With four stems, a learned choreography head may rerank role-ownership programs; it does not invent raw DSP outside the immutable program contract.

## Rehearsal budget

The configured finalist count is one total per-edge render budget shared across route families. Neural ranking is intended to spend that budget more intelligently, not increase it without bound.

## Review-needed and set replanning

A candidate that does not pass rendered rehearsal and professional judgement cannot enter the automatic final plan. After bounded family search fails, the directed pair is blocked and the Executive owns the route repair/replan loop. A learned next-track model may contribute a bounded value residual, but hard feasibility and rendered evidence remain downstream gates.

## Emergency boundary

Only after alternate family/edge/order search is exhausted may automatic delivery try a separate emergency escape. Emergency candidates are themselves rendered and judged. A deterministic final phrase-aligned escape remains the last technical delivery boundary for valid audio.

## Telemetry

Measurement telemetry is reconciled after the final track order is known. Rehearsals discarded by global replanning remain evidence but are marked rejected/superseded; only final junction execution signatures keep `accepted=true`.
