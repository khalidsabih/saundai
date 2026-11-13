# Performance and quality-loop architecture

Current execution and authority boundaries are defined in [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md).

## Phase 0 foundation

DJAI reuses one `PairEvidenceMatrix` for every directional track pair. Safe, creative, and emergency searches do not repeat style-independent compatibility calculations. Track analysis and directed graph construction can use deterministic process pools.

Worker values:

```text
0 or unset  conservative automatic policy
1           serial execution
2+          explicit worker count
```

Graph workers receive the analysis library once during process initialization. Worker completion order cannot alter graph contents, optimized order, scores, cues, techniques, or execution signatures.

## Why the graph does not render candidates

For N tracks, the graph contains N x (N - 1) directional edges. Rendering top-K candidates for every graph edge would erase the Phase 0 speedup and produce unnecessary audio work.

DJAI therefore uses two levels:

```text
Entire graph: analytical evidence and executable plans only
Selected final path: render and measure top-K candidates per junction
```

This changes the expensive audio work from approximately O(N² x K) candidate renders to O((N - 1) x K) for the final sequence.

## Decode reuse

For one final directional pair, both files are decoded once at the measurement sample rate. All top-K candidate plans reuse those loaded arrays and the production DSP path.

## Default measurement budget

```text
Top-K candidates: 4
Measurement sample rate: 22.05 kHz
Context before/after: 2 seconds
```

Final rendering remains 44.1 kHz stereo PCM_24.

## Performance metadata

Set plans contain Phase 0 analysis/graph statistics and Phase 1 quality-loop metadata:

```json
{
  "performance": {
    "analysis": {},
    "planning": {}
  },
  "quality_loop": {
    "enabled": true,
    "top_k": 4,
    "sample_rate": 22050,
    "checked_transitions": 11,
    "passed_transitions": 11,
    "measurement_version": "3.0.0"
  }
}
```

## Benchmark

```bash
./scripts/benchmark.sh audio \
  --analysis-workers 4 \
  --graph-workers 6 \
  --compare-serial
```

The Phase 0 benchmark intentionally excludes final audio measurement so it can compare serial and parallel graph construction fairly. Use normal `set-plan` or `set` timing to measure the additional final-path quality-loop cost.
