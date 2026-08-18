# Development

## Backend

```bash
./scripts/setup-native.sh
source .venv/bin/activate
pytest -q
```

Run one CLI command:

```bash
./scripts/djai.sh --help
```

## Phase 0 worker configuration

Independent audio analyses and directed track-pair searches can run in process pools. Worker values use the same convention everywhere:

```text
0 or unset = automatic
1          = serial
2+         = explicit process count
```

Environment variables:

```bash
export DJAI_ANALYSIS_WORKERS=4
export DJAI_GRAPH_WORKERS=6
export DJAI_PROCESS_START_METHOD=auto
```

CLI overrides:

```bash
./scripts/djai.sh set-plan audio \
  --analysis-workers 4 \
  --graph-workers 6 \
  --output output/set-plan.json
```

The automatic analysis cap is intentionally conservative because every worker can hold waveforms, STFTs, chroma and structural matrices in memory. Numerical libraries are constrained to one internal thread per process to avoid oversubscription.

## Benchmark

```bash
./scripts/benchmark.sh audio \
  --analysis-workers 4 \
  --graph-workers 6 \
  --compare-serial
```

The benchmark checks deterministic serial/parallel graph equality and reports analysis time, graph time, cache hits, worker counts and compatibility-evaluation counts.

## Frontend

```bash
cd frontend
npm install
npm run typecheck
npm run dev
```

Generate the lockfile once npm registry access is available:

```bash
./scripts/lock-frontend.sh
```

## Docker source workflow

The repository is bind-mounted into the backend services, so Python source edits do not normally require a rebuild. Rebuild when dependencies, Dockerfiles or frontend production assets change.

```bash
docker compose build api test frontend
docker compose up -d api frontend
```

## Tests

The test suite targets the current 2.2 contract:

- local analysis and source evidence
- compatibility/key behavior
- intent-aware executable pair plans
- execution signatures and tamper detection
- nested serialization and versioned cache invalidation
- one compatibility calculation per unique cue pair
- graph evidence reuse during sequence compilation
- stat-aware learned/genre model caches
- deterministic serial/parallel analysis and graph planning
- path-dependent set compilation
- rendered-audio measurement and top-K reranking
- measurement logging and typed serialization
- graph isolation from expensive measurement
- pair and full-set measured audio integration
- pair audio rendering
- API version, path safety and non-cache job headers
- release metadata consistency
- frontend source contract

## Release

```bash
./scripts/verify.sh
./scripts/package-release.sh
```

The packaging script excludes local music, outputs, ratings, caches, secrets, virtual environments, node modules and build products.
