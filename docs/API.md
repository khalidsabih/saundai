# SAUND.AI API

Interactive OpenAPI documentation is served at `/docs` when the FastAPI service is running.

Current product/version, authority and schema contracts are defined in [CANONICAL_ARCHITECTURE.md](CANONICAL_ARCHITECTURE.md).

## Product endpoints

The existing upload, analysis, automatic-set, manual/pro, job/SSE, preview/render and feedback endpoints remain available.

## Intelligence endpoints

- `POST /api/intelligence/listening-session` — create blind rehearsed Taste Lab candidates.
- `POST /api/intelligence/preference` — record A/B/Tie/Neither and optional reason tags; `Neither` also writes rejection memory.
- `POST /api/intelligence/ranker/train` — fit the pairwise preference model.
- `GET /api/intelligence/ranker/status` — validation/authority status.
- `GET /api/brain/status` — professional-brain configuration, preference/imitation readiness, rejection-memory count, data-provenance audit, curriculum and Arena state.

## Live endpoints

- `GET /api/live/manifest?plan_path=...` — timeline plus prepared primary/backup/emergency transition metadata.
- `GET /api/live/stream?plan_path=...` — progressive MP3 of the already-prepared primary immutable set.

## Safety

The API does not permit learned scores to bypass execution-plan validation. Paths are resolved through the existing project/job boundaries before rendering.
