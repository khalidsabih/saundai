# DJAI 4.0.0 security notes

DJAI is configured primarily for trusted local use.

## Existing controls

- all user-supplied paths are resolved and required to remain inside `DJAI_WORKSPACE`;
- upload names are sanitized;
- supported audio extensions are enforced;
- upload size is limited by `DJAI_MAX_UPLOAD_BYTES`;
- CORS is restricted by `DJAI_CORS_ORIGINS`;
- Pro confirmation tokens are bound to the exact plan payload;
- immutable execution signatures detect plan changes;
- job state and live manifest/stream responses disable caching;
- generated media is served only from the workspace;
- stem jobs have a dedicated queue so expensive separation does not starve set jobs.

## Before exposing beyond localhost

Add or configure:

```text
authentication and authorization
per-user workspace isolation
rate limiting
request/job quotas
storage quotas and retention
strong random confirmation secret
TLS/reverse proxy
non-root containers
network egress policy
content and licensing policy
monitoring/audit logs
stream concurrency limits
```

## Resource exhaustion

Audio analysis, time stretching, candidate measurement, stem separation, and live encoding are CPU/memory intensive. Limit:

```text
DJAI_JOB_WORKERS
DJAI_STEM_JOB_WORKERS
DJAI_ANALYSIS_WORKERS
DJAI_GRAPH_WORKERS
DJAI_MAX_UPLOAD_BYTES
```

Do not expose unlimited Demucs or live-stream jobs to untrusted users.

## Privacy

The default installation stores locally:

```text
uploaded audio
generated mixes
analysis/stem caches
measurement logs
feedback events
listener profiles
learned models
```

Use `scripts/clean.sh --all` to remove local runtime artifacts. Listener identifiers should be pseudonymous and must not contain sensitive personal information.

## Audio rights

Users are responsible for having the rights required to upload, analyze, transform, perform, stream, and redistribute source material and generated mixes.
