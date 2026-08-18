# SAUND.AI hosted/private-beta deployment

SAUND.AI 1.5 introduces a conservative hosted profile without changing the local-first workflow.

## Deployment shape

- **Frontend:** Next.js can run on Vercel or any Node container platform.
- **API / DJ compute:** one durable SAUND container with a persistent `/workspace` volume.
- **Database:** managed PostgreSQL via `SAUND_DATABASE_URL`.
- **Object storage:** private S3-compatible bucket (Cloudflare R2 or AWS S3) for direct browser uploads and published result WAVs.
- **Training corpora:** never copied into the runtime image.
- **Model promotion:** remains governed/default-deny. Public/production deployments should not make research bootstrap models authoritative unless separately approved.

The 1.5 hosted profile is deliberately **single-node for DJ jobs**. It is hostable with durable job history/output because job state lives on a persistent workspace. If the process is interrupted, an in-flight job is marked failed and can be retried; 1.5 does not yet resume a partially executed job or distribute it across workers. `/api/hosting/status` exposes this honestly.

## Why uploads bypass the frontend/API proxy

Large audio files should go directly from the browser to the private object store using a short-lived presigned PUT. The SAUND API only creates the upload intent, verifies the finalized object, and materializes it into the user's private durable library for analysis. Finished WAVs are uploaded to object storage and returned with short-lived signed GET URLs.

For Cloudflare R2, create an S3 API token for the bucket and use the endpoint:

`https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

The R2 bucket also needs browser CORS for your app origin. Example:

```json
[
  {
    "AllowedOrigins": ["https://app.example.com"],
    "AllowedMethods": ["GET", "PUT", "HEAD"],
    "AllowedHeaders": ["Content-Type"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Keep the bucket private. SAUND uses presigned URLs; do not make the bucket public.

## First deployment

1. Provision a managed PostgreSQL database with TLS.
2. Create a private R2/S3 bucket and configure CORS.
3. Copy `deploy/hosted.env.example` into your secret manager/platform environment and replace every placeholder.
4. Ensure `/workspace` is a persistent volume. Set `SAUND_PERSISTENT_WORKSPACE=1` only after this is true.
5. Build the `hosted` Docker target.
6. Run `python scripts/hosted-audit.py` inside the configured environment.
7. Start the API and verify both `/api/health` and `/api/ready`.
8. Deploy the Next.js frontend and configure its API origin/rewrite.

For a single-host smoke deployment:

```bash
cp deploy/hosted.env.example .env.hosted
# edit .env.hosted
set -a; . ./.env.hosted; set +a
docker compose -f compose.hosted.yaml build
docker compose -f compose.hosted.yaml up -d
curl -fsS http://localhost:8000/api/ready | jq .
```

A managed database is still expected; `compose.hosted.yaml` intentionally does not hide production database state in a local container.

## Vercel frontend

Deploy `frontend/` as the Vercel project. Two supported patterns exist:

- Set `SAUND_INTERNAL_API_URL=https://api.example.com` and keep browser `/api/*` calls same-origin through the Next.js rewrite.
- Or set `NEXT_PUBLIC_SAUND_API_URL=https://api.example.com` and configure `DJAI_CORS_ORIGINS=https://app.example.com` on the API.

Large audio does not pass through that control-plane request path: the browser PUTs it directly to R2/S3.

## Private beta signup

Hosted mode defaults `SAUND_SIGNUP_MODE=invite`. Configure one or more beta codes in `SAUND_INVITE_CODES`. Set `open` only when you intentionally want public registration; set `closed` to stop new accounts while keeping login available.

The built-in authentication rate limiter is a single-node backstop. Also enable your CDN/WAF/provider rate limiting for `/api/auth/*`.

## Security and privacy defaults

Hosted startup fails closed when required release conditions are missing, including secure cookies, durable workspace confirmation, required Postgres/object storage, HTTPS CORS origins, a non-default confirmation secret, or an unintentionally enabled Training Lab.

Hosted libraries are isolated per account. Another user cannot list or stream a different account's source tracks. Account deletion removes the user's materialized library and attempts to remove that user's object-store prefix before deleting the account row.

Raw audio is not automatically model-training consent. Keep model-improvement consent and uploaded music rights as separate decisions.

## Operational limitation intentionally kept in 1.5

Job execution remains `persistent-single-node`: FastAPI and its job executor share the durable workspace. This avoids a risky rewrite of the path-centric render engine before private beta. The next infrastructure step, if load requires it, is to replace the in-process executor with a durable queue and object-addressed worker artifacts. That change should be benchmarked separately from DJ-brain quality work.
