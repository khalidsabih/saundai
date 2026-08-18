# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund --ignore-scripts

FROM node:22-bookworm-slim AS builder
WORKDIR /app
ARG SAUND_INTERNAL_API_URL=http://api:8000
ARG NEXT_PUBLIC_SAUND_API_URL=
ARG SAUND_BUILD_COMMIT=
ARG SAUND_BUILD_TIME_UTC=
ENV NEXT_TELEMETRY_DISABLED=1 \
    SAUND_INTERNAL_API_URL=${SAUND_INTERNAL_API_URL} \
    NEXT_PUBLIC_SAUND_API_URL=${NEXT_PUBLIC_SAUND_API_URL} \
    SAUND_BUILD_COMMIT=${SAUND_BUILD_COMMIT} \
    SAUND_BUILD_TIME_UTC=${SAUND_BUILD_TIME_UTC}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run typecheck && npm run build

FROM node:22-bookworm-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    SAUND_INTERNAL_API_URL=http://api:8000
RUN groupadd --system --gid 1001 nodejs \
    && useradd --system --uid 1001 --gid nodejs nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
