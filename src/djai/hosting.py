from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HostingConfig:
    environment: str
    hosted_mode: bool
    public_origin: str | None
    api_origin: str | None
    persistent_workspace_confirmed: bool
    require_postgres: bool
    require_object_store: bool
    upload_auth_required: bool
    signup_mode: str
    invite_codes_configured: bool
    allow_hosted_training_lab: bool
    trust_proxy_headers: bool
    auth_rate_limit_per_minute: int

    @classmethod
    def from_mapping(cls, env: Mapping[str, str] | None = None) -> "HostingConfig":
        source = os.environ if env is None else env
        environment = (source.get("SAUND_ENV", "development").strip().lower() or "development")
        hosted = _truthy(source.get("SAUND_HOSTED_MODE"), default=environment == "production")
        signup_mode = (source.get("SAUND_SIGNUP_MODE", "invite" if hosted else "open").strip().lower() or "open")
        if signup_mode not in {"open", "invite", "closed"}:
            raise RuntimeError("SAUND_SIGNUP_MODE must be one of: open, invite, closed.")
        return cls(
            environment=environment,
            hosted_mode=hosted,
            public_origin=source.get("SAUND_PUBLIC_ORIGIN", "").strip().rstrip("/") or None,
            api_origin=source.get("SAUND_API_ORIGIN", "").strip().rstrip("/") or None,
            persistent_workspace_confirmed=_truthy(source.get("SAUND_PERSISTENT_WORKSPACE"), default=not hosted),
            require_postgres=_truthy(source.get("SAUND_HOSTED_REQUIRE_POSTGRES"), default=hosted),
            require_object_store=_truthy(source.get("SAUND_HOSTED_REQUIRE_OBJECT_STORE"), default=hosted),
            upload_auth_required=_truthy(source.get("SAUND_UPLOAD_AUTH_REQUIRED"), default=hosted),
            signup_mode=signup_mode,
            invite_codes_configured=bool([value.strip() for value in source.get("SAUND_INVITE_CODES", "").split(",") if value.strip()]),
            allow_hosted_training_lab=_truthy(source.get("SAUND_ALLOW_HOSTED_TRAINING_LAB")),
            trust_proxy_headers=_truthy(source.get("SAUND_TRUST_PROXY_HEADERS"), default=hosted),
            auth_rate_limit_per_minute=max(5, min(600, int(source.get("SAUND_AUTH_RATE_LIMIT_PER_MINUTE", "30")))),
        )

    def validation_errors(
        self,
        *,
        database_backend: str,
        object_store_enabled: bool,
        cookie_secure: bool,
        cors_origins: list[str],
        training_lab_enabled: bool,
        confirmation_secret: str,
        workspace: Path,
    ) -> list[str]:
        if not self.hosted_mode:
            return []
        errors: list[str] = []
        if not self.persistent_workspace_confirmed:
            errors.append("SAUND_PERSISTENT_WORKSPACE=1 must confirm a durable workspace volume in hosted mode.")
        if not self.public_origin:
            errors.append("SAUND_PUBLIC_ORIGIN is required in hosted mode.")
        if not self.api_origin:
            errors.append("SAUND_API_ORIGIN is required in hosted mode.")
        if not cors_origins:
            errors.append("At least one HTTPS SAUND_CORS_ORIGINS entry is required in hosted mode.")
        if not self.upload_auth_required:
            errors.append("Hosted mode requires authenticated uploads (SAUND_UPLOAD_AUTH_REQUIRED=1).")
        if self.signup_mode == "invite" and not self.invite_codes_configured:
            errors.append("SAUND_SIGNUP_MODE=invite requires at least one SAUND_INVITE_CODES value.")
        if self.require_postgres and database_backend != "postgres":
            errors.append("Hosted mode requires SAUND_DATABASE_URL=postgresql://... unless SAUND_HOSTED_REQUIRE_POSTGRES=0.")
        if self.require_object_store and not object_store_enabled:
            errors.append("Hosted mode requires S3-compatible object storage unless SAUND_HOSTED_REQUIRE_OBJECT_STORE=0.")
        if not cookie_secure:
            errors.append("Hosted mode requires secure authentication cookies (SAUND_AUTH_COOKIE_SECURE=1).")
        if len(confirmation_secret) < 32 or "local-development-secret" in confirmation_secret or confirmation_secret.startswith("change-this"):
            errors.append("DJAI_CONFIRMATION_SECRET must be a non-default secret of at least 32 characters in hosted mode.")
        if any(origin.startswith("http://") for origin in cors_origins):
            errors.append("Hosted CORS origins must use HTTPS.")
        if self.public_origin and not self.public_origin.startswith("https://"):
            errors.append("SAUND_PUBLIC_ORIGIN must use HTTPS in hosted mode.")
        if self.api_origin and not self.api_origin.startswith("https://"):
            errors.append("SAUND_API_ORIGIN must use HTTPS in hosted mode.")
        if training_lab_enabled and not self.allow_hosted_training_lab:
            errors.append("Training Lab must be disabled on hosted deployments unless SAUND_ALLOW_HOSTED_TRAINING_LAB=1 is explicitly set.")
        try:
            workspace.mkdir(parents=True, exist_ok=True)
            probe = workspace / ".saund-host-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError:
            errors.append("DJAI_WORKSPACE is not writable by the API process.")
        return errors

    def public_status(self, *, database_backend: str, object_store_enabled: bool) -> dict[str, object]:
        return {
            "schema_version": "1.5",
            "hosted_mode": self.hosted_mode,
            "environment": self.environment,
            "deployment_profile": "private-beta-single-node" if self.hosted_mode else "local",
            "database_backend": database_backend,
            "object_store": "s3" if object_store_enabled else "local-filesystem",
            "direct_uploads": bool(self.hosted_mode and object_store_enabled),
            "upload_auth_required": self.upload_auth_required,
            "signup_mode": self.signup_mode,
            "invite_codes_configured": self.invite_codes_configured,
            "persistent_workspace_required": self.hosted_mode,
            "job_execution": "persistent-single-node",
            "horizontal_worker_scaling": False,
            "public_origin": self.public_origin,
            "api_origin": self.api_origin,
        }


class SlidingWindowRateLimiter:
    """Small single-node limiter for auth endpoints.

    This is intentionally an application backstop, not a replacement for CDN/WAF
    rate limiting. It matches the 1.5 hosted execution profile: one durable API
    node plus persistent workspace.
    """

    def __init__(self, *, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = [stamp for stamp in self._events.get(key, []) if stamp >= cutoff]
            if len(events) >= self.limit:
                self._events[key] = events
                return False
            events.append(now)
            self._events[key] = events
            if len(self._events) > 4096:
                stale = [item for item, stamps in self._events.items() if not stamps or stamps[-1] < cutoff]
                for item in stale[:1024]:
                    self._events.pop(item, None)
            return True


def invite_code_allowed(code: str | None, *, mode: str, configured_codes: str | None = None) -> bool:
    if mode == "open":
        return True
    if mode == "closed":
        return False
    candidate = (code or "").strip()
    if not candidate:
        return False
    codes = [value.strip() for value in (configured_codes if configured_codes is not None else os.environ.get("SAUND_INVITE_CODES", "")).split(",") if value.strip()]
    return any(secrets.compare_digest(candidate, allowed) for allowed in codes)
