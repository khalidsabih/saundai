#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from djai.accounts import AccountStore  # noqa: E402
from djai.hosting import HostingConfig  # noqa: E402
from djai.object_store import ObjectStore  # noqa: E402


def main() -> int:
    workspace = Path(os.environ.get("DJAI_WORKSPACE", ROOT)).expanduser().resolve()
    hosting = HostingConfig.from_mapping()
    database_url = os.environ.get("SAUND_DATABASE_URL", "").strip() or None
    account_path = Path(os.environ.get("SAUND_ACCOUNT_DB", "accounts/accounts.sqlite3")).expanduser()
    if not account_path.is_absolute():
        account_path = workspace / account_path
    try:
        account_store = AccountStore(account_path if database_url is None else None, database_url=database_url)
        database_status = account_store.status()
    except Exception as exc:
        database_status = {"backend": "postgres" if database_url else "sqlite", "ready": False, "error": f"{type(exc).__name__}: {exc}"}
        account_store = None

    try:
        object_store = ObjectStore()
        object_status = object_store.status()
        object_status["ready"] = object_store.ready()
    except Exception as exc:
        object_status = {"backend": "unknown", "enabled": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"}
        object_store = None

    cors = [value.strip() for value in os.environ.get("DJAI_CORS_ORIGINS", "").split(",") if value.strip()]
    cookie_secure = os.environ.get("SAUND_AUTH_COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes", "on"}
    training_lab = os.environ.get("DJAI_TRAINING_LAB_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    errors = hosting.validation_errors(
        database_backend=str(database_status.get("backend", "unknown")),
        object_store_enabled=bool(object_status.get("enabled", False)),
        cookie_secure=cookie_secure,
        cors_origins=cors,
        training_lab_enabled=training_lab,
        confirmation_secret=os.environ.get("DJAI_CONFIRMATION_SECRET", ""),
        workspace=workspace,
    )
    if account_store is not None and not database_status.get("ready"):
        errors.append("Database connection check failed.")
    if object_store is not None and object_store.enabled and not object_status.get("ready"):
        errors.append("Object-store bucket check failed.")

    payload = {
        "schema_version": "1.5",
        "ready": not errors,
        "hosting": hosting.public_status(
            database_backend=str(database_status.get("backend", "unknown")),
            object_store_enabled=bool(object_status.get("enabled", False)),
        ),
        "database": database_status,
        "object_store": object_status,
        "checks": {
            "secure_cookie": cookie_secure,
            "training_lab_enabled": training_lab,
            "cors_origins": cors,
            "workspace": str(workspace),
        },
        "errors": errors,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
