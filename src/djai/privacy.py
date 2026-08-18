from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

PRIVACY_VERSION = "1.0"


def pseudonymous_listener_id(user_id: str) -> str:
    """Stable account-scoped listener key without exposing the account UUID."""
    pepper = os.environ.get("SAUND_PRIVACY_PEPPER", "saund-ai-local-personalization-v1")
    digest = hmac.new(pepper.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"acct_{digest[:24]}"


def sanitize_feedback_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    blocked = {"email", "password", "token", "session", "phone", "address", "name", "user_id"}
    clean: dict[str, Any] = {}
    for raw_key, value in dict(metadata or {}).items():
        key = str(raw_key)[:80]
        if key.casefold() in blocked:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value[:500] if isinstance(value, str) else value
    return clean


def purge_listener_profile(listener: str, path: str | Path) -> bool:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    records = payload.get("profiles", payload) if isinstance(payload, dict) else {}
    if not isinstance(records, dict) or listener not in records:
        return False
    records.pop(listener, None)
    if isinstance(payload, dict) and "profiles" in payload:
        payload["profiles"] = records
    else:
        payload = records
    temp = source.with_suffix(source.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(source)
    return True

def purge_jsonl_listener(listener: str, path: str | Path) -> int:
    """Delete JSONL records belonging to one pseudonymous account listener."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return 0
    kept: list[str] = []
    removed = 0
    for raw in source.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            kept.append(raw)
            continue
        if isinstance(payload, dict) and str(payload.get("listener", "")) == listener:
            removed += 1
        else:
            kept.append(raw)
    temp = source.with_suffix(source.suffix + ".tmp")
    temp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    temp.replace(source)
    return removed
