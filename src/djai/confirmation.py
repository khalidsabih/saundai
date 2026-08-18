from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any


def _secret() -> bytes:
    return os.environ.get("DJAI_CONFIRMATION_SECRET", "djai-local-confirmation-v1").encode("utf-8")


def create_confirmation_token(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(_secret(), canonical, hashlib.sha256).hexdigest()
    return digest


def verify_confirmation_token(payload: dict[str, Any], token: str) -> bool:
    expected = create_confirmation_token(payload)
    return hmac.compare_digest(expected, token)
