from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
from typing import Any

LIVE_TELEMETRY_VERSION = "1.0"
_TERMINAL_STATUSES = {"complete", "failed", "closed"}
_SESSION_TTL_SECONDS = 60.0 * 60.0
_MAX_SESSIONS = 128


@dataclass(slots=True)
class LiveTelemetryEvent:
    sequence: int
    session_id: str
    plan_id: str
    status: str
    updated_at_utc: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LIVE_TELEMETRY_VERSION,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "plan_id": self.plan_id,
            "status": self.status,
            "updated_at_utc": self.updated_at_utc,
            **deepcopy(self.payload),
        }


class _LiveTelemetrySession:
    def __init__(self, *, plan_id: str, session_id: str) -> None:
        self.plan_id = str(plan_id)
        self.session_id = str(session_id)
        self._condition = threading.Condition()
        self._sequence = -1
        self._latest: LiveTelemetryEvent | None = None
        self._updated_monotonic = time.monotonic()

    @property
    def updated_monotonic(self) -> float:
        with self._condition:
            return self._updated_monotonic

    def publish(self, payload: dict[str, Any], *, status: str) -> LiveTelemetryEvent:
        with self._condition:
            self._sequence += 1
            self._updated_monotonic = time.monotonic()
            self._latest = LiveTelemetryEvent(
                sequence=self._sequence,
                session_id=self.session_id,
                plan_id=self.plan_id,
                status=str(status),
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
                payload=deepcopy(payload),
            )
            self._condition.notify_all()
            return self._latest

    def wait_after(self, sequence: int, *, timeout: float) -> LiveTelemetryEvent | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= int(sequence):
                self._condition.wait(timeout=max(0.05, float(timeout)))
            if self._latest is None or self._latest.sequence <= int(sequence):
                return None
            return deepcopy(self._latest)

    def latest(self) -> LiveTelemetryEvent | None:
        with self._condition:
            return deepcopy(self._latest)


_registry_lock = threading.RLock()
_sessions: dict[tuple[str, str], _LiveTelemetrySession] = {}


def _key(plan_id: str, session_id: str) -> tuple[str, str]:
    return str(plan_id), str(session_id)


def _prune_locked(now: float) -> None:
    expired = [
        key
        for key, session in _sessions.items()
        if now - session.updated_monotonic > _SESSION_TTL_SECONDS
    ]
    for key in expired:
        _sessions.pop(key, None)
    if len(_sessions) <= _MAX_SESSIONS:
        return
    oldest = sorted(_sessions.items(), key=lambda item: item[1].updated_monotonic)
    for key, _ in oldest[: len(_sessions) - _MAX_SESSIONS]:
        _sessions.pop(key, None)


def ensure_live_telemetry_session(plan_id: str, session_id: str) -> None:
    with _registry_lock:
        now = time.monotonic()
        _prune_locked(now)
        key = _key(plan_id, session_id)
        if key not in _sessions:
            _sessions[key] = _LiveTelemetrySession(plan_id=plan_id, session_id=session_id)


def publish_live_telemetry(
    plan_id: str,
    session_id: str | None,
    payload: dict[str, Any],
    *,
    status: str = "streaming",
) -> dict[str, Any] | None:
    if not session_id:
        return None
    ensure_live_telemetry_session(plan_id, session_id)
    with _registry_lock:
        session = _sessions[_key(plan_id, session_id)]
    return session.publish(payload, status=status).to_dict()


def wait_for_live_telemetry(
    plan_id: str,
    session_id: str,
    after: int,
    *,
    timeout: float = 12.0,
) -> dict[str, Any] | None:
    ensure_live_telemetry_session(plan_id, session_id)
    with _registry_lock:
        session = _sessions[_key(plan_id, session_id)]
    event = session.wait_after(after, timeout=timeout)
    return None if event is None else event.to_dict()


def latest_live_telemetry(plan_id: str, session_id: str) -> dict[str, Any] | None:
    with _registry_lock:
        session = _sessions.get(_key(plan_id, session_id))
    if session is None:
        return None
    event = session.latest()
    return None if event is None else event.to_dict()


def live_telemetry_terminal(event: dict[str, Any] | None) -> bool:
    return bool(event and str(event.get("status", "")) in _TERMINAL_STATUSES)
