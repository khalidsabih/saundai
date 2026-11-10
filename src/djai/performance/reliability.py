from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from ..set_models import SetPlan, TransitionEdge

VENUE_RELIABILITY_VERSION = "15.0"


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    PROMOTE_HOT_BACKUP = "promote_hot_backup"
    REPLAN_FUTURE = "replan_future"
    HOLD_CURRENT = "hold_current"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True, slots=True)
class AssetHealth:
    track_index: int
    path: str
    exists: bool
    readable: bool
    size_bytes: int
    reason: str = ""

    @property
    def healthy(self) -> bool:
        return self.exists and self.readable and self.size_bytes > 0

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "healthy": self.healthy}


def inspect_track_asset(plan: SetPlan, track_index: int) -> AssetHealth:
    path = Path(plan.tracks[int(track_index)].path).expanduser()
    exists = path.is_file()
    readable = bool(exists and os.access(path, os.R_OK))
    size = 0
    reason = ""
    if exists:
        try:
            size = int(path.stat().st_size)
        except OSError as exc:
            reason = f"stat_failed:{type(exc).__name__}"
    else:
        reason = "missing"
    if exists and not readable:
        reason = "unreadable"
    if exists and readable and size <= 0:
        reason = "empty"
    return AssetHealth(int(track_index), str(path), exists, readable, size, reason)


def select_hot_backup(edge: TransitionEdge):
    """Choose an already-prepared non-primary executable alternative."""

    primary = str(edge.plan.execution_signature or "")
    for candidate in edge.alternatives:
        signature = str(candidate.execution_signature or "")
        if signature and signature != primary:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class LiveRecoveryDecision:
    action: str
    reason: str
    failed_track: int | None = None
    backup_execution_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_asset_recovery(
    *,
    failed_track: int,
    current_track: int,
    next_edge: TransitionEdge | None,
) -> LiveRecoveryDecision:
    """Classify asset failure by NOW/NEXT/FUTURE authority."""

    failed = int(failed_track)
    current = int(current_track)
    if failed == current:
        return LiveRecoveryDecision(
            RecoveryAction.HOLD_CURRENT.value,
            "current audible asset failed; preserve buffered audio and move to controlled safe-stop if continuity cannot be maintained",
            failed,
        )
    if next_edge is not None and failed == int(next_edge.to_index):
        backup = select_hot_backup(next_edge)
        if backup is not None:
            return LiveRecoveryDecision(
                RecoveryAction.PROMOTE_HOT_BACKUP.value,
                "locked NEXT asset/program failed; promote already-prepared hot backup",
                failed,
                str(backup.execution_signature or ""),
            )
        return LiveRecoveryDecision(
            RecoveryAction.HOLD_CURRENT.value,
            "locked NEXT failed without a hot backup; hold outgoing owner rather than enter invalid material",
            failed,
        )
    return LiveRecoveryDecision(
        RecoveryAction.REPLAN_FUTURE.value,
        "failed asset is FUTURE-only and can be removed from the replannable suffix",
        failed,
    )


@dataclass(frozen=True, slots=True)
class WatchdogLimits:
    heartbeat_timeout_seconds: float = 3.0
    audio_progress_timeout_seconds: float = 1.5
    minimum_buffer_seconds: float = 0.20
    maximum_cpu_ratio: float = 0.99


@dataclass(frozen=True, slots=True)
class WatchdogObservation:
    monotonic_seconds: float
    heartbeat_age_seconds: float | None = None
    audio_progress_age_seconds: float | None = None
    buffer_seconds: float | None = None
    cpu_ratio: float | None = None
    fallback_ready: bool = False


@dataclass(frozen=True, slots=True)
class WatchdogDecision:
    action: str
    reason: str
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VenueWatchdog:
    """Deliberately non-musical watchdog suitable for a separate supervisor."""

    def __init__(self, limits: WatchdogLimits | None = None) -> None:
        self.limits = limits or WatchdogLimits()

    def evaluate(self, observation: WatchdogObservation) -> WatchdogDecision:
        checks = asdict(observation)
        if (
            observation.audio_progress_age_seconds is not None
            and observation.audio_progress_age_seconds > self.limits.audio_progress_timeout_seconds
        ):
            return WatchdogDecision(
                RecoveryAction.PROMOTE_HOT_BACKUP.value if observation.fallback_ready else RecoveryAction.SAFE_STOP.value,
                "audio progress heartbeat stalled",
                checks,
            )
        if (
            observation.heartbeat_age_seconds is not None
            and observation.heartbeat_age_seconds > self.limits.heartbeat_timeout_seconds
        ):
            return WatchdogDecision(
                RecoveryAction.PROMOTE_HOT_BACKUP.value if observation.fallback_ready else RecoveryAction.SAFE_STOP.value,
                "executor heartbeat stalled",
                checks,
            )
        if observation.buffer_seconds is not None and observation.buffer_seconds < self.limits.minimum_buffer_seconds:
            return WatchdogDecision(
                RecoveryAction.PROMOTE_HOT_BACKUP.value if observation.fallback_ready else RecoveryAction.HOLD_CURRENT.value,
                "audio reserve below watchdog threshold",
                checks,
            )
        if observation.cpu_ratio is not None and observation.cpu_ratio > self.limits.maximum_cpu_ratio:
            return WatchdogDecision(
                RecoveryAction.HOLD_CURRENT.value,
                "system CPU pressure exceeds venue watchdog envelope",
                checks,
            )
        return WatchdogDecision(RecoveryAction.CONTINUE.value, "runtime healthy", checks)


class SessionJournal:
    """Atomic safe-boundary journal for restart/recovery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(
        self,
        *,
        plan_id: str,
        commit_digest: str,
        schedule_generation: int,
        current_track: int,
        next_track: int | None,
        source_cursor: float,
        set_time: float,
        edge_position: int,
        prepared_backup_signatures: Sequence[str] = (),
    ) -> Path:
        payload = {
            "version": VENUE_RELIABILITY_VERSION,
            "recorded_at_unix": time.time(),
            "plan_id": str(plan_id),
            "commit_digest": str(commit_digest),
            "schedule_generation": int(schedule_generation),
            "current_track": int(current_track),
            "next_track": (int(next_track) if next_track is not None else None),
            "source_cursor": round(float(source_cursor), 6),
            "set_time": round(float(set_time), 6),
            "edge_position": int(edge_position),
            "prepared_backup_signatures": [str(item) for item in prepared_backup_signatures if item],
            "recovery_semantics": "safe_boundary_only",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        return self.path

    def read(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if str(payload.get("version")) != VENUE_RELIABILITY_VERSION:
            return None
        return payload
