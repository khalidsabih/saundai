from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..set_models import SetPlan

SAFETY_GOVERNOR_VERSION = "1.0"


class SafetyViolation(RuntimeError):
    """Raised when deterministic execution violates a hard runtime safety invariant."""


class SafetyAction(str, Enum):
    ALLOW = "allow"
    HOLD = "hold"
    USE_BACKUP = "use_backup"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Bounded runtime limits used by the non-musical safety governor."""

    warning_abs_peak: float = 1.0001
    maximum_abs_peak: float = 2.0
    maximum_cpu_ratio: float = 0.98
    minimum_buffer_seconds: float = 0.25
    minimum_deadline_seconds: float = 0.15
    maximum_disk_stall_seconds: float = 0.75
    maximum_prefetch_seconds: float = 2.5
    require_local_assets: bool = True

    @classmethod
    def from_environment(cls) -> "SafetyLimits":
        def number(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return bool(default)
            return str(raw).strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            warning_abs_peak=max(0.05, number("DJAI_SAFETY_WARNING_ABS_PEAK", 1.0001)),
            maximum_abs_peak=max(0.10, number("DJAI_SAFETY_MAX_ABS_PEAK", 2.0)),
            maximum_cpu_ratio=max(0.10, min(4.0, number("DJAI_SAFETY_MAX_CPU_RATIO", 0.98))),
            minimum_buffer_seconds=max(0.0, number("DJAI_SAFETY_MIN_BUFFER_SECONDS", 0.25)),
            minimum_deadline_seconds=max(0.0, number("DJAI_SAFETY_MIN_DEADLINE_SECONDS", 0.15)),
            maximum_disk_stall_seconds=max(0.05, number("DJAI_SAFETY_MAX_DISK_STALL_SECONDS", 0.75)),
            maximum_prefetch_seconds=max(0.05, number("DJAI_SAFETY_MAX_PREFETCH_SECONDS", 2.5)),
            require_local_assets=flag("DJAI_SAFETY_REQUIRE_LOCAL_ASSETS", True),
        )


@dataclass(frozen=True, slots=True)
class SafetyTelemetry:
    """Optional observations supplied by the execution runtime.

    Missing values are treated as unknown, not as failures. This lets the same
    governor protect offline rendering today and accept richer live telemetry
    later without changing the musical plan.
    """

    cpu_ratio: float | None = None
    buffer_seconds: float | None = None
    deadline_seconds: float | None = None
    prefetch_ready: bool | None = None
    prefetch_latency_seconds: float | None = None
    disk_stall_seconds: float | None = None
    audio_underruns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    sequence: int
    action: str
    reason: str
    command_kind: str | None = None
    command_id: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyGovernor:
    """Fail-closed guard between Executive commitment and DSP execution.

    This component has safety authority only. It never chooses tracks,
    transition techniques, cue points, stems, or set direction. Hard integrity
    violations stop execution. Soft resource pressure is surfaced as HOLD or
    USE_BACKUP. Preparation may continue while held, but the deterministic
    Executor treats any non-ALLOW decision as an audible TRANSITION barrier.
    Live playback can therefore keep the outgoing owner safe while resources
    recover instead of advancing into NEXT under pressure.
    """

    def __init__(
        self,
        plan: SetPlan,
        *,
        sample_rate: int,
        limits: SafetyLimits | None = None,
    ) -> None:
        self.plan = plan
        self.sample_rate = int(sample_rate)
        self.limits = limits or SafetyLimits.from_environment()
        self._decisions: list[SafetyDecision] = []
        self._sequence = 0
        self._last_action = SafetyAction.ALLOW
        self._preflight()

    @property
    def last_action(self) -> str:
        return self._last_action.value

    def _record(
        self,
        action: SafetyAction,
        reason: str,
        *,
        command_kind: str | None = None,
        command_id: str | None = None,
        checks: dict[str, Any] | None = None,
    ) -> SafetyDecision:
        self._sequence += 1
        self._last_action = action
        decision = SafetyDecision(
            sequence=self._sequence,
            action=action.value,
            reason=str(reason),
            command_kind=command_kind,
            command_id=command_id,
            checks=dict(checks or {}),
        )
        self._decisions.append(decision)
        return decision

    def _preflight(self) -> None:
        if self.sample_rate <= 0:
            decision = self._record(SafetyAction.STOP, "invalid executor sample rate")
            raise SafetyViolation(decision.reason)
        missing: list[str] = []
        if self.limits.require_local_assets:
            for index in self.plan.order:
                path = Path(self.plan.tracks[index].path).expanduser()
                if not path.is_file():
                    missing.append(str(path))
        if missing:
            decision = self._record(
                SafetyAction.STOP,
                "one or more committed audio assets are unavailable",
                checks={"missing_assets": missing},
            )
            raise SafetyViolation(decision.reason + ": " + ", ".join(missing[:3]))
        self._record(
            SafetyAction.ALLOW,
            "preflight passed",
            checks={
                "sample_rate": self.sample_rate,
                "track_count": len(self.plan.order),
                "local_assets_required": self.limits.require_local_assets,
            },
        )

    def hard_stop(
        self,
        reason: str,
        *,
        command_kind: str | None = None,
        command_id: str | None = None,
        checks: dict[str, Any] | None = None,
    ) -> SafetyDecision:
        return self._record(
            SafetyAction.STOP,
            reason,
            command_kind=command_kind,
            command_id=command_id,
            checks=checks,
        )

    def evaluate_telemetry(
        self,
        telemetry: SafetyTelemetry | None,
        *,
        command_kind: str | None = None,
        command_id: str | None = None,
        backup_available: bool = False,
    ) -> SafetyDecision:
        telemetry = telemetry or SafetyTelemetry()
        checks = telemetry.to_dict()

        if int(telemetry.audio_underruns) > 0:
            action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
            return self._record(
                action,
                "audio underrun observed; do not advance into unprotected material",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        if (
            telemetry.disk_stall_seconds is not None
            and telemetry.disk_stall_seconds > self.limits.maximum_disk_stall_seconds
        ):
            action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
            return self._record(
                action,
                "storage read stall exceeds configured venue safety envelope",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        if (
            telemetry.prefetch_latency_seconds is not None
            and telemetry.prefetch_latency_seconds > self.limits.maximum_prefetch_seconds
            and telemetry.prefetch_ready is False
        ):
            action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
            return self._record(
                action,
                "incoming asset prefetch is too slow for the live safety envelope",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        if telemetry.cpu_ratio is not None and telemetry.cpu_ratio > self.limits.maximum_cpu_ratio:
            action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
            return self._record(
                action,
                "runtime CPU pressure exceeds configured safety envelope",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        if telemetry.buffer_seconds is not None and telemetry.buffer_seconds < self.limits.minimum_buffer_seconds:
            action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
            return self._record(
                action,
                "prepared audio buffer is below configured safety reserve",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        if telemetry.deadline_seconds is not None and telemetry.deadline_seconds < self.limits.minimum_deadline_seconds:
            if telemetry.prefetch_ready is False:
                action = SafetyAction.USE_BACKUP if backup_available else SafetyAction.HOLD
                return self._record(
                    action,
                    "next execution deadline is too close and required material is not ready",
                    command_kind=command_kind,
                    command_id=command_id,
                    checks=checks,
                )
        return self._record(
            SafetyAction.ALLOW,
            "runtime telemetry within safety envelope",
            command_kind=command_kind,
            command_id=command_id,
            checks=checks,
        )

    def admit_command(
        self,
        *,
        command_kind: str,
        command_id: str,
        track_index: int | None,
        source_cursor: float,
        telemetry: SafetyTelemetry | None = None,
        backup_available: bool = False,
    ) -> SafetyDecision:
        checks: dict[str, Any] = {
            "track_index": track_index,
            "source_cursor": round(float(source_cursor), 6),
        }
        if not math.isfinite(float(source_cursor)) or float(source_cursor) < -1e-7:
            decision = self._record(
                SafetyAction.STOP,
                "executor source cursor is invalid",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
            raise SafetyViolation(decision.reason)
        if track_index is not None:
            if track_index < 0 or track_index >= len(self.plan.tracks):
                decision = self._record(
                    SafetyAction.STOP,
                    "executor referenced a track outside the committed crate",
                    command_kind=command_kind,
                    command_id=command_id,
                    checks=checks,
                )
                raise SafetyViolation(decision.reason)
            if self.limits.require_local_assets:
                path = Path(self.plan.tracks[track_index].path).expanduser()
                checks["asset"] = str(path)
                checks["asset_exists"] = path.is_file()
                if not path.is_file():
                    decision = self._record(
                        SafetyAction.STOP,
                        "committed audio asset disappeared before execution",
                        command_kind=command_kind,
                        command_id=command_id,
                        checks=checks,
                    )
                    raise SafetyViolation(decision.reason)
        decision = self.evaluate_telemetry(
            telemetry,
            command_kind=command_kind,
            command_id=command_id,
            backup_available=backup_available,
        )
        return decision

    def observe_audio(
        self,
        audio: np.ndarray,
        *,
        command_kind: str,
        command_id: str,
    ) -> SafetyDecision:
        block = np.asarray(audio)
        if block.size == 0:
            return self._record(
                SafetyAction.ALLOW,
                "empty audio block accepted",
                command_kind=command_kind,
                command_id=command_id,
                checks={"frames": 0, "peak": 0.0},
            )
        finite = bool(np.isfinite(block).all())
        peak = float(np.max(np.abs(block))) if finite else float("inf")
        checks = {
            "frames": int(block.shape[-1]),
            "finite": finite,
            "peak": peak if math.isfinite(peak) else "non_finite",
            "warning_abs_peak": self.limits.warning_abs_peak,
            "maximum_abs_peak": self.limits.maximum_abs_peak,
        }
        if not finite:
            decision = self._record(
                SafetyAction.STOP,
                "DSP backend produced non-finite audio",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
            raise SafetyViolation(decision.reason)
        if peak > self.limits.maximum_abs_peak + 1e-7:
            decision = self._record(
                SafetyAction.STOP,
                "DSP backend exceeded configured hard absolute peak safety limit",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
            raise SafetyViolation(decision.reason)
        if peak > self.limits.warning_abs_peak + 1e-7:
            return self._record(
                SafetyAction.HOLD,
                "audio peak exceeded the advisory clipping-risk threshold",
                command_kind=command_kind,
                command_id=command_id,
                checks=checks,
            )
        return self._record(
            SafetyAction.ALLOW,
            "audio block within safety envelope",
            command_kind=command_kind,
            command_id=command_id,
            checks=checks,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SAFETY_GOVERNOR_VERSION,
            "authority": "safety_only",
            "musical_authority": False,
            "can_choose_track": False,
            "can_choose_technique": False,
            "can_modify_committed_program": False,
            "actions": [item.value for item in SafetyAction],
            "last_action": self.last_action,
            "limits": asdict(self.limits),
            "decisions": [item.to_dict() for item in self._decisions],
        }
