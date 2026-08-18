from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EXECUTIVE_CONTRACTS_VERSION = "1.0"


class StaleExecutiveResultError(RuntimeError):
    """Raised when a specialist result was computed from an obsolete state snapshot."""


@dataclass(frozen=True, slots=True)
class ExecutiveSnapshot:
    """Immutable state passed from the Executive to specialist services."""

    state_version: int
    source: str
    journey: str
    listener: str
    track_count: int
    crowd_state: dict[str, Any] = field(default_factory=dict)
    blocked_pairs: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocked_pairs"] = [list(pair) for pair in self.blocked_pairs]
        payload["contract_version"] = EXECUTIVE_CONTRACTS_VERSION
        return payload


@dataclass(frozen=True, slots=True)
class ExecutiveServiceResult:
    """Version-bound result returned by an Executive-owned specialist service."""

    component: str
    snapshot_version: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_current_state_version(*, result_version: int, current_version: int, component: str) -> None:
    if int(result_version) != int(current_version):
        raise StaleExecutiveResultError(
            f"Discarded stale {component} result from state version {result_version}; "
            f"current Executive state version is {current_version}."
        )
