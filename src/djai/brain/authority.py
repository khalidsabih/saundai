from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..set_models import SetPlan, TransitionEdge

EXECUTIVE_COMMIT_CONTRACT_VERSION = "1.0"
EXECUTIVE_LIVE_COMMIT_VERSION = "1.0"
EXECUTIVE_AUTHORITY = "dj_executive_v5"


class ExecutiveAuthorityError(RuntimeError):
    """Raised when an executable plan does not carry a valid Executive commit receipt."""


def _digest_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class ExecutiveSetCommit:
    version: str
    authority: str
    state_version: int
    order: tuple[int, ...]
    transaction_ids: tuple[str, ...]
    execution_signatures: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["order"] = list(self.order)
        payload["transaction_ids"] = list(self.transaction_ids)
        payload["execution_signatures"] = list(self.execution_signatures)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutiveFutureCommit:
    version: str
    authority: str
    base_schedule_generation: int
    replace_from_edge: int
    anchor_track: int
    anchor_source_cursor: float
    order_suffix: tuple[int, ...]
    execution_signatures: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["order_suffix"] = list(self.order_suffix)
        payload["execution_signatures"] = list(self.execution_signatures)
        return payload


def build_future_commit(
    *,
    base_schedule_generation: int,
    replace_from_edge: int,
    anchor_track: int,
    anchor_source_cursor: float,
    order_suffix: Sequence[int],
    transitions: Sequence[TransitionEdge],
    authority: str = EXECUTIVE_AUTHORITY,
) -> ExecutiveFutureCommit:
    normalized_order = tuple(int(item) for item in order_suffix)
    signatures = tuple(str(edge.plan.execution_signature or "") for edge in transitions)
    if not normalized_order or normalized_order[0] != int(anchor_track):
        raise ExecutiveAuthorityError("Live Executive commit must begin at the locked NEXT anchor track.")
    if len(signatures) != max(0, len(normalized_order) - 1):
        raise ExecutiveAuthorityError("Live Executive commit has an inconsistent order/transition suffix.")
    if any(not signature for signature in signatures):
        raise ExecutiveAuthorityError("Live Executive commit cannot bind an unsigned transition program.")
    for position, edge in enumerate(transitions):
        if int(edge.from_index) != normalized_order[position] or int(edge.to_index) != normalized_order[position + 1]:
            raise ExecutiveAuthorityError("Live Executive commit transition endpoints do not match its order suffix.")
    payload = {
        "version": EXECUTIVE_LIVE_COMMIT_VERSION,
        "authority": authority,
        "base_schedule_generation": int(base_schedule_generation),
        "replace_from_edge": int(replace_from_edge),
        "anchor_track": int(anchor_track),
        "anchor_source_cursor": round(float(anchor_source_cursor), 6),
        "order_suffix": list(normalized_order),
        "execution_signatures": list(signatures),
    }
    return ExecutiveFutureCommit(
        version=EXECUTIVE_LIVE_COMMIT_VERSION,
        authority=authority,
        base_schedule_generation=int(base_schedule_generation),
        replace_from_edge=int(replace_from_edge),
        anchor_track=int(anchor_track),
        anchor_source_cursor=round(float(anchor_source_cursor), 6),
        order_suffix=normalized_order,
        execution_signatures=signatures,
        digest=_digest_payload(payload),
    )


def validate_future_commit(
    commit: ExecutiveFutureCommit,
    *,
    current_schedule_generation: int,
    current_edge_position: int,
    current_track: int,
    current_source_cursor: float,
    transitions: Sequence[TransitionEdge],
) -> None:
    if commit.version != EXECUTIVE_LIVE_COMMIT_VERSION:
        raise ExecutiveAuthorityError("Live Executive commit version is incompatible with this executor.")
    if commit.authority != EXECUTIVE_AUTHORITY:
        raise ExecutiveAuthorityError("Executor rejected a live schedule not committed by DJ Executive.")
    if int(commit.base_schedule_generation) != int(current_schedule_generation):
        raise ExecutiveAuthorityError("Live Executive commit was computed from a stale schedule generation.")
    if int(commit.replace_from_edge) != int(current_edge_position):
        raise ExecutiveAuthorityError("Live Executive commit targets a boundary that is no longer FUTURE-safe.")
    if int(commit.anchor_track) != int(current_track):
        raise ExecutiveAuthorityError("Live Executive commit anchor track does not match executor state.")
    if abs(float(commit.anchor_source_cursor) - float(current_source_cursor)) > 1e-5:
        raise ExecutiveAuthorityError("Live Executive commit anchor source cursor does not match executor state.")
    expected = build_future_commit(
        base_schedule_generation=commit.base_schedule_generation,
        replace_from_edge=commit.replace_from_edge,
        anchor_track=commit.anchor_track,
        anchor_source_cursor=commit.anchor_source_cursor,
        order_suffix=commit.order_suffix,
        transitions=transitions,
        authority=commit.authority,
    )
    if commit.execution_signatures != expected.execution_signatures or commit.digest != expected.digest:
        raise ExecutiveAuthorityError("Live Executive commit digest or execution signatures are invalid.")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    version: str
    authority: str
    commit_digest: str
    state_version: int
    success: bool
    duration_seconds: float
    peak: float
    verified_transitions: int
    transition_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_set_commit_receipt(
    *,
    order: Sequence[int],
    transitions: Sequence[TransitionEdge],
    transaction_ids: Sequence[str],
    authority: str = EXECUTIVE_AUTHORITY,
) -> ExecutiveSetCommit:
    normalized_order = tuple(int(item) for item in order)
    signatures = tuple(str(edge.plan.execution_signature or "") for edge in transitions)
    txns = tuple(str(item) for item in transaction_ids)
    if len(signatures) != max(0, len(normalized_order) - 1):
        raise ExecutiveAuthorityError("Executive set commit cannot bind an inconsistent order/transition sequence.")
    if len(txns) != len(signatures):
        raise ExecutiveAuthorityError("Executive set commit requires one transaction receipt per transition.")
    if any(not signature for signature in signatures):
        raise ExecutiveAuthorityError("Executive set commit cannot bind an unsigned transition program.")
    payload = {
        "version": EXECUTIVE_COMMIT_CONTRACT_VERSION,
        "authority": authority,
        "state_version": len(signatures),
        "order": list(normalized_order),
        "transaction_ids": list(txns),
        "execution_signatures": list(signatures),
    }
    return ExecutiveSetCommit(
        version=EXECUTIVE_COMMIT_CONTRACT_VERSION,
        authority=authority,
        state_version=len(signatures),
        order=normalized_order,
        transaction_ids=txns,
        execution_signatures=signatures,
        digest=_digest_payload(payload),
    )


def validate_set_commit_receipt(plan: SetPlan) -> ExecutiveSetCommit | None:
    """Validate the Executive->Executor boundary for authoritative plans.

    Legacy/observe plans are intentionally accepted without a receipt so old
    saved plans and non-authoritative test modes remain renderable. Once a plan
    declares the Executive authoritative, however, execution fails closed.
    """

    report = dict(plan.metadata.get("dj_executive", {}) or {})
    if not bool(report.get("authoritative")):
        return None
    raw = dict(report.get("set_commit", {}) or {})
    if not raw:
        raise ExecutiveAuthorityError("Authoritative set has no Executive commit receipt.")
    try:
        receipt = ExecutiveSetCommit(
            version=str(raw["version"]),
            authority=str(raw["authority"]),
            state_version=int(raw["state_version"]),
            order=tuple(int(item) for item in raw["order"]),
            transaction_ids=tuple(str(item) for item in raw["transaction_ids"]),
            execution_signatures=tuple(str(item) for item in raw["execution_signatures"]),
            digest=str(raw["digest"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutiveAuthorityError("Executive commit receipt is malformed.") from exc
    if receipt.version != EXECUTIVE_COMMIT_CONTRACT_VERSION:
        raise ExecutiveAuthorityError("Executive commit receipt version is incompatible with this executor.")
    if receipt.authority != EXECUTIVE_AUTHORITY:
        raise ExecutiveAuthorityError("Executor rejected a command not committed by the current DJ Executive.")
    report_transaction_ids = tuple(
        str(item.get("transaction_id", ""))
        for item in list(report.get("transactions", []) or [])
    )
    if receipt.transaction_ids != report_transaction_ids:
        raise ExecutiveAuthorityError("Executive commit transaction ledger does not match the set receipt.")
    expected = build_set_commit_receipt(
        order=plan.order,
        transitions=plan.transitions,
        transaction_ids=receipt.transaction_ids,
        authority=receipt.authority,
    )
    if receipt.order != tuple(plan.order):
        raise ExecutiveAuthorityError("Executive commit order does not match the executable set plan.")
    current_signatures = tuple(str(edge.plan.execution_signature or "") for edge in plan.transitions)
    if receipt.execution_signatures != current_signatures:
        raise ExecutiveAuthorityError("Executive commit signatures do not match the executable transition programs.")
    if receipt.state_version != len(plan.transitions):
        raise ExecutiveAuthorityError("Executive commit state version does not match the committed transition count.")
    if receipt.digest != expected.digest:
        raise ExecutiveAuthorityError("Executive commit receipt digest is invalid.")
    return receipt


def record_execution_result(
    plan: SetPlan,
    *,
    duration_seconds: float,
    peak: float,
    verified_transitions: int,
) -> dict[str, Any] | None:
    """Return deterministic execution feedback to the authoritative Executive ledger."""

    receipt = validate_set_commit_receipt(plan)
    if receipt is None:
        return None
    result = ExecutionResult(
        version=EXECUTIVE_COMMIT_CONTRACT_VERSION,
        authority=receipt.authority,
        commit_digest=receipt.digest,
        state_version=receipt.state_version,
        success=int(verified_transitions) == len(plan.transitions),
        duration_seconds=round(float(duration_seconds), 6),
        peak=round(float(peak), 6),
        verified_transitions=int(verified_transitions),
        transition_count=len(plan.transitions),
    ).to_dict()
    report = dict(plan.metadata.get("dj_executive", {}) or {})
    report["execution_result"] = result
    report["execution_feedback_closed_loop"] = True
    plan.metadata["dj_executive"] = report
    return result


def record_runtime_feedback(
    plan: SetPlan,
    *,
    executor_snapshot: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    """Attach non-musical runtime observations for the next Executive cycle.

    V6 prepared-set execution records this feedback but does not yet permit it
    to replan committed audio. The contract is intentionally ready for the
    future NOW/NEXT/FUTURE live controller.
    """

    receipt = validate_set_commit_receipt(plan)
    if receipt is None:
        return None
    safety = dict(executor_snapshot.get("safety_governor", {}) or {})
    horizons = dict(executor_snapshot.get("performance_horizons", {}) or {})
    playback = dict(horizons.get("playback_state", {}) or {})
    live_executive = dict(executor_snapshot.get("live_executive", {}) or {})
    continuous = bool(executor_snapshot.get("continuous_replanning", False))
    feedback = {
        "version": "2.0",
        "authority": receipt.authority,
        "commit_digest": receipt.digest,
        "mode": str(mode),
        "executor_phase": str(executor_snapshot.get("phase", "unknown")),
        "safety_action": str(safety.get("last_action", "unknown")),
        "playback_state": playback,
        "performance_horizons": horizons,
        "continuous_replanning": continuous,
        "may_change_committed_now": False,
        "may_change_committed_next": False,
        "future_replannable": continuous,
        "schedule_generation": int(executor_snapshot.get("schedule_generation", 0) or 0),
        "active_order": list(executor_snapshot.get("active_order", []) or []),
        "live_executive": live_executive,
        "feedback_ready_for_next_executive_cycle": True,
    }
    report = dict(plan.metadata.get("dj_executive", {}) or {})
    report["runtime_feedback"] = feedback
    report["runtime_feedback_connected"] = True
    report["runtime_feedback_authority"] = "active_future_only" if continuous else "observe"
    plan.metadata["dj_executive"] = report
    return feedback
