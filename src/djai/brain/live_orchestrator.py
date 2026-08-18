from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from ..models import TransitionPlan
from ..set_models import TransitionEdge
from .authority import ExecutiveFutureCommit, build_future_commit
from .crowd_state import crowd_directive_from_mapping

LIVE_EXECUTIVE_VERSION = "1.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class LiveExecutiveConfig:
    enabled: bool = True
    future_depth: int = 3
    replan_every_edges: int = 1
    max_replans: int = 2
    minimum_future_tracks: int = 2

    @classmethod
    def from_environment(cls) -> "LiveExecutiveConfig":
        return cls(
            enabled=_env_bool("DJAI_LIVE_EXECUTIVE_REPLAN", True),
            future_depth=_env_int("DJAI_LIVE_FUTURE_DEPTH", 3, minimum=1, maximum=4),
            replan_every_edges=_env_int("DJAI_LIVE_REPLAN_EVERY_EDGES", 1, minimum=1, maximum=8),
            max_replans=_env_int("DJAI_LIVE_MAX_REPLANS", 2, minimum=0, maximum=8),
            minimum_future_tracks=_env_int("DJAI_LIVE_MIN_FUTURE_TRACKS", 2, minimum=1, maximum=8),
        )


@dataclass(frozen=True, slots=True)
class LiveReplanRequest:
    cycle: int
    schedule_generation: int
    replace_from_edge: int
    anchor_track: int
    anchor_source_cursor: float
    anchor_effective_bpm: float
    anchor_render_rate: float
    history_order: tuple[int, ...]
    history_execution_signatures: tuple[str, ...]
    remaining_indices: tuple[int, ...]
    fixed_closer: int | None
    journey: str
    listener: str
    reason: str
    crowd_state: dict[str, Any] = field(default_factory=dict)
    crowd_directive: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["history_order"] = list(self.history_order)
        payload["history_execution_signatures"] = list(self.history_execution_signatures)
        payload["remaining_indices"] = list(self.remaining_indices)
        return payload


@dataclass(frozen=True, slots=True)
class LiveFutureCandidate:
    cycle: int
    schedule_generation: int
    replace_from_edge: int
    anchor_track: int
    anchor_source_cursor: float
    order_suffix: tuple[int, ...]
    transitions: tuple[TransitionEdge, ...]
    planning_revision: int
    blocked_pairs: tuple[tuple[int, int], ...] = ()
    decisions: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "schedule_generation": self.schedule_generation,
            "replace_from_edge": self.replace_from_edge,
            "anchor_track": self.anchor_track,
            "anchor_source_cursor": round(float(self.anchor_source_cursor), 6),
            "order_suffix": list(self.order_suffix),
            "execution_signatures": [str(edge.plan.execution_signature or "") for edge in self.transitions],
            "planning_revision": self.planning_revision,
            "blocked_pairs": [list(pair) for pair in self.blocked_pairs],
            "decisions": list(self.decisions),
        }


@dataclass(frozen=True, slots=True)
class LiveFutureCommitResult:
    commit: ExecutiveFutureCommit
    transitions: tuple[TransitionEdge, ...]
    candidate: LiveFutureCandidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LiveExecutiveEvent:
    sequence: int
    action: str
    cycle: int
    schedule_generation: int
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveExecutiveController:
    """Single-writer control loop for safe receding-horizon live replanning.

    NOW and the currently committed NEXT transition are never rewritten. A
    background planner may reconsider only the FUTURE suffix. The candidate is
    accepted only at the following safe boundary, before that former FUTURE
    edge becomes NEXT. Late or stale work is discarded and the already
    prepared schedule continues unchanged.
    """

    def __init__(self, *, config: LiveExecutiveConfig | None = None) -> None:
        self.config = config or LiveExecutiveConfig.from_environment()
        self._cycle = 0
        self._event_sequence = 0
        self._events: list[LiveExecutiveEvent] = []
        self._last_requested_edge = -1
        self._last_committed_generation = 0

    def _record(
        self,
        action: str,
        *,
        schedule_generation: int,
        detail: str,
        **payload: Any,
    ) -> None:
        self._event_sequence += 1
        self._events.append(
            LiveExecutiveEvent(
                sequence=self._event_sequence,
                action=str(action),
                cycle=self._cycle,
                schedule_generation=int(schedule_generation),
                detail=str(detail),
                payload=dict(payload),
            )
        )

    def make_request(
        self,
        *,
        schedule_generation: int,
        current_edge_position: int,
        locked_next: TransitionEdge,
        history_order: Sequence[int],
        history_transitions: Sequence[TransitionEdge],
        remaining_indices: Sequence[int],
        journey: str,
        listener: str,
        crowd_state: dict[str, Any] | None = None,
        reason: str = "periodic_receding_horizon",
    ) -> LiveReplanRequest | None:
        if not self.config.enabled:
            return None
        remaining = tuple(int(item) for item in remaining_indices)
        if len(remaining) < self.config.minimum_future_tracks:
            return None
        if current_edge_position < 0:
            return None
        if current_edge_position % self.config.replan_every_edges != 0:
            return None
        replace_from_edge = int(current_edge_position) + 1
        if replace_from_edge == self._last_requested_edge:
            return None
        self._cycle += 1
        self._last_requested_edge = replace_from_edge
        anchor_track = int(locked_next.to_index)
        plan = locked_next.plan
        fixed_closer = int(remaining[-1]) if remaining else None
        crowd_payload = dict(crowd_state or {})
        crowd_directive = crowd_directive_from_mapping(crowd_payload)
        request = LiveReplanRequest(
            cycle=self._cycle,
            schedule_generation=int(schedule_generation),
            replace_from_edge=replace_from_edge,
            anchor_track=anchor_track,
            anchor_source_cursor=round(float(plan.post_transition_source_end_b), 6),
            anchor_effective_bpm=round(
                float(plan.post_transition_bpm_b or plan.source_bpm_b * plan.post_transition_rate_b),
                6,
            ),
            anchor_render_rate=round(float(plan.post_transition_rate_b), 8),
            history_order=tuple(int(item) for item in history_order),
            history_execution_signatures=tuple(
                str(edge.plan.execution_signature or "") for edge in history_transitions
            ),
            remaining_indices=remaining,
            fixed_closer=fixed_closer,
            journey=str(journey),
            listener=str(listener),
            reason=str(reason),
            crowd_state=crowd_payload,
            crowd_directive=crowd_directive.to_dict(),
        )
        self._record(
            "replan_requested",
            schedule_generation=schedule_generation,
            detail="Executive requested a FUTURE-only route reconsideration",
            request=request.to_dict(),
            crowd_directive=crowd_directive.to_dict(),
        )
        return request

    def discard(self, request: LiveReplanRequest, *, reason: str) -> None:
        self._record(
            "replan_discarded",
            schedule_generation=request.schedule_generation,
            detail=reason,
            replace_from_edge=request.replace_from_edge,
        )

    def commit_candidate(
        self,
        request: LiveReplanRequest,
        candidate: LiveFutureCandidate,
        *,
        current_schedule_generation: int,
    ) -> LiveFutureCommitResult | None:
        if candidate.cycle != request.cycle:
            self.discard(request, reason="planner result belongs to a different Executive cycle")
            return None
        if candidate.schedule_generation != request.schedule_generation:
            self.discard(request, reason="planner result was computed from a different schedule generation")
            return None
        if int(current_schedule_generation) != int(request.schedule_generation):
            self.discard(request, reason="planner result became stale after a newer live schedule commit")
            return None
        if candidate.replace_from_edge != request.replace_from_edge:
            self.discard(request, reason="planner result targets the wrong future boundary")
            return None
        if candidate.anchor_track != request.anchor_track:
            self.discard(request, reason="planner result does not preserve the locked NEXT anchor track")
            return None
        if abs(candidate.anchor_source_cursor - request.anchor_source_cursor) > 1e-5:
            self.discard(request, reason="planner result does not preserve the locked NEXT source state")
            return None
        signatures = tuple(str(edge.plan.execution_signature or "") for edge in candidate.transitions)
        if any(not value for value in signatures):
            self.discard(request, reason="planner returned an unsigned future transition")
            return None
        commit = build_future_commit(
            base_schedule_generation=request.schedule_generation,
            replace_from_edge=request.replace_from_edge,
            anchor_track=request.anchor_track,
            anchor_source_cursor=request.anchor_source_cursor,
            order_suffix=candidate.order_suffix,
            transitions=candidate.transitions,
        )
        self._last_committed_generation = request.schedule_generation + 1
        self._record(
            "future_committed",
            schedule_generation=request.schedule_generation,
            detail="Executive committed a replacement FUTURE schedule",
            commit=commit.to_dict(),
        )
        return LiveFutureCommitResult(
            commit=commit,
            transitions=candidate.transitions,
            candidate=candidate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LIVE_EXECUTIVE_VERSION,
            "enabled": self.config.enabled,
            "single_writer_authority": True,
            "now_mutable": False,
            "next_mutable": False,
            "future_replannable": True,
            "crowd_state_authority": "bounded_advisory",
            "transition_policy_authority": "bounded_advisory",
            "late_work_policy": "discard_keep_prepared_schedule",
            "future_depth": self.config.future_depth,
            "replan_every_edges": self.config.replan_every_edges,
            "max_replans": self.config.max_replans,
            "cycle": self._cycle,
            "last_committed_generation": self._last_committed_generation,
            "events": [event.to_dict() for event in self._events],
        }


def history_plans_from_edges(edges: Sequence[TransitionEdge]) -> list[TransitionPlan]:
    return [edge.plan for edge in edges]
