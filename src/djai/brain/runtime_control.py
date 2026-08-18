from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..set_models import SetPlan, TransitionEdge

RUNTIME_CONTROL_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Observed execution state returned to the Executive control loop."""

    state_version: int
    schedule_generation: int
    executor_phase: str
    current_track: int
    source_cursor: float
    set_time: float
    edge_position: int
    expected_command: str | None
    commit_digest: str
    safety_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HorizonSlot:
    horizon: str
    status: str
    track_index: int | None
    edge_position: int | None
    from_index: int | None
    to_index: int | None
    execution_signature: str | None
    replannable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceHorizonController:
    """NOW/NEXT/FUTURE state shared by the Executor and live Executive.

    Prepared/offline execution keeps the historical immutable behavior. Live
    execution may replace only FUTURE edges. NOW and NEXT are always reported
    non-replannable so the audio currently executing and the immediately armed
    transition remain stable.
    """

    def __init__(
        self,
        plan: SetPlan,
        *,
        future_depth: int = 3,
        continuous_replanning: bool = False,
    ) -> None:
        self.plan = plan
        self.future_depth = max(1, int(future_depth))
        self.continuous_replanning = bool(continuous_replanning)
        self.state_version = 0
        self.schedule_generation = 0
        self._order = list(plan.order)
        self._transitions = list(plan.transitions)
        self._playback = PlaybackState(
            state_version=0,
            schedule_generation=0,
            executor_phase="idle",
            current_track=int(plan.order[0]),
            source_cursor=0.0,
            set_time=0.0,
            edge_position=0,
            expected_command="load_track",
            commit_digest="",
            safety_action="allow",
        )

    def replace_schedule(
        self,
        *,
        order: Sequence[int],
        transitions: Sequence[TransitionEdge],
        schedule_generation: int,
    ) -> None:
        self._order = [int(item) for item in order]
        self._transitions = list(transitions)
        self.schedule_generation = int(schedule_generation)
        self._playback = PlaybackState(
            state_version=self._playback.state_version,
            schedule_generation=self.schedule_generation,
            executor_phase=self._playback.executor_phase,
            current_track=self._playback.current_track,
            source_cursor=self._playback.source_cursor,
            set_time=self._playback.set_time,
            edge_position=self._playback.edge_position,
            expected_command=self._playback.expected_command,
            commit_digest=self._playback.commit_digest,
            safety_action=self._playback.safety_action,
        )

    def observe(
        self,
        *,
        executor_phase: str,
        current_track: int,
        source_cursor: float,
        set_time: float,
        edge_position: int,
        expected_command: str | None,
        commit_digest: str,
        safety_action: str,
        schedule_generation: int | None = None,
        advance_version: bool = False,
    ) -> PlaybackState:
        if advance_version:
            self.state_version += 1
        if schedule_generation is not None:
            self.schedule_generation = int(schedule_generation)
        self._playback = PlaybackState(
            state_version=self.state_version,
            schedule_generation=self.schedule_generation,
            executor_phase=str(executor_phase),
            current_track=int(current_track),
            source_cursor=round(float(source_cursor), 6),
            set_time=round(float(set_time), 6),
            edge_position=max(0, int(edge_position)),
            expected_command=expected_command,
            commit_digest=str(commit_digest),
            safety_action=str(safety_action),
        )
        return self._playback

    def playback_state(self) -> PlaybackState:
        return self._playback

    def _edge_slot(
        self,
        edge_position: int,
        *,
        horizon: str,
        status: str,
        replannable: bool,
    ) -> HorizonSlot:
        if edge_position < 0 or edge_position >= len(self._transitions):
            return HorizonSlot(
                horizon=horizon,
                status="empty",
                track_index=None,
                edge_position=None,
                from_index=None,
                to_index=None,
                execution_signature=None,
                replannable=False,
            )
        edge = self._transitions[edge_position]
        return HorizonSlot(
            horizon=horizon,
            status=status,
            track_index=int(edge.to_index),
            edge_position=edge_position,
            from_index=int(edge.from_index),
            to_index=int(edge.to_index),
            execution_signature=str(edge.plan.execution_signature or ""),
            replannable=bool(replannable),
        )

    def snapshot(self) -> dict[str, Any]:
        playback = self._playback
        edge_position = min(playback.edge_position, len(self._transitions))
        now = HorizonSlot(
            horizon="now",
            status=(
                "executing"
                if playback.executor_phase not in {"complete", "failed", "idle"}
                else playback.executor_phase
            ),
            track_index=playback.current_track,
            edge_position=(edge_position if edge_position < len(self._transitions) else None),
            from_index=playback.current_track,
            to_index=(
                int(self._transitions[edge_position].to_index)
                if edge_position < len(self._transitions)
                else None
            ),
            execution_signature=(
                str(self._transitions[edge_position].plan.execution_signature or "")
                if edge_position < len(self._transitions)
                else None
            ),
            replannable=False,
        )
        next_slot = self._edge_slot(
            edge_position,
            horizon="next",
            status="prepared_committed",
            replannable=False,
        )
        future: list[dict[str, Any]] = []
        for offset in range(1, self.future_depth + 1):
            slot = self._edge_slot(
                edge_position + offset,
                horizon="future",
                status=("prepared_replannable" if self.continuous_replanning else "prepared_committed"),
                replannable=self.continuous_replanning,
            )
            if slot.status == "empty":
                break
            future.append(slot.to_dict())
        return {
            "version": RUNTIME_CONTROL_VERSION,
            "mode": "continuous_live" if self.continuous_replanning else "prepared_set",
            "continuous_replanning": self.continuous_replanning,
            "future_replannable": self.continuous_replanning,
            "now_replannable": False,
            "next_replannable": False,
            "feedback_ready_for_executive": True,
            "schedule_generation": self.schedule_generation,
            "active_order": list(self._order),
            "playback_state": playback.to_dict(),
            "now": now.to_dict(),
            "next": next_slot.to_dict(),
            "future": future,
        }
