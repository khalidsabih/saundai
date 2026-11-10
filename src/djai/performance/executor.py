from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence

from ..brain.authority import (
    ExecutiveFutureCommit,
    validate_future_commit,
    validate_set_commit_receipt,
)
from ..brain.runtime_control import PerformanceHorizonController
from ..decision import validate_execution_plan
from ..set_models import SetPlan, TransitionEdge
from .safety import SafetyGovernor, SafetyTelemetry
from .reliability import SessionJournal

EXECUTOR_CONTRACT_VERSION = "1.0"


class ExecutorProtocolError(RuntimeError):
    """Raised when deterministic execution deviates from the committed schedule."""


class ExecutorSafetyHold(ExecutorProtocolError):
    """A non-fatal Safety Governor barrier prevented an audible transition."""

    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = dict(decision)
        super().__init__(
            f"Safety Governor blocked audible transition: {self.decision.get('action', 'hold')} - "
            f"{self.decision.get('reason', 'runtime safety barrier')}"
        )


class ExecutorPhase(str, Enum):
    IDLE = "idle"
    ARMED = "armed"
    LOADING = "loading"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class ExecutionCommandKind(str, Enum):
    LOAD_TRACK = "load_track"
    BODY = "body"
    TRANSITION = "transition"
    TEMPO_RELEASE = "tempo_release"
    FINAL_BODY = "final_body"


@dataclass(frozen=True, slots=True)
class ExecutionCommand:
    version: str
    command_id: str
    sequence: int
    kind: str
    edge_position: int | None
    track_index: int | None
    from_index: int | None
    to_index: int | None
    execution_signature: str | None
    expected_source_cursor: float
    target_source_cursor: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutorEvent:
    sequence: int
    phase: str
    action: str
    command_id: str | None
    command_kind: str | None
    track_index: int | None
    edge_position: int | None
    source_cursor: float
    set_time: float
    frames: int
    verified: bool | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _close(left: float, right: float, *, tolerance: float = 1e-5) -> bool:
    return abs(float(left) - float(right)) <= tolerance


class DeterministicSetExecutor:
    """State-machine authority for an already committed set.

    The Executive chooses the music and commits immutable TransitionPrograms.
    This object makes no musical or strategic decisions. It only enforces the
    order in which committed assets and DSP programs may be executed, validates
    source-cursor continuity, and records deterministic execution receipts.

    Heavy DSP remains in the renderer/performance engine. This class is the
    scheduler/control surface immediately above that backend.
    """

    def __init__(
        self,
        plan: SetPlan,
        *,
        sample_rate: int,
        transitions: Sequence[TransitionEdge] | None = None,
        safety_governor: SafetyGovernor | None = None,
        continuous_replanning: bool = False,
    ) -> None:
        if not plan.order:
            raise ExecutorProtocolError("Executor cannot arm an empty set plan.")
        if int(sample_rate) <= 0:
            raise ExecutorProtocolError("Executor requires a positive sample rate.")

        receipt = validate_set_commit_receipt(plan)
        self.plan = plan
        self.sample_rate = int(sample_rate)
        self.transitions = list(transitions if transitions is not None else plan.transitions)
        self.active_order = list(plan.order)
        self.schedule_generation = 0
        self._live_future_commits: list[dict[str, Any]] = []
        self.continuous_replanning = bool(continuous_replanning)
        self.commit_digest = receipt.digest if receipt is not None else f"legacy:{plan.plan_id}"
        self.authoritative = receipt is not None
        self.phase = ExecutorPhase.IDLE
        self.current_track = int(plan.order[0])
        self.source_cursor = 0.0
        self.edge_position = 0
        self.verified_transitions = 0
        self._sequence = 0
        self._event_sequence = 0
        self._pending: ExecutionCommand | None = None
        self._expected_kind = ExecutionCommandKind.LOAD_TRACK
        self._initial_track_loaded = False
        self._events: list[ExecutorEvent] = []
        self._telemetry = SafetyTelemetry()
        self.safety = safety_governor or SafetyGovernor(plan, sample_rate=self.sample_rate)
        self.horizons = PerformanceHorizonController(
            plan,
            continuous_replanning=self.continuous_replanning,
        )
        journal_path = os.environ.get("DJAI_SESSION_JOURNAL", "").strip()
        self._session_journal = SessionJournal(journal_path) if journal_path else None

    @property
    def pending(self) -> ExecutionCommand | None:
        return self._pending

    @property
    def expected_kind(self) -> str | None:
        return self._expected_kind.value if self._expected_kind is not None else None

    def update_telemetry(
        self,
        *,
        cpu_ratio: float | None = None,
        buffer_seconds: float | None = None,
        deadline_seconds: float | None = None,
        prefetch_ready: bool | None = None,
        prefetch_latency_seconds: float | None = None,
        disk_stall_seconds: float | None = None,
        audio_underruns: int = 0,
    ) -> dict[str, Any]:
        self._telemetry = SafetyTelemetry(
            cpu_ratio=cpu_ratio,
            buffer_seconds=buffer_seconds,
            deadline_seconds=deadline_seconds,
            prefetch_ready=prefetch_ready,
            prefetch_latency_seconds=prefetch_latency_seconds,
            disk_stall_seconds=disk_stall_seconds,
            audio_underruns=max(0, int(audio_underruns)),
        )
        decision = self.safety.evaluate_telemetry(
            self._telemetry,
            backup_available=self._backup_available(self.edge_position),
        )
        self.horizons.observe(
            executor_phase=self.phase.value,
            current_track=self.current_track,
            source_cursor=self.source_cursor,
            set_time=(self._events[-1].set_time if self._events else 0.0),
            edge_position=self.edge_position,
            expected_command=self.expected_kind,
            commit_digest=self.commit_digest,
            safety_action=decision.action,
            schedule_generation=self.schedule_generation,
        )
        return decision.to_dict()

    def observe_audio(self, audio: Any, command: ExecutionCommand) -> dict[str, Any]:
        decision = self.safety.observe_audio(
            audio,
            command_kind=command.kind,
            command_id=command.command_id,
        )
        return decision.to_dict()

    def _backup_available(self, edge_position: int | None) -> bool:
        if edge_position is None or edge_position < 0 or edge_position >= len(self.transitions):
            return False
        edge = self.transitions[edge_position]
        primary = str(edge.plan.execution_signature or "")
        return any(
            str(candidate.execution_signature or "") and str(candidate.execution_signature or "") != primary
            for candidate in edge.alternatives
        )

    def _record(
        self,
        *,
        action: str,
        command: ExecutionCommand | None = None,
        set_time: float = 0.0,
        frames: int = 0,
        verified: bool | None = None,
        detail: str = "",
    ) -> None:
        self._event_sequence += 1
        self._events.append(
            ExecutorEvent(
                sequence=self._event_sequence,
                phase=self.phase.value,
                action=str(action),
                command_id=command.command_id if command is not None else None,
                command_kind=command.kind if command is not None else None,
                track_index=(
                    command.track_index
                    if command is not None and command.track_index is not None
                    else self.current_track
                ),
                edge_position=command.edge_position if command is not None else None,
                source_cursor=round(float(self.source_cursor), 6),
                set_time=round(float(set_time), 6),
                frames=max(0, int(frames)),
                verified=verified,
                detail=str(detail),
            )
        )
        self.horizons.observe(
            executor_phase=self.phase.value,
            current_track=self.current_track,
            source_cursor=self.source_cursor,
            set_time=set_time,
            edge_position=self.edge_position,
            expected_command=self.expected_kind,
            commit_digest=self.commit_digest,
            safety_action=self.safety.last_action,
            schedule_generation=self.schedule_generation,
            advance_version=action in {"command_completed", "execution_complete", "execution_failed"},
        )
        if self._session_journal is not None and action in {
            "armed",
            "command_completed",
            "future_schedule_replaced",
            "execution_complete",
        }:
            next_track = None
            backup_signatures: list[str] = []
            if self.edge_position < len(self.transitions):
                edge = self.transitions[self.edge_position]
                next_track = int(edge.to_index)
                primary = str(edge.plan.execution_signature or "")
                backup_signatures = [
                    str(candidate.execution_signature or "")
                    for candidate in edge.alternatives
                    if str(candidate.execution_signature or "")
                    and str(candidate.execution_signature or "") != primary
                ]
            self._session_journal.write(
                plan_id=self.plan.plan_id,
                commit_digest=self.commit_digest,
                schedule_generation=self.schedule_generation,
                current_track=self.current_track,
                next_track=next_track,
                source_cursor=self.source_cursor,
                set_time=set_time,
                edge_position=self.edge_position,
                prepared_backup_signatures=backup_signatures,
            )

    def arm(self) -> None:
        if self.phase is not ExecutorPhase.IDLE:
            raise ExecutorProtocolError(f"Executor can only arm from idle, not {self.phase.value}.")
        self.phase = ExecutorPhase.ARMED
        self._record(action="armed", detail="committed set accepted by deterministic executor")

    def _command(
        self,
        kind: ExecutionCommandKind,
        *,
        edge_position: int | None = None,
        track_index: int | None = None,
        from_index: int | None = None,
        to_index: int | None = None,
        execution_signature: str | None = None,
        expected_source_cursor: float | None = None,
        target_source_cursor: float | None = None,
    ) -> ExecutionCommand:
        if self.phase is not ExecutorPhase.ARMED:
            raise ExecutorProtocolError(
                f"Executor command {kind.value} cannot start while phase is {self.phase.value}."
            )
        if self._pending is not None:
            raise ExecutorProtocolError("Executor already has an in-flight command.")
        if self._expected_kind is not kind:
            expected = self._expected_kind.value if self._expected_kind is not None else "none"
            raise ExecutorProtocolError(
                f"Executor expected {expected}, received {kind.value}; committed schedule cannot be reordered."
            )
        cursor = self.source_cursor if expected_source_cursor is None else float(expected_source_cursor)
        if not _close(cursor, self.source_cursor):
            raise ExecutorProtocolError(
                f"Executor source cursor mismatch ({cursor:.6f} != {self.source_cursor:.6f})."
            )
        self._sequence += 1
        payload = {
            "version": EXECUTOR_CONTRACT_VERSION,
            "commit_digest": self.commit_digest,
            "sequence": self._sequence,
            "kind": kind.value,
            "edge_position": edge_position,
            "track_index": track_index,
            "from_index": from_index,
            "to_index": to_index,
            "execution_signature": execution_signature,
            "expected_source_cursor": round(float(cursor), 6),
            "target_source_cursor": (
                None if target_source_cursor is None else round(float(target_source_cursor), 6)
            ),
        }
        command = ExecutionCommand(
            version=EXECUTOR_CONTRACT_VERSION,
            command_id=_digest(payload),
            sequence=self._sequence,
            kind=kind.value,
            edge_position=edge_position,
            track_index=track_index,
            from_index=from_index,
            to_index=to_index,
            execution_signature=execution_signature,
            expected_source_cursor=float(cursor),
            target_source_cursor=(None if target_source_cursor is None else float(target_source_cursor)),
        )
        safety_decision = self.safety.admit_command(
            command_kind=kind.value,
            command_id=command.command_id,
            track_index=track_index,
            source_cursor=cursor,
            telemetry=self._telemetry,
            backup_available=self._backup_available(edge_position),
        )
        # Loading and preparation are allowed to continue under HOLD so the
        # system can recover. Beginning an audible transition is different:
        # Safety is an actual barrier, not telemetry decoration.
        if kind is ExecutionCommandKind.TRANSITION and safety_decision.action != "allow":
            self.horizons.observe(
                executor_phase=self.phase.value,
                current_track=self.current_track,
                source_cursor=self.source_cursor,
                set_time=(self._events[-1].set_time if self._events else 0.0),
                edge_position=self.edge_position,
                expected_command=self.expected_kind,
                commit_digest=self.commit_digest,
                safety_action=safety_decision.action,
                schedule_generation=self.schedule_generation,
            )
            raise ExecutorSafetyHold(safety_decision.to_dict())
        self._pending = command
        self.phase = (
            ExecutorPhase.LOADING
            if kind is ExecutionCommandKind.LOAD_TRACK
            else ExecutorPhase.EXECUTING
        )
        self._record(action="command_started", command=command)
        return command

    def _require_pending(self, command: ExecutionCommand, kind: ExecutionCommandKind) -> None:
        if self._pending is None or self._pending.command_id != command.command_id:
            raise ExecutorProtocolError("Executor completion does not match the in-flight command.")
        if command.kind != kind.value:
            raise ExecutorProtocolError(
                f"Executor completion expected {kind.value}, received {command.kind}."
            )

    def _complete(
        self,
        command: ExecutionCommand,
        *,
        kind: ExecutionCommandKind,
        next_kind: ExecutionCommandKind | None,
        source_cursor_after: float,
        set_time: float,
        frames: int,
        verified: bool | None = None,
        detail: str = "",
    ) -> None:
        self._require_pending(command, kind)
        if command.target_source_cursor is not None and not _close(
            source_cursor_after, command.target_source_cursor
        ):
            raise ExecutorProtocolError(
                "DSP backend reported a source cursor that differs from the committed command "
                f"({source_cursor_after:.6f} != {command.target_source_cursor:.6f})."
            )
        self.source_cursor = float(source_cursor_after)
        self._pending = None
        self._expected_kind = next_kind
        self.phase = ExecutorPhase.VERIFYING if next_kind is None else ExecutorPhase.ARMED
        self._record(
            action="command_completed",
            command=command,
            set_time=set_time,
            frames=frames,
            verified=verified,
            detail=detail,
        )

    def record_hold_audio(self, *, set_time: float, frames: int, detail: str) -> None:
        """Record audible outgoing-deck hold material without advancing source state."""
        if self.phase is not ExecutorPhase.ARMED or self._pending is not None:
            raise ExecutorProtocolError("Safety hold audio can only play at an armed command boundary.")
        self._record(
            action="safety_hold_audio",
            set_time=set_time,
            frames=frames,
            verified=True,
            detail=detail,
        )

    def begin_load_track(self, *, edge_position: int | None = None) -> ExecutionCommand:
        if not self._initial_track_loaded:
            if edge_position is not None:
                raise ExecutorProtocolError("Initial track load cannot be associated with a transition edge.")
            track_index = int(self.plan.order[0])
        else:
            if edge_position is None or edge_position != self.edge_position:
                raise ExecutorProtocolError("Incoming track load must match the current edge position.")
            if edge_position >= len(self.transitions):
                raise ExecutorProtocolError("Executor has no incoming track left to load.")
            track_index = int(self.transitions[edge_position].to_index)
        return self._command(
            ExecutionCommandKind.LOAD_TRACK,
            edge_position=edge_position,
            track_index=track_index,
            target_source_cursor=self.source_cursor,
        )

    def complete_load_track(
        self,
        command: ExecutionCommand,
        *,
        loaded_sample_rate: int,
        set_time: float = 0.0,
    ) -> None:
        self._require_pending(command, ExecutionCommandKind.LOAD_TRACK)
        if int(loaded_sample_rate) != self.sample_rate:
            self.safety.hard_stop(
                "loaded audio sample rate does not match executor contract",
                command_kind=command.kind,
                command_id=command.command_id,
                checks={"loaded_sample_rate": int(loaded_sample_rate), "executor_sample_rate": self.sample_rate},
            )
            raise ExecutorProtocolError(
                f"Loaded sample rate {loaded_sample_rate} does not match executor rate {self.sample_rate}."
            )
        if command.track_index is None:
            raise ExecutorProtocolError("Load command has no track index.")
        if not self._initial_track_loaded:
            if command.track_index != int(self.active_order[0]):
                raise ExecutorProtocolError("Initial load does not match the first committed track.")
            self.current_track = int(command.track_index)
            self._initial_track_loaded = True
            next_kind = (
                ExecutionCommandKind.BODY
                if self.transitions
                else ExecutionCommandKind.FINAL_BODY
            )
        else:
            edge = self.transitions[self.edge_position]
            if command.track_index != int(edge.to_index):
                raise ExecutorProtocolError("Incoming load does not match the committed transition target.")
            next_kind = ExecutionCommandKind.TRANSITION
        self._complete(
            command,
            kind=ExecutionCommandKind.LOAD_TRACK,
            next_kind=next_kind,
            source_cursor_after=self.source_cursor,
            set_time=set_time,
            frames=0,
            detail="audio asset loaded and sample rate verified",
        )

    def replace_future_schedule(
        self,
        commit: ExecutiveFutureCommit,
        transitions: Sequence[TransitionEdge],
    ) -> None:
        """Install an Executive-committed FUTURE suffix at a safe boundary.

        This may only run while armed, with no command in flight, immediately
        before the current edge body begins. Executed history is immutable.
        The currently playing track/source cursor must match the future commit
        anchor exactly.
        """

        if not self.continuous_replanning:
            raise ExecutorProtocolError("Executor was not armed for continuous FUTURE replanning.")
        if self.phase is not ExecutorPhase.ARMED or self._pending is not None:
            raise ExecutorProtocolError("FUTURE schedule replacement requires an idle armed executor boundary.")
        if self._expected_kind not in {ExecutionCommandKind.BODY, ExecutionCommandKind.FINAL_BODY}:
            raise ExecutorProtocolError("FUTURE schedule replacement is only allowed before an outgoing body begins.")
        replacement = list(transitions)
        validate_future_commit(
            commit,
            current_schedule_generation=self.schedule_generation,
            current_edge_position=self.edge_position,
            current_track=self.current_track,
            current_source_cursor=self.source_cursor,
            transitions=replacement,
        )
        if not replacement:
            if len(commit.order_suffix) != 1:
                raise ExecutorProtocolError("Empty FUTURE replacement must contain only the current anchor track.")
        else:
            if int(replacement[0].from_index) != self.current_track:
                raise ExecutorProtocolError("Replacement FUTURE does not begin at the current executor track.")
            for edge in replacement:
                validate_execution_plan(edge.plan)
            for left, right in zip(replacement, replacement[1:]):
                if int(left.to_index) != int(right.from_index):
                    raise ExecutorProtocolError("Replacement FUTURE transition chain is discontinuous.")

        prefix_transitions = list(self.transitions[: self.edge_position])
        prefix_order = list(self.active_order[: self.edge_position + 1])
        suffix_order = list(commit.order_suffix)
        if not suffix_order or suffix_order[0] != self.current_track:
            raise ExecutorProtocolError("Replacement FUTURE order does not preserve the current anchor track.")
        self.transitions = prefix_transitions + replacement
        self.active_order = prefix_order[:-1] + suffix_order
        if len(self.transitions) != max(0, len(self.active_order) - 1):
            raise ExecutorProtocolError("Replacement FUTURE produced an inconsistent active schedule.")
        self.schedule_generation += 1
        self._live_future_commits.append(commit.to_dict())
        self.horizons.replace_schedule(
            order=self.active_order,
            transitions=self.transitions,
            schedule_generation=self.schedule_generation,
        )
        self._record(
            action="future_schedule_replaced",
            set_time=(self._events[-1].set_time if self._events else 0.0),
            verified=True,
            detail=(
                f"Executive installed FUTURE generation {self.schedule_generation} "
                f"from edge {self.edge_position}"
            ),
        )

    def begin_body(self, *, edge_position: int, source_cursor: float) -> ExecutionCommand:
        if edge_position != self.edge_position or edge_position >= len(self.transitions):
            raise ExecutorProtocolError("Body command does not match the current committed edge.")
        edge = self.transitions[edge_position]
        if int(edge.from_index) != self.current_track:
            raise ExecutorProtocolError("Body command source track is not the executor's current track.")
        if not _close(source_cursor, self.source_cursor):
            raise ExecutorProtocolError("Renderer and executor disagree about the current source cursor.")
        return self._command(
            ExecutionCommandKind.BODY,
            edge_position=edge_position,
            track_index=self.current_track,
            from_index=int(edge.from_index),
            to_index=int(edge.to_index),
            execution_signature=str(edge.plan.execution_signature or ""),
            expected_source_cursor=source_cursor,
            target_source_cursor=float(edge.plan.exit_time_a),
        )

    def complete_body(
        self,
        command: ExecutionCommand,
        *,
        source_cursor_after: float,
        set_time: float,
        frames: int,
    ) -> None:
        self._complete(
            command,
            kind=ExecutionCommandKind.BODY,
            next_kind=ExecutionCommandKind.LOAD_TRACK,
            source_cursor_after=source_cursor_after,
            set_time=set_time,
            frames=frames,
        )

    def begin_transition(self, *, edge_position: int) -> ExecutionCommand:
        if edge_position != self.edge_position or edge_position >= len(self.transitions):
            raise ExecutorProtocolError("Transition command does not match the current committed edge.")
        edge = self.transitions[edge_position]
        if int(edge.from_index) != self.current_track:
            raise ExecutorProtocolError("Transition source track does not match executor state.")
        signature = str(edge.plan.execution_signature or "")
        if not signature:
            raise ExecutorProtocolError("Transition command requires an immutable execution signature.")
        return self._command(
            ExecutionCommandKind.TRANSITION,
            edge_position=edge_position,
            track_index=self.current_track,
            from_index=int(edge.from_index),
            to_index=int(edge.to_index),
            execution_signature=signature,
            target_source_cursor=float(edge.plan.source_end_b),
        )

    def complete_transition(
        self,
        command: ExecutionCommand,
        *,
        source_cursor_after: float,
        set_time: float,
        frames: int,
        verified: bool,
    ) -> None:
        if not bool(verified):
            self.safety.hard_stop(
                "transition verification failed",
                command_kind=command.kind,
                command_id=command.command_id,
            )
            raise ExecutorProtocolError("Transition verification failed; executor refuses to advance deck state.")
        self._require_pending(command, ExecutionCommandKind.TRANSITION)
        if command.to_index is None:
            raise ExecutorProtocolError("Transition command has no incoming track index.")
        self.current_track = int(command.to_index)
        self.verified_transitions += 1
        self._complete(
            command,
            kind=ExecutionCommandKind.TRANSITION,
            next_kind=ExecutionCommandKind.TEMPO_RELEASE,
            source_cursor_after=source_cursor_after,
            set_time=set_time,
            frames=frames,
            verified=True,
            detail="immutable transition completed and execution state advanced",
        )

    def begin_tempo_release(self, *, edge_position: int) -> ExecutionCommand:
        if edge_position != self.edge_position or edge_position >= len(self.transitions):
            raise ExecutorProtocolError("Tempo-release command does not match the current committed edge.")
        edge = self.transitions[edge_position]
        if int(edge.to_index) != self.current_track:
            raise ExecutorProtocolError("Tempo-release target track does not match executor state.")
        return self._command(
            ExecutionCommandKind.TEMPO_RELEASE,
            edge_position=edge_position,
            track_index=self.current_track,
            from_index=int(edge.from_index),
            to_index=int(edge.to_index),
            execution_signature=str(edge.plan.execution_signature or ""),
            target_source_cursor=float(edge.plan.post_transition_source_end_b),
        )

    def complete_tempo_release(
        self,
        command: ExecutionCommand,
        *,
        source_cursor_after: float,
        set_time: float,
        frames: int,
    ) -> None:
        is_last = self.edge_position + 1 >= len(self.transitions)
        next_kind = ExecutionCommandKind.FINAL_BODY if is_last else ExecutionCommandKind.BODY
        self.edge_position += 1
        self._complete(
            command,
            kind=ExecutionCommandKind.TEMPO_RELEASE,
            next_kind=next_kind,
            source_cursor_after=source_cursor_after,
            set_time=set_time,
            frames=frames,
        )

    def begin_final_body(self, *, source_cursor: float) -> ExecutionCommand:
        if int(self.active_order[-1]) != self.current_track:
            raise ExecutorProtocolError("Final body track does not match the last committed track.")
        if not _close(source_cursor, self.source_cursor):
            raise ExecutorProtocolError("Final body source cursor does not match executor state.")
        duration = float(self.plan.tracks[self.current_track].duration_seconds)
        return self._command(
            ExecutionCommandKind.FINAL_BODY,
            track_index=self.current_track,
            expected_source_cursor=source_cursor,
            target_source_cursor=duration,
        )

    def complete_final_body(
        self,
        command: ExecutionCommand,
        *,
        source_cursor_after: float,
        set_time: float,
        frames: int,
    ) -> None:
        self._complete(
            command,
            kind=ExecutionCommandKind.FINAL_BODY,
            next_kind=None,
            source_cursor_after=source_cursor_after,
            set_time=set_time,
            frames=frames,
        )

    def finish(self, *, set_time: float) -> None:
        if self.phase is not ExecutorPhase.VERIFYING or self._pending is not None:
            raise ExecutorProtocolError("Executor can only finish after the final body has completed.")
        if self.verified_transitions != len(self.transitions):
            raise ExecutorProtocolError(
                "Executor cannot complete because not every committed transition was verified."
            )
        self.phase = ExecutorPhase.COMPLETE
        self._record(
            action="execution_complete",
            set_time=set_time,
            verified=True,
            detail="all committed commands executed in deterministic order",
        )

    def fail(self, exc: BaseException, *, set_time: float = 0.0) -> None:
        if self.phase is ExecutorPhase.COMPLETE:
            raise ExecutorProtocolError("Completed execution cannot be changed to failed.")
        if self.safety.last_action != "stop":
            self.safety.hard_stop(f"executor failed: {type(exc).__name__}: {exc}")
        self.phase = ExecutorPhase.FAILED
        self._record(
            action="execution_failed",
            command=self._pending,
            set_time=set_time,
            verified=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
        self._pending = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EXECUTOR_CONTRACT_VERSION,
            "phase": self.phase.value,
            "authoritative_commit": self.authoritative,
            "commit_digest": self.commit_digest,
            "sample_rate": self.sample_rate,
            "deterministic": True,
            "neural_control": False,
            "strategic_decisions": False,
            "safety_governor": self.safety.to_dict(),
            "performance_horizons": self.horizons.snapshot(),
            "runtime_feedback_ready": True,
            "live_safety": {
                "telemetry": self._telemetry.to_dict(),
                "hot_backup_available": self._backup_available(self.edge_position),
                "deadline_protected": True,
                "underrun_protected": True,
                "storage_stall_protected": True,
            },
            "continuous_replanning": self.continuous_replanning,
            "schedule_generation": self.schedule_generation,
            "active_order": list(self.active_order),
            "live_future_commits": list(self._live_future_commits),
            "session_journal_enabled": self._session_journal is not None,
            "restart_recovery_semantics": "safe_boundary_only",
            "current_track": self.current_track,
            "source_cursor": round(float(self.source_cursor), 6),
            "edge_position": self.edge_position,
            "expected_kind": self.expected_kind,
            "verified_transitions": self.verified_transitions,
            "transition_count": len(self.transitions),
            "command_count": self._sequence,
            "events": [event.to_dict() for event in self._events],
        }
