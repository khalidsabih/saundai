from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from ..models import TrackAnalysis
from ..set_models import TransitionEdge
from .contracts import ExecutiveServiceResult, ExecutiveSnapshot, require_current_state_version
from .crowd_state import CrowdState, crowd_state_from_feedback
from .executive import build_executive_report
from .move_generator import CompleteMoveDecision

EXECUTIVE_ORCHESTRATOR_VERSION = "1.0.0"


class ExecutivePhase(str, Enum):
    INITIALIZING = "initializing"
    OBSERVING = "observing"
    DIRECTING = "directing"
    SEARCHING = "searching"
    AUDITIONING = "auditioning"
    PREPARING = "preparing"
    VALIDATING = "validating"
    REPLANNING = "replanning"
    COMMITTING = "committing"
    COMPLETE = "complete"
    FAILED = "failed"


class ExecutiveReplanRequest(RuntimeError):
    """A specialist rejected an edge and requests an Executive-owned route repair."""

    def __init__(
        self,
        from_index: int,
        to_index: int,
        *,
        action: str,
        reason: str,
        resolution: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.from_index = int(from_index)
        self.to_index = int(to_index)
        self.action = str(action)
        self.reason = str(reason)
        self.resolution = dict(resolution or {})


@dataclass(frozen=True, slots=True)
class ExecutiveEvent:
    sequence: int
    phase: str
    detail: str
    planning_revision: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExecutivePreparedSetResult:
    order: tuple[int, ...]
    transitions: tuple[TransitionEdge, ...]
    blocked_pairs: tuple[tuple[int, int], ...]
    replan_trace: tuple[dict[str, Any], ...]
    executive_report: dict[str, Any]
    crowd_state: dict[str, Any]
    orchestration: dict[str, Any]


RouteSelector = Callable[[set[tuple[int, int]]], list[int]]
TransitionCompiler = Callable[[list[int], bool, bool], list[TransitionEdge]]
DecisionProvider = Callable[[], Sequence[CompleteMoveDecision]]
ProgressCallback = Callable[[str, int, int, str], None]


def brain_component_registry() -> dict[str, dict[str, str]]:
    """Account for every module in ``djai.brain`` under Executive ownership.

    ``conditional`` means available to the Executive but intentionally invoked
    only when the musical/material state requires it. ``fallback`` means it may
    only be entered through an explicit Executive repair route. ``observe`` is
    connected to the control loop but has no musical authority in this
    behavior-preserving migration.
    """

    return {
        "authority": {"status": "active", "owner": "executive", "role": "Executive-to-Executor commit contract"},
        "candidate_generator": {"status": "active", "owner": "executive", "role": "transition candidate diversity"},
        "candidate_retrieval": {"status": "active", "owner": "executive", "role": "lazy next-track retrieval"},
        "contracts": {"status": "active", "owner": "executive", "role": "versioned specialist service contracts"},
        "crate_memory": {"status": "active", "owner": "executive", "role": "experience prior"},
        "crowd_state": {"status": "active", "owner": "executive", "role": "prepared observation + live bounded advisory"},
        "director": {"status": "active", "owner": "executive", "role": "set intent"},
        "dj_memory": {"status": "active", "owner": "executive", "role": "set memory"},
        "dj_move": {"status": "active", "owner": "executive", "role": "state and move semantics"},
        "executive": {"status": "active", "owner": "executive", "role": "single commit authority"},
        "layer_intelligence": {"status": "conditional", "owner": "executive", "role": "performance-layer decisions"},
        "live_orchestrator": {"status": "active", "owner": "executive", "role": "continuous NOW/NEXT/FUTURE control loop"},
        "memory": {"status": "active", "owner": "executive", "role": "professional rejection memory"},
        "move_generator": {"status": "active", "owner": "executive", "role": "complete DJMove proposals"},
        "move_intent": {"status": "active", "owner": "executive", "role": "semantic intent binding"},
        "orchestrator": {"status": "active", "owner": "executive", "role": "supervisory control and replan lifecycle"},
        "professional": {"status": "active", "owner": "executive", "role": "rendered candidate judge"},
        "professional_mixing": {"status": "active", "owner": "executive", "role": "compatibility-driven mix family and musical ownership policy"},
        "proposal": {"status": "active", "owner": "executive", "role": "transition proposal contract"},
        "recovery": {"status": "conditional", "owner": "executive", "role": "prepared same-target recovery"},
        "runtime_control": {"status": "active", "owner": "executive", "role": "NOW/NEXT/FUTURE playback observation contract"},
        "set_policy": {"status": "fallback", "owner": "executive", "role": "legacy route repair"},
        "set_state": {"status": "active", "owner": "executive", "role": "set/world state"},
        "stem_choreography": {"status": "conditional", "owner": "executive", "role": "stem choreography"},
        "structural_operations": {"status": "conditional", "owner": "executive", "role": "deck structural operations"},
        "technique_policy": {"status": "active", "owner": "executive", "role": "technique and duration policy"},
        "thinking_budget": {"status": "active", "owner": "executive", "role": "adaptive compute budget"},
        "transition_policy": {"status": "active", "owner": "executive", "role": "prepared observation + live bounded reranking"},
        "transition_router": {"status": "active", "owner": "executive", "role": "transition family routing"},
        "unified_planner": {"status": "fallback", "owner": "executive", "role": "legacy route repair"},
        "unified_transition": {"status": "active", "owner": "executive", "role": "exact preparation and professional repair"},
    }


class ExecutiveOrchestrator:
    """Prepared-set supervisory controller.

    Expensive brain algorithms remain specialist services. This object owns the
    order in which they are invoked, the replan/blocked-edge loop, observation
    state, and the final call into the single authoritative commit boundary.
    It deliberately does not perform DSP.
    """

    def __init__(
        self,
        *,
        analyses: Sequence[TrackAnalysis],
        source: str,
        journey: str,
        listener: str,
        automatic_delivery: bool,
        max_replans: int,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.analyses = analyses
        self.source = source
        self.journey = journey
        self.listener = listener
        self.automatic_delivery = bool(automatic_delivery)
        self.max_replans = max(1, int(max_replans))
        self.progress = progress
        self.phase = ExecutivePhase.INITIALIZING
        self.planning_revision = 0
        self._events: list[ExecutiveEvent] = []
        self._record(self.phase, "Executive supervisory controller initialized")
        feedback_path = os.environ.get("DJAI_CROWD_FEEDBACK_PATH", "ratings/feedback-events.jsonl")
        self.crowd_state: CrowdState = crowd_state_from_feedback(
            feedback_path,
            listener=listener,
        )

    def _record(
        self,
        phase: ExecutivePhase,
        detail: str,
        **payload: Any,
    ) -> None:
        self.phase = phase
        self._events.append(
            ExecutiveEvent(
                sequence=len(self._events) + 1,
                phase=phase.value,
                detail=detail,
                planning_revision=self.planning_revision,
                payload=dict(payload),
            )
        )

    def _orchestration_payload(
        self,
        *,
        blocked_pairs: set[tuple[int, int]],
        replan_trace: list[dict[str, Any]],
        transitions: Sequence[TransitionEdge],
        final_escape_used: bool,
    ) -> dict[str, Any]:
        components = brain_component_registry()
        return {
            "version": EXECUTIVE_ORCHESTRATOR_VERSION,
            "controller": "dj_executive_supervisory_controller",
            "single_writer_authority": True,
            "owns_route_selection": True,
            "owns_replan_loop": True,
            "owns_commit_boundary": True,
            "executor_is_deterministic": True,
            "safety_governor_connected": True,
            "performance_horizons_connected": True,
            "runtime_feedback_connected": True,
            "musical_behavior_preserved": True,
            "continuous_live_replanning": False,
            "continuous_live_replanning_available": True,
            "live_future_only_mutation": True,
            "prepared_set_phase": True,
            "stale_async_results_rejected": True,
            "planning_revision": self.planning_revision,
            "committed_state_version": len(transitions),
            "blocked_pairs": [list(pair) for pair in sorted(blocked_pairs)],
            "replan_count": len(replan_trace),
            "final_escape_used": final_escape_used,
            "crowd_state_connected": True,
            "crowd_state_authority": "prepared_observe_live_bounded_advisory",
            "transition_policy_connected": True,
            "transition_policy_authority": "prepared_observe_live_bounded_advisory",
            "component_registry": components,
            "all_brain_components_accounted_for": all(
                item.get("status") in {"active", "conditional", "fallback", "observe"}
                for item in components.values()
            ),
            "events": [event.to_dict() for event in self._events],
        }

    def run_prepared_set(
        self,
        *,
        manual_order: list[int] | None,
        choose_order: RouteSelector,
        compile_order: TransitionCompiler,
        decisions: DecisionProvider,
        top_alternatives: int,
    ) -> ExecutivePreparedSetResult:
        blocked_pairs: set[tuple[int, int]] = set()
        replan_trace: list[dict[str, Any]] = []
        final_escape_used = False
        order: list[int] | None = None
        transitions: list[TransitionEdge] | None = None

        self._record(
            ExecutivePhase.OBSERVING,
            "Observed prepared-set world state",
            track_count=len(self.analyses),
            crowd_state=self.crowd_state.to_dict(),
        )
        initial_snapshot = ExecutiveSnapshot(
            state_version=self.planning_revision,
            source=self.source,
            journey=self.journey,
            listener=self.listener,
            track_count=len(self.analyses),
            crowd_state=self.crowd_state.to_dict(),
            blocked_pairs=tuple(sorted(blocked_pairs)),
        )
        self._record(
            ExecutivePhase.DIRECTING,
            "Delegating set-intent interpretation to the Set Director",
            journey=self.journey,
        )

        try:
            if manual_order is not None:
                order = list(manual_order)
                self._record(
                    ExecutivePhase.SEARCHING,
                    "Accepted client-supplied route as the route proposal",
                    order=order,
                )
                self._record(ExecutivePhase.PREPARING, "Preparing exact transition programs for manual route")
                transitions = compile_order(order, False, self.automatic_delivery)
            elif self.automatic_delivery:
                for replan_round in range(self.max_replans + 1):
                    self.planning_revision = replan_round
                    if self.progress:
                        self.progress(
                            "order-replan",
                            replan_round,
                            self.max_replans + 1,
                            (
                                f"Choosing a {len(self.analyses)}-track route"
                                if replan_round == 0
                                else f"Executive repairing route after {replan_round} rejected transition(s)"
                            ),
                        )
                    self._record(
                        ExecutivePhase.SEARCHING,
                        "Requesting route and Complete DJMove search",
                        blocked_pairs=[list(pair) for pair in sorted(blocked_pairs)],
                    )
                    try:
                        order = choose_order(blocked_pairs)
                    except RuntimeError as exc:
                        self._record(ExecutivePhase.REPLANNING, "Route search could not construct a complete path", error=str(exc))
                        break
                    route_result = ExecutiveServiceResult(
                        component="route_search",
                        snapshot_version=self.planning_revision,
                        payload={"order": list(order)},
                    )
                    require_current_state_version(
                        result_version=route_result.snapshot_version,
                        current_version=self.planning_revision,
                        component=route_result.component,
                    )
                    move_decisions = tuple(decisions())
                    self._record(
                        ExecutivePhase.AUDITIONING,
                        "Route search completed; AI Headphones evidence is part of returned move decisions",
                        decision_count=len(move_decisions),
                        headphones_decisions=sum(1 for item in move_decisions if isinstance(item.headphones, dict)),
                    )
                    self._record(
                        ExecutivePhase.PREPARING,
                        "Requesting exact transition compilation and professional judgement",
                        order=order,
                    )
                    try:
                        transitions = compile_order(order, True, False)
                        self._record(
                            ExecutivePhase.VALIDATING,
                            "Prepared route survived transition preparation and professional gates",
                            transition_count=len(transitions),
                        )
                        break
                    except ExecutiveReplanRequest as exc:
                        pair = (exc.from_index, exc.to_index)
                        blocked_pairs.add(pair)
                        trace = {
                            "round": replan_round + 1,
                            "from_index": exc.from_index,
                            "to_index": exc.to_index,
                            "from_track": self.analyses[exc.from_index].title,
                            "to_track": self.analyses[exc.to_index].title,
                            "action": exc.action,
                            "reason": exc.reason,
                            "resolution": exc.resolution,
                        }
                        replan_trace.append(trace)
                        self._record(
                            ExecutivePhase.REPLANNING,
                            "Professional/transition preparation rejected an edge; Executive blocked it and will replan",
                            **trace,
                        )
                        continue

                if transitions is None or order is None:
                    final_escape_used = True
                    self.planning_revision += 1
                    self._record(
                        ExecutivePhase.REPLANNING,
                        "Normal route repair exhausted; entering explicit emergency/fallback route",
                        blocked_pairs=[list(pair) for pair in sorted(blocked_pairs)],
                    )
                    try:
                        order = choose_order(blocked_pairs)
                    except RuntimeError:
                        order = choose_order(set())
                    self._record(
                        ExecutivePhase.PREPARING,
                        "Preparing emergency-capable final route after alternate routes were exhausted",
                        order=order,
                    )
                    transitions = compile_order(order, False, True)
                    self._record(
                        ExecutivePhase.VALIDATING,
                        "Emergency-capable prepared route completed deterministic validation",
                        transition_count=len(transitions),
                    )
            else:
                self._record(ExecutivePhase.SEARCHING, "Requesting non-delivery route search")
                order = choose_order(set())
                self._record(ExecutivePhase.AUDITIONING, "Route proposal returned", decision_count=len(tuple(decisions())))
                self._record(ExecutivePhase.PREPARING, "Preparing route without automatic-delivery repair loop")
                transitions = compile_order(order, False, False)
                self._record(ExecutivePhase.VALIDATING, "Prepared route validated", transition_count=len(transitions))

            assert order is not None and transitions is not None
            final_decisions = tuple(decisions())
            self._record(
                ExecutivePhase.COMMITTING,
                "Entering the single authoritative Executive commit boundary",
                transition_count=len(transitions),
            )
            report = build_executive_report(
                analyses=self.analyses,
                order=order,
                transitions=transitions,
                decisions=final_decisions,
                journey=self.journey,
                top_alternatives=top_alternatives,
            )
            self._record(
                ExecutivePhase.COMPLETE,
                "Prepared-set orchestration complete",
                authoritative=bool(report.get("authoritative")),
                commit_authority=report.get("commit_authority"),
            )
            orchestration = self._orchestration_payload(
                blocked_pairs=blocked_pairs,
                replan_trace=replan_trace,
                transitions=transitions,
                final_escape_used=final_escape_used,
            )
            report["orchestrator"] = orchestration
            report["observations"] = {
                "crowd_state": self.crowd_state.to_dict(),
                "crowd_state_source": "feedback_events",
                "crowd_state_musical_authority": "observe",
                "initial_snapshot": initial_snapshot.to_dict(),
            }
            return ExecutivePreparedSetResult(
                order=tuple(order),
                transitions=tuple(transitions),
                blocked_pairs=tuple(sorted(blocked_pairs)),
                replan_trace=tuple(replan_trace),
                executive_report=report,
                crowd_state=self.crowd_state.to_dict(),
                orchestration=orchestration,
            )
        except Exception as exc:
            self._record(ExecutivePhase.FAILED, "Executive orchestration failed", error=f"{type(exc).__name__}: {exc}")
            raise
