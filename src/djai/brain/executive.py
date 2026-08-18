from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from ..decision import validate_execution_plan
from ..models import TrackAnalysis, TransitionPlan
from ..set_models import TransitionEdge
from .authority import EXECUTIVE_AUTHORITY, build_set_commit_receipt
from .dj_move import DJState, build_dj_state
from .move_generator import CompleteMoveDecision
from .move_intent import (
    DJ_MOVE_INTENT_VERSION,
    DJMoveIntent,
    IntentBindingError,
    PreparedIntentBinding,
    bind_prepared_transition_to_intent,
    build_prepared_plan_intent,
)
from .set_state import build_set_state
from .thinking_budget import adaptive_thinking_status
from .recovery import PREPARED_RECOVERY_VERSION, PreparedRecoveryPlan, build_prepared_recovery_plan

DJ_EXECUTIVE_VERSION = "2.5.0"
ExecutiveMode = Literal["off", "observe", "authoritative"]
TransactionStatus = Literal["proposed", "prepared", "committed", "rejected"]


class ExecutiveCommitError(RuntimeError):
    """Raised when the authoritative Executive cannot safely commit a final move."""


def _mode() -> ExecutiveMode:
    raw = os.environ.get("DJAI_EXECUTIVE_MODE", "authoritative").strip().lower()
    aliases = {
        "commit": "authoritative",
        "committing": "authoritative",
        "on": "authoritative",
        "1": "authoritative",
        "true": "authoritative",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in {"off", "observe", "authoritative"} else "authoritative"  # type: ignore[return-value]


def _stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def _headline(track: TrackAnalysis) -> str:
    return str(track.title or track.path)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class BrainSnapshot:
    decision_id: str
    sequence_position: int
    current_index: int
    next_index: int
    current_track: str
    next_track: str
    journey: str
    dj_state: dict[str, Any] = field(default_factory=dict)
    projected_state: dict[str, Any] = field(default_factory=dict)
    layer_state: dict[str, Any] = field(default_factory=dict)
    projected_layer_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BrainEvidence:
    retrieval_available: int = 0
    retrieval_selected: tuple[int, ...] = ()
    generated_moves: int = 0
    lookahead_depth: int = 0
    projected_path: tuple[int, ...] = ()
    ai_headphones_used: bool = False
    ai_headphones_authoritative: bool = False
    ai_headphones_rendered_candidates: int = 0
    crate_memory_adjustment: float = 0.0
    taste_shadow_authority: float = 0.0
    layer_move: bool = False
    layer_strategy: str | None = None
    generated_execution_signature: str = ""
    compiled_execution_signature: str = ""
    execution_signature_match: bool = False
    quality_checked: bool = False
    measurement_passed: bool = False
    professional_judge_score: float | None = None
    professional_judge_uncertainty: float | None = None
    hard_audio_failures: tuple[str, ...] = ()
    render_peak: float | None = None
    clipping_fraction: float | None = None
    fallback_level: int = 0
    thinking_budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retrieval_selected"] = list(self.retrieval_selected)
        payload["projected_path"] = list(self.projected_path)
        payload["hard_audio_failures"] = list(self.hard_audio_failures)
        return payload


@dataclass(frozen=True, slots=True)
class ExecutiveTransaction:
    transaction_id: str
    decision_id: str
    move_id: str
    intent_id: str
    move_intent: dict[str, Any]
    intent_binding: dict[str, Any]
    intent_binding_status: str
    semantic_intent_preserved: bool
    fallback_binding_used: bool
    sequence_position: int
    current_index: int
    next_index: int
    status: TransactionStatus
    proposed_execution_signature: str
    prepared_execution_signature: str
    proposal_changed_during_prepare: bool
    prepared_exact_program: bool
    validation_passed: bool
    validation_checks: tuple[str, ...]
    phase_history: tuple[str, ...]
    rejection_reason: str | None
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    thinking_budget: dict[str, Any] = field(default_factory=dict)
    prepared_recovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation_checks"] = list(self.validation_checks)
        payload["phase_history"] = list(self.phase_history)
        return payload


@dataclass(frozen=True, slots=True)
class BrainDecision:
    decision_id: str
    move_id: str
    status: str
    observer_authoritative: bool
    commit_authority: str
    snapshot: BrainSnapshot
    action: str
    technique: str
    intent: str
    selection_score: float
    commit_score: float
    rationale: tuple[str, ...]
    move_intent: dict[str, Any]
    evidence: BrainEvidence
    alternatives: tuple[dict[str, Any], ...] = ()
    transaction_id: str | None = None
    transaction_status: str | None = None
    prepared_recovery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "move_id": self.move_id,
            "status": self.status,
            # Kept for V1 API compatibility. In V5 authoritative mode this now
            # means the DJ Executive owns the final commit boundary.
            "observer_authoritative": self.observer_authoritative,
            "commit_authority": self.commit_authority,
            "snapshot": self.snapshot.to_dict(),
            "action": self.action,
            "technique": self.technique,
            "intent": self.intent,
            "selection_score": round(self.selection_score, 6),
            "commit_score": round(self.commit_score, 6),
            "rationale": list(self.rationale),
            "move_intent": dict(self.move_intent),
            "evidence": self.evidence.to_dict(),
            "alternatives": [dict(item) for item in self.alternatives],
            "transaction_id": self.transaction_id,
            "transaction_status": self.transaction_status,
            "prepared_recovery": dict(self.prepared_recovery),
        }


@dataclass(frozen=True, slots=True)
class BrainRunReport:
    version: str
    mode: ExecutiveMode
    enabled: bool
    authoritative: bool
    musical_behavior_changes: bool
    commit_authority: str
    decisions: tuple[BrainDecision, ...] = ()
    transactions: tuple[ExecutiveTransaction, ...] = ()
    utilization: dict[str, Any] = field(default_factory=dict)
    lifecycle: tuple[str, ...] = (
        "observe", "propose", "assess_brittleness", "allocate_thinking",
        "search", "audition", "prepare", "prepare_backup", "validate", "commit"
    )
    intent_lifecycle: tuple[str, ...] = (
        "propose", "brittleness_assessed", "thinking_budget_allocated",
        "intent_created", "prepare", "intent_bound", "prepare_backup", "validate", "commit"
    )
    initial_authoritative_state: dict[str, Any] = field(default_factory=dict)
    final_authoritative_state: dict[str, Any] = field(default_factory=dict)
    set_commit: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "enabled": self.enabled,
            "authoritative": self.authoritative,
            "musical_behavior_changes": self.musical_behavior_changes,
            "single_commit_authority": self.authoritative,
            "commit_authority": self.commit_authority,
            "decision_count": len(self.decisions),
            "transaction_count": len(self.transactions),
            "lifecycle": list(self.lifecycle),
            "intent_lifecycle": list(self.intent_lifecycle),
            "utilization": dict(self.utilization),
            "initial_authoritative_state": dict(self.initial_authoritative_state),
            "final_authoritative_state": dict(self.final_authoritative_state),
            "set_commit": dict(self.set_commit),
            "transactions": [item.to_dict() for item in self.transactions],
            "decisions": [item.to_dict() for item in self.decisions],
            "error": self.error,
        }


def executive_status() -> dict[str, Any]:
    mode = _mode()
    authoritative = mode == "authoritative"
    try:
        from .crowd_state import CrowdControlConfig
        from .live_orchestrator import LiveExecutiveConfig
        from .transition_policy import TransitionPolicyConfig

        crowd_control = CrowdControlConfig.from_environment()
        live_control = LiveExecutiveConfig.from_environment()
        transition_policy = TransitionPolicyConfig.from_environment()
    except Exception:
        crowd_control = None
        live_control = None
        transition_policy = None
    return {
        "version": DJ_EXECUTIVE_VERSION,
        "mode": mode,
        "enabled": mode != "off",
        "authoritative": authoritative,
        "musical_behavior_changes": bool(adaptive_thinking_status()["enabled"]),
        "single_commit_authority": authoritative,
        "commit_authority": EXECUTIVE_AUTHORITY if authoritative else "existing_rc1_pipeline",
        "supervisory_orchestrator": "dj_executive_v6",
        "orchestration_model": "single_writer_supervisory_control",
        "executor_contract": "validated_immutable_command_only",
        "crowd_state_connected": True,
        "crowd_state_authority": "live_bounded_advisory",
        "crowd_control_enabled": bool(crowd_control.enabled) if crowd_control is not None else False,
        "crowd_minimum_confidence": (
            crowd_control.minimum_confidence if crowd_control is not None else None
        ),
        "crowd_max_track_adjustment": (
            crowd_control.max_track_adjustment if crowd_control is not None else None
        ),
        "transition_policy_connected": True,
        "transition_policy_authority": "live_bounded_advisory",
        "transition_policy_enabled": (
            bool(transition_policy.enabled) if transition_policy is not None else False
        ),
        "continuous_live_replanning": (
            bool(live_control.enabled) if live_control is not None else False
        ),
        "projected_search_state_is_non_authoritative": True,
        "propose_prepare_commit": authoritative,
        "djmove_intent_version": DJ_MOVE_INTENT_VERSION,
        "djmove_intent_required_for_commit": authoritative,
        "flexible_prepare_binding": authoritative,
        "exact_proposal_signature_required": False,
        "same_target_safe_fallback_binding": authoritative,
        "adaptive_thinking": adaptive_thinking_status(),
        "prepared_recovery_version": PREPARED_RECOVERY_VERSION,
        "prepared_recovery_capable": authoritative,
        "backup_requirement_by_brittleness": {"easy": 0, "normal": 1, "brittle": 1},
        "maximum_backups_by_brittleness": {"easy": 1, "normal": 1, "brittle": 2},
        "same_target_recovery_only": True,
        "alternate_target_recovery_requires_plan_repair": True,
        "recovery_readiness_levels": ["prepared", "auditioned", "hot"],
        "brittleness_controls_compute_only": True,
        "hard_safety_budget_never_reduced": True,
        "exact_compiled_program_required_for_commit": authoritative,
        "commit_validation_fail_closed": authoritative,
        "orchestrates_complete_djmove": authoritative,
        "orchestrates_ai_headphones": authoritative,
        "orchestrates_professional_brain": authoritative,
        "orchestrates_layer_intelligence": authoritative,
        "orchestrates_compiled_execution": authoritative,
        # Backward-compatible observation flags remain for existing clients.
        "observes_complete_djmove": True,
        "observes_ai_headphones": True,
        "observes_professional_brain": True,
        "observes_layer_intelligence": True,
        "observes_compiled_execution": True,
        "telemetry_failure_is_nonfatal": not authoritative,
        "observed_lifecycle": [
            "observe", "propose", "assess_brittleness", "allocate_thinking",
            "search", "audition", "prepare", "prepare_backup", "validate", "commit"
        ],
        "intent_lifecycle": [
            "propose", "brittleness_assessed", "thinking_budget_allocated",
            "intent_created", "prepare", "intent_bound", "prepare_backup", "validate", "commit"
        ],
    }


def _candidate_summary(candidate: Any) -> dict[str, Any]:
    move = candidate.move
    return {
        "next_index": int(candidate.next_index),
        "selection_score": round(_safe_float(candidate.selection_score), 6),
        "technique": str(move.technique),
        "action": str(move.action),
        "execution_signature": str(candidate.plan.execution_signature or ""),
        "layer_move": bool(getattr(move, "layer_choreography", {})),
    }


def _match_decision(
    decisions: Sequence[CompleteMoveDecision],
    from_index: int,
    to_index: int,
) -> CompleteMoveDecision | None:
    matches = [
        decision
        for decision in decisions
        if int(decision.current_index) == from_index and int(decision.winner.next_index) == to_index
    ]
    return matches[-1] if matches else None


def _initial_state(analyses: Sequence[TrackAnalysis], order: Sequence[int]) -> DJState | None:
    if not order:
        return None
    first_index = int(order[0])
    first_track = analyses[first_index]
    return build_dj_state(
        first_track,
        source_cursor=0.0,
        effective_bpm=first_track.bpm,
        render_rate=1.0,
        set_state=build_set_state([first_track]),
        tempo_owner="A",
    )


def _state_after_committed_plan(
    *,
    analyses: Sequence[TrackAnalysis],
    order: Sequence[int],
    committed_plans: Sequence[TransitionPlan],
    position: int,
    plan: TransitionPlan,
) -> DJState:
    to_index = int(order[position + 1])
    track_b = analyses[to_index]
    committed_tracks = [analyses[int(index)] for index in order[: position + 2]]
    set_state = build_set_state(committed_tracks, list(committed_plans))
    source_cursor = float(plan.post_transition_source_end_b)
    if source_cursor <= 0.0:
        source_cursor = float(plan.source_end_b)
    effective_bpm = float(plan.post_transition_bpm_b or plan.rendered_bpm_b or track_b.bpm)
    render_rate = float(plan.post_transition_rate_b or plan.render_rate_b or 1.0)
    return build_dj_state(
        track_b,
        source_cursor=source_cursor,
        effective_bpm=effective_bpm,
        render_rate=render_rate,
        set_state=set_state,
        tempo_owner="B",
    )


def _validate_prepared_transition(
    *,
    plan: TransitionPlan,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
) -> tuple[str, ...]:
    checks: list[str] = []
    if str(plan.track_a) != str(track_a.path):
        raise ExecutiveCommitError("Prepared transition source A does not match the authoritative sequence.")
    checks.append("source_a_matches_sequence")
    if str(plan.track_b) != str(track_b.path):
        raise ExecutiveCommitError("Prepared transition source B does not match the authoritative sequence.")
    checks.append("source_b_matches_sequence")
    if not plan.compiled_execution:
        raise ExecutiveCommitError("Prepared transition is not compiled for deterministic execution.")
    checks.append("compiled_execution")
    if not str(plan.performance_program_fingerprint or ""):
        raise ExecutiveCommitError("Prepared transition has no immutable PerformanceProgram fingerprint.")
    checks.append("performance_program_fingerprint")
    if not str(plan.execution_signature or ""):
        raise ExecutiveCommitError("Prepared transition has no execution signature.")
    checks.append("execution_signature_present")
    try:
        # The final renderer validates asset fingerprints again. The Executive
        # owns semantic commitment and therefore validates the exact immutable
        # execution contract here without turning planning into a second disk
        # verification pass for every asset.
        validate_execution_plan(plan, verify_assets=False)
    except Exception as exc:
        raise ExecutiveCommitError(f"Prepared execution contract is invalid: {exc}") from exc
    checks.append("execution_contract_valid")
    if plan.measurement is not None and plan.measurement.hard_failures:
        failures = ", ".join(str(item) for item in plan.measurement.hard_failures)
        raise ExecutiveCommitError(f"Prepared transition has hard rendered-audio failures: {failures}")
    checks.append("no_hard_audio_failures")
    if plan.quality_checked and not plan.measurement_passed:
        raise ExecutiveCommitError("Quality-checked prepared transition did not pass its rendered measurement gate.")
    if plan.quality_checked:
        checks.append("rendered_measurement_passed")
    else:
        checks.append("rendered_measurement_not_required_by_current_rc1_path")
    return tuple(checks)


def _decision_payload(
    *,
    analyses: Sequence[TrackAnalysis],
    position: int,
    edge: TransitionEdge,
    matched: CompleteMoveDecision | None,
    journey: str,
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    authoritative: bool,
    transaction: ExecutiveTransaction | None,
    top_alternatives: int,
) -> BrainDecision:
    from_index = int(edge.from_index)
    to_index = int(edge.to_index)
    plan = edge.plan
    generated_signature = ""
    projected_state: dict[str, Any] = {}
    retrieval_available = 0
    retrieval_selected: tuple[int, ...] = ()
    generated_moves = 0
    lookahead_depth = 0
    projected_path: tuple[int, ...] = ()
    headphones_used = False
    headphones_authoritative = False
    headphones_rendered = 0
    crate_adjustment = 0.0
    taste_authority = 0.0
    selection_score = _safe_float(plan.metadata.get("selection_utility", plan.score))
    commit_score = selection_score
    action = "transition"
    rationale: tuple[str, ...] = tuple(str(item) for item in plan.rationale)
    alternatives: tuple[dict[str, Any], ...] = ()
    thinking_budget: dict[str, Any] = {}

    if matched is not None:
        winner = matched.winner
        generated_signature = str(winner.plan.execution_signature or "")
        projected_state = winner.move.projected_state.to_dict()
        action = str(winner.move.action)
        rationale = tuple(str(item) for item in winner.move.rationale)
        selection_score = _safe_float(winner.selection_score)
        commit_score = _safe_float(matched.commit_score or matched.search_score or winner.selection_score)
        generated_moves = len(matched.candidates)
        lookahead_depth = int(matched.lookahead_depth)
        projected_path = tuple(int(item) for item in matched.projected_path)
        if matched.retrieval is not None:
            retrieval_available = int(matched.retrieval.total_available)
            retrieval_selected = tuple(int(item) for item in matched.retrieval.selected)
        if isinstance(matched.headphones, dict):
            headphones_used = True
            headphones_authoritative = bool(matched.headphones.get("authoritative"))
            headphones_rendered = int(matched.headphones.get("rendered_candidates", 0) or 0)
        if isinstance(winner.crate_memory, dict):
            crate_adjustment = _safe_float(winner.crate_memory.get("adjustment"))
        if isinstance(winner.taste_shadow, dict):
            taste_authority = _safe_float(winner.taste_shadow.get("authority"))
        if matched.thinking_budget is not None:
            thinking_budget = matched.thinking_budget.to_dict()
        candidate_pool = [winner] + [
            candidate
            for candidate in matched.candidates
            if candidate.plan.execution_signature != winner.plan.execution_signature
        ]
        alternatives = tuple(
            {**_candidate_summary(candidate), "selected": candidate.plan.execution_signature == winner.plan.execution_signature}
            for candidate in candidate_pool[: max(1, int(top_alternatives))]
        )

    compiled_signature = str(plan.execution_signature or "")
    layer_choreography = dict(plan.metadata.get("layer_choreography", {}) or {})
    judge_score_raw = plan.metadata.get("professional_judge_score")
    judge_uncertainty_raw = plan.metadata.get("professional_judge_uncertainty")
    decision_id = (
        transaction.decision_id
        if transaction is not None
        else _stable_id("decision", position, from_index, to_index, generated_signature, compiled_signature)
    )
    move_id = (
        transaction.move_id
        if transaction is not None
        else _stable_id("move", from_index, to_index, generated_signature or compiled_signature)
    )
    snapshot = BrainSnapshot(
        decision_id=decision_id,
        sequence_position=position,
        current_index=from_index,
        next_index=to_index,
        current_track=_headline(analyses[from_index]),
        next_track=_headline(analyses[to_index]),
        journey=str(journey),
        dj_state=dict(state_before),
        projected_state=projected_state or dict(state_after),
        layer_state=dict(state_before.get("layer_state", {}) or {}),
        projected_layer_state=dict((projected_state or state_after).get("layer_state", {}) or {}),
    )
    if transaction is not None:
        move_intent_payload = dict(transaction.move_intent)
    elif matched is not None and matched.move_intent is not None:
        move_intent_payload = matched.move_intent.to_dict()
    else:
        move_intent_payload = build_prepared_plan_intent(
            current_index=from_index,
            next_index=to_index,
            track_a=analyses[from_index],
            track_b=analyses[to_index],
            plan=plan,
        ).to_dict()
    evidence = BrainEvidence(
        retrieval_available=retrieval_available,
        retrieval_selected=retrieval_selected,
        generated_moves=generated_moves,
        lookahead_depth=lookahead_depth,
        projected_path=projected_path,
        ai_headphones_used=headphones_used,
        ai_headphones_authoritative=headphones_authoritative,
        ai_headphones_rendered_candidates=headphones_rendered,
        crate_memory_adjustment=crate_adjustment,
        taste_shadow_authority=taste_authority,
        layer_move=bool(layer_choreography) or str(plan.technique).startswith("layer_"),
        layer_strategy=(str(layer_choreography.get("strategy")) if layer_choreography.get("strategy") else None),
        generated_execution_signature=generated_signature,
        compiled_execution_signature=compiled_signature,
        execution_signature_match=bool(generated_signature and compiled_signature and generated_signature == compiled_signature),
        quality_checked=bool(plan.quality_checked),
        measurement_passed=bool(plan.measurement_passed),
        professional_judge_score=(_safe_float(judge_score_raw) if judge_score_raw is not None else None),
        professional_judge_uncertainty=(
            _safe_float(judge_uncertainty_raw) if judge_uncertainty_raw is not None else None
        ),
        hard_audio_failures=tuple(
            str(item) for item in (plan.measurement.hard_failures if plan.measurement is not None else ())
        ),
        render_peak=(_safe_float(plan.measurement.peak) if plan.measurement is not None else None),
        clipping_fraction=(_safe_float(plan.measurement.clipping_fraction) if plan.measurement is not None else None),
        fallback_level=int(plan.fallback_level),
        thinking_budget=thinking_budget,
    )
    return BrainDecision(
        decision_id=decision_id,
        move_id=move_id,
        status="executive_committed" if authoritative else "observed_existing_commit",
        observer_authoritative=authoritative,
        commit_authority=EXECUTIVE_AUTHORITY if authoritative else "existing_rc1_pipeline",
        snapshot=snapshot,
        action=action,
        technique=str(plan.technique),
        intent=str(plan.intent),
        selection_score=selection_score,
        commit_score=commit_score,
        rationale=rationale,
        move_intent=move_intent_payload,
        evidence=evidence,
        alternatives=alternatives,
        transaction_id=(transaction.transaction_id if transaction is not None else None),
        transaction_status=(transaction.status if transaction is not None else None),
        prepared_recovery=(dict(transaction.prepared_recovery) if transaction is not None else {}),
    )


def _utilization(
    *,
    transitions: Sequence[TransitionEdge],
    decisions: Sequence[CompleteMoveDecision],
    records: Sequence[BrainDecision],
    order: Sequence[int],
    transactions: Sequence[ExecutiveTransaction],
) -> dict[str, Any]:
    return {
        "final_transitions": len(transitions),
        "complete_move_decisions_correlated": sum(
            1
            for edge in transitions
            if _match_decision(decisions, int(edge.from_index), int(edge.to_index)) is not None
        ),
        "complete_moves_generated": sum(item.evidence.generated_moves for item in records),
        "maximum_lookahead_depth": max((item.evidence.lookahead_depth for item in records), default=0),
        "ai_headphones_decisions": sum(1 for item in records if item.evidence.ai_headphones_used),
        "ai_headphones_rendered_candidates": sum(item.evidence.ai_headphones_rendered_candidates for item in records),
        "crate_memory_informed_decisions": sum(
            1 for item in records if abs(item.evidence.crate_memory_adjustment) > 1e-9
        ),
        "layer_moves_committed": sum(1 for item in records if item.evidence.layer_move),
        "quality_checked_transitions": sum(1 for item in records if item.evidence.quality_checked),
        "measurement_passed_transitions": sum(1 for item in records if item.evidence.measurement_passed),
        "hard_audio_failure_transitions": sum(1 for item in records if item.evidence.hard_audio_failures),
        "generated_compiled_signature_matches": sum(1 for item in records if item.evidence.execution_signature_match),
        "executive_transactions": len(transactions),
        "executive_commits": sum(1 for item in transactions if item.status == "committed"),
        "proposal_substitutions_during_prepare": sum(
            1 for item in transactions if item.proposal_changed_during_prepare
        ),
        "djmove_intents": sum(1 for item in records if item.move_intent),
        "exact_intent_bindings": sum(1 for item in transactions if item.intent_binding_status == "exact"),
        "flexible_intent_bindings": sum(1 for item in transactions if item.intent_binding_status == "flexible"),
        "fallback_intent_bindings": sum(1 for item in transactions if item.fallback_binding_used),
        "semantic_intent_preserved_commits": sum(1 for item in transactions if item.semantic_intent_preserved),
        "thinking_budget_decisions": sum(1 for item in records if item.evidence.thinking_budget),
        "thinking_budget_levels": {
            level: sum(
                1 for item in records
                if item.evidence.thinking_budget.get("brittleness", {}).get("level") == level
            )
            for level in ("easy", "normal", "brittle")
        },
        "thinking_budget_deep_search_escalations": sum(
            1 for item in records if bool(item.evidence.thinking_budget.get("lookahead_escalated"))
        ),
        "thinking_budget_headphones_maximum_requested": sum(
            int(item.evidence.thinking_budget.get("headphones_maximum_renders", 0) or 0)
            for item in records
        ),
        "prepared_recovery_transactions": sum(1 for item in transactions if item.prepared_recovery),
        "recovery_ready_transactions": sum(
            1 for item in transactions if bool(item.prepared_recovery.get("recovery_ready"))
        ),
        "prepared_backup_moves": sum(
            len(item.prepared_recovery.get("prepared_backups", [])) for item in transactions
        ),
        "hot_backup_moves": sum(
            sum(1 for backup in item.prepared_recovery.get("prepared_backups", []) if backup.get("readiness") == "hot")
            for item in transactions
        ),
        "auditioned_backup_moves": sum(
            sum(1 for backup in item.prepared_recovery.get("prepared_backups", []) if backup.get("readiness") == "auditioned")
            for item in transactions
        ),
        "final_order": [int(item) for item in order],
    }


def build_executive_observation(
    *,
    analyses: Sequence[TrackAnalysis],
    order: Sequence[int],
    transitions: Sequence[TransitionEdge],
    decisions: Sequence[CompleteMoveDecision],
    journey: str,
    top_alternatives: int = 3,
) -> BrainRunReport:
    mode = _mode()
    if mode == "off":
        return BrainRunReport(
            version=DJ_EXECUTIVE_VERSION,
            mode=mode,
            enabled=False,
            authoritative=False,
            musical_behavior_changes=bool(adaptive_thinking_status()["enabled"]),
            commit_authority="existing_rc1_pipeline",
        )

    initial = _initial_state(analyses, order)
    current_state = initial
    committed_plans: list[TransitionPlan] = []
    records: list[BrainDecision] = []
    for position, edge in enumerate(transitions):
        before = current_state.to_dict() if current_state is not None else {}
        committed_plans.append(edge.plan)
        current_state = _state_after_committed_plan(
            analyses=analyses,
            order=order,
            committed_plans=committed_plans,
            position=position,
            plan=edge.plan,
        )
        after = current_state.to_dict()
        records.append(
            _decision_payload(
                analyses=analyses,
                position=position,
                edge=edge,
                matched=_match_decision(decisions, int(edge.from_index), int(edge.to_index)),
                journey=journey,
                state_before=before,
                state_after=after,
                authoritative=False,
                transaction=None,
                top_alternatives=top_alternatives,
            )
        )

    return BrainRunReport(
        version=DJ_EXECUTIVE_VERSION,
        mode=mode,
        enabled=True,
        authoritative=False,
        musical_behavior_changes=bool(adaptive_thinking_status()["enabled"]),
        commit_authority="existing_rc1_pipeline",
        decisions=tuple(records),
        utilization=_utilization(
            transitions=transitions,
            decisions=decisions,
            records=records,
            order=order,
            transactions=(),
        ),
        initial_authoritative_state=(initial.to_dict() if initial is not None else {}),
        final_authoritative_state=(current_state.to_dict() if current_state is not None else {}),
    )


def build_executive_authoritative_commit(
    *,
    analyses: Sequence[TrackAnalysis],
    order: Sequence[int],
    transitions: Sequence[TransitionEdge],
    decisions: Sequence[CompleteMoveDecision],
    journey: str,
    top_alternatives: int = 3,
) -> BrainRunReport:
    if len(order) != len(transitions) + 1 and order:
        raise ExecutiveCommitError(
            f"Authoritative sequence is inconsistent: {len(order)} ordered tracks but {len(transitions)} transitions."
        )
    initial = _initial_state(analyses, order)
    current_state = initial
    committed_plans: list[TransitionPlan] = []
    records: list[BrainDecision] = []
    transactions: list[ExecutiveTransaction] = []

    for position, edge in enumerate(transitions):
        expected_from = int(order[position])
        expected_to = int(order[position + 1])
        if int(edge.from_index) != expected_from or int(edge.to_index) != expected_to:
            raise ExecutiveCommitError(
                "Final transition sequence does not match final track order at "
                f"position {position}: expected {expected_from}->{expected_to}, "
                f"got {edge.from_index}->{edge.to_index}."
            )
        matched = _match_decision(decisions, expected_from, expected_to)
        proposed_signature = str(matched.winner.plan.execution_signature or "") if matched is not None else ""
        prepared_signature = str(edge.plan.execution_signature or "")
        move_intent: DJMoveIntent = (
            matched.move_intent
            if matched is not None and matched.move_intent is not None
            else build_prepared_plan_intent(
                current_index=expected_from,
                next_index=expected_to,
                track_a=analyses[expected_from],
                track_b=analyses[expected_to],
                plan=edge.plan,
            )
        )
        try:
            intent_binding: PreparedIntentBinding = bind_prepared_transition_to_intent(
                intent=move_intent,
                plan=edge.plan,
                track_a=analyses[expected_from],
                track_b=analyses[expected_to],
            )
        except IntentBindingError as exc:
            raise ExecutiveCommitError(f"Prepared transition cannot bind to DJMoveIntent: {exc}") from exc
        decision_id = _stable_id(
            "decision",
            position,
            expected_from,
            expected_to,
            proposed_signature,
            prepared_signature,
        )
        move_id = _stable_id("move", expected_from, expected_to, proposed_signature or prepared_signature)
        transaction_id = _stable_id("txn", decision_id, move_intent.intent_id, prepared_signature)
        before = current_state.to_dict() if current_state is not None else {}

        checks = _validate_prepared_transition(
            plan=edge.plan,
            track_a=analyses[expected_from],
            track_b=analyses[expected_to],
        ) + (f"djmove_intent_bound:{intent_binding.status}",)
        # PREPARE is complete only after validation of the exact immutable
        # program. No authoritative state has changed before this point.
        candidate_plans = [*committed_plans, edge.plan]
        after_state = _state_after_committed_plan(
            analyses=analyses,
            order=order,
            committed_plans=candidate_plans,
            position=position,
            plan=edge.plan,
        )
        after = after_state.to_dict()
        thinking_budget = (
            matched.thinking_budget.to_dict()
            if matched is not None and matched.thinking_budget is not None
            else {}
        )
        brittleness_level = str(
            thinking_budget.get("brittleness", {}).get("level", "normal")
            if thinking_budget else "normal"
        )
        available_alternatives = max(0, len(matched.candidates) - 1) if matched is not None else 0
        required_backups = 0 if available_alternatives <= 0 or brittleness_level == "easy" else 1
        maximum_backups = 2 if brittleness_level == "brittle" else 1
        recovery_plan: PreparedRecoveryPlan = build_prepared_recovery_plan(
            current_index=expected_from,
            next_index=expected_to,
            track_a=analyses[expected_from],
            track_b=analyses[expected_to],
            intent=move_intent,
            primary_plan=edge.plan,
            candidates=(matched.candidates if matched is not None else ()),
            headphones_report=(matched.headphones if matched is not None and isinstance(matched.headphones, dict) else None),
            required_backups=required_backups,
            maximum_backups=maximum_backups,
        )
        transaction = ExecutiveTransaction(
            transaction_id=transaction_id,
            decision_id=decision_id,
            move_id=move_id,
            intent_id=move_intent.intent_id,
            move_intent=move_intent.to_dict(),
            intent_binding=intent_binding.to_dict(),
            intent_binding_status=intent_binding.status,
            semantic_intent_preserved=intent_binding.semantic_intent_preserved,
            fallback_binding_used=intent_binding.fallback_used,
            sequence_position=position,
            current_index=expected_from,
            next_index=expected_to,
            status="committed",
            proposed_execution_signature=proposed_signature,
            prepared_execution_signature=prepared_signature,
            proposal_changed_during_prepare=bool(
                proposed_signature and prepared_signature and proposed_signature != prepared_signature
            ),
            prepared_exact_program=True,
            validation_passed=True,
            validation_checks=checks,
            phase_history=(
                "proposed", "brittleness_assessed", "thinking_budget_allocated",
                "intent_created", "prepared", "intent_bound",
                ("backup_prepared" if recovery_plan.prepared_backups else "backup_unavailable"),
                "validated", "committed"
            ),
            rejection_reason=None,
            state_before=before,
            state_after=after,
            thinking_budget=thinking_budget,
            prepared_recovery=recovery_plan.to_dict(),
        )
        # This assignment is the one authoritative state advance. V6 keeps the
        # commit ledger here while the supervisory orchestrator owns when this
        # boundary may be entered.
        committed_plans.append(edge.plan)
        current_state = after_state
        transactions.append(transaction)
        records.append(
            _decision_payload(
                analyses=analyses,
                position=position,
                edge=edge,
                matched=matched,
                journey=journey,
                state_before=before,
                state_after=after,
                authoritative=True,
                transaction=transaction,
                top_alternatives=top_alternatives,
            )
        )

    set_commit = build_set_commit_receipt(
        order=order,
        transitions=transitions,
        transaction_ids=[item.transaction_id for item in transactions],
        authority=EXECUTIVE_AUTHORITY,
    )
    return BrainRunReport(
        version=DJ_EXECUTIVE_VERSION,
        mode="authoritative",
        enabled=True,
        authoritative=True,
        musical_behavior_changes=bool(adaptive_thinking_status()["enabled"]),
        commit_authority=EXECUTIVE_AUTHORITY,
        decisions=tuple(records),
        transactions=tuple(transactions),
        utilization=_utilization(
            transitions=transitions,
            decisions=decisions,
            records=records,
            order=order,
            transactions=transactions,
        ),
        initial_authoritative_state=(initial.to_dict() if initial is not None else {}),
        final_authoritative_state=(current_state.to_dict() if current_state is not None else {}),
        set_commit=set_commit.to_dict(),
    )


def build_executive_report(
    *,
    analyses: Sequence[TrackAnalysis],
    order: Sequence[int],
    transitions: Sequence[TransitionEdge],
    decisions: Sequence[CompleteMoveDecision],
    journey: str,
    top_alternatives: int = 3,
) -> dict[str, Any]:
    mode = _mode()
    if mode == "authoritative":
        # Authoritative commitment is intentionally fail-closed. A broken
        # transaction must never be downgraded to mere telemetry and allowed to
        # masquerade as a committed DJState.
        return build_executive_authoritative_commit(
            analyses=analyses,
            order=order,
            transitions=transitions,
            decisions=decisions,
            journey=journey,
            top_alternatives=top_alternatives,
        ).to_dict()
    return build_executive_observation_safely(
        analyses=analyses,
        order=order,
        transitions=transitions,
        decisions=decisions,
        journey=journey,
        top_alternatives=top_alternatives,
    )


def build_executive_observation_safely(**kwargs: Any) -> dict[str, Any]:
    """V1 observation remains non-fatal; it never owns authoritative state."""

    try:
        return build_executive_observation(**kwargs).to_dict()
    except Exception as exc:  # pragma: no cover - contract verified by fault injection.
        mode = _mode()
        return BrainRunReport(
            version=DJ_EXECUTIVE_VERSION,
            mode=mode,
            enabled=mode != "off",
            authoritative=False,
            musical_behavior_changes=bool(adaptive_thinking_status()["enabled"]),
            commit_authority="existing_rc1_pipeline",
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()
