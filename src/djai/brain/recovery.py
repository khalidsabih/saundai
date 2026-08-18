from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from ..decision import validate_execution_plan
from ..models import TrackAnalysis, TransitionPlan
from .move_intent import DJMoveIntent, IntentBindingError, bind_prepared_transition_to_intent

PREPARED_RECOVERY_VERSION = "1.0"
RecoveryReadiness = Literal["prepared", "auditioned", "hot"]


@dataclass(frozen=True, slots=True)
class PreparedBackupMove:
    backup_id: str
    next_index: int
    execution_signature: str
    technique: str
    selection_score: float
    readiness: RecoveryReadiness
    intent_binding_status: str
    semantic_intent_preserved: bool
    quality_checked: bool
    measurement_passed: bool
    headphones_evidence: bool
    fallback_level: int
    rationale: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rationale"] = list(self.rationale)
        return payload


@dataclass(frozen=True, slots=True)
class PreparedRecoveryPlan:
    version: str
    recovery_id: str
    primary_execution_signature: str
    next_index: int
    required_backups: int
    prepared_backups: tuple[PreparedBackupMove, ...]
    alternate_target_advisories: tuple[dict[str, Any], ...]
    recovery_ready: bool
    best_readiness: str | None
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "recovery_id": self.recovery_id,
            "primary_execution_signature": self.primary_execution_signature,
            "next_index": self.next_index,
            "required_backups": self.required_backups,
            "prepared_backups": [item.to_dict() for item in self.prepared_backups],
            "alternate_target_advisories": [dict(item) for item in self.alternate_target_advisories],
            "recovery_ready": self.recovery_ready,
            "best_readiness": self.best_readiness,
            "policy": self.policy,
        }


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(item) for item in parts)
    return f"{prefix}-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _contains_signature(value: object, signature: str) -> bool:
    if not signature:
        return False
    if isinstance(value, str):
        return value == signature
    if isinstance(value, dict):
        return any(_contains_signature(item, signature) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_signature(item, signature) for item in value)
    return False


def _candidate_is_executable(plan: TransitionPlan) -> bool:
    if not bool(plan.compiled_execution):
        return False
    if not str(plan.performance_program_fingerprint or ""):
        return False
    if not str(plan.execution_signature or ""):
        return False
    if plan.measurement is not None and plan.measurement.hard_failures:
        return False
    if plan.quality_checked and not plan.measurement_passed:
        return False
    try:
        validate_execution_plan(plan, verify_assets=False)
    except Exception:
        return False
    return True


def _readiness(plan: TransitionPlan, *, headphones_evidence: bool) -> RecoveryReadiness:
    if bool(plan.quality_checked and plan.measurement_passed):
        return "hot"
    if headphones_evidence:
        return "auditioned"
    return "prepared"


def _readiness_rank(value: RecoveryReadiness) -> int:
    return {"prepared": 1, "auditioned": 2, "hot": 3}[value]


def build_prepared_recovery_plan(
    *,
    current_index: int,
    next_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    intent: DJMoveIntent,
    primary_plan: TransitionPlan,
    candidates: Sequence[Any],
    headphones_report: dict[str, Any] | None = None,
    required_backups: int = 1,
    maximum_backups: int = 2,
) -> PreparedRecoveryPlan:
    """Prepare same-target fallback executions from already-generated candidates.

    V1 deliberately does not change the route. Different-target alternatives are
    retained as advisory evidence for a later commitment-horizon/local-repair
    milestone, while same-target exact programs can become recovery-ready now.
    No new rendering is triggered here; readiness reflects evidence that already
    exists in the current planning run.
    """

    primary_signature = str(primary_plan.execution_signature or "")
    backups: list[PreparedBackupMove] = []
    alternate_targets: list[dict[str, Any]] = []
    seen_signatures: set[str] = {primary_signature} if primary_signature else set()

    for candidate in candidates:
        candidate_next = int(getattr(candidate, "next_index", -1))
        plan = getattr(candidate, "plan", None)
        move = getattr(candidate, "move", None)
        if plan is None:
            continue
        signature = str(getattr(plan, "execution_signature", "") or "")
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        score = _safe_float(getattr(candidate, "selection_score", 0.0))
        technique = str(getattr(move, "technique", getattr(plan, "technique", "")))

        if candidate_next != int(next_index):
            alternate_targets.append(
                {
                    "next_index": candidate_next,
                    "execution_signature": signature,
                    "technique": technique,
                    "selection_score": round(score, 6),
                    "status": "advisory_only_route_change_required",
                }
            )
            continue
        if str(getattr(plan, "track_a", "")) != str(track_a.path):
            continue
        if str(getattr(plan, "track_b", "")) != str(track_b.path):
            continue
        if not _candidate_is_executable(plan):
            continue
        try:
            binding = bind_prepared_transition_to_intent(
                intent=intent,
                plan=plan,
                track_a=track_a,
                track_b=track_b,
            )
        except IntentBindingError:
            continue

        heard = _contains_signature(headphones_report or {}, signature)
        readiness = _readiness(plan, headphones_evidence=heard)
        rationale = tuple(str(item) for item in getattr(move, "rationale", ()) or ())
        backups.append(
            PreparedBackupMove(
                backup_id=_stable_id("backup", current_index, next_index, signature),
                next_index=int(next_index),
                execution_signature=signature,
                technique=technique,
                selection_score=score,
                readiness=readiness,
                intent_binding_status=str(binding.status),
                semantic_intent_preserved=bool(binding.semantic_intent_preserved),
                quality_checked=bool(plan.quality_checked),
                measurement_passed=bool(plan.measurement_passed),
                headphones_evidence=heard,
                fallback_level=int(getattr(plan, "fallback_level", 0) or 0),
                rationale=rationale,
            )
        )

    backups.sort(
        key=lambda item: (
            _readiness_rank(item.readiness),
            int(item.semantic_intent_preserved),
            item.selection_score,
            -item.fallback_level,
        ),
        reverse=True,
    )
    kept = tuple(backups[: max(0, int(maximum_backups))])
    required = max(0, int(required_backups))
    best = kept[0].readiness if kept else None
    recovery_id = _stable_id("recovery", current_index, next_index, primary_signature, *(item.execution_signature for item in kept))
    return PreparedRecoveryPlan(
        version=PREPARED_RECOVERY_VERSION,
        recovery_id=recovery_id,
        primary_execution_signature=primary_signature,
        next_index=int(next_index),
        required_backups=required,
        prepared_backups=kept,
        alternate_target_advisories=tuple(alternate_targets[:3]),
        recovery_ready=bool(kept) and len(kept) >= max(1, required),
        best_readiness=best,
        policy="same_target_prepared_backup_first; alternate_target_requires_future_plan_repair",
    )


def select_prepared_backup(
    recovery: PreparedRecoveryPlan,
    *,
    failed_execution_signature: str | None = None,
) -> PreparedBackupMove | None:
    """Return the strongest prevalidated same-target backup receipt."""

    for backup in recovery.prepared_backups:
        if failed_execution_signature and backup.execution_signature == failed_execution_signature:
            continue
        return backup
    return None
