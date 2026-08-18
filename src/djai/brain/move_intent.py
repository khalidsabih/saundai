from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from ..models import TrackAnalysis, TransitionPlan
from ..music.timeline import build_musical_timeline
from .dj_move import DJMove

DJ_MOVE_INTENT_VERSION = "1.0"
IntentBindingStatus = Literal["exact", "flexible", "fallback"]


class IntentBindingError(RuntimeError):
    """Raised when a prepared transition cannot represent the proposed DJ intent."""


@dataclass(frozen=True, slots=True)
class BarWindow:
    start_bar: int
    end_bar: int

    def contains(self, bar: int) -> bool:
        return int(self.start_bar) <= int(bar) <= int(self.end_bar)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DJMoveIntent:
    """A musical intention that may bind to more than one exact execution.

    Complete DJMove still projects one exact candidate for ranking/lookahead,
    but the Executive commits only after PREPARE binds a safe compiled program
    to this intention. This separates *what the DJ wants to accomplish* from
    *the exact DSP/source/timing realization* chosen later.
    """

    version: str
    intent_id: str
    current_index: int
    next_index: int
    current_track: str
    next_track: str
    action: str
    musical_objective: str
    preferred_exit_bar: int
    preferred_entry_bar: int
    preferred_transition_bars: int
    acceptable_exit_window: BarWindow
    acceptable_entry_window: BarWindow
    acceptable_transition_bars: tuple[int, ...]
    allowed_techniques: tuple[str, ...]
    allowed_layer_strategies: tuple[str, ...]
    tempo_strategy: str
    phase_strategy: str
    stem_strategy: str
    hard_constraints: tuple[str, ...]
    fallback_policy: str
    proposal_execution_signature: str
    candidate_execution_signatures: tuple[str, ...]
    rationale: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DJMoveIntent":
        return cls(
            version=str(payload.get("version", DJ_MOVE_INTENT_VERSION)),
            intent_id=str(payload["intent_id"]),
            current_index=int(payload["current_index"]),
            next_index=int(payload["next_index"]),
            current_track=str(payload["current_track"]),
            next_track=str(payload["next_track"]),
            action=str(payload["action"]),
            musical_objective=str(payload["musical_objective"]),
            preferred_exit_bar=int(payload["preferred_exit_bar"]),
            preferred_entry_bar=int(payload["preferred_entry_bar"]),
            preferred_transition_bars=int(payload["preferred_transition_bars"]),
            acceptable_exit_window=BarWindow(**{key: int(value) for key, value in dict(payload["acceptable_exit_window"]).items()}),
            acceptable_entry_window=BarWindow(**{key: int(value) for key, value in dict(payload["acceptable_entry_window"]).items()}),
            acceptable_transition_bars=tuple(int(value) for value in payload.get("acceptable_transition_bars", ())),
            allowed_techniques=tuple(str(value) for value in payload.get("allowed_techniques", ())),
            allowed_layer_strategies=tuple(str(value) for value in payload.get("allowed_layer_strategies", ())),
            tempo_strategy=str(payload.get("tempo_strategy", "")),
            phase_strategy=str(payload.get("phase_strategy", "")),
            stem_strategy=str(payload.get("stem_strategy", "")),
            hard_constraints=tuple(str(value) for value in payload.get("hard_constraints", ())),
            fallback_policy=str(payload.get("fallback_policy", "same_target_compiled_safe_fallback")),
            proposal_execution_signature=str(payload.get("proposal_execution_signature", "")),
            candidate_execution_signatures=tuple(str(value) for value in payload.get("candidate_execution_signatures", ())),
            rationale=tuple(str(value) for value in payload.get("rationale", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["acceptable_exit_window"] = self.acceptable_exit_window.to_dict()
        payload["acceptable_entry_window"] = self.acceptable_entry_window.to_dict()
        payload["acceptable_transition_bars"] = list(self.acceptable_transition_bars)
        payload["allowed_techniques"] = list(self.allowed_techniques)
        payload["allowed_layer_strategies"] = list(self.allowed_layer_strategies)
        payload["hard_constraints"] = list(self.hard_constraints)
        payload["candidate_execution_signatures"] = list(self.candidate_execution_signatures)
        payload["rationale"] = list(self.rationale)
        return payload


@dataclass(frozen=True, slots=True)
class PreparedIntentBinding:
    intent_id: str
    status: IntentBindingStatus
    semantic_intent_preserved: bool
    exact_proposal_preserved: bool
    fallback_used: bool
    bound_exit_bar: int
    bound_entry_bar: int
    bound_transition_bars: int
    bound_technique: str
    bound_layer_strategy: str | None
    within_exit_window: bool
    within_entry_window: bool
    transition_bars_allowed: bool
    technique_allowed: bool
    layer_strategy_allowed: bool
    objective_match: bool
    prepared_execution_signature: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _stable_intent_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "intent-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _phrase_window(track: TrackAnalysis, bar: int) -> BarWindow:
    phrase = build_musical_timeline(track).phrase_for_bar(int(bar))
    if phrase is None:
        return BarWindow(int(bar), int(bar))
    return BarWindow(int(phrase.start_bar), max(int(phrase.start_bar), int(phrase.end_bar) - 1))


def _layer_strategy(move: DJMove) -> str | None:
    choreography = dict(move.layer_choreography or {})
    value = choreography.get("strategy")
    return str(value) if value else None


def build_dj_move_intent(
    *,
    current_index: int,
    next_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    winner_move: DJMove,
    winner_plan: TransitionPlan,
    candidate_moves: Sequence[DJMove] = (),
    candidate_plans: Sequence[TransitionPlan] = (),
) -> DJMoveIntent:
    """Build a phrase-bounded intent from the winning Complete DJMove.

    The flexibility envelope is intentionally conservative in V1: timing may
    vary inside the preferred source/entry phrases, and duration/technique may
    use alternatives that the current Complete Move search already considered
    for the same target track. A safe same-target compiled fallback remains
    legal during PREPARE and is explicitly marked as fallback rather than
    silently treated as the original proposal.
    """

    moves = [winner_move, *candidate_moves]
    plans = [winner_plan, *candidate_plans]
    same_target_moves = [move for move in moves if str(move.next_track) == str(winner_move.next_track)]
    same_target_plans = [plan for plan in plans if str(plan.track_b) == str(winner_plan.track_b)]

    techniques = tuple(dict.fromkeys(str(plan.technique) for plan in same_target_plans if plan.technique))
    if str(winner_move.technique) not in techniques:
        techniques = (str(winner_move.technique), *techniques)
    durations = tuple(sorted({int(move.transition_bars) for move in same_target_moves if int(move.transition_bars) > 0}))
    if not durations:
        durations = (int(winner_move.transition_bars),)
    layer_strategies = tuple(
        dict.fromkeys(
            strategy
            for strategy in (_layer_strategy(move) for move in same_target_moves)
            if strategy
        )
    )
    signatures = tuple(
        dict.fromkeys(str(plan.execution_signature) for plan in same_target_plans if str(plan.execution_signature or ""))
    )

    hard_constraints = ["target_track_locked", "compiled_execution_required", "hard_audio_safety_required"]
    if bool(winner_move.payoff_satisfied) and not bool(winner_move.protected_payoff_violation):
        hard_constraints.append("preserve_protected_payoff")
    if layer_strategies:
        hard_constraints.append("layer_move_must_keep_master_fallback")

    seed = {
        "current_index": int(current_index),
        "next_index": int(next_index),
        "track_a": str(track_a.path),
        "track_b": str(track_b.path),
        "objective": str(winner_move.intent),
        "exit": int(winner_move.exit_bar),
        "entry": int(winner_move.entry_bar),
        "bars": int(winner_move.transition_bars),
        "signature": str(winner_plan.execution_signature or ""),
    }
    return DJMoveIntent(
        version=DJ_MOVE_INTENT_VERSION,
        intent_id=_stable_intent_id(seed),
        current_index=int(current_index),
        next_index=int(next_index),
        current_track=str(track_a.path),
        next_track=str(track_b.path),
        action=str(winner_move.action),
        musical_objective=str(winner_move.intent),
        preferred_exit_bar=int(winner_move.exit_bar),
        preferred_entry_bar=int(winner_move.entry_bar),
        preferred_transition_bars=int(winner_move.transition_bars),
        acceptable_exit_window=_phrase_window(track_a, int(winner_move.exit_bar)),
        acceptable_entry_window=_phrase_window(track_b, int(winner_move.entry_bar)),
        acceptable_transition_bars=durations,
        allowed_techniques=techniques,
        allowed_layer_strategies=layer_strategies,
        tempo_strategy=str(winner_move.tempo_strategy),
        phase_strategy=str(winner_move.phase_strategy),
        stem_strategy=str(winner_move.stem_strategy),
        hard_constraints=tuple(hard_constraints),
        fallback_policy="same_target_compiled_safe_fallback",
        proposal_execution_signature=str(winner_plan.execution_signature or ""),
        candidate_execution_signatures=signatures,
        rationale=tuple(str(item) for item in winner_move.rationale),
    )


def build_prepared_plan_intent(
    *,
    current_index: int,
    next_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
) -> DJMoveIntent:
    """Create a narrow intent when no Complete-Move proposal exists.

    Manual order and legacy/emergency compilation still receive an explicit
    intent contract, but V1 does not invent flexibility that search never
    proposed.
    """

    exit_bar = build_musical_timeline(track_a).bar_for_time(float(plan.exit_time_a))
    entry_bar = build_musical_timeline(track_b).bar_for_time(float(plan.entry_time_b))
    choreography = dict(plan.metadata.get("layer_choreography", {}) or {})
    layer_strategy = str(choreography.get("strategy")) if choreography.get("strategy") else None
    seed = {
        "current_index": int(current_index),
        "next_index": int(next_index),
        "track_a": str(track_a.path),
        "track_b": str(track_b.path),
        "objective": str(plan.intent),
        "exit": int(exit_bar),
        "entry": int(entry_bar),
        "bars": int(plan.transition_bars),
        "signature": str(plan.execution_signature or ""),
    }
    return DJMoveIntent(
        version=DJ_MOVE_INTENT_VERSION,
        intent_id=_stable_intent_id(seed),
        current_index=int(current_index),
        next_index=int(next_index),
        current_track=str(track_a.path),
        next_track=str(track_b.path),
        action="transition",
        musical_objective=str(plan.intent),
        preferred_exit_bar=int(exit_bar),
        preferred_entry_bar=int(entry_bar),
        preferred_transition_bars=int(plan.transition_bars),
        acceptable_exit_window=BarWindow(int(exit_bar), int(exit_bar)),
        acceptable_entry_window=BarWindow(int(entry_bar), int(entry_bar)),
        acceptable_transition_bars=(int(plan.transition_bars),),
        allowed_techniques=(str(plan.technique),),
        allowed_layer_strategies=((layer_strategy,) if layer_strategy else ()),
        tempo_strategy=str(plan.tempo_strategy),
        phase_strategy="compiled",
        stem_strategy=str(plan.metadata.get("stem_strategy", "prepared")),
        hard_constraints=("target_track_locked", "compiled_execution_required", "hard_audio_safety_required"),
        fallback_policy="exact_prepared_plan_only",
        proposal_execution_signature=str(plan.execution_signature or ""),
        candidate_execution_signatures=(str(plan.execution_signature or ""),) if plan.execution_signature else (),
        rationale=tuple(str(item) for item in plan.rationale),
    )


def bind_prepared_transition_to_intent(
    *,
    intent: DJMoveIntent,
    plan: TransitionPlan,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
) -> PreparedIntentBinding:
    """Bind the exact prepared program to a previously proposed musical intent."""

    if str(plan.track_a) != str(intent.current_track) or str(track_a.path) != str(intent.current_track):
        raise IntentBindingError("Prepared source A does not match the DJMoveIntent current track.")
    if str(plan.track_b) != str(intent.next_track) or str(track_b.path) != str(intent.next_track):
        raise IntentBindingError("Prepared source B does not match the DJMoveIntent target track.")

    exit_bar = build_musical_timeline(track_a).bar_for_time(float(plan.exit_time_a))
    entry_bar = build_musical_timeline(track_b).bar_for_time(float(plan.entry_time_b))
    choreography = dict(plan.metadata.get("layer_choreography", {}) or {})
    layer_strategy = str(choreography.get("strategy")) if choreography.get("strategy") else None

    within_exit = intent.acceptable_exit_window.contains(exit_bar)
    within_entry = intent.acceptable_entry_window.contains(entry_bar)
    duration_allowed = int(plan.transition_bars) in set(intent.acceptable_transition_bars)
    technique_allowed = str(plan.technique) in set(intent.allowed_techniques)
    layer_allowed = layer_strategy is None or layer_strategy in set(intent.allowed_layer_strategies)
    objective_match = str(plan.intent) == str(intent.musical_objective)
    exact = bool(
        intent.proposal_execution_signature
        and str(plan.execution_signature or "") == intent.proposal_execution_signature
    )

    preferred_semantics = all(
        (within_exit, within_entry, duration_allowed, technique_allowed, layer_allowed, objective_match)
    )
    reasons: list[str] = []
    if exact:
        status: IntentBindingStatus = "exact"
        reasons.append("Prepared execution exactly matches the Complete DJMove proposal.")
    elif preferred_semantics:
        status = "flexible"
        reasons.append("Prepared execution changed while remaining inside the DJMoveIntent flexibility envelope.")
    else:
        if intent.fallback_policy == "exact_prepared_plan_only":
            raise IntentBindingError("Prepared execution falls outside an exact-only DJMoveIntent.")
        status = "fallback"
        if not within_exit:
            reasons.append("Prepared exit moved outside the preferred phrase window.")
        if not within_entry:
            reasons.append("Prepared entry moved outside the preferred phrase window.")
        if not duration_allowed:
            reasons.append("Prepared transition duration was not in the proposed duration set.")
        if not technique_allowed:
            reasons.append("Prepared technique was not among the proposed techniques.")
        if not layer_allowed:
            reasons.append("Prepared layer choreography was not among the proposed layer strategies.")
        if not objective_match:
            reasons.append("Prepared transition intent label differs from the proposed musical objective.")
        reasons.append("Same-target safe fallback is allowed, but is recorded as a fallback binding.")

    return PreparedIntentBinding(
        intent_id=intent.intent_id,
        status=status,
        semantic_intent_preserved=bool(preferred_semantics or exact),
        exact_proposal_preserved=exact,
        fallback_used=status == "fallback",
        bound_exit_bar=int(exit_bar),
        bound_entry_bar=int(entry_bar),
        bound_transition_bars=int(plan.transition_bars),
        bound_technique=str(plan.technique),
        bound_layer_strategy=layer_strategy,
        within_exit_window=within_exit,
        within_entry_window=within_entry,
        transition_bars_allowed=duration_allowed,
        technique_allowed=technique_allowed,
        layer_strategy_allowed=layer_allowed,
        objective_match=objective_match,
        prepared_execution_signature=str(plan.execution_signature or ""),
        reasons=tuple(reasons),
    )
