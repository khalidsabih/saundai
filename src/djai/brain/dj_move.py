from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from ..models import TrackAnalysis, TransitionCandidate, TransitionPlan
from ..performance.operations import ProgramDeckOperation
from ..music.timeline import (
    MusicalCommitment,
    MusicalTimeline,
    PhraseMoment,
    build_musical_timeline,
    is_hard_song_form_commitment,
)
from .layer_intelligence import (
    LAYER_INTELLIGENCE_VERSION,
    LayerBrainConfig,
    LayerState,
    layer_state_for_track,
)
from .set_state import SetState
from .structural_operations import compile_move_deck_operations, structural_operation_status

DJ_MOVE_VERSION = "1.4"
MoveAction = Literal["transition", "wait_then_transition", "wait"]


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class MoveBrainConfig:
    enabled: bool = True
    order_weight: float = 0.30
    plan_weight: float = 0.36
    payoff_protection: float = 0.86
    protected_importance: float = 0.74
    wait_threshold_bars: int = 4
    weak_clock_weight: float = 0.05
    minimum_body_bars: int = 8
    soft_target_bars: int = 48
    soft_latest_bars: int = 80

    @classmethod
    def from_environment(cls) -> "MoveBrainConfig":
        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return default

        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default

        return cls(
            enabled=_enabled("DJAI_MOVE_BRAIN", "1"),
            order_weight=_clamp(_float("DJAI_MOVE_ORDER_WEIGHT", 0.30)),
            plan_weight=_clamp(_float("DJAI_MOVE_PLAN_WEIGHT", 0.36)),
            payoff_protection=_clamp(_float("DJAI_MOVE_PAYOFF_PROTECTION", 0.86)),
            protected_importance=_clamp(_float("DJAI_MOVE_PROTECTED_IMPORTANCE", 0.74)),
            wait_threshold_bars=max(1, min(32, _int("DJAI_MOVE_WAIT_THRESHOLD_BARS", 4))),
            weak_clock_weight=max(0.0, min(0.15, _float("DJAI_MOVE_CLOCK_WEIGHT", 0.05))),
            minimum_body_bars=max(4, min(32, _int("DJAI_MOVE_MIN_BODY_BARS", 8))),
            soft_target_bars=max(16, min(128, _int("DJAI_MOVE_SOFT_TARGET_BARS", 48))),
            soft_latest_bars=max(24, min(160, _int("DJAI_MOVE_SOFT_LATEST_BARS", 80))),
        )


@dataclass(frozen=True, slots=True)
class DJState:
    version: str
    current_track: str
    source_cursor_seconds: float
    source_bar: int
    phrase_index: int
    phrase_role: str
    effective_bpm: float
    native_bpm: float
    render_rate: float
    tempo_owner: str
    current_energy: float
    current_tension: float
    current_vocal_probability: float
    current_bass_activity: float
    set_energy: float
    set_tension: float
    set_groove: float
    set_vocal_density: float
    pending_commitments: tuple[MusicalCommitment, ...] = ()
    layer_state: LayerState = field(default_factory=LayerState)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pending_commitments"] = [item.to_dict() for item in self.pending_commitments]
        return payload


@dataclass(frozen=True, slots=True)
class DJMove:
    version: str
    action: MoveAction
    current_track: str
    next_track: str
    exit_bar: int
    entry_bar: int
    exit_phrase: int
    entry_phrase: int
    transition_bars: int
    wait_bars: int
    matched_bpm: float
    incoming_native_bpm: float
    post_transition_bpm: float
    tempo_release_seconds: float
    phase_nudge_seconds: float
    technique: str
    intent: str
    tempo_strategy: str
    phase_strategy: str
    stem_strategy: str
    stem_profile: str | None
    structural_operations: tuple[str, ...]
    deck_operations: tuple[ProgramDeckOperation, ...]
    protected_payoff_violation: bool
    payoff_satisfied: bool
    landing_phrase: int
    future_optionality: float
    projected_state: DJState
    score: float
    protected_commitment_kind: str | None = None
    components: dict[str, float] = field(default_factory=dict)
    rationale: tuple[str, ...] = ()
    layer_choreography: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rationale"] = list(self.rationale)
        payload["structural_operations"] = list(self.structural_operations)
        payload["deck_operations"] = [item.to_dict() for item in self.deck_operations]
        payload["projected_state"] = self.projected_state.to_dict()
        return payload


def _bar_for_time(track: TrackAnalysis, seconds: float) -> int:
    return build_musical_timeline(track).bar_for_time(seconds)


def build_dj_state(
    track: TrackAnalysis,
    *,
    source_cursor: float,
    effective_bpm: float | None = None,
    render_rate: float = 1.0,
    set_state: SetState | None = None,
    timeline: MusicalTimeline | None = None,
    tempo_owner: str = "A",
) -> DJState:
    timeline = timeline or build_musical_timeline(track)
    bar = timeline.bar_for_time(max(0.0, source_cursor))
    phrase = timeline.phrase_for_bar(bar)
    pending = tuple(
        item
        for item in timeline.commitments
        if item.protect_until_bar > bar
        and (
            item.target_bar <= bar < item.protect_until_bar
            or bar <= item.target_bar <= bar + 16
        )
    )
    return DJState(
        version=DJ_MOVE_VERSION,
        current_track=track.path,
        source_cursor_seconds=round(float(source_cursor), 6),
        source_bar=bar,
        phrase_index=phrase.phrase_index if phrase else 0,
        phrase_role=phrase.role if phrase else "unknown",
        effective_bpm=round(float(effective_bpm or track.bpm), 4),
        native_bpm=round(float(track.bpm), 4),
        render_rate=round(float(render_rate), 6),
        tempo_owner=str(tempo_owner),
        current_energy=phrase.energy if phrase else float(track.energy),
        current_tension=phrase.tension if phrase else 0.35,
        current_vocal_probability=phrase.vocal_probability if phrase else float(track.vocal_probability),
        current_bass_activity=phrase.bass_activity if phrase else float(track.bass_activity),
        set_energy=float(set_state.energy if set_state is not None else track.energy),
        set_tension=float(set_state.tension if set_state is not None else (phrase.tension if phrase else 0.35)),
        set_groove=float(set_state.groove if set_state is not None else track.percussive_ratio),
        set_vocal_density=float(set_state.vocal_density if set_state is not None else track.vocal_probability),
        pending_commitments=pending,
        layer_state=layer_state_for_track(track, owner=tempo_owner),
    )


def _protected_violation(state: DJState, exit_bar: int, config: MoveBrainConfig) -> tuple[bool, MusicalCommitment | None]:
    for commitment in state.pending_commitments:
        if commitment.confidence < 0.55:
            continue
        if (
            commitment.importance < config.protected_importance
            and commitment.kind not in {"pending_drop_payoff", "first_body_payoff", "protected_hook"}
        ):
            continue
        if exit_bar < commitment.protect_until_bar:
            return True, commitment
    return False, None


def _timeline_protected_violation(
    timeline: MusicalTimeline,
    exit_bar: int,
    config: MoveBrainConfig,
) -> tuple[bool, MusicalCommitment | None]:
    """Protect the musical promise local to a future exit candidate.

    Order planning often evaluates an exit many phrases ahead of the current
    playback state. Limiting protection to the current state's short horizon
    would let a pre-drop exit escape the gate simply because that build has not
    started yet. This local check follows the candidate's own phrase context.
    """
    for commitment in timeline.commitments:
        if commitment.confidence < 0.55:
            continue
        if (
            commitment.importance < config.protected_importance
            and commitment.kind not in {"pending_drop_payoff", "first_body_payoff", "protected_hook"}
        ):
            continue
        if commitment.kind in {"pending_drop_payoff", "first_body_payoff"}:
            try:
                protect_from = timeline.moments[commitment.source_phrase].start_bar
            except (IndexError, TypeError):
                protect_from = commitment.target_bar
        else:
            protect_from = commitment.target_bar
        if int(protect_from) <= exit_bar < commitment.protect_until_bar:
            return True, commitment
    return False, None


def exit_candidate_musical_score(
    track: TrackAnalysis,
    candidate: TransitionCandidate,
    state: DJState,
    *,
    config: MoveBrainConfig | None = None,
    timeline: MusicalTimeline | None = None,
) -> tuple[float, dict[str, Any]]:
    cfg = config or MoveBrainConfig.from_environment()
    timeline = timeline or build_musical_timeline(track)
    moment = timeline.phrase_for_bar(int(candidate.bar_index))
    if moment is None:
        return _clamp(candidate.score), {"available": False}
    violation, commitment = _protected_violation(state, int(candidate.bar_index), cfg)
    if not violation:
        violation, commitment = _timeline_protected_violation(timeline, int(candidate.bar_index), cfg)
    payoff_satisfied = bool(
        state.pending_commitments
        and all(int(candidate.bar_index) >= item.protect_until_bar for item in state.pending_commitments if item.confidence >= 0.55)
    )
    importance_exitability = 1.0 - 0.72 * moment.importance
    if moment.role in {"outro", "breakdown", "bridge"}:
        importance_exitability = min(1.0, importance_exitability + 0.20)
    # Song-form semantics refine exitability without replacing the coarse
    # structural role.  Verse openings and hooks should not look like generic
    # high-quality boundaries merely because their onset is structurally strong.
    if moment.song_role == "hook":
        importance_exitability = min(
            importance_exitability,
            0.20 if moment.song_role_index <= 1 else 0.42,
        )
    elif moment.song_role == "verse":
        importance_exitability = min(importance_exitability, 0.46)
    elif moment.song_role in {"instrumental", "outro"}:
        importance_exitability = min(1.0, importance_exitability + 0.10)
    vocal_clean = 1.0 - float(candidate.vocal_probability)
    musical = _clamp(
        0.34 * moment.transition_readiness
        + 0.20 * float(candidate.score)
        + 0.14 * float(candidate.phrase_strength)
        + 0.10 * float(candidate.boundary_strength)
        + 0.10 * importance_exitability
        + 0.07 * vocal_clean
        + 0.05 * (1.0 if payoff_satisfied else 0.5)
    )
    if violation:
        # This is a musical acceptability gate, not a tiny linear preference.
        # A deliberate drop replacement can still be considered later by the
        # complete plan scorer, but ordinary exits should fall out of the pool.
        musical *= max(0.04, 1.0 - cfg.payoff_protection)
    return _clamp(musical), {
        "available": True,
        "phrase_index": moment.phrase_index,
        "role": moment.role,
        "song_role": moment.song_role,
        "song_role_index": moment.song_role_index,
        "song_form_confidence": round(moment.song_form_confidence, 4),
        "importance": round(moment.importance, 4),
        "transition_readiness": round(moment.transition_readiness, 4),
        "protected_payoff_violation": violation,
        "payoff_satisfied": payoff_satisfied,
        "commitment": commitment.to_dict() if commitment is not None else None,
        "hard_song_form_protection": is_hard_song_form_commitment(commitment),
    }


def entry_candidate_musical_score(
    track: TrackAnalysis,
    candidate: TransitionCandidate,
    *,
    timeline: MusicalTimeline | None = None,
) -> tuple[float, dict[str, Any]]:
    timeline = timeline or build_musical_timeline(track)
    moment = timeline.phrase_for_bar(int(candidate.bar_index))
    if moment is None:
        return _clamp(candidate.score), {"available": False}
    role_fit = {
        "intro": 0.82,
        "groove": 0.96,
        "bridge": 0.92,
        "build": 0.82,
        "pre_drop": 0.72,
        "drop": 0.78,
        "breakdown": 0.66,
        "outro": 0.20,
        "unknown": 0.50,
    }.get(moment.role, 0.50)
    vocal_space = 1.0 - 0.72 * float(candidate.vocal_probability)
    musical = _clamp(
        0.28 * float(candidate.score)
        + 0.20 * moment.transition_readiness
        + 0.16 * float(candidate.phrase_strength)
        + 0.12 * float(candidate.boundary_strength)
        + 0.14 * role_fit
        + 0.10 * vocal_space
    )
    return musical, {
        "available": True,
        "phrase_index": moment.phrase_index,
        "role": moment.role,
        "song_role": moment.song_role,
        "song_role_index": moment.song_role_index,
        "song_form_confidence": round(moment.song_form_confidence, 4),
        "importance": round(moment.importance, 4),
        "transition_readiness": round(moment.transition_readiness, 4),
    }


def _tempo_fit(track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    a = max(float(track_a.bpm), 1.0)
    b = max(float(track_b.bpm), 1.0)
    ratios = [b / a, (b * 0.5) / a, (b * 2.0) / a]
    delta = min(abs(value - 1.0) for value in ratios)
    return _clamp(math.exp(-((delta / 0.065) ** 2)))


def _future_optionality(timeline: MusicalTimeline, landing_bar: int) -> float:
    upcoming = [
        moment
        for moment in timeline.moments
        if moment.start_bar >= landing_bar and moment.start_bar <= landing_bar + 40
    ][:4]
    if not upcoming:
        return 0.25
    values = [
        _clamp(0.68 * moment.transition_readiness + 0.32 * (1.0 - 0.65 * moment.importance))
        for moment in upcoming
    ]
    return _clamp(0.65 * max(values) + 0.35 * float(np.mean(values)))


def move_opportunity_score(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    timeline_a: MusicalTimeline | None = None,
    timeline_b: MusicalTimeline | None = None,
) -> tuple[float, dict[str, Any]]:
    """Cheap phrase-level prior used before committing to a track order.

    This is intentionally not the final transition score. It asks a human-DJ
    question earlier in planning: *does this record actually offer a good
    musical move from the current record?*
    """
    cfg = MoveBrainConfig.from_environment()
    if not cfg.enabled:
        return 0.5, {"available": False, "reason": "disabled"}
    timeline_a = timeline_a or build_musical_timeline(track_a)
    timeline_b = timeline_b or build_musical_timeline(track_b)
    state = build_dj_state(track_a, source_cursor=0.0, timeline=timeline_a)
    exit_ranked: list[tuple[float, TransitionCandidate, dict[str, Any]]] = []
    for candidate in track_a.exit_candidates:
        value, meta = exit_candidate_musical_score(track_a, candidate, state, config=cfg, timeline=timeline_a)
        exit_ranked.append((value, candidate, meta))
    entry_ranked: list[tuple[float, TransitionCandidate, dict[str, Any]]] = []
    for candidate in track_b.entry_candidates:
        value, meta = entry_candidate_musical_score(track_b, candidate, timeline=timeline_b)
        entry_ranked.append((value, candidate, meta))
    if not exit_ranked or not entry_ranked:
        return 0.38, {"available": False, "reason": "missing_cue_candidates"}
    exit_ranked.sort(key=lambda item: item[0], reverse=True)
    entry_ranked.sort(key=lambda item: item[0], reverse=True)
    safe_exit_ranked = [item for item in exit_ranked if not bool(item[2].get("hard_song_form_protection", False))]
    if not safe_exit_ranked:
        return 0.12, {
            "available": False,
            "reason": "no_post_song_form_exit",
            "protected_payoff_violation": True,
            "hard_song_form_protection": True,
        }
    best_exit = safe_exit_ranked[0]
    best_entry = entry_ranked[0]
    tempo = _tempo_fit(track_a, track_b)
    vocal = 1.0 - min(1.0, float(best_exit[1].vocal_probability) * float(best_entry[1].vocal_probability) * 1.4)
    future = _future_optionality(timeline_b, int(best_entry[1].bar_index) + 8)
    score = _clamp(
        0.32 * best_exit[0]
        + 0.27 * best_entry[0]
        + 0.17 * tempo
        + 0.12 * vocal
        + 0.12 * future
    )
    return score, {
        "available": True,
        "score": round(score, 6),
        "best_exit_bar": int(best_exit[1].bar_index),
        "best_entry_bar": int(best_entry[1].bar_index),
        "exit_phrase": best_exit[2].get("phrase_index"),
        "entry_phrase": best_entry[2].get("phrase_index"),
        "tempo_fit": round(tempo, 4),
        "vocal_space": round(vocal, 4),
        "future_optionality": round(future, 4),
        "protected_payoff_violation": bool(best_exit[2].get("protected_payoff_violation", False)),
    }


def score_transition_plan_as_move(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    state: DJState,
    config: MoveBrainConfig | None = None,
) -> DJMove:
    cfg = config or MoveBrainConfig.from_environment()
    exit_bar = int(plan.metadata.get("exit_bar_index", _bar_for_time(track_a, plan.exit_time_a)))
    entry_bar = int(plan.metadata.get("entry_bar_index", _bar_for_time(track_b, plan.entry_time_b)))
    timeline_a = build_musical_timeline(track_a)
    timeline_b = build_musical_timeline(track_b)
    exit_moment = timeline_a.phrase_for_bar(exit_bar)
    entry_moment = timeline_b.phrase_for_bar(entry_bar)
    violation, commitment = _protected_violation(state, exit_bar, cfg)
    if not violation:
        violation, commitment = _timeline_protected_violation(timeline_a, exit_bar, cfg)
    hard_song_form_protection = bool(violation and is_hard_song_form_commitment(commitment))
    deliberate_payoff_replacement = bool(
        not hard_song_form_protection
        and (
            plan.intent in {"breakdown_to_drop", "dramatic_switch"}
            or plan.technique in {"double_drop", "echo_out_drop_in", "backspin"}
        )
    )
    if violation and hard_song_form_protection:
        violation_penalty = 1.0
    elif violation and deliberate_payoff_replacement:
        # A human may intentionally replace the outgoing payoff with the
        # incoming payoff. Keep it possible, but still make the decision visible.
        violation_penalty = 0.28
    elif violation:
        violation_penalty = cfg.payoff_protection
    else:
        violation_penalty = 0.0

    exit_readiness = float(exit_moment.transition_readiness if exit_moment else plan.cue_confidence)
    entry_readiness = float(entry_moment.transition_readiness if entry_moment else plan.cue_confidence)
    phrase_fit = _clamp(float(getattr(plan.compatibility, "phrase", 0.5)))
    tempo = _tempo_fit(track_a, track_b)
    vocal_overlap = float(plan.metadata.get("exit_vocal_probability", track_a.vocal_probability)) * float(
        plan.metadata.get("entry_vocal_probability", track_b.vocal_probability)
    )
    if plan.technique in {
        "stem_choreography",
        "stem_vocal_overlay",
        "layer_vocal_bridge",
        "layer_incoming_vocal_handoff",
        "layer_instrumental_handoff",
    }:
        vocal_overlap *= 0.45
    vocal_space = _clamp(1.0 - 1.25 * vocal_overlap)
    landing_bar = entry_bar + max(1, int(plan.transition_bars))
    future = _future_optionality(timeline_b, landing_bar)
    duration_fit = 1.0
    if plan.transition_bars >= 16 and vocal_overlap > 0.30:
        duration_fit -= 0.22
    if plan.transition_bars <= 4 and abs(track_b.bpm - state.effective_bpm) / max(state.effective_bpm, 1.0) > 0.04:
        duration_fit -= 0.18
    duration_fit = _clamp(duration_fit)
    base_quality = _clamp(float(plan.metadata.get("selection_utility", plan.score)))
    score = _clamp(
        0.20 * exit_readiness
        + 0.17 * entry_readiness
        + 0.15 * phrase_fit
        + 0.12 * tempo
        + 0.11 * vocal_space
        + 0.10 * duration_fit
        + 0.09 * future
        + 0.06 * base_quality
        - violation_penalty
    )
    wait_bars = max(0, exit_bar - state.source_bar)
    payoff_satisfied = bool(
        state.pending_commitments
        and all(exit_bar >= item.protect_until_bar for item in state.pending_commitments if item.confidence >= 0.55)
    )
    action: MoveAction = "wait_then_transition" if wait_bars >= cfg.wait_threshold_bars else "transition"
    rationale: list[str] = []
    if action == "wait_then_transition":
        rationale.append(f"Wait {wait_bars} bars for the stronger musical opportunity instead of forcing an immediate mix.")
    if payoff_satisfied:
        rationale.append("Outgoing musical commitment/payoff is completed before the handoff.")
    if violation:
        if hard_song_form_protection:
            rationale.append(
                "Candidate would leave before the first real song body/hook promise completes; this song-form guard cannot be replaced by a drop-oriented move."
            )
        else:
            rationale.append(
                "Candidate intersects a protected payoff; only an explicit payoff-replacement move should survive."
                if deliberate_payoff_replacement
                else "Candidate would leave before a protected musical payoff completes."
            )
    rationale.append(f"Landing leaves the incoming track with {future:.0%} near-term transition optionality.")
    projected_state = build_dj_state(
        track_b,
        source_cursor=plan.post_transition_source_end_b,
        effective_bpm=plan.post_transition_bpm_b or track_b.bpm,
        render_rate=plan.post_transition_rate_b,
        tempo_owner="B",
    )
    stem_payload = plan.metadata.get("stem_choreography", {})
    stem_profile = str(stem_payload.get("profile")) if isinstance(stem_payload, dict) and stem_payload.get("profile") else None
    operations: list[str] = []
    if action == "wait_then_transition":
        operations.append("WAIT")
    if plan.beatmatched:
        operations.append("TEMPO_MATCH")
    if abs(float(plan.phase_nudge_seconds_b)) >= 0.001:
        operations.append("PHASE_NUDGE")
    if stem_profile:
        operations.append("STEM_HANDOFF")
    if plan.technique in {
        "layer_vocal_bridge",
        "layer_incoming_vocal_handoff",
        "layer_instrumental_handoff",
    }:
        operations.append("LAYER_HANDOFF")
    if plan.technique == "layer_incoming_vocal_handoff":
        operations.append("VOCAL_OVER_INSTRUMENTAL")
    if float(plan.tempo_release_seconds_b) > 0.05:
        operations.append("TEMPO_RELEASE")
    deck_operations = compile_move_deck_operations(
        track_a=track_a,
        track_b=track_b,
        plan=plan,
        source_cursor_seconds=state.source_cursor_seconds,
        source_bar=state.source_bar,
        exit_bar=exit_bar,
        entry_bar=entry_bar,
        wait_bars=wait_bars,
        timeline_a=timeline_a,
        timeline_b=timeline_b,
        stem_profile=stem_profile,
        wait_threshold_bars=cfg.wait_threshold_bars,
    )
    return DJMove(
        version=DJ_MOVE_VERSION,
        action=action,
        current_track=track_a.path,
        next_track=track_b.path,
        exit_bar=exit_bar,
        entry_bar=entry_bar,
        exit_phrase=exit_moment.phrase_index if exit_moment else 0,
        entry_phrase=entry_moment.phrase_index if entry_moment else 0,
        transition_bars=int(plan.transition_bars),
        wait_bars=wait_bars,
        matched_bpm=round(float(plan.target_bpm), 4),
        incoming_native_bpm=round(float(track_b.bpm), 4),
        post_transition_bpm=round(float(plan.post_transition_bpm_b or track_b.bpm), 4),
        tempo_release_seconds=round(float(plan.tempo_release_seconds_b), 6),
        phase_nudge_seconds=round(float(plan.phase_nudge_seconds_b), 6),
        technique=str(plan.technique),
        intent=str(plan.intent),
        tempo_strategy=str(plan.tempo_strategy),
        phase_strategy=("phase_nudge" if abs(float(plan.phase_nudge_seconds_b)) >= 0.001 else "beat_grid_locked"),
        stem_strategy=(
            stem_profile
            or (
                "performance_layers"
                if plan.technique
                in {
                    "layer_vocal_bridge",
                    "layer_incoming_vocal_handoff",
                    "layer_instrumental_handoff",
                }
                else "master"
            )
        ),
        stem_profile=stem_profile,
        structural_operations=tuple(operations),
        deck_operations=deck_operations,
        protected_payoff_violation=violation,
        payoff_satisfied=payoff_satisfied,
        landing_phrase=(timeline_b.phrase_for_bar(landing_bar).phrase_index if timeline_b.phrase_for_bar(landing_bar) else 0),
        future_optionality=round(future, 4),
        projected_state=projected_state,
        score=round(score, 6),
        protected_commitment_kind=(commitment.kind if violation and commitment is not None else None),
        components={
            "exit_readiness": round(exit_readiness, 4),
            "entry_readiness": round(entry_readiness, 4),
            "phrase_fit": round(phrase_fit, 4),
            "tempo_fit": round(tempo, 4),
            "vocal_space": round(vocal_space, 4),
            "duration_fit": round(duration_fit, 4),
            "future_optionality": round(future, 4),
            "base_quality": round(base_quality, 4),
            "payoff_penalty": round(violation_penalty, 4),
            "song_form_hard_guard": float(hard_song_form_protection),
        },
        rationale=tuple(rationale),
        layer_choreography=dict(plan.metadata.get("layer_choreography", {}) or {}),
    )


def rank_transition_plans_as_moves(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    *,
    state: DJState,
    config: MoveBrainConfig | None = None,
) -> list[TransitionPlan]:
    cfg = config or MoveBrainConfig.from_environment()
    if not cfg.enabled:
        return list(plans)
    ranked: list[tuple[float, TransitionPlan]] = []
    for plan in plans:
        move = score_transition_plan_as_move(track_a, track_b, plan, state=state, config=cfg)
        old_utility = _clamp(float(plan.metadata.get("selection_utility", plan.score)))
        # Hierarchy: ordinary protected-payoff violations are capped before the
        # learned/heuristic preference blend. We do not let a high generic pair
        # score buy its way through a musically unacceptable early exit.
        if move.protected_payoff_violation and move.protected_commitment_kind in {"first_body_payoff", "protected_hook"}:
            utility = 0.0
        elif move.protected_payoff_violation and move.intent not in {"breakdown_to_drop", "dramatic_switch"}:
            utility = min(0.24, old_utility)
        else:
            utility = (1.0 - cfg.plan_weight) * old_utility + cfg.plan_weight * move.score
        plan.metadata = {
            **plan.metadata,
            "pre_move_selection_utility": round(old_utility, 6),
            "selection_utility": round(utility, 6),
            "dj_move": move.to_dict(),
            "dj_move_brain": {
                "version": DJ_MOVE_VERSION,
                "plan_weight": round(cfg.plan_weight, 4),
                "hard_audio_safety_authoritative": True,
                "song_form_hard_guard": bool(
                    move.protected_payoff_violation
                    and move.protected_commitment_kind in {"first_body_payoff", "protected_hook"}
                ),
            },
        }
        ranked.append((utility, plan))
    ranked.sort(key=lambda item: (item[0], item[1].score, item[1].execution_signature), reverse=True)
    return [plan for _, plan in ranked]


def move_brain_status() -> dict[str, Any]:
    cfg = MoveBrainConfig.from_environment()
    layer_cfg = LayerBrainConfig.from_environment()
    try:
        from .move_generator import complete_move_status
        complete_move = complete_move_status()
    except ImportError:
        complete_move = {"version": "unavailable"}
    return {
        "schema_version": DJ_MOVE_VERSION,
        "enabled": cfg.enabled,
        "planning_unit": "DJState -> DJMove -> DJState",
        "coordinates": "bar_phrase_native",
        "order_weight": round(cfg.order_weight, 4),
        "plan_weight": round(cfg.plan_weight, 4),
        "payoff_protection": round(cfg.payoff_protection, 4),
        "protected_importance": round(cfg.protected_importance, 4),
        "explicit_wait_actions": True,
        "musical_commitments": True,
        "song_form_roles": True,
        "first_body_protection": True,
        "hook_protection": True,
        "first_hook_pre_entry_protection": True,
        "song_form_hard_guard": True,
        "future_optionality": True,
        "time_seconds_role": "weak_sanity_prior",
        "minimum_body_bars": cfg.minimum_body_bars,
        "soft_target_bars": cfg.soft_target_bars,
        "soft_latest_bars": cfg.soft_latest_bars,
        "canonical_musical_timeline": True,
        "complete_move_schema": True,
        "joint_track_timing_technique_decision": True,
        "track_world_model_projection": True,
        "layer_intelligence": {
            "version": LAYER_INTELLIGENCE_VERSION,
            "enabled": layer_cfg.enabled,
            "ownership_in_dj_state": True,
            "incoming_vocal_over_outgoing_instrumental": True,
            "full_master_fallback": True,
            "planning_triggers_separation": False,
            "rendered_headroom_authoritative": True,
        },
        "hard_audio_safety_authoritative": True,
        "complete_move_generator": complete_move,
        "structural_deck_operations": structural_operation_status(),
        "fallback": "legacy planner remains available when complete-move evidence is missing",
    }
