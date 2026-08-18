from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..candidate_renderer import (
    MeasuredSelection,
    MeasuredTransitionError,
    QualityGateConfig,
    evaluate_transition_candidates,
)
from ..decision import seal_execution_plan, validate_execution_plan
from ..fallback import build_last_resort_plan
from ..execution import TransitionRenderContext
from ..judge.professional import DJJudgement, judge_transition
from ..models import TrackAnalysis, TransitionPlan
from ..performance.program import attach_performance_program
from ..stems import find_cached_stems
from .candidate_generator import (
    CandidateGenerationConfig,
    diversify_transition_candidates,
    generate_transition_ideas,
)
from .memory import DEFAULT_REJECTION_PATH
from .set_state import SetState


@dataclass(frozen=True, slots=True)
class ProfessionalBrainConfig:
    enabled: bool = True
    analytical_pool: int = 48
    finalists: int = 8
    retry_rounds: int = 2
    diversity_weight: float = 0.34
    minimum_judge_score: float = 0.56
    measurement_sample_rate: int = 44_100
    measurement_context_seconds: float = 2.0
    measurement_minimum: float = 0.34
    adaptive_rehearsal: bool = True
    adaptive_pilot_finalists: int = 2
    adaptive_minimum_score: float = 0.66
    adaptive_minimum_naturalness: float = 0.58
    adaptive_maximum_uncertainty: float = 0.22
    rejection_path: str = DEFAULT_REJECTION_PATH

    @classmethod
    def from_environment(cls, **overrides: Any) -> "ProfessionalBrainConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                return default

        def _bool(name: str, default: bool) -> bool:
            value = os.environ.get(name)
            if value is None:
                return default
            return value.strip().lower() not in {"0", "false", "no", "off"}

        enabled = os.environ.get("DJAI_PROFESSIONAL_BRAIN", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        values = dict(
            enabled=enabled,
            analytical_pool=_int("DJAI_BRAIN_ANALYTICAL_POOL", 48),
            finalists=_int("DJAI_BRAIN_FINALISTS", 8),
            retry_rounds=_int("DJAI_BRAIN_RETRY_ROUNDS", 2),
            diversity_weight=_float("DJAI_BRAIN_DIVERSITY_WEIGHT", 0.34),
            minimum_judge_score=_float("DJAI_BRAIN_MINIMUM_SCORE", 0.56),
            measurement_sample_rate=_int("DJAI_MEASUREMENT_SAMPLE_RATE", 44_100),
            measurement_context_seconds=max(
                _float("DJAI_MEASUREMENT_PRE_ROLL", 2.0),
                _float("DJAI_MEASUREMENT_POST_ROLL", 2.0),
            ),
            measurement_minimum=_float("DJAI_BRAIN_REHEARSAL_MINIMUM", 0.34),
            adaptive_rehearsal=_bool("DJAI_ADAPTIVE_REHEARSAL", True),
            adaptive_pilot_finalists=_int("DJAI_ADAPTIVE_REHEARSAL_PILOT", 2),
            adaptive_minimum_score=_float("DJAI_ADAPTIVE_REHEARSAL_MIN_SCORE", 0.66),
            adaptive_minimum_naturalness=_float("DJAI_ADAPTIVE_REHEARSAL_MIN_NATURALNESS", 0.58),
            adaptive_maximum_uncertainty=_float("DJAI_ADAPTIVE_REHEARSAL_MAX_UNCERTAINTY", 0.22),
            rejection_path=os.environ.get("DJAI_REJECTION_MEMORY", DEFAULT_REJECTION_PATH),
        )
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values).normalized()

    def normalized(self) -> "ProfessionalBrainConfig":
        return ProfessionalBrainConfig(
            enabled=bool(self.enabled),
            analytical_pool=max(12, min(240, int(self.analytical_pool))),
            finalists=max(3, min(24, int(self.finalists))),
            retry_rounds=max(1, min(4, int(self.retry_rounds))),
            diversity_weight=max(0.0, min(0.75, float(self.diversity_weight))),
            minimum_judge_score=max(0.0, min(1.0, float(self.minimum_judge_score))),
            measurement_sample_rate=max(8_000, min(44_100, int(self.measurement_sample_rate))),
            measurement_context_seconds=max(0.5, min(8.0, float(self.measurement_context_seconds))),
            measurement_minimum=max(0.0, min(1.0, float(self.measurement_minimum))),
            adaptive_rehearsal=bool(self.adaptive_rehearsal),
            adaptive_pilot_finalists=max(1, min(8, int(self.adaptive_pilot_finalists))),
            adaptive_minimum_score=max(0.0, min(1.0, float(self.adaptive_minimum_score))),
            adaptive_minimum_naturalness=max(0.0, min(1.0, float(self.adaptive_minimum_naturalness))),
            adaptive_maximum_uncertainty=max(0.0, min(1.0, float(self.adaptive_maximum_uncertainty))),
            rejection_path=str(self.rejection_path),
        )


@dataclass(slots=True)
class BrainSelection:
    selected: TransitionPlan
    rehearsed: list[TransitionPlan]
    judgements: list[DJJudgement]
    attempts: int
    fallback_used: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "brain": "professional_dj_brain",
            "attempts": self.attempts,
            "rehearsed_candidates": len(self.rehearsed),
            "accepted_candidates": sum(item.accepted for item in self.judgements),
            "fallback_used": self.fallback_used,
            "selected_score": float(self.selected.metadata.get("professional_judge_score", self.selected.score)),
            "selected_uncertainty": float(self.selected.metadata.get("professional_judge_uncertainty", 1.0)),
            "candidate_trace": [
                {
                    "execution_signature": plan.execution_signature,
                    "style": plan.style,
                    "intent": plan.intent,
                    "technique": plan.technique,
                    "exit_time_a": plan.exit_time_a,
                    "entry_time_b": plan.entry_time_b,
                    "bars": plan.transition_bars,
                    "analytical_score": plan.analytical_score,
                    "measured_score": plan.measured_score,
                    "naturalness_score": plan.metadata.get("transition_naturalness_score"),
                    "judge": plan.metadata.get("professional_judge"),
                }
                for plan in self.rehearsed
            ],
        }


def professional_brain_enabled() -> bool:
    return ProfessionalBrainConfig.from_environment().enabled


def _attach_cached_stems(track_a: TrackAnalysis, track_b: TrackAnalysis, plans: Sequence[TransitionPlan]) -> None:
    stems_a = find_cached_stems(track_a.path)
    stems_b = find_cached_stems(track_b.path)
    for plan in plans:
        changed = False
        if not plan.stem_paths_a and stems_a:
            plan.stem_paths_a = dict(stems_a)
            changed = True
        if not plan.stem_paths_b and stems_b:
            plan.stem_paths_b = dict(stems_b)
            changed = True
        if changed:
            # 6.0 source ownership is encoded inside the executable IR, so a
            # source upgrade must recompile the program before the plan is
            # content-addressed and sealed.
            attach_performance_program(plan)
            seal_execution_plan(plan)


def _quality_config(cfg: ProfessionalBrainConfig, *, top_k: int, log_path: str | None) -> QualityGateConfig:
    return QualityGateConfig(
        enabled=True,
        top_k=max(1, top_k),
        sample_rate=cfg.measurement_sample_rate,
        context_seconds=cfg.measurement_context_seconds,
        analytical_weight=0.42,
        measured_weight=0.58,
        minimum_measured_score=cfg.measurement_minimum,
        log_path=log_path,
    ).normalized()


def _rehearse(
    plans: Sequence[TransitionPlan],
    *,
    cfg: ProfessionalBrainConfig,
    context: dict[str, Any],
    log_path: str | None,
    render_context: TransitionRenderContext | None = None,
) -> MeasuredSelection:
    scores = {
        plan.execution_signature: float(plan.metadata.get("selection_utility", plan.score))
        for plan in plans
    }
    quality = _quality_config(cfg, top_k=len(plans), log_path=log_path)
    try:
        return evaluate_transition_candidates(
            plans,
            config=quality,
            analytical_scores=scores,
            context={"professional_brain": True, **context},
            render_context=render_context,
        )
    except MeasuredTransitionError as exc:
        evaluated = exc.evaluated or ([exc.best] if exc.best is not None else [])
        if not evaluated:
            raise
        return MeasuredSelection(
            selected=copy.deepcopy(exc.best or evaluated[0]),
            evaluated=list(evaluated),
            log_path=exc.log_path,
            rejected_count=len(evaluated),
            metadata={"all_failed_measurement_gate": True},
        )


def _judge_all(
    plans: Sequence[TransitionPlan],
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    state: SetState | None,
    journey: str,
    progress: float,
    cfg: ProfessionalBrainConfig,
    listener: str | None,
) -> tuple[list[TransitionPlan], list[DJJudgement]]:
    judged: list[tuple[TransitionPlan, DJJudgement]] = []
    for plan in plans:
        judgement = judge_transition(
            plan,
            track_a=track_a,
            track_b=track_b,
            set_state=state,
            journey=journey,
            journey_progress=progress,
            rejection_path=cfg.rejection_path,
            listener=listener,
            minimum_score=cfg.minimum_judge_score,
        )
        judged.append((plan, judgement))
    judged.sort(
        key=lambda item: (
            item[1].accepted,
            item[1].feasible,
            item[1].overall - 0.08 * item[1].uncertainty,
            item[0].measured_score or 0.0,
            item[0].analytical_score or item[0].score,
            item[0].execution_signature,
        ),
        reverse=True,
    )
    return [item[0] for item in judged], [item[1] for item in judged]


def _selection_from_measured(
    measured: MeasuredSelection,
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    state: SetState | None,
    journey: str,
    progress: float,
    cfg: ProfessionalBrainConfig,
    listener: str | None,
    adaptive: dict[str, Any] | None = None,
) -> BrainSelection:
    ranked, judgements = _judge_all(
        measured.evaluated,
        track_a=track_a,
        track_b=track_b,
        state=state,
        journey=journey,
        progress=progress,
        cfg=cfg,
        listener=listener,
    )
    selected = ranked[0]
    selected.metadata = {
        **selected.metadata,
        "pre_professional_score": selected.score,
    }
    selected.score = round(float(selected.metadata.get("professional_judge_score", selected.score)), 4)
    selection = BrainSelection(
        selected=selected,
        rehearsed=ranked,
        judgements=judgements,
        attempts=1,
    )
    summary = selection.summary()
    if adaptive is not None:
        summary["adaptive_rehearsal"] = adaptive
    selected.metadata = {
        **selected.metadata,
        "professional_brain": summary,
    }
    return selection


def rehearse_candidate_pool(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    *,
    config: ProfessionalBrainConfig | None = None,
    set_state: SetState | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
    listener: str | None = None,
    measurement_log_path: str | None = None,
    context: dict[str, Any] | None = None,
    render_context: TransitionRenderContext | None = None,
) -> BrainSelection:
    """Rehearse actual audio with an adaptive bounded render budget.

    Normal automatic planning starts with the two strongest/diverse finalists.
    If one is already technically feasible, accepted, high-scoring and
    low-uncertainty, the remaining candidates are not rendered. Ambiguous or
    rejected edges automatically expand to the full configured finalist pool.
    Taste Lab always evaluates the requested pool so training evidence is not
    truncated by the production latency policy.
    """
    cfg = (config or ProfessionalBrainConfig.from_environment()).normalized()
    if not plans:
        raise ValueError("The professional brain requires at least one transition candidate.")
    resolved_context = dict(context or {})
    diverse = diversify_transition_candidates(
        list(plans),
        count=min(cfg.finalists, len(plans)),
        diversity_weight=cfg.diversity_weight,
    )
    _attach_cached_stems(track_a, track_b, diverse)

    pilot_count = min(cfg.adaptive_pilot_finalists, len(diverse))
    adaptive_allowed = (
        cfg.adaptive_rehearsal
        and resolved_context.get("mode") != "taste_lab"
        and pilot_count < len(diverse)
    )
    if adaptive_allowed:
        pilot = diverse[:pilot_count]
        # Do not write final measurement telemetry yet: an ambiguous pilot may
        # expand. Unified set planning logs the final accepted rehearsal set.
        pilot_measured = _rehearse(
            pilot,
            cfg=cfg,
            context={**resolved_context, "adaptive_rehearsal_stage": "pilot"},
            log_path=None,
            render_context=render_context,
        )
        pilot_selection = _selection_from_measured(
            pilot_measured,
            track_a=track_a,
            track_b=track_b,
            state=set_state,
            journey=journey,
            progress=journey_progress,
            cfg=cfg,
            listener=listener,
            adaptive={
                "enabled": True,
                "pilot_finalists": pilot_count,
                "available_finalists": len(diverse),
                "expanded": False,
            },
        )
        top = pilot_selection.judgements[0]
        if (
            top.accepted
            and top.feasible
            and top.overall >= max(cfg.minimum_judge_score, cfg.adaptive_minimum_score)
            and float(getattr(top, "naturalness", 1.0)) >= cfg.adaptive_minimum_naturalness
            and top.uncertainty <= cfg.adaptive_maximum_uncertainty
        ):
            # Direct callers may request a measurement log. Re-run the pilot
            # only for persistence; the rehearsal cache makes this a cache hit.
            if measurement_log_path:
                logged = _rehearse(
                    pilot,
                    cfg=cfg,
                    context={**resolved_context, "adaptive_rehearsal_stage": "accepted_pilot"},
                    log_path=measurement_log_path,
                    render_context=render_context,
                )
                pilot_selection = _selection_from_measured(
                    logged,
                    track_a=track_a,
                    track_b=track_b,
                    state=set_state,
                    journey=journey,
                    progress=journey_progress,
                    cfg=cfg,
                    listener=listener,
                    adaptive={
                        "enabled": True,
                        "pilot_finalists": pilot_count,
                        "available_finalists": len(diverse),
                        "expanded": False,
                        "early_accept": True,
                    },
                )
            else:
                pilot_selection.selected.metadata["professional_brain"]["adaptive_rehearsal"]["early_accept"] = True
            return pilot_selection

    measured = _rehearse(
        diverse,
        cfg=cfg,
        context={
            **resolved_context,
            **({"adaptive_rehearsal_stage": "expanded"} if adaptive_allowed else {}),
        },
        log_path=measurement_log_path,
        render_context=render_context,
    )
    return _selection_from_measured(
        measured,
        track_a=track_a,
        track_b=track_b,
        state=set_state,
        journey=journey,
        progress=journey_progress,
        cfg=cfg,
        listener=listener,
        adaptive=(
            {
                "enabled": True,
                "pilot_finalists": pilot_count,
                "available_finalists": len(diverse),
                "expanded": True,
                "early_accept": False,
            }
            if adaptive_allowed
            else {"enabled": False, "reason": "disabled_or_full_evidence_mode"}
        ),
    )


def plan_professional_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    config: ProfessionalBrainConfig | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    evidence_matrix=None,
    set_state: SetState | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
    listener: str | None = None,
    measurement_log_path: str | None = None,
    source_cursor: float = 0.0,
    preferred_exit_time: float | None = None,
    render_context: TransitionRenderContext | None = None,
) -> TransitionPlan:
    """Unified pair solver: route → duration search → rehearse → accept or controlled escape."""
    cfg = (config or ProfessionalBrainConfig.from_environment()).normalized()
    if evidence_matrix is None:
        from ..planning_context import build_pair_evidence_matrix

        evidence_matrix = build_pair_evidence_matrix(track_a, track_b)

    # Import lazily because unified_transition depends on the rehearsal helpers
    # in this module.  The runtime dependency is one-way after module loading.
    from .unified_transition import resolve_emergency_escape, resolve_transition

    resolution = resolve_transition(
        track_a,
        track_b,
        evidence_matrix=evidence_matrix,
        professional_config=cfg,
        tempo_policy=tempo_policy,
        preferred_style="auto",
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        set_state=set_state,
        journey=journey,
        journey_progress=journey_progress,
        listener=listener,
        measurement_log_path=measurement_log_path,
        context={"mode": "pair"},
        render_context=render_context,
    )
    if resolution.accepted:
        assert resolution.selected is not None
        validate_execution_plan(resolution.selected)
        return resolution.selected

    emergency = resolve_emergency_escape(
        track_a,
        track_b,
        evidence_matrix=evidence_matrix,
        professional_config=cfg,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        set_state=set_state,
        journey=journey,
        journey_progress=journey_progress,
        listener=listener,
        measurement_log_path=measurement_log_path,
        context={"mode": "pair_final_escape"},
        render_context=render_context,
    )
    if emergency.accepted:
        assert emergency.selected is not None
        validate_execution_plan(emergency.selected)
        return emergency.selected

    fallback = build_last_resort_plan(
        track_a,
        track_b,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        source_cursor=source_cursor,
        preferred_exit_time=preferred_exit_time,
        reason=(
            "Unified pair solver rejected every normal candidate and every separately rehearsed emergency escape; "
            "deterministic final escape used."
        ),
        render_context=(
            render_context
            if render_context is not None
            else TransitionRenderContext(sample_rate=cfg.measurement_sample_rate)
        ),
        verify_audio=True,
    )
    fallback.metadata = {
        **fallback.metadata,
        "unified_transition_resolution": resolution.to_dict(),
        "unified_emergency_resolution": emergency.to_dict(),
        "deterministic_final_escape": True,
    }
    return fallback
