from __future__ import annotations

"""SAUND.AI 1.8.0 AI Headphones.

The complete-move planner stays cheap while exploring the future. Immediately
before committing the current action, AI Headphones privately renders a small,
diverse set of complete moves through the same deterministic transition engine
used by the production quality gate. Existing critics, rendered-audio neural
judgement, rejection memory, and taste evidence then choose among only those
real rehearsals. Hard audio safety remains authoritative later in final set
compilation as well.
"""

import copy
import os
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from ..brain.move_generator import CompleteMoveCandidate
from ..brain.pair_adaptive_mixing import pair_adaptive_variants
from ..brain.set_state import SetState
from ..candidate_renderer import (
    MeasuredTransitionError,
    QualityGateConfig,
    measure_and_select_transition,
)
from ..judge.professional import DJJudgement, judge_transition
from ..models import TrackAnalysis, TransitionMeasurement, TransitionPlan
from ..version import MEASUREMENT_VERSION

AI_HEADPHONES_VERSION = "1.8.0"


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True)
class AIHeadphonesConfig:
    enabled: bool = True
    serious_pool: int = 12
    minimum_renders: int = 2
    maximum_renders: int = 4
    ambiguous_margin: float = 0.045
    moderate_margin: float = 0.090
    uncertainty_expand_threshold: float = 0.28
    rendered_margin_expand: float = 0.035
    rendered_weight: float = 0.72
    structural_weight: float = 0.28
    sample_rate: int = 22_050
    context_seconds: float = 2.0
    minimum_measured_score: float = 0.34
    minimum_judge_score: float = 0.56
    measurement_log: str = "ratings/headphones-measurements.jsonl"

    @classmethod
    def from_environment(cls) -> "AIHeadphonesConfig":
        rendered = max(0.0, _float_env("DJAI_HEADPHONES_RENDERED_WEIGHT", 0.72))
        structural = max(0.0, _float_env("DJAI_HEADPHONES_STRUCTURAL_WEIGHT", 0.28))
        total = rendered + structural or 1.0
        return cls(
            enabled=_bool_env("DJAI_HEADPHONES", True),
            serious_pool=max(4, min(32, _int_env("DJAI_HEADPHONES_SERIOUS_POOL", 12))),
            minimum_renders=max(1, min(4, _int_env("DJAI_HEADPHONES_MIN_RENDERS", 2))),
            maximum_renders=max(2, min(6, _int_env("DJAI_HEADPHONES_MAX_RENDERS", 4))),
            ambiguous_margin=max(0.0, min(0.25, _float_env("DJAI_HEADPHONES_AMBIGUOUS_MARGIN", 0.045))),
            moderate_margin=max(0.0, min(0.35, _float_env("DJAI_HEADPHONES_MODERATE_MARGIN", 0.090))),
            uncertainty_expand_threshold=_clamp(_float_env("DJAI_HEADPHONES_UNCERTAINTY_EXPAND", 0.28)),
            rendered_margin_expand=max(0.0, min(0.20, _float_env("DJAI_HEADPHONES_RENDERED_MARGIN_EXPAND", 0.035))),
            rendered_weight=rendered / total,
            structural_weight=structural / total,
            sample_rate=max(8_000, min(44_100, _int_env("DJAI_HEADPHONES_SAMPLE_RATE", 22_050))),
            context_seconds=max(0.5, min(6.0, _float_env("DJAI_HEADPHONES_CONTEXT_SECONDS", 2.0))),
            minimum_measured_score=_clamp(_float_env("DJAI_HEADPHONES_MEASUREMENT_MINIMUM", 0.34)),
            minimum_judge_score=_clamp(_float_env("DJAI_HEADPHONES_JUDGE_MINIMUM", 0.56)),
            measurement_log=os.environ.get(
                "DJAI_HEADPHONES_MEASUREMENT_LOG",
                "ratings/headphones-measurements.jsonl",
            ),
        ).normalized()

    def normalized(self) -> "AIHeadphonesConfig":
        minimum = max(1, min(4, int(self.minimum_renders)))
        maximum = max(minimum, min(6, int(self.maximum_renders)))
        moderate = max(float(self.ambiguous_margin), float(self.moderate_margin))
        rendered = max(0.0, float(self.rendered_weight))
        structural = max(0.0, float(self.structural_weight))
        total = rendered + structural or 1.0
        return AIHeadphonesConfig(
            enabled=bool(self.enabled),
            serious_pool=max(maximum, min(32, int(self.serious_pool))),
            minimum_renders=minimum,
            maximum_renders=maximum,
            ambiguous_margin=max(0.0, min(0.25, float(self.ambiguous_margin))),
            moderate_margin=max(0.0, min(0.35, moderate)),
            uncertainty_expand_threshold=_clamp(self.uncertainty_expand_threshold),
            rendered_margin_expand=max(0.0, min(0.20, float(self.rendered_margin_expand))),
            rendered_weight=rendered / total,
            structural_weight=structural / total,
            sample_rate=max(8_000, min(44_100, int(self.sample_rate))),
            context_seconds=max(0.5, min(6.0, float(self.context_seconds))),
            minimum_measured_score=_clamp(self.minimum_measured_score),
            minimum_judge_score=_clamp(self.minimum_judge_score),
            measurement_log=str(self.measurement_log),
        )


@dataclass(frozen=True, slots=True)
class AIHeadphonesEvaluation:
    candidate: CompleteMoveCandidate = field(compare=False, repr=False)
    plan: TransitionPlan = field(compare=False, repr=False)
    judgement: DJJudgement = field(compare=False, repr=False)
    headphone_score: float = 0.0
    render_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        measurement = self.plan.measurement
        taste = self.plan.metadata.get("global_taste_ranker_shadow", {})
        state_move_taste = self.plan.metadata.get("state_move_taste_shadow", {})
        crate_memory = self.plan.metadata.get("crate_memory", {})
        neural = self.plan.metadata.get("neural_rendered_judge", {})
        return {
            "next_index": self.candidate.next_index,
            "execution_signature": self.plan.execution_signature,
            "structural_score": round(float(self.candidate.selection_score), 6),
            "rendered_measurement": round(float(measurement.overall), 6) if measurement is not None else None,
            "professional_judge": round(float(self.judgement.overall), 6),
            "naturalness": round(float(self.judgement.naturalness), 6),
            "uncertainty": round(float(self.judgement.uncertainty), 6),
            "feasible": bool(self.judgement.feasible),
            "measurement_passed": bool(self.plan.measurement_passed),
            "judge_accepted": bool(self.judgement.accepted),
            "accepted": bool(self.judgement.accepted and self.judgement.feasible and self.plan.measurement_passed),
            "critic_scores": dict(self.judgement.critic_scores),
            "failure_domains": list(self.judgement.failure_domains),
            "rendered_judge": neural,
            "taste_shadow": taste,
            "state_move_taste_shadow": state_move_taste,
            "crate_memory": crate_memory,
            "headphone_score": round(float(self.headphone_score), 6),
            "rehearsal_cache_hit": bool(self.plan.metadata.get("rehearsal_cache_hit", False)),
            "render_error": self.render_error,
        }


@dataclass(frozen=True, slots=True)
class AIHeadphonesDecision:
    winner: CompleteMoveCandidate
    preferred_before_listening: CompleteMoveCandidate = field(compare=False, repr=False)
    evaluations: tuple[AIHeadphonesEvaluation, ...] = field(default_factory=tuple, compare=False, repr=False)
    serious_candidates: int = 0
    render_budget: int = 0
    authoritative: bool = False
    reason: str = ""

    @property
    def winner_score(self) -> float:
        for evaluation in self.evaluations:
            if evaluation.plan.execution_signature == self.winner.plan.execution_signature:
                return float(evaluation.headphone_score)
        return float(self.winner.selection_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": AI_HEADPHONES_VERSION,
            "authoritative": bool(self.authoritative),
            "reason": self.reason,
            "serious_candidates": int(self.serious_candidates),
            "render_budget": int(self.render_budget),
            "rendered_candidates": len(self.evaluations),
            "preferred_before_listening": {
                "next_index": self.preferred_before_listening.next_index,
                "execution_signature": self.preferred_before_listening.plan.execution_signature,
                "selection_score": round(float(self.preferred_before_listening.selection_score), 6),
            },
            "winner": {
                "next_index": self.winner.next_index,
                "execution_signature": self.winner.plan.execution_signature,
                "headphone_score": round(float(self.winner_score), 6),
                "changed_after_listening": self.winner.plan.execution_signature
                != self.preferred_before_listening.plan.execution_signature,
            },
            "evaluations": [item.to_dict() for item in self.evaluations],
            "hard_audio_safety_authoritative": True,
            "learned_models_may_not_override_hard_failures": True,
        }


def _serious_pool(
    candidates: Sequence[CompleteMoveCandidate],
    *,
    preferred: CompleteMoveCandidate,
    limit: int,
) -> list[CompleteMoveCandidate]:
    """Cheap-filter candidates while retaining next-track and technique diversity."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.selection_score,
            item.move.future_optionality,
            item.move.score,
            -item.next_index,
        ),
        reverse=True,
    )
    chosen: list[CompleteMoveCandidate] = []
    seen_signatures: set[str] = set()

    def add(item: CompleteMoveCandidate) -> None:
        signature = item.plan.execution_signature
        if signature in seen_signatures or len(chosen) >= limit:
            return
        chosen.append(item)
        seen_signatures.add(signature)

    add(preferred)
    seen_tracks = {preferred.next_index}
    for item in ranked:
        if item.next_index not in seen_tracks:
            add(item)
            seen_tracks.add(item.next_index)
        if len(chosen) >= limit:
            break
    seen_techniques = {item.move.technique for item in chosen}
    for item in ranked:
        if item.move.technique not in seen_techniques:
            add(item)
            seen_techniques.add(item.move.technique)
        if len(chosen) >= limit:
            break
    for item in ranked:
        add(item)
        if len(chosen) >= limit:
            break
    return chosen


def _render_budget(pool: Sequence[CompleteMoveCandidate], cfg: AIHeadphonesConfig) -> int:
    if not pool:
        return 0
    if len(pool) == 1:
        return 1
    ranked = sorted(pool, key=lambda item: item.selection_score, reverse=True)
    margin = max(0.0, float(ranked[0].selection_score - ranked[1].selection_score))
    if margin <= cfg.ambiguous_margin:
        target = cfg.maximum_renders
    elif margin <= cfg.moderate_margin:
        target = min(cfg.maximum_renders, cfg.minimum_renders + 1)
    else:
        target = cfg.minimum_renders
    return min(len(pool), max(1, target))


def _render_finalists(
    pool: Sequence[CompleteMoveCandidate],
    *,
    preferred: CompleteMoveCandidate,
    budget: int,
) -> list[CompleteMoveCandidate]:
    if budget <= 0:
        return []
    ranked = sorted(
        pool,
        key=lambda item: (
            item.selection_score,
            item.move.future_optionality,
            item.move.score,
            -item.next_index,
        ),
        reverse=True,
    )
    chosen: list[CompleteMoveCandidate] = []
    seen: set[str] = set()

    def add(item: CompleteMoveCandidate) -> None:
        signature = item.plan.execution_signature
        if signature in seen or len(chosen) >= budget:
            return
        chosen.append(item)
        seen.add(signature)

    add(preferred)
    # First pass prefers different tracks, then a different technique, then raw rank.
    used_tracks = {preferred.next_index}
    for item in ranked:
        if item.next_index not in used_tracks:
            add(item)
            used_tracks.add(item.next_index)
        if len(chosen) >= budget:
            return chosen
    used_techniques = {item.move.technique for item in chosen}
    for item in ranked:
        if item.move.technique not in used_techniques:
            add(item)
            used_techniques.add(item.move.technique)
        if len(chosen) >= budget:
            return chosen
    for item in ranked:
        add(item)
        if len(chosen) >= budget:
            break
    return chosen


def _failed_plan(candidate: CompleteMoveCandidate, exc: Exception) -> tuple[TransitionPlan, str]:
    if isinstance(exc, MeasuredTransitionError) and exc.best is not None:
        return exc.best, f"{type(exc).__name__}: {exc}"
    plan = copy.deepcopy(candidate.plan)
    plan.measurement = TransitionMeasurement(
        version=MEASUREMENT_VERSION,
        sample_rate=0,
        duration_seconds=0.0,
        overall=0.0,
        hard_failures=[f"AI Headphones render failed: {type(exc).__name__}: {exc}"],
    )
    plan.quality_checked = True
    plan.measurement_passed = False
    plan.measured_score = 0.0
    return plan, f"{type(exc).__name__}: {exc}"


def listen_to_complete_moves(
    *,
    preferred: CompleteMoveCandidate,
    candidates: Sequence[CompleteMoveCandidate],
    analyses: Sequence[TrackAnalysis],
    set_state: SetState,
    journey: str,
    progress: float,
    listener: str | None = None,
    config: AIHeadphonesConfig | None = None,
) -> AIHeadphonesDecision:
    """Privately render/judge the best current complete moves before commit.

    Future branches are never rendered here. Receding-horizon search already
    narrowed the current decision; Headphones spends real audio only at the
    commit boundary and returns one candidate. If every rehearsal is rejected
    or rendering is unavailable, the cheap planner winner is retained and the
    later final compiler/rehearsal loop remains the recovery authority.
    """
    cfg = (config or AIHeadphonesConfig.from_environment()).normalized()
    if not cfg.enabled or not candidates:
        return AIHeadphonesDecision(
            winner=preferred,
            preferred_before_listening=preferred,
            serious_candidates=len(candidates),
            render_budget=0,
            authoritative=False,
            reason="disabled" if not cfg.enabled else "no_candidates",
        )

    pool = _serious_pool(candidates, preferred=preferred, limit=cfg.serious_pool)
    initial_budget = _render_budget(pool, cfg)
    finalists = _render_finalists(pool, preferred=preferred, budget=min(len(pool), cfg.maximum_renders))
    evaluations: list[AIHeadphonesEvaluation] = []
    gate = QualityGateConfig(
        enabled=True,
        top_k=1,
        sample_rate=cfg.sample_rate,
        context_seconds=cfg.context_seconds,
        analytical_weight=0.35,
        measured_weight=0.65,
        minimum_measured_score=cfg.minimum_measured_score,
        log_path=cfg.measurement_log,
    ).normalized()

    target_budget = max(1, initial_budget)
    for candidate in finalists:
        track_a = analyses[candidate.current_index]
        track_b = analyses[candidate.next_index]
        render_error: str | None = None
        try:
            adaptive_plans = pair_adaptive_variants(
                track_a,
                track_b,
                candidate.plan,
                maximum_variants=3,
            )
            adaptive_gate = replace(gate, top_k=max(1, len(adaptive_plans)))
            measured = measure_and_select_transition(
                track_a,
                track_b,
                adaptive_plans,
                config=adaptive_gate,
                context={
                    "mode": "ai_headphones",
                    "capture_render_embeddings": True,
                    "current_index": candidate.current_index,
                    "next_index": candidate.next_index,
                    "journey": journey,
                    "journey_progress": round(float(progress), 6),
                    "pair_adaptive_mixing": True,
                    "pair_adaptive_variant_count": len(adaptive_plans),
                },
            )
            plan = measured.selected
            plan.metadata = {
                **plan.metadata,
                "pair_adaptive_rehearsal": {
                    "variant_count": len(adaptive_plans),
                    "rendered_variant_count": len(getattr(measured, "evaluated", adaptive_plans)),
                    "selected_variant": str(
                        dict(plan.metadata.get("pair_adaptive_mix", {}) or {}).get(
                            "variant",
                            "legacy",
                        )
                    ),
                    "selected_by_real_audio_quality_gate": True,
                },
            }
        except Exception as exc:  # render/measure failure is evidence, not a crash
            plan, render_error = _failed_plan(candidate, exc)
        judgement = judge_transition(
            plan,
            track_a=track_a,
            track_b=track_b,
            set_state=set_state,
            journey=journey,
            journey_progress=progress,
            listener=listener,
            minimum_score=cfg.minimum_judge_score,
        )
        structural_value = _clamp(
            0.82 * candidate.selection_score + 0.18 * candidate.move.future_optionality
        )
        score = _clamp(
            cfg.rendered_weight * judgement.overall
            + cfg.structural_weight * structural_value
        )
        effective_accept = bool(judgement.accepted and judgement.feasible and plan.measurement_passed)
        if not judgement.feasible or not plan.measurement_passed:
            score = 0.0
        elif not effective_accept:
            score = min(score, cfg.minimum_judge_score - 0.001)
        plan.metadata = {
            **plan.metadata,
            "ai_headphones": {
                "version": AI_HEADPHONES_VERSION,
                "structural_score": round(float(candidate.selection_score), 6),
                "future_optionality": round(float(candidate.move.future_optionality), 6),
                "rendered_judge_score": round(float(judgement.overall), 6),
                "headphone_score": round(float(score), 6),
                "accepted": effective_accept,
                "judge_accepted": bool(judgement.accepted),
                "measurement_passed": bool(plan.measurement_passed),
                "feasible": bool(judgement.feasible),
                "render_error": render_error,
                "hard_audio_safety_authoritative": True,
                "pair_adaptive_mixing": bool(plan.metadata.get("pair_adaptive_mix")),
                "pair_adaptive_selected_variant": str(
                    dict(plan.metadata.get("pair_adaptive_mix", {}) or {}).get("variant", "legacy")
                ),
            },
        }
        evaluations.append(
            AIHeadphonesEvaluation(
                candidate=candidate,
                plan=plan,
                judgement=judgement,
                headphone_score=round(score, 6),
                render_error=render_error,
            )
        )
        if len(evaluations) < target_budget:
            continue
        if target_budget >= min(len(finalists), cfg.maximum_renders):
            break
        accepted_now = sorted(
            (
                item
                for item in evaluations
                if item.judgement.accepted and item.judgement.feasible and item.plan.measurement_passed
            ),
            key=lambda item: (item.headphone_score, item.judgement.overall),
            reverse=True,
        )
        should_expand = not accepted_now
        if accepted_now and accepted_now[0].judgement.uncertainty >= cfg.uncertainty_expand_threshold:
            should_expand = True
        if len(accepted_now) >= 2 and (
            accepted_now[0].headphone_score - accepted_now[1].headphone_score
        ) <= cfg.rendered_margin_expand:
            should_expand = True
        if should_expand:
            target_budget = min(target_budget + 1, len(finalists), cfg.maximum_renders)
        else:
            break

    accepted = [
        item
        for item in evaluations
        if item.judgement.accepted and item.judgement.feasible and item.plan.measurement_passed
    ]
    if not accepted:
        return AIHeadphonesDecision(
            winner=preferred,
            preferred_before_listening=preferred,
            evaluations=tuple(evaluations),
            serious_candidates=len(pool),
            render_budget=len(evaluations),
            authoritative=False,
            reason="no_rendered_candidate_passed; final compiler retains recovery authority",
        )

    best = max(
        accepted,
        key=lambda item: (
            item.headphone_score,
            item.judgement.overall,
            -item.judgement.uncertainty,
            item.plan.measured_score,
            item.candidate.selection_score,
            -item.candidate.next_index,
        ),
    )
    winner = replace(best.candidate, plan=best.plan)
    return AIHeadphonesDecision(
        winner=winner,
        preferred_before_listening=preferred,
        evaluations=tuple(evaluations),
        serious_candidates=len(pool),
        render_budget=len(evaluations),
        authoritative=True,
        reason="rendered finalists judged; best accepted move committed",
    )


def ai_headphones_status(*, planner_mode: str | None = None) -> dict[str, Any]:
    cfg = AIHeadphonesConfig.from_environment()
    mode = planner_mode or os.environ.get("DJAI_PLANNER_MODE", "move_shadow").strip().lower()
    return {
        "version": AI_HEADPHONES_VERSION,
        "enabled": cfg.enabled,
        "active_for_current_mode": bool(cfg.enabled and mode == "move_authoritative"),
        "planner_mode": mode,
        "commit_boundary_only": True,
        "lookahead_rendering": False,
        "serious_pool": cfg.serious_pool,
        "render_budget": {"minimum": cfg.minimum_renders, "maximum": cfg.maximum_renders},
        "uncertainty_aware_budget": True,
        "uncertainty_expand_threshold": cfg.uncertainty_expand_threshold,
        "rendered_margin_expand": cfg.rendered_margin_expand,
        "sample_rate": cfg.sample_rate,
        "context_seconds": cfg.context_seconds,
        "uses_real_audio_renderer": True,
        "uses_multi_critic_judge": True,
        "uses_rendered_dj_judge": True,
        "uses_taste_evidence": True,
        "uses_state_move_taste_shadow": True,
        "uses_crate_memory_prior": True,
        "rehearsal_cache": True,
        "hard_audio_safety_authoritative": True,
        "fallback": "cheap complete-move winner remains provisional when every private rehearsal is rejected",
    }
