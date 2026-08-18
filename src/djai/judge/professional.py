from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..brain.memory import DEFAULT_REJECTION_PATH, rejection_penalty
from ..brain.set_state import SetState, next_track_context_score
from ..intelligence.transition_ranker import RankerPrediction, predict_plan
from ..intelligence.expert_policy import ExpertPrior, expert_action_prior
from ..models import TrackAnalysis, TransitionMeasurement, TransitionPlan
from ..naturalness import annotate_transition_naturalness
from ..performance.mix_contract import evaluate_professional_mix_contract
from .critics import CriticEnsemble, evaluate_critic_ensemble


@dataclass(slots=True)
class DJJudgement:
    feasible: bool
    accepted: bool
    technical_quality: float
    rhythm_quality: float
    harmonic_quality: float
    vocal_quality: float
    bass_quality: float
    artistic_payoff: float
    set_context_value: float
    novelty: float
    naturalness: float
    professional_mix_quality: float
    expert_preference: float
    expert_authority: float
    expert_imitation: float
    expert_imitation_authority: float
    neural_rendered_preference: float
    neural_rendered_authority: float
    rejection_penalty: float
    uncertainty: float
    overall: float
    reasons: list[str] = field(default_factory=list)
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    known_rejection: dict[str, Any] | None = None
    critic_scores: dict[str, float] = field(default_factory=dict)
    failure_domains: list[str] = field(default_factory=list)
    remedies: list[str] = field(default_factory=list)
    set_intent: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _measurement(plan: TransitionPlan) -> TransitionMeasurement | None:
    return plan.measurement


def _state_novelty(plan: TransitionPlan, state: SetState | None) -> float:
    diversity = float(plan.metadata.get("candidate_diversity", 0.5))
    if state is None:
        return _clamp(diversity)
    technique_repetition = state.recent_techniques[-3:].count(plan.technique)
    intent_repetition = state.recent_intents[-3:].count(plan.intent)
    recent_lengths = list(state.dj_memory.get("recent_transition_lengths", []))[-4:]
    length_repetition = recent_lengths.count(int(plan.transition_bars))
    repetition_penalty = min(
        0.62,
        0.16 * technique_repetition + 0.10 * intent_repetition + 0.055 * length_repetition,
    )
    return _clamp(0.64 * diversity + 0.36 * (1.0 - repetition_penalty))


def _artistic_payoff(plan: TransitionPlan, measurement: TransitionMeasurement | None) -> float:
    energy_fit = measurement.energy_intent_fit if measurement else 0.5
    drop_payoff = measurement.drop_payoff if measurement else 0.5
    intent_bonus = plan.intent_score
    technique_bonus = plan.technique_score
    if plan.intent in {"energy_lift", "breakdown_to_drop"} or plan.technique == "double_drop":
        return _clamp(0.30 * intent_bonus + 0.22 * technique_bonus + 0.22 * energy_fit + 0.26 * drop_payoff)
    return _clamp(0.38 * intent_bonus + 0.28 * technique_bonus + 0.24 * energy_fit + 0.10 * drop_payoff)


def judge_transition(
    plan: TransitionPlan,
    *,
    track_a: TrackAnalysis | None = None,
    track_b: TrackAnalysis | None = None,
    set_state: SetState | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
    rejection_path: str | Path = DEFAULT_REJECTION_PATH,
    listener: str | None = None,
    minimum_score: float = 0.50,
) -> DJJudgement:
    measurement = _measurement(plan)
    mix_contract = evaluate_professional_mix_contract(plan)
    hard_failures = list(measurement.hard_failures if measurement else [])
    hard_failures.extend(mix_contract.hard_failures)
    critic_ensemble: CriticEnsemble = evaluate_critic_ensemble(
        plan,
        measurement=measurement,
        track_b=track_b,
        set_state=set_state,
        journey=journey,
        journey_progress=journey_progress,
    )
    feasible = not hard_failures
    technical = _clamp(critic_ensemble.scores["technical"])
    rhythm = _clamp(critic_ensemble.scores["rhythm"])
    harmonic = _clamp(critic_ensemble.scores["harmony"])
    vocal = _clamp(critic_ensemble.scores["vocals"])
    bass = _clamp(critic_ensemble.scores["bass"])
    musical = _clamp(measurement.musical_overall if measurement else 0.5)
    artistic = _artistic_payoff(plan, measurement)
    if set_state is not None and track_b is not None:
        context_score, context_parts = next_track_context_score(
            set_state,
            track_b,
            journey=journey,
            progress=journey_progress,
        )
        director_score = _clamp(critic_ensemble.scores["set_context"])
        context_score = _clamp(0.72 * context_score + 0.28 * director_score)
        context_parts = {**context_parts, "set_director": round(director_score, 4)}
    else:
        context_score, context_parts = 0.5, {}
    novelty = _state_novelty(plan, set_state)
    naturalness_assessment = annotate_transition_naturalness(
        plan, track_a=track_a, track_b=track_b
    )
    naturalness = _clamp(naturalness_assessment.score)

    penalty, rejected = rejection_penalty(plan, path=rejection_path, listener=listener)
    prediction: RankerPrediction = predict_plan(plan)
    expert = _clamp(prediction.learned_score)
    expert_authority = _clamp(prediction.authority)
    shadow_global_taste: dict[str, Any] = {"available": False, "authority": 0.0}
    try:
        from ..training.taste_learning import shadow_global_taste_prediction

        shadow_global_taste = shadow_global_taste_prediction(plan)
    except Exception as exc:
        shadow_global_taste = {
            "available": False,
            "authority": 0.0,
            "reason": f"global taste shadow unavailable: {exc}",
        }
    imitation: ExpertPrior = expert_action_prior(plan, journey=journey)
    neural_rendered_score=0.5; neural_rendered_authority=0.0; neural_rendered_meta: dict[str,Any]={"available":False,"reason":"track context unavailable"}
    if track_a is not None and track_b is not None:
        try:
            from ..intelligence.neural_policy import rendered_judge_prediction
            npred=rendered_judge_prediction(track_a,track_b,plan,set_state=set_state); neural_rendered_score=_clamp(npred.score); neural_rendered_authority=_clamp(npred.authority); neural_rendered_meta=npred.to_dict()
        except Exception as exc:
            neural_rendered_meta={"available":False,"reason":f"rendered judge unavailable: {exc}"}

    router = plan.metadata.get("transition_router", {})
    route_fit = _clamp(float(router.get("route_fit", 0.5))) if isinstance(router, dict) else 0.5
    bridge_preference = _clamp(float(router.get("bridge_preference", 0.0))) if isinstance(router, dict) else 0.0
    professional_mix_quality = _clamp(mix_contract.quality)
    deterministic = _clamp(
        0.16 * technical
        + 0.13 * musical
        + 0.10 * rhythm
        + 0.06 * harmonic
        + 0.06 * vocal
        + 0.07 * bass
        + 0.08 * artistic
        + 0.055 * context_score
        + 0.025 * novelty
        + 0.045 * route_fit
        + 0.10 * naturalness
        + 0.115 * professional_mix_quality
    )
    # Learned taste is a bounded residual. It cannot override feasibility and
    # earns authority only from held-out preference validation.
    learned_authority_sum=expert_authority+imitation.authority+neural_rendered_authority
    learned_mix=min(0.65,learned_authority_sum)
    if learned_mix>0.0:
        learned_target=(expert_authority*expert+imitation.authority*imitation.score+neural_rendered_authority*neural_rendered_score)/max(learned_authority_sum,1e-9)
        overall=(1.0-learned_mix)*deterministic+learned_mix*learned_target
    else:
        overall=deterministic

    # Until held-out expert taste earns authority, avoid letting a flashy
    # creative/emergency program beat a clean foundational blend by a tiny
    # difference in hand-written metrics. Creative moves can still win when
    # their measured/artistic advantage is clear. As learned authority grows,
    # this conservative prior fades away.
    style_prior = {"safe": 0.028, "creative": -0.030, "emergency": -0.120}.get(plan.style, 0.0)
    if plan.transition_mode == "emergency_cut":
        style_prior -= 0.045
    overall += (1.0 - learned_mix) * style_prior

    # Effects are execution details, not artistic goals.  Until validated
    # expert taste has authority, charge a small intervention cost so a more
    # complex technique must produce a materially better rendered outcome to
    # displace a clean foundational blend.
    from ..brain.technique_policy import intervention_cost

    technique_intervention = intervention_cost(plan.technique)
    intervention_penalty = (1.0 - learned_mix) * 0.08 * technique_intervention
    overall -= intervention_penalty

    # A bridge-preferred edge is not forbidden: a truly excellent rendered
    # reset can still win. But ordinary direct blends must clear a higher bar
    # when the Set Brain has evidence that an intermediate track is preferable.
    overall -= (1.0 - learned_mix) * 0.10 * bridge_preference * (1.0 - route_fit)
    overall -= 0.48 * penalty
    overall = _clamp(overall)

    metric_values = np.asarray(
        [technical, musical, rhythm, harmonic, vocal, bass, artistic, context_score, novelty],
        dtype=np.float64,
    )
    disagreement = float(np.std(metric_values))
    uncertainty = _clamp(
        0.50 * (1.0 - prediction.confidence)
        + 0.25 * min(1.0, disagreement / 0.28)
        + 0.15 * (1.0 - plan.cue_confidence)
        + 0.10 * (1.0 - naturalness)
    )
    known_rejection = rejected.to_dict() if rejected is not None else None

    reasons = [
        f"Technical rendered quality {technical:.0%}.",
        f"Musical execution quality {musical:.0%}; rhythmic quality {rhythm:.0%}.",
        f"Artistic intent/payoff {artistic:.0%}.",
        f"Transition naturalness {naturalness:.0%}: phrase/structure, duration fit, tempo strain, source cleanliness, and rendered continuity.",
        f"Professional mix contract {professional_mix_quality:.0%}: {mix_contract.family.replace('_', ' ')}, source mode {mix_contract.source_mode}, dual-deck overlap {mix_contract.dual_deck_overlap_fraction:.0%}.",
        f"Transition-router fit {route_fit:.0%}; bridge preference {bridge_preference:.0%}.",
    ]
    if set_state is not None:
        reasons.append(f"Set-context value {context_score:.0%} for the current {journey} journey.")
    if expert_authority > 0.0:
        reasons.append(f"Held-out preference model contributes {expert_authority:.0%} authority with score {expert:.0%}.")
    else:
        reasons.append("Expert preference model has no validated authority yet; deterministic judgment remains dominant.")
    if shadow_global_taste.get("available"):
        reasons.append(
            f"Global Taste Ranker shadow predicts {float(shadow_global_taste.get('score', 0.5)):.0%}; "
            "it is logged for comparison and has zero decision authority."
        )
    if imitation.authority > 0.0:
        reasons.append(f"Expert demonstration prior contributes {imitation.authority:.0%} authority with score {imitation.score:.0%}.")
    if neural_rendered_authority > 0.0:
        reasons.append(f"Validated rendered-audio neural judge contributes {neural_rendered_authority:.0%} authority with score {neural_rendered_score:.0%}.")
    elif neural_rendered_meta.get("available"):
        reasons.append("Rendered-audio neural judge is running in shadow mode; its prediction is recorded but cannot change acceptance.")
    if abs(style_prior) > 1e-9 and learned_mix < 0.95:
        reasons.append(
            f"Untrained-taste operational prior adjusts {plan.style} style by {((1.0 - learned_mix) * style_prior):+.1%}; this fades as validated expert authority grows."
        )
    if intervention_penalty > 1e-9:
        reasons.append(
            f"Technique intervention cost {technique_intervention:.0%} applies a {intervention_penalty:.1%} simplicity penalty; complex DSP must earn a clearly better rendered result."
        )
    if penalty > 0.0:
        reasons.append(f"Known-rejection similarity applies a {penalty:.0%} memory penalty.")
    if critic_ensemble.set_intent is not None:
        reasons.append(
            f"Set Director intent is {str(critic_ensemble.set_intent.get('name', 'unknown')).replace('_', ' ')}; "
            f"director fit {critic_ensemble.scores['set_context']:.0%}."
        )
    if critic_ensemble.failure_domains:
        reasons.append(
            "Critic ensemble requests targeted replanning for: "
            + ", ".join(critic_ensemble.failure_domains)
            + "."
        )

    warnings = list(measurement.warnings if measurement else [])
    warnings.extend(mix_contract.warnings)
    if uncertainty > 0.62:
        warnings.append("Judgment uncertainty is high; the brain should rehearse additional alternatives.")
    if penalty >= 0.70:
        warnings.append("This candidate is substantially similar to a transition previously rejected by a listener.")

    accepted = feasible and penalty < 0.92 and overall >= minimum_score
    judgement = DJJudgement(
        feasible=feasible,
        accepted=accepted,
        technical_quality=round(technical, 4),
        rhythm_quality=round(rhythm, 4),
        harmonic_quality=round(harmonic, 4),
        vocal_quality=round(vocal, 4),
        bass_quality=round(bass, 4),
        artistic_payoff=round(artistic, 4),
        set_context_value=round(context_score, 4),
        novelty=round(novelty, 4),
        naturalness=round(naturalness, 4),
        professional_mix_quality=round(professional_mix_quality, 4),
        expert_preference=round(expert, 4),
        expert_authority=round(expert_authority, 4),
        expert_imitation=round(imitation.score, 4),
        expert_imitation_authority=round(imitation.authority, 4),
        neural_rendered_preference=round(neural_rendered_score,4),
        neural_rendered_authority=round(neural_rendered_authority,4),
        rejection_penalty=round(penalty, 4),
        uncertainty=round(uncertainty, 4),
        overall=round(overall, 4),
        reasons=reasons,
        hard_failures=hard_failures,
        warnings=list(dict.fromkeys(warnings)),
        known_rejection=known_rejection,
        critic_scores=critic_ensemble.scores,
        failure_domains=list(critic_ensemble.failure_domains),
        remedies=list(critic_ensemble.remedies),
        set_intent=critic_ensemble.set_intent,
    )
    plan.metadata = {
        **plan.metadata,
        "professional_judge": judgement.to_dict(),
        "professional_judge_score": judgement.overall,
        "professional_judge_uncertainty": judgement.uncertainty,
        "professional_judge_naturalness": judgement.naturalness,
        "professional_mix_contract": mix_contract.to_dict(),
        "professional_mix_quality": judgement.professional_mix_quality,
        "set_context_score": judgement.set_context_value,
        "rejection_penalty": judgement.rejection_penalty,
        "set_context_components": context_parts,
        "neural_rendered_judge": neural_rendered_meta,
        "global_taste_ranker_shadow": shadow_global_taste,
        "critic_ensemble": critic_ensemble.to_dict(),
        "replanning_remedies": list(critic_ensemble.remedies),
    }
    return judgement
