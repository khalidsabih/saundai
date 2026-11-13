from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Literal

from ..execution import TransitionRenderContext
from ..models import TrackAnalysis, TransitionPlan
from ..measurement_store import append_measurement_records, measurement_record
from ..planning_context import PairEvidenceMatrix
from ..planner import DifficultTransitionError, rank_transition_candidates
from .candidate_generator import diversify_transition_candidates
from .crate_topk import crate_aware_topk_enabled
from .crowd_state import CrowdDirective
from .professional import BrainSelection, ProfessionalBrainConfig, rehearse_candidate_pool
from .proposal import attach_transition_proposal
from .set_state import SetState
from .dj_move import DJState, rank_transition_plans_as_moves
from .technique_policy import duration_candidates, intervention_cost, technique_policy
from .transition_policy import TransitionPolicyContext, apply_bounded_transition_policy
from .transition_router import TransitionRoute, route_family_sequence, route_transition

ResolutionAction = Literal["accept", "replan_edge", "replan_track", "emergency_escape"]


@dataclass(frozen=True, slots=True)
class UnifiedTransitionConfig:
    finalists: int = 10
    candidates_per_duration: int = 4
    retry_rounds: int = 2
    minimum_judge_score: float = 0.56
    maximum_duration_candidates: int = 6
    bridge_exception_score: float = 0.78
    bridge_exception_route_fit: float = 0.68
    high_uncertainty: float = 0.64

    @classmethod
    def from_professional(cls, config: ProfessionalBrainConfig) -> "UnifiedTransitionConfig":
        cfg = config.normalized()
        try:
            unified_floor = float(os.environ.get("DJAI_UNIFIED_MINIMUM_SCORE", "0.56"))
        except ValueError:
            unified_floor = 0.56
        unified_floor = max(0.0, min(1.0, unified_floor))
        return cls(
            finalists=max(4, min(20, cfg.finalists)),
            candidates_per_duration=max(2, min(8, max(2, cfg.finalists // 2))),
            retry_rounds=max(1, min(4, cfg.retry_rounds)),
            # The unified floor is intentionally separate from the legacy/general
            # judge threshold so an upgraded 4.2 .env with 0.50 cannot silently
            # restore the old audience-facing acceptance behavior.
            minimum_judge_score=max(cfg.minimum_judge_score, unified_floor),
            maximum_duration_candidates=6,
        )


@dataclass(slots=True)
class TransitionResolution:
    action: ResolutionAction
    route: TransitionRoute
    selected: TransitionPlan | None
    rehearsed: list[TransitionPlan] = field(default_factory=list)
    judgements: list[dict[str, Any]] = field(default_factory=list)
    attempted_durations: list[int] = field(default_factory=list)
    attempts: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.action == "accept" and self.selected is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "accepted": self.accepted,
            "route": self.route.to_dict(),
            "selected_execution_signature": self.selected.execution_signature if self.selected else None,
            "attempted_durations": list(self.attempted_durations),
            "attempts": self.attempts,
            "rehearsed_candidates": len(self.rehearsed),
            "judgements": list(self.judgements),
            "reasons": list(self.reasons),
            "contract": "neural_rehearsal_intelligence_v2",
        }


def _styles_for_route(route: TransitionRoute, preferred_style: str) -> tuple[str, ...]:
    if preferred_style in {"safe", "creative", "emergency"}:
        return (preferred_style,)
    if route.family == "direct_blend":
        return ("safe", "creative")
    if route.family == "layer_handoff":
        return ("safe", "creative")
    # Force=True allows intentional cut programs to appear in creative search;
    # they do not need to masquerade as emergency fallbacks.
    return ("creative", "safe")


def _annotate_duration_search(
    plan: TransitionPlan,
    *,
    durations: tuple[int, ...],
    requested: int,
    neural_policy: dict[str, Any] | None = None,
    bootstrap_policy: dict[str, Any] | None = None,
) -> None:
    plan.metadata = {
        **plan.metadata,
        "duration_search": {
            "candidate_bars": list(durations),
            "requested_exact_bars": int(requested),
            "selected_bars": int(plan.transition_bars),
            "source": "policy_plus_phrase_structure_plus_learned_priors",
            "bootstrap_policy": dict(bootstrap_policy or {}),
            "neural_policy": dict(neural_policy or {}),
        },
    }


def _candidate_pool(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    route: TransitionRoute,
    evidence_matrix: PairEvidenceMatrix,
    config: UnifiedTransitionConfig,
    tempo_policy: str,
    preferred_style: str,
    effective_bpm_a: float | None,
    render_rate_a: float,
    stems_available: bool,
    retry_round: int,
    finalist_budget: int,
    set_state: SetState | None = None,
    dj_state: DJState | None = None,
    retry_remedies: tuple[str, ...] = (),
    previous_failed_bars: int | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
    transition_policy_context: TransitionPolicyContext | None = None,
    crowd_directive: CrowdDirective | None = None,
) -> tuple[list[TransitionPlan], tuple[int, ...]]:
    durations = duration_candidates(
        route,
        evidence_matrix,
        maximum_candidates=min(
            config.maximum_duration_candidates,
            3 + retry_round * 2,
        ),
    )
    duration_bootstrap: dict[str, Any] = {}
    try:
        from ..intelligence.bootstrap_runtime import rank_duration_values as rank_bootstrap_duration_values

        durations, duration_bootstrap = rank_bootstrap_duration_values(track_a, track_b, durations)
    except Exception as exc:
        duration_bootstrap = {"available": False, "reason": f"bootstrap duration prior unavailable: {exc}"}

    duration_neural: dict[str, Any] = {}
    try:
        from ..intelligence.neural_policy import rank_duration_values
        durations,duration_neural=rank_duration_values(track_a,track_b,route,durations,set_state=set_state)
    except Exception as exc:
        duration_neural={"available":False,"reason":f"duration policy unavailable: {exc}"}
    remedy_set = set(retry_remedies)
    if previous_failed_bars is not None and "shorten_overlap" in remedy_set:
        shorter = tuple(value for value in durations if value < previous_failed_bars)
        if shorter:
            durations = shorter + tuple(value for value in durations if value not in shorter)
    if previous_failed_bars is not None and "change_transition_length" in remedy_set:
        changed = tuple(value for value in durations if value != previous_failed_bars)
        if changed:
            durations = changed + ((previous_failed_bars,) if previous_failed_bars in durations else ())

    styles = _styles_for_route(route, preferred_style)
    plans: list[TransitionPlan] = []
    per_style = min(6, max(1, min(config.candidates_per_duration, finalist_budget)))
    for bars in durations:
        try:
            generated = rank_transition_candidates(
                track_a,
                track_b,
                styles=styles,
                force=True,
                requested_bars=bars,
                exact_bars=True,
                tempo_policy=tempo_policy,
                effective_bpm_a=effective_bpm_a,
                render_rate_a=render_rate_a,
                stems_available=stems_available,
                evidence_matrix=evidence_matrix,
                limit_per_style=per_style,
                transition_route=route,
            )
        except (DifficultTransitionError, RuntimeError, ValueError):
            continue
        for plan in generated:
            _annotate_duration_search(
                plan,
                durations=durations,
                requested=bars,
                neural_policy=duration_neural,
                bootstrap_policy=duration_bootstrap,
            )
            set_intent_name = None
            if set_state is not None:
                try:
                    from .director import direct_set_intent
                    set_intent_name = direct_set_intent(
                        journey,
                        journey_progress,
                        set_state,
                    ).name
                except (ValueError, TypeError):
                    set_intent_name = None
            attach_transition_proposal(plan, set_intent=set_intent_name)
            plans.append(plan)

    unique: dict[str, TransitionPlan] = {}
    for plan in plans:
        unique[plan.execution_signature] = plan
    move_ranked = list(unique.values())
    if dj_state is not None:
        move_ranked = rank_transition_plans_as_moves(
            track_a, track_b, move_ranked, state=dj_state
        )
    if dj_state is not None:
        move_ranked = [
            plan
            for plan in move_ranked
            if not bool((plan.metadata.get("dj_move_brain", {}) or {}).get("song_form_hard_guard", False))
        ]
        if not move_ranked:
            return [], durations
    if transition_policy_context is not None:
        apply_bounded_transition_policy(
            move_ranked,
            context=transition_policy_context,
            crowd_directive=crowd_directive,
        )
    ranked = sorted(
        move_ranked,
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            -intervention_cost(plan.technique),
            plan.score,
            plan.execution_signature,
        ),
        reverse=True,
    )
    if not ranked:
        return [], durations
    # Give the authorized, previously trained taste ranker first choice of the
    # rehearsal pilot. Without validated model authority, its existing helper
    # returns the deterministic ordering unchanged. Audio gates remain final.
    score_key = "selection_utility"
    if crate_aware_topk_enabled():
        from ..intelligence.transition_ranker import rank_transition_plans

        ranked = rank_transition_plans(ranked)
        # The taste model must carry verified authority. Otherwise the entire
        # rehearsal shortlist stays on the established deterministic ranking.
        if any(
            float(plan.metadata.get("transition_ranker_authority", 0.0)) > 0.0
            for plan in ranked
        ):
            score_key = "transition_ranker_blended_score"
    count = min(len(ranked), max(1, int(finalist_budget)))
    selected = diversify_transition_candidates(
        ranked, count=count, diversity_weight=0.30, score_key=score_key,
    )
    # This is selection telemetry, not an assertion of predictive accuracy:
    # unrendered candidates have no observed acceptance labels.
    ranked_by_score = sorted(
        ranked,
        key=lambda plan: (
            float(plan.metadata.get(score_key, plan.metadata.get("selection_utility", plan.score))),
            plan.execution_signature,
        ),
        reverse=True,
    )
    score_ranks = {
        plan.execution_signature: index + 1
        for index, plan in enumerate(ranked_by_score)
    }
    for plan in selected:
        plan.metadata = {
            **plan.metadata,
            "topk_selection": {
                "ranking_source": "validated_model" if score_key != "selection_utility" else "deterministic",
                "rank_in_analytical_pool": score_ranks[plan.execution_signature],
                "analytical_pool_size": len(ranked),
                "family_finalist_budget": count,
            },
        }
    return selected, durations



def _judge_records(selection: BrainSelection) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for plan, judgement in zip(selection.rehearsed, selection.judgements, strict=False):
        records.append(
            {
                "execution_signature": plan.execution_signature,
                "technique": plan.technique,
                "intent": plan.intent,
                "bars": plan.transition_bars,
                "accepted": judgement.accepted,
                "feasible": judgement.feasible,
                "overall": judgement.overall,
                "uncertainty": judgement.uncertainty,
                "intervention_cost": intervention_cost(plan.technique),
                "route_fit": float(dict(plan.metadata.get("transition_router", {})).get("route_fit", 0.5)),
                "transition_family": str(dict(plan.metadata.get("transition_router", {})).get("family", "unknown")),
                "critic_scores": dict(judgement.critic_scores),
                "failure_domains": list(judgement.failure_domains),
                "remedies": list(judgement.remedies),
            }
        )
    return records


def _log_resolution_measurements(
    log_path: str | None,
    plans: list[TransitionPlan],
    *,
    accepted_signature: str | None,
    context: dict[str, Any],
) -> None:
    """Persist rehearsal telemetry with *brain* acceptance semantics.

    CandidateRenderer's local measurement winner is not necessarily the DJ
    brain's final winner.  A record is accepted only when the whole unified
    transition solver commits that exact execution signature.
    """
    if not log_path or not plans:
        return
    unique: dict[str, TransitionPlan] = {}
    for plan in plans:
        unique[plan.execution_signature] = plan
    ranked = sorted(
        unique.values(),
        key=lambda plan: (
            float(plan.metadata.get("professional_judge_score", -1.0)),
            float(plan.measured_score or -1.0),
            float(plan.analytical_score or plan.score),
            plan.execution_signature,
        ),
        reverse=True,
    )
    records = [
        measurement_record(
            plan,
            accepted=accepted_signature is not None and plan.execution_signature == accepted_signature,
            rank=index,
            candidate_count=len(ranked),
            context={
                **context,
                "unified_transition_final_acceptance": accepted_signature is not None,
                "unified_transition_selected_signature": accepted_signature,
            },
        )
        for index, plan in enumerate(ranked, start=1)
        if plan.measurement is not None
    ]
    if records:
        try:
            destination = append_measurement_records(log_path, records)
        except OSError:
            # Training telemetry must never invalidate an otherwise valid set.
            return
        if accepted_signature is not None:
            for plan in ranked:
                if plan.execution_signature == accepted_signature:
                    plan.metadata["measurement_log_path"] = str(destination)
                    break


def _bridge_exception_allowed(
    route: TransitionRoute,
    plan: TransitionPlan,
    judgement: dict[str, Any],
    config: UnifiedTransitionConfig,
) -> bool:
    if route.family != "bridge_preferred":
        return True
    router = dict(plan.metadata.get("transition_router", {}))
    route_fit = float(router.get("route_fit", 0.0))
    overall = float(judgement.get("overall", 0.0))
    reset_technique = plan.technique in {
        "echo_out_drop_in",
        "quick_cut",
        "layer_vocal_bridge",
        "layer_incoming_vocal_handoff",
        "layer_instrumental_handoff",
    }
    return (
        reset_technique
        and overall >= config.bridge_exception_score
        and route_fit >= config.bridge_exception_route_fit
    )


def resolve_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    evidence_matrix: PairEvidenceMatrix,
    professional_config: ProfessionalBrainConfig,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    set_state: SetState | None = None,
    dj_state: DJState | None = None,
    journey: str = "balanced",
    journey_progress: float = 0.5,
    listener: str | None = None,
    measurement_log_path: str | None = None,
    context: dict[str, Any] | None = None,
    render_context: TransitionRenderContext | None = None,
    transition_policy_context: TransitionPolicyContext | None = None,
    crowd_directive: CrowdDirective | None = None,
) -> TransitionResolution:
    """Solve one edge with a strict rehearse-before-commit contract.

    Search is hierarchical rather than effect-random: the primary musical
    transition family is rehearsed first, then bounded adjacent families are
    tried only when the higher-priority family fails rendered-audio acceptance.
    A non-accepted judgement is never returned as an accepted plan.
    """
    primary_route = route_transition(track_a, track_b)
    routes = route_family_sequence(track_a, track_b)
    cfg = UnifiedTransitionConfig.from_professional(professional_config)
    all_rehearsed: list[TransitionPlan] = []
    all_judgements: list[dict[str, Any]] = []
    attempted: list[int] = []
    reasons: list[str] = [*primary_route.reasons, *technique_policy(primary_route).reasons]
    retry_remedies: tuple[str, ...] = ()
    previous_failed_bars: int | None = None

    # retry_rounds means "how many additional musical strategies may be tried",
    # not "repeat the same bad blend".  A two-round configuration therefore
    # explores primary + up to two adjacent families.
    route_budget = min(len(routes), 1 + cfg.retry_rounds)
    scheduled_routes = routes[:route_budget]

    # A single edge gets one bounded render budget shared by all strategy
    # families.  Retrying musically must not multiply render cost by the
    # number of families.  Production default 8 therefore becomes roughly
    # 4 primary + 2 + 2 fallback; test default 4 becomes 2 + 2.
    total_budget = max(3, int(cfg.finalists))
    if len(scheduled_routes) == 1:
        family_budgets = [total_budget]
    else:
        primary_budget = max(2, int(round(total_budget * 0.55)))
        primary_budget = min(primary_budget, total_budget - (len(scheduled_routes) - 1))
        remaining = total_budget - primary_budget
        family_budgets = [primary_budget]
        for index in range(len(scheduled_routes) - 1):
            slots_left = len(scheduled_routes) - 1 - index
            budget = max(1, remaining // max(1, slots_left))
            family_budgets.append(budget)
            remaining -= budget
        if remaining > 0:
            family_budgets[-1] += remaining

    for attempt, (route, family_budget) in enumerate(zip(scheduled_routes, family_budgets, strict=True), start=1):
        policy = technique_policy(route)
        if attempt > 1:
            reasons.append(
                f"Strategy retry {attempt}: switching from the previous family to {route.family.replace('_', ' ')}; the rejected family is not simply regenerated."
            )
            reasons.extend(policy.reasons)

        pool, durations = _candidate_pool(
            track_a,
            track_b,
            route=route,
            evidence_matrix=evidence_matrix,
            config=cfg,
            tempo_policy=tempo_policy,
            preferred_style=preferred_style,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            stems_available=stems_available,
            retry_round=attempt,
            finalist_budget=family_budget,
            set_state=set_state,
            dj_state=dj_state,
            retry_remedies=retry_remedies,
            previous_failed_bars=previous_failed_bars,
            journey=journey,
            journey_progress=journey_progress,
            transition_policy_context=transition_policy_context,
            crowd_directive=crowd_directive,
        )
        attempted.extend(value for value in durations if value not in attempted)
        if not pool:
            reasons.append(
                f"Strategy {attempt} ({route.family}): no policy-feasible transition programs were generated."
            )
            continue

        brain_cfg = ProfessionalBrainConfig(
            **{
                **asdict(professional_config.normalized()),
                "finalists": min(len(pool), max(1, family_budget)),
                # Crucially, the acceptance bar never gets lowered on retry.
                "minimum_judge_score": cfg.minimum_judge_score,
            }
        )
        selection = rehearse_candidate_pool(
            track_a,
            track_b,
            pool,
            config=brain_cfg,
            set_state=set_state,
            journey=journey,
            journey_progress=journey_progress,
            listener=listener,
            measurement_log_path=None,
            context={
                **dict(context or {}),
                "unified_transition": True,
                "primary_transition_family": primary_route.family,
                "transition_family": route.family,
                "strategy_attempt": attempt,
                "duration_candidates": list(durations),
                "family_finalist_budget": family_budget,
                "total_edge_finalist_budget": total_budget,
            },
            render_context=render_context,
        )
        records = _judge_records(selection)
        all_rehearsed.extend(selection.rehearsed)
        all_judgements.extend(records)
        if not records:
            reasons.append(f"Strategy {attempt} ({route.family}): rehearsal returned no judge records.")
            continue

        top = records[0]
        selected = selection.selected
        if bool(top["accepted"]):
            if not _bridge_exception_allowed(primary_route, selected, top, cfg):
                reasons.append(
                    "The direct edge passed a generic threshold but the primary router strongly prefers a bridge; offline planning should re-route."
                )
                _log_resolution_measurements(
                    measurement_log_path,
                    all_rehearsed,
                    accepted_signature=None,
                    context={**dict(context or {}), "resolution_action": "replan_track"},
                )
                return TransitionResolution(
                    action="replan_track",
                    route=primary_route,
                    selected=None,
                    rehearsed=all_rehearsed,
                    judgements=all_judgements,
                    attempted_durations=attempted,
                    attempts=attempt,
                    reasons=reasons,
                )
            if float(top["uncertainty"]) > cfg.high_uncertainty and attempt < len(scheduled_routes):
                reasons.append(
                    f"Strategy {attempt}: candidate passed but uncertainty remained high ({float(top['uncertainty']):.0%}); trying the next musical family before commitment."
                )
                continue
            selected.requires_confirmation = False
            selected.auto_approved = True
            selected.metadata = {
                **selected.metadata,
                "unified_transition_resolution": {
                    "action": "accept",
                    "primary_family": primary_route.family,
                    "selected_family": route.family,
                    "strategy_attempt": attempt,
                    "family_sequence": [candidate.family for candidate in scheduled_routes],
                    "family_budgets": list(family_budgets),
                    "duration_candidates": list(attempted),
                    "review_needed_is_replan": True,
                    "simple_pro_shared_brain": True,
                    "principle": "simplest_professional_program_that_survives_rendered_audio_judging",
                },
            }
            _log_resolution_measurements(
                measurement_log_path,
                all_rehearsed,
                accepted_signature=selected.execution_signature,
                context={
                    **dict(context or {}),
                    "resolution_action": "accept",
                    "selected_family": route.family,
                },
            )
            return TransitionResolution(
                action="accept",
                route=route,
                selected=selected,
                rehearsed=all_rehearsed,
                judgements=all_judgements,
                attempted_durations=attempted,
                attempts=attempt,
                reasons=reasons,
            )

        retry_remedies = tuple(str(value) for value in top.get("remedies", []) if value)
        previous_failed_bars = int(top["bars"])
        reasons.append(
            f"Strategy {attempt} ({route.family}): best rendered candidate was not accepted; review-needed candidates are ineligible for the final plan."
        )
        if retry_remedies:
            reasons.append(
                "Critic-guided retry remedies: " + ", ".join(value.replace("_", " ") for value in retry_remedies) + "."
            )

    action: ResolutionAction = "replan_track" if primary_route.bridge_preference >= 0.45 else "replan_edge"
    reasons.append(
        "All bounded transition families failed rendered-audio acceptance; control returns to the Set Brain instead of performing a blind fade."
    )
    _log_resolution_measurements(
        measurement_log_path,
        all_rehearsed,
        accepted_signature=None,
        context={**dict(context or {}), "resolution_action": action},
    )
    return TransitionResolution(
        action=action,
        route=primary_route,
        selected=None,
        rehearsed=all_rehearsed,
        judgements=all_judgements,
        attempted_durations=attempted,
        attempts=len(scheduled_routes),
        reasons=reasons,
    )


def resolve_emergency_escape(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    evidence_matrix: PairEvidenceMatrix,
    professional_config: ProfessionalBrainConfig,
    effective_bpm_a: float | None,
    render_rate_a: float,
    stems_available: bool,
    set_state: SetState | None,
    journey: str,
    journey_progress: float,
    listener: str | None,
    measurement_log_path: str | None,
    context: dict[str, Any] | None = None,
    render_context: TransitionRenderContext | None = None,
) -> TransitionResolution:
    """Generate and judge a controlled reset after global replanning is exhausted."""
    route = route_transition(track_a, track_b)
    cfg = UnifiedTransitionConfig.from_professional(professional_config)
    plans: list[TransitionPlan] = []
    attempted = [4, 2, 1]
    for bars in attempted:
        try:
            plans.extend(
                rank_transition_candidates(
                    track_a,
                    track_b,
                    styles=("creative", "emergency"),
                    force=True,
                    requested_bars=bars,
                    exact_bars=True,
                    tempo_policy="cut",
                    effective_bpm_a=effective_bpm_a,
                    render_rate_a=render_rate_a,
                    stems_available=stems_available,
                    evidence_matrix=evidence_matrix,
                    limit_per_style=3,
                )
            )
        except (DifficultTransitionError, RuntimeError, ValueError):
            continue
    allowed = {
        "echo_out_drop_in",
        "quick_cut",
        "layer_vocal_bridge",
        "layer_incoming_vocal_handoff",
        "layer_instrumental_handoff",
        "backspin",
    }
    plans = [plan for plan in plans if plan.technique in allowed]
    unique = {plan.execution_signature: plan for plan in plans}
    pool = diversify_transition_candidates(
        list(unique.values()),
        count=min(max(3, cfg.finalists // 2), len(unique)) if unique else 0,
        diversity_weight=0.28,
    ) if unique else []
    if not pool:
        return TransitionResolution(
            action="emergency_escape",
            route=route,
            selected=None,
            attempted_durations=attempted,
            attempts=1,
            reasons=["No separately executable emergency escape candidate could be generated."],
        )

    emergency_minimum = max(0.42, professional_config.minimum_judge_score - 0.06)
    brain_cfg = ProfessionalBrainConfig(
        **{
            **asdict(professional_config.normalized()),
            "finalists": len(pool),
            "minimum_judge_score": emergency_minimum,
        }
    )
    selection = rehearse_candidate_pool(
        track_a,
        track_b,
        pool,
        config=brain_cfg,
        set_state=set_state,
        journey=journey,
        journey_progress=journey_progress,
        listener=listener,
        measurement_log_path=None,
        context={**dict(context or {}), "unified_transition_emergency": True},
        render_context=render_context,
    )
    records = _judge_records(selection)
    if records and bool(records[0]["accepted"]):
        selected = selection.selected
        selected.requires_confirmation = False
        selected.auto_approved = True
        selected.fallback_level = max(selected.fallback_level, 4)
        selected.fallback_reason = "Global track-order replanning was exhausted; a separately rendered and judged emergency escape was accepted."
        selected.metadata = {
            **selected.metadata,
            "unified_transition_resolution": {
                "action": "emergency_escape",
                "separately_rehearsed": True,
                "review_needed_is_replan": True,
                "duration_candidates": attempted,
            },
        }
        _log_resolution_measurements(
            measurement_log_path,
            list(selection.rehearsed),
            accepted_signature=selected.execution_signature,
            context={**dict(context or {}), "resolution_action": "emergency_escape_accept"},
        )
        return TransitionResolution(
            action="accept",
            route=route,
            selected=selected,
            rehearsed=selection.rehearsed,
            judgements=records,
            attempted_durations=attempted,
            attempts=1,
            reasons=["A separately rendered emergency escape cleared the emergency acceptance gate."],
        )
    _log_resolution_measurements(
        measurement_log_path,
        list(selection.rehearsed),
        accepted_signature=None,
        context={**dict(context or {}), "resolution_action": "emergency_escape_rejected"},
    )
    return TransitionResolution(
        action="emergency_escape",
        route=route,
        selected=None,
        rehearsed=selection.rehearsed,
        judgements=records,
        attempted_durations=attempted,
        attempts=1,
        reasons=["Emergency candidates were rendered but none cleared the emergency acceptance gate."],
    )
