from __future__ import annotations

import copy
from dataclasses import replace
import os
from typing import Iterable, Sequence

from .performance.program import attach_performance_program
from .music.timeline import build_musical_timeline
from .decision import assess_transition, execution_signature, seal_execution_plan
from .execution import TransitionRenderContext
from .models import (
    TempoPolicy,
    TrackAnalysis,
    TransitionCandidate,
    TransitionPlan,
    TransitionStyle,
)
from .planning_context import (
    CandidatePairEvidence,
    PairEvidenceMatrix,
    build_pair_evidence_matrix,
)
from .version import DECISION_VERSION

MINIMUM_AUTOMATIC_SCORE = 0.48
MINIMUM_BEAT_GRID_CONFIDENCE = 0.38


class DifficultTransitionError(RuntimeError):
    def __init__(self, score: float, message: str) -> None:
        super().__init__(message)
        self.score = score


def _manual_candidate(
    track: TrackAnalysis,
    *,
    kind: str,
    requested_time: float,
) -> TransitionCandidate:
    """Snap a user cue to the nearest estimated downbeat and reuse local evidence."""
    requested = max(0.0, min(float(requested_time), track.duration_seconds))
    templates = track.exit_candidates if kind == "exit" else track.entry_candidates
    if not templates:
        raise RuntimeError(f"No {kind} candidates are available for manual cueing.")

    template = min(templates, key=lambda item: abs(item.time_seconds - requested))
    timeline = build_musical_timeline(track)
    bar_index = timeline.nearest_bar(requested)
    snapped = min(track.duration_seconds, timeline.bar_start_seconds(bar_index))

    bar_seconds = max(3, track.time_signature) * 60.0 / max(track.bpm, 1.0)
    available_bars = max(0, int((track.duration_seconds - snapped) // bar_seconds))
    return replace(
        template,
        kind=kind,  # type: ignore[arg-type]
        time_seconds=round(snapped, 6),
        bar_index=bar_index,
        available_bars=available_bars,
    )


def _validate_tempo_policy(value: str) -> TempoPolicy:
    if value not in {"safe", "stretch", "cut"}:
        raise ValueError("tempo_policy must be one of: safe, stretch, cut")
    return value  # type: ignore[return-value]


def _validate_style(value: str) -> TransitionStyle:
    if value not in {"safe", "creative", "emergency"}:
        raise ValueError("style must be one of: safe, creative, emergency")
    return value  # type: ignore[return-value]


def _resolve_evidence_matrix(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    exit_time_a: float | None,
    entry_time_b: float | None,
    exit_candidates: Sequence[TransitionCandidate] | None,
    entry_candidates: Sequence[TransitionCandidate] | None,
    evidence_matrix: PairEvidenceMatrix | None,
) -> PairEvidenceMatrix:
    if evidence_matrix is not None:
        if exit_time_a is not None or entry_time_b is not None:
            raise ValueError("Manual cue times cannot be combined with a precomputed evidence matrix.")
        evidence_matrix.validate_tracks(track_a, track_b)
        if exit_candidates is not None or entry_candidates is not None:
            return evidence_matrix.subset(
                exit_candidates=exit_candidates,
                entry_candidates=entry_candidates,
            )
        return evidence_matrix

    exits: Sequence[TransitionCandidate] | None = exit_candidates
    entries: Sequence[TransitionCandidate] | None = entry_candidates
    if exit_time_a is not None:
        if exit_candidates is not None:
            raise ValueError("Provide either exit_time_a or exit_candidates, not both.")
        exits = [_manual_candidate(track_a, kind="exit", requested_time=exit_time_a)]
    if entry_time_b is not None:
        if entry_candidates is not None:
            raise ValueError("Provide either entry_time_b or entry_candidates, not both.")
        entries = [_manual_candidate(track_b, kind="entry", requested_time=entry_time_b)]

    return build_pair_evidence_matrix(
        track_a,
        track_b,
        exit_candidates=exits,
        entry_candidates=entries,
    )


def _build_plan(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    evidence: CandidatePairEvidence,
    *,
    style: TransitionStyle,
    tempo_policy: TempoPolicy,
    decision: object,
    force: bool,
    matrix_size: int,
    model_fingerprint: str | None,
) -> TransitionPlan:
    from .decision import TransitionDecision

    selected = decision
    if not isinstance(selected, TransitionDecision):
        raise TypeError("Expected a transition decision.")

    exit_candidate = evidence.exit_candidate
    entry_candidate = evidence.entry_candidate
    assessment = selected.assessment
    from .intelligence.embeddings import embedding_similarity
    timeline_a = build_musical_timeline(track_a)
    timeline_b = build_musical_timeline(track_b)
    exit_phrase = timeline_a.phrase_for_bar(exit_candidate.bar_index)
    entry_phrase = timeline_b.phrase_for_bar(entry_candidate.bar_index)
    embedding_score = embedding_similarity(
        track_a.section_embeddings.get(exit_candidate.section_label, track_a.music_embedding),
        track_b.section_embeddings.get(entry_candidate.section_label, track_b.music_embedding),
    )

    plan = TransitionPlan(
        track_a=track_a.path,
        track_b=track_b.path,
        exit_time_a=round(exit_candidate.time_seconds, 6),
        entry_time_b=round(entry_candidate.time_seconds, 6),
        transition_bars=assessment.transition_bars,
        transition_seconds=selected.transition_seconds,
        target_bpm=selected.target_bpm,
        time_stretch_rate_b=selected.render_rate_b,
        time_stretch_applied=selected.time_stretch_applied,
        technique=assessment.technique,
        score=selected.final_score,
        compatibility=evidence.breakdown,
        tempo_policy=tempo_policy,
        tempo_strategy=assessment.tempo_strategy,
        transition_mode=assessment.transition_mode,
        beatmatched=assessment.beatmatched,
        source_bpm_a=round(track_a.bpm, 4),
        source_bpm_b=round(track_b.bpm, 4),
        rendered_bpm_b=selected.effective_bpm_b,
        warning=selected.warning,
        forced=force and selected.requires_confirmation,
        style=style,
        alternative_id=style,
        rationale=list(selected.rationale),
        exit_section=exit_candidate.section_label,
        entry_section=entry_candidate.section_label,
        cue_confidence=round(
            0.32 * (selected.assessment.score or 0.0)
            + 0.68
            * min(
                exit_candidate.score,
                entry_candidate.score,
                exit_candidate.beat_grid_confidence or 0.55,
                entry_candidate.beat_grid_confidence or 0.55,
            ),
            4,
        ),
        requires_confirmation=selected.requires_confirmation,
        quality=selected.quality,
        genre_pair=(track_a.genre.label, track_b.genre.label),
        intent=selected.intent,
        intent_score=selected.intent_score,
        technique_score=selected.assessment.score,
        technique_assessments=list(selected.assessments),
        effective_bpm_a=selected.effective_bpm_a,
        effective_bpm_b=selected.effective_bpm_b,
        render_rate_a=selected.render_rate_a,
        render_rate_b=selected.render_rate_b,
        source_duration_a=selected.source_duration_a,
        source_duration_b=selected.source_duration_b,
        compiled_execution=True,
        analytical_score=selected.final_score,
        metadata={
            "exit_bar_index": exit_candidate.bar_index,
            "entry_bar_index": entry_candidate.bar_index,
            "exit_vocal_probability": round(exit_candidate.vocal_probability, 6),
            "entry_vocal_probability": round(entry_candidate.vocal_probability, 6),
            "exit_bass_activity": round(exit_candidate.bass_activity, 6),
            "entry_bass_activity": round(entry_candidate.bass_activity, 6),
            "section_pair": f"{exit_candidate.section_label}->{entry_candidate.section_label}",
            "musical_timeline_version": timeline_a.version,
            "exit_timeline_fingerprint": timeline_a.fingerprint,
            "entry_timeline_fingerprint": timeline_b.fingerprint,
            "exit_phrase_id": exit_phrase.phrase_id if exit_phrase else None,
            "entry_phrase_id": entry_phrase.phrase_id if entry_phrase else None,
            "exit_phrase_role": exit_phrase.role if exit_phrase else "unknown",
            "entry_phrase_role": entry_phrase.role if entry_phrase else "unknown",
            "exit_song_role": exit_phrase.song_role if exit_phrase else "unknown",
            "entry_song_role": entry_phrase.song_role if entry_phrase else "unknown",
            "exit_song_role_index": exit_phrase.song_role_index if exit_phrase else 0,
            "entry_song_role_index": entry_phrase.song_role_index if entry_phrase else 0,
            "exit_song_form_confidence": round(exit_phrase.song_form_confidence, 4) if exit_phrase else 0.0,
            "entry_song_form_confidence": round(entry_phrase.song_form_confidence, 4) if entry_phrase else 0.0,
            "exit_phrase_readiness": round(exit_phrase.transition_readiness, 4) if exit_phrase else 0.0,
            "entry_phrase_readiness": round(entry_phrase.transition_readiness, 4) if entry_phrase else 0.0,
            "decision_version": DECISION_VERSION,
            "compatibility_matrix_size": matrix_size,
            "compatibility_matrix_reused": True,
            "scoring_model_fingerprint": model_fingerprint,
            "embedding_similarity": round(embedding_score, 6),
            "embedding_backend_a": track_a.embedding_backend,
            "embedding_backend_b": track_b.embedding_backend,
        },
    )
    from .performance.deck import apply_deck_performance
    apply_deck_performance(plan, track_a, track_b, entry_candidate=entry_candidate)
    attach_performance_program(plan)
    seal_execution_plan(plan)
    return plan


def _rank_style_plans(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    evidence_matrix: PairEvidenceMatrix,
    style: TransitionStyle,
    tempo_policy: TempoPolicy,
    force: bool,
    requested_bars: int | None,
    exact_bars: bool,
    effective_bpm_a: float | None,
    render_rate_a: float,
    stems_available: bool,
    excluded_pairs: set[tuple[int, int]] | None,
) -> list[TransitionPlan]:
    """Return analytically ranked executable plans for one user-facing style."""
    evidence_matrix.validate_tracks(track_a, track_b)
    if not evidence_matrix.pairs:
        raise RuntimeError("No valid transition candidates were found.")

    excluded = excluded_pairs or set()
    ranked: list[tuple[float, TransitionPlan, tuple[int, int]]] = []
    allow_fallback = force or style == "emergency" or tempo_policy == "cut"

    # SAUND.AI 1.1: evaluate the already-trained bootstrap cue prior once for
    # each side of the pair.  In local/advisory mode this is a bounded ranking
    # residual; in production/shadow mode it is telemetry only.  The model can
    # never make an infeasible candidate feasible or bypass rendered judging.
    bootstrap_exit_priors = {}
    bootstrap_entry_priors = {}
    bootstrap_cue_pair = None
    try:
        from .intelligence.bootstrap_runtime import cue_candidate_priors, cue_pair_prediction

        bootstrap_exit_priors = cue_candidate_priors(track_a, evidence_matrix.exit_candidates)
        bootstrap_entry_priors = cue_candidate_priors(track_b, evidence_matrix.entry_candidates)
        bootstrap_cue_pair = cue_pair_prediction
    except Exception:
        # External research priors must be isolated from the deterministic DJ
        # engine: a bad/missing artifact is equivalent to no learned prior.
        bootstrap_exit_priors = {}
        bootstrap_entry_priors = {}
        bootstrap_cue_pair = None

    for evidence in evidence_matrix.pairs:
        signature = evidence.cue_signature
        if signature in excluded:
            continue
        decision = assess_transition(
            track_a,
            track_b,
            evidence.exit_candidate,
            evidence.entry_candidate,
            evidence.breakdown,
            style=style,
            tempo_policy=tempo_policy,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            requested_bars=requested_bars,
            stems_available=stems_available,
            allow_fallback=allow_fallback,
            exact_bars=exact_bars,
        )
        if decision is None:
            continue
        plan = _build_plan(
            track_a,
            track_b,
            evidence,
            style=style,
            tempo_policy=tempo_policy,
            decision=decision,
            force=force,
            matrix_size=evidence_matrix.compatibility_evaluations,
            model_fingerprint=evidence_matrix.scoring_model_fingerprint,
        )
        heuristic_utility = (
            0.70 * plan.score
            + 0.16 * plan.cue_confidence
            + 0.08 * plan.compatibility.boundary
            + 0.06 * plan.compatibility.phrase
        )
        learned_cue = None
        if bootstrap_cue_pair is not None:
            try:
                learned_cue = bootstrap_cue_pair(
                    track_a,
                    track_b,
                    evidence.exit_candidate,
                    evidence.entry_candidate,
                    exit_priors=bootstrap_exit_priors,
                    entry_priors=bootstrap_entry_priors,
                )
            except Exception:
                learned_cue = None
        learned_weight = float(getattr(learned_cue, "effective_weight", 0.0) or 0.0)
        learned_score = float(getattr(learned_cue, "score", 0.5) or 0.5)
        utility = (1.0 - learned_weight) * heuristic_utility + learned_weight * learned_score
        bootstrap_meta = dict(plan.metadata.get("bootstrap_learning", {}))
        if learned_cue is not None:
            bootstrap_meta["cue_point_prior"] = learned_cue.to_dict()
        plan.metadata = {
            **plan.metadata,
            "selection_utility": round(utility, 6),
            "heuristic_selection_utility": round(heuristic_utility, 6),
            "bootstrap_learning": bootstrap_meta,
        }
        ranked.append((utility, plan, signature))

    if not ranked:
        bars_detail = (
            f" The requested {requested_bars}-bar limit is unsupported or unavailable for every feasible technique."
            if requested_bars is not None
            else ""
        )
        raise DifficultTransitionError(
            0.0,
            "No technique-feasible transition was found for the selected cues and tempo policy."
            + bars_detail,
        )

    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].transition_mode != "emergency_cut",
            item[1].score,
            item[1].cue_confidence,
            item[1].execution_signature,
        ),
        reverse=True,
    )
    return [item[1] for item in ranked]


def _search_style(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    evidence_matrix: PairEvidenceMatrix,
    style: TransitionStyle,
    tempo_policy: TempoPolicy,
    force: bool,
    requested_bars: int | None,
    effective_bpm_a: float | None,
    render_rate_a: float,
    stems_available: bool,
    excluded_pairs: set[tuple[int, int]] | None,
    enforce_safety: bool,
) -> TransitionPlan:
    plans = _rank_style_plans(
        track_a,
        track_b,
        evidence_matrix=evidence_matrix,
        style=style,
        tempo_policy=tempo_policy,
        force=force or not enforce_safety,
        requested_bars=requested_bars,
        exact_bars=False,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        excluded_pairs=excluded_pairs,
    )
    plan = plans[0]

    if enforce_safety and not force:
        if plan.transition_mode == "emergency_cut" and style != "emergency":
            raise DifficultTransitionError(
                plan.score,
                "A transparent transition is not feasible. Confirm a fallback cut or choose emergency mode.",
            )
        if plan.score < MINIMUM_AUTOMATIC_SCORE:
            raise DifficultTransitionError(
                plan.score,
                f"Transition score {plan.score:.0%} is below the automatic safety threshold.",
            )
        if plan.compatibility.beat_grid < MINIMUM_BEAT_GRID_CONFIDENCE and plan.beatmatched:
            raise DifficultTransitionError(
                plan.score,
                f"Local beat-grid confidence is only {plan.compatibility.beat_grid:.0%}.",
            )
        if plan.requires_confirmation:
            raise DifficultTransitionError(
                plan.score,
                plan.warning or "The transition requires explicit confirmation.",
            )
    return plan


def rank_transition_candidates(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    styles: Iterable[str] = ("safe", "creative", "emergency"),
    force: bool = True,
    requested_bars: int | None = None,
    exact_bars: bool = False,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    exit_candidates: Sequence[TransitionCandidate] | None = None,
    entry_candidates: Sequence[TransitionCandidate] | None = None,
    evidence_matrix: PairEvidenceMatrix | None = None,
    limit_per_style: int = 3,
    transition_route: "TransitionRoute | None" = None,
) -> list[TransitionPlan]:
    """Return analytical candidates without rendering audio.

    Phase 1 uses this as the cheap filter before rendering only the top-K plans.
    """
    validated_styles = [_validate_style(style) for style in styles]
    if not validated_styles:
        return []
    policy = _validate_tempo_policy(tempo_policy)
    matrix = _resolve_evidence_matrix(
        track_a,
        track_b,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        exit_candidates=exit_candidates,
        entry_candidates=entry_candidates,
        evidence_matrix=evidence_matrix,
    )
    limit = max(1, int(limit_per_style))
    candidates: list[TransitionPlan] = []
    for style in validated_styles:
        style_policy: TempoPolicy = "cut" if style == "emergency" else policy
        try:
            ranked = _rank_style_plans(
                track_a,
                track_b,
                evidence_matrix=matrix,
                style=style,
                tempo_policy=style_policy,
                force=force,
                requested_bars=requested_bars,
                exact_bars=exact_bars,
                effective_bpm_a=effective_bpm_a,
                render_rate_a=render_rate_a,
                stems_available=stems_available,
                excluded_pairs=None,
            )
        except DifficultTransitionError:
            continue
        candidates.extend(ranked[:limit])

    # Supplied full/instrumental/vocal bundles create additional executable
    # ideas after the normal analytical cue/tempo safety search. They are not
    # mislabeled as four-way stems and therefore cannot accidentally unlock
    # bass/drum stem techniques that the sources cannot support.
    from .layers import augment_layer_transition_plans
    from .brain.stem_choreography import augment_stem_choreography_plans
    from .brain.transition_router import TransitionRoute, apply_transition_route, route_transition
    from .brain.technique_policy import apply_technique_policy
    from .brain.professional_mixing import apply_professional_mixing_policy

    route = transition_route or route_transition(track_a, track_b)
    master_fallback = None
    if candidates:
        pure_master_techniques = {
            "equal_power_blend",
            "echo_out_drop_in",
            "quick_cut",
            "backspin",
            "loop_roll",
        }
        fallback_pool = [
            plan for plan in candidates if plan.technique in pure_master_techniques
        ] or list(candidates)
        master_fallback = copy.deepcopy(
            max(
                fallback_pool,
                key=lambda plan: (
                    float(plan.metadata.get("selection_utility", plan.score)),
                    plan.score,
                    plan.cue_confidence,
                ),
            )
        )
    candidates = augment_layer_transition_plans(track_a, track_b, candidates)
    layer_intelligence_active = any(
        plan.technique == "layer_incoming_vocal_handoff" for plan in candidates
    )
    if stems_available:
        candidates = augment_stem_choreography_plans(track_a, track_b, candidates, route=route)
    route, candidates = apply_transition_route(track_a, track_b, candidates, route=route)
    candidates = apply_technique_policy(track_a, track_b, route, candidates)
    candidates = apply_professional_mixing_policy(track_a, track_b, candidates, route=route)

    # A layer-aware brain may prefer every isolated-source candidate over the
    # original master programs.  Keep one pre-layer full-master candidate in
    # the pool so AI Headphones/final compilation always has a deterministic
    # escape that does not depend on separated material.
    has_master_fallback = any(
        str(dict(plan.performance_program or {}).get("source_mode", "master")) == "master"
        for plan in candidates
    )
    if layer_intelligence_active and master_fallback is not None and not has_master_fallback:
        master_fallback.metadata = {
            **master_fallback.metadata,
            "layer_fallback_candidate": True,
            "layer_fallback_for": "layer_incoming_vocal_handoff",
            "fallback_principle": "full_master_transition_remains_available",
        }
        candidates.append(master_fallback)

    unique: dict[str, TransitionPlan] = {}
    for plan in candidates:
        unique[plan.execution_signature] = plan
    return sorted(
        unique.values(),
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            plan.score,
            plan.cue_confidence,
            plan.execution_signature,
        ),
        reverse=True,
    )


def plan_transition_candidate_pool(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    styles: Iterable[str] = ("safe", "creative", "emergency"),
    per_style: int = 3,
    force: bool = True,
    requested_bars: int | None = None,
    exact_bars: bool = False,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    exit_candidates: Sequence[TransitionCandidate] | None = None,
    entry_candidates: Sequence[TransitionCandidate] | None = None,
    evidence_matrix: PairEvidenceMatrix | None = None,
) -> list[TransitionPlan]:
    """Compatibility alias for the Phase 1 analytical candidate pool."""
    return rank_transition_candidates(
        track_a,
        track_b,
        styles=styles,
        force=force,
        requested_bars=requested_bars,
        exact_bars=exact_bars,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        tempo_policy=tempo_policy,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        exit_candidates=exit_candidates,
        entry_candidates=entry_candidates,
        evidence_matrix=evidence_matrix,
        limit_per_style=per_style,
    )


def plan_transition_measured(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    force: bool = False,
    requested_bars: int | None = None,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    style: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    evidence_matrix: PairEvidenceMatrix | None = None,
    measurement_top_k: int = 3,
    measurement_sample_rate: int = 22_050,
    measurement_log_path: str | None = None,
) -> TransitionPlan:
    """Render and measure top analytical candidates before final selection."""
    from .candidate_renderer import QualityGateConfig, evaluate_transition_candidates

    validated_style = _validate_style(style)
    candidates = rank_transition_candidates(
        track_a,
        track_b,
        styles=(validated_style,),
        force=True,
        requested_bars=requested_bars,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        tempo_policy=tempo_policy,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        evidence_matrix=evidence_matrix,
        limit_per_style=max(3, measurement_top_k * 2),
    )
    if not candidates:
        raise DifficultTransitionError(0.0, "No measurable transition candidates were found.")
    scores = {
        plan.execution_signature: float(plan.metadata.get("selection_utility", plan.score))
        for plan in candidates
    }
    selection = evaluate_transition_candidates(
        candidates,
        config=QualityGateConfig.from_environment(
            enabled=True,
            top_k=measurement_top_k,
            sample_rate=measurement_sample_rate,
            log_path=measurement_log_path,
        ),
        analytical_scores=scores,
        context={"mode": "pair", "style": validated_style},
    )
    plan = selection.selected
    if not force:
        if plan.score < MINIMUM_AUTOMATIC_SCORE:
            raise DifficultTransitionError(
                plan.score,
                f"Measured transition score {plan.score:.0%} is below the automatic safety threshold.",
            )
        if plan.requires_confirmation:
            raise DifficultTransitionError(
                plan.score,
                plan.warning or "The measured transition requires explicit confirmation.",
            )
    return plan


def plan_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    force: bool = False,
    requested_bars: int | None = None,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    style: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    evidence_matrix: PairEvidenceMatrix | None = None,
) -> TransitionPlan:
    matrix = _resolve_evidence_matrix(
        track_a,
        track_b,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        exit_candidates=None,
        entry_candidates=None,
        evidence_matrix=evidence_matrix,
    )
    plan = _search_style(
        track_a,
        track_b,
        evidence_matrix=matrix,
        style=_validate_style(style),
        tempo_policy=_validate_tempo_policy(tempo_policy),
        force=force,
        requested_bars=requested_bars,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        excluded_pairs=None,
        enforce_safety=True,
    )
    from .brain.professional_mixing import apply_professional_mixing_policy
    from .brain.transition_router import route_transition

    return apply_professional_mixing_policy(
        track_a, track_b, [plan], route=route_transition(track_a, track_b)
    )[0]


def plan_transition_styles(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    styles: Iterable[str],
    force: bool = False,
    requested_bars: int | None = None,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    exit_candidates: Sequence[TransitionCandidate] | None = None,
    entry_candidates: Sequence[TransitionCandidate] | None = None,
    evidence_matrix: PairEvidenceMatrix | None = None,
    diversify_cues: bool = True,
) -> list[TransitionPlan]:
    validated_styles = [_validate_style(style) for style in styles]
    if not validated_styles:
        return []
    policy = _validate_tempo_policy(tempo_policy)
    matrix = _resolve_evidence_matrix(
        track_a,
        track_b,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        exit_candidates=exit_candidates,
        entry_candidates=entry_candidates,
        evidence_matrix=evidence_matrix,
    )

    plans: list[TransitionPlan] = []
    excluded: set[tuple[int, int]] = set()
    for style in validated_styles:
        style_policy: TempoPolicy = "cut" if style == "emergency" else policy
        try:
            plan = _search_style(
                track_a,
                track_b,
                evidence_matrix=matrix,
                style=style,
                tempo_policy=style_policy,
                force=force,
                requested_bars=requested_bars,
                effective_bpm_a=effective_bpm_a,
                render_rate_a=render_rate_a,
                stems_available=stems_available,
                excluded_pairs=excluded if diversify_cues else None,
                enforce_safety=False,
            )
        except DifficultTransitionError:
            if not diversify_cues:
                raise
            plan = _search_style(
                track_a,
                track_b,
                evidence_matrix=matrix,
                style=style,
                tempo_policy=style_policy,
                force=force,
                requested_bars=requested_bars,
                effective_bpm_a=effective_bpm_a,
                render_rate_a=render_rate_a,
                stems_available=stems_available,
                excluded_pairs=None,
                enforce_safety=False,
            )
        plans.append(plan)
        if diversify_cues:
            excluded.add(
                (
                    int(plan.metadata.get("exit_bar_index", round(plan.exit_time_a))),
                    int(plan.metadata.get("entry_bar_index", round(plan.entry_time_b))),
                )
            )
    from .brain.professional_mixing import apply_professional_mixing_policy
    from .brain.transition_router import route_transition

    return apply_professional_mixing_policy(
        track_a, track_b, plans, route=route_transition(track_a, track_b)
    )


def plan_transition_alternatives(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    force: bool = False,
    requested_bars: int | None = None,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    exit_candidates: Sequence[TransitionCandidate] | None = None,
    entry_candidates: Sequence[TransitionCandidate] | None = None,
    evidence_matrix: PairEvidenceMatrix | None = None,
) -> list[TransitionPlan]:
    """Return safe, creative, and emergency plans over one evidence matrix."""
    return plan_transition_styles(
        track_a,
        track_b,
        styles=("safe", "creative", "emergency"),
        force=force,
        requested_bars=requested_bars,
        exit_time_a=exit_time_a,
        entry_time_b=entry_time_b,
        tempo_policy=tempo_policy,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        exit_candidates=exit_candidates,
        entry_candidates=entry_candidates,
        evidence_matrix=evidence_matrix,
        diversify_cues=True,
    )


def plan_transition_automatic(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    requested_bars: int | None = None,
    exit_time_a: float | None = None,
    entry_time_b: float | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    evidence_matrix: PairEvidenceMatrix | None = None,
    measurement_top_k: int = 4,
    measurement_sample_rate: int = 22_050,
    measurement_log_path: str | None = None,
    source_cursor: float = 0.0,
) -> TransitionPlan:
    """Plan a fully automatic transition with an internal fallback ladder."""
    from .candidate_renderer import QualityGateConfig
    from .fallback import select_automatic_transition
    from .brain.professional import ProfessionalBrainConfig, plan_professional_transition

    if os.environ.get("DJAI_PROFESSIONAL_BRAIN", "1").strip().lower() not in {"0", "false", "no", "off"} and requested_bars is None and exit_time_a is None and entry_time_b is None:
        return plan_professional_transition(
            track_a,
            track_b,
            config=ProfessionalBrainConfig.from_environment(
                finalists=(
                    int(os.environ["DJAI_BRAIN_FINALISTS"])
                    if os.environ.get("DJAI_BRAIN_FINALISTS", "").isdigit()
                    else max(4, measurement_top_k)
                ),
                measurement_sample_rate=measurement_sample_rate,
            ),
            tempo_policy=tempo_policy,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            stems_available=stems_available,
            evidence_matrix=evidence_matrix,
            measurement_log_path=measurement_log_path,
            source_cursor=source_cursor,
            preferred_exit_time=exit_time_a,
        )

    try:
        candidates = rank_transition_candidates(
            track_a,
            track_b,
            styles=("safe", "creative", "emergency"),
            force=True,
            requested_bars=requested_bars,
            exit_time_a=exit_time_a,
            entry_time_b=entry_time_b,
            tempo_policy=tempo_policy,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            stems_available=stems_available,
            evidence_matrix=evidence_matrix,
            limit_per_style=max(3, measurement_top_k * 2),
        )
    except (DifficultTransitionError, RuntimeError, ValueError) as exc:
        from .fallback import build_last_resort_plan

        return build_last_resort_plan(
            track_a,
            track_b,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            source_cursor=source_cursor,
            preferred_exit_time=exit_time_a,
            reason=f"Automatic planning used the last-resort cut: {exc}",
            render_context=TransitionRenderContext(sample_rate=measurement_sample_rate),
            verify_audio=True,
        )
    if not candidates:
        from .fallback import build_last_resort_plan

        return build_last_resort_plan(
            track_a,
            track_b,
            effective_bpm_a=effective_bpm_a,
            render_rate_a=render_rate_a,
            source_cursor=source_cursor,
            preferred_exit_time=exit_time_a,
            reason="Automatic planning found no technique-feasible candidates.",
            render_context=TransitionRenderContext(sample_rate=measurement_sample_rate),
            verify_audio=True,
        )
    scores = {
        plan.execution_signature: float(plan.metadata.get("selection_utility", plan.score))
        for plan in candidates
    }
    return select_automatic_transition(
        track_a,
        track_b,
        candidates,
        analytical_scores=scores,
        quality_config=QualityGateConfig.from_environment(
            enabled=True,
            top_k=measurement_top_k,
            sample_rate=measurement_sample_rate,
            log_path=measurement_log_path,
        ),
        context={"mode": "pair", "automatic_delivery": True},
        source_cursor=source_cursor,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
    )
