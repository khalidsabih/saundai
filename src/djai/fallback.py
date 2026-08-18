from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Sequence

from .candidate_renderer import (
    MeasuredTransitionError,
    QualityGateConfig,
    evaluate_transition_candidates,
)
from .compatibility import harmonic_compatibility, score_transition, tempo_compatibility
from .performance.program import attach_performance_program
from .decision import execution_signature, seal_execution_plan
from .execution import TransitionRenderContext
from .models import (
    TechniqueAssessment,
    TrackAnalysis,
    TransitionCandidate,
    TransitionPlan,
)


@dataclass(frozen=True, slots=True)
class AutomaticFallbackPolicy:
    initial_top_k: int = 4
    expanded_top_k: int = 8
    minimum_measured_score: float = 0.42
    relaxed_measured_score: float = 0.28
    emergency_seconds: float = 1.8

    @classmethod
    def from_environment(cls) -> "AutomaticFallbackPolicy":
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

        return cls(
            initial_top_k=max(1, min(16, _int("DJAI_AUTO_TOP_K", 4))),
            expanded_top_k=max(2, min(24, _int("DJAI_AUTO_EXPANDED_TOP_K", 8))),
            minimum_measured_score=max(0.0, min(1.0, _float("DJAI_AUTO_MEASUREMENT_MINIMUM", 0.42))),
            relaxed_measured_score=max(0.0, min(1.0, _float("DJAI_AUTO_RELAXED_MINIMUM", 0.28))),
            emergency_seconds=max(0.4, min(4.0, _float("DJAI_AUTO_EMERGENCY_SECONDS", 1.8))),
        )


def _fallback_candidate(
    track: TrackAnalysis,
    *,
    kind: str,
    time_seconds: float,
) -> TransitionCandidate:
    templates = track.exit_candidates if kind == "exit" else track.entry_candidates
    if templates:
        source = min(templates, key=lambda item: abs(item.time_seconds - time_seconds))
        return copy.deepcopy(source)
    return TransitionCandidate(
        kind=kind,  # type: ignore[arg-type]
        time_seconds=time_seconds,
        bar_index=0,
        score=0.25,
        local_energy=track.energy,
        stability=0.35,
        available_bars=1,
        beat_grid_confidence=track.beat_grid_confidence,
        local_bpm=track.bpm,
        section_label="unknown",
        vocal_probability=track.vocal_probability,
        bass_activity=track.bass_activity,
    )




def _fallback_entry_candidate(track: TrackAnalysis) -> TransitionCandidate | None:
    """Pick a structurally useful emergency entry, never time-zero by default."""
    if not track.entry_candidates:
        return None
    bar_duration = max(0.5, track.time_signature * 60.0 / max(track.bpm, 1.0))
    mature_after = min(4.0 * bar_duration, max(2.0, track.duration_seconds * 0.08))
    preferred_before = min(135.0, track.duration_seconds * 0.48)
    mature = [
        candidate for candidate in track.entry_candidates
        if mature_after <= candidate.time_seconds <= preferred_before
    ]
    pool = mature or list(track.entry_candidates)
    role = {
        "groove": 1.00,
        "drop": 0.96,
        "build": 0.90,
        "breakdown": 0.74,
        "intro": 0.62,
        "outro": 0.12,
        "unknown": 0.48,
    }
    return max(
        pool,
        key=lambda candidate: (
            0.38 * candidate.score
            + 0.19 * candidate.boundary_strength
            + 0.16 * candidate.phrase_strength
            + 0.15 * role.get(candidate.section_label, 0.5)
            + 0.12 * (1.0 - max(0.0, min(1.0, candidate.vocal_probability)))
        ),
    )


def _hard_cuts_allowed() -> bool:
    return os.environ.get("DJAI_ALLOW_HARD_CUTS", "0").strip().lower() in {"1", "true", "yes", "on"}


_SECTION_HANDOFF_FIT: dict[tuple[str, str], float] = {
    ("outro", "intro"): 1.00,
    ("outro", "groove"): 0.90,
    ("groove", "intro"): 0.84,
    ("groove", "groove"): 0.96,
    ("groove", "build"): 0.88,
    ("groove", "breakdown"): 0.82,
    ("build", "drop"): 1.00,
    ("build", "groove"): 0.82,
    ("breakdown", "drop"): 1.00,
    ("breakdown", "groove"): 0.88,
    ("drop", "groove"): 0.84,
    ("drop", "breakdown"): 0.86,
    ("drop", "intro"): 0.72,
}


def _rescue_bars(track_a: TrackAnalysis, track_b: TrackAnalysis) -> tuple[int, str]:
    explicit = os.environ.get("DJAI_AUTO_RESCUE_BARS")
    if explicit not in {None, ""}:
        try:
            return max(2, min(8, int(explicit))), "explicit"
        except ValueError:
            pass
    tempo_score, _ = tempo_compatibility(track_a.bpm, track_b.bpm)
    harmonic_score = harmonic_compatibility(track_a.key, track_b.key)
    source_conflict = max(track_a.vocal_probability, track_b.vocal_probability)
    # A rescue exists because the richer programs failed.  Do not force a long
    # coexistence across a weak tempo/key/vocal relationship. Two bars still
    # means a real audible overlap (with a 4 second floor) while sounding much
    # more deliberate on difficult pairs.
    if tempo_score < 0.58 or harmonic_score < 0.44 or source_conflict > 0.68:
        return 2, "compact"
    return 4, "balanced"


def _natural_rescue_pair(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    source_cursor: float,
    latest_exit: float,
    body_target: float,
    transition_seconds: float,
) -> tuple[TransitionCandidate | None, TransitionCandidate | None]:
    exits = [
        candidate for candidate in track_a.exit_candidates
        if source_cursor + 0.25 <= candidate.time_seconds <= latest_exit
    ]
    entries = [
        candidate for candidate in track_b.entry_candidates
        if candidate.time_seconds <= max(0.0, track_b.duration_seconds - transition_seconds)
    ]
    if not exits or not entries:
        return (max(exits, key=lambda item: item.score) if exits else None, _fallback_entry_candidate(track_b))

    exits = sorted(
        exits,
        key=lambda candidate: (
            0.34 * candidate.score
            + 0.24 * candidate.phrase_strength
            + 0.20 * candidate.boundary_strength
            + 0.12 * (1.0 - min(1.0, abs(candidate.time_seconds - body_target) / 90.0))
            + 0.10 * (1.0 - candidate.vocal_probability)
        ),
        reverse=True,
    )[:8]
    entries = sorted(
        entries,
        key=lambda candidate: (
            0.34 * candidate.score
            + 0.24 * candidate.phrase_strength
            + 0.20 * candidate.boundary_strength
            + 0.12 * (1.0 - candidate.vocal_probability)
            + 0.10 * (1.0 - candidate.bass_activity)
        ),
        reverse=True,
    )[:8]

    best: tuple[float, TransitionCandidate, TransitionCandidate] | None = None
    for exit_candidate in exits:
        for entry_candidate in entries:
            breakdown = score_transition(track_a, track_b, exit_candidate, entry_candidate)
            phrase = min(exit_candidate.phrase_strength, entry_candidate.phrase_strength)
            boundary = min(exit_candidate.boundary_strength, entry_candidate.boundary_strength)
            source_clean = 0.56 * breakdown.vocal_overlap + 0.44 * breakdown.bass_conflict
            section_fit = _SECTION_HANDOFF_FIT.get(
                (exit_candidate.section_label, entry_candidate.section_label),
                0.56 if "unknown" in {exit_candidate.section_label, entry_candidate.section_label} else 0.48,
            )
            timing = 1.0 - min(1.0, abs(exit_candidate.time_seconds - body_target) / 100.0)
            utility = (
                0.19 * phrase
                + 0.15 * boundary
                + 0.17 * section_fit
                + 0.12 * breakdown.tempo
                + 0.08 * breakdown.harmonic
                + 0.15 * source_clean
                + 0.08 * min(exit_candidate.score, entry_candidate.score)
                + 0.06 * timing
            )
            if best is None or utility > best[0]:
                best = (utility, exit_candidate, entry_candidate)
    return (best[1], best[2]) if best is not None else (None, None)


def build_last_resort_plan(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    source_cursor: float = 0.0,
    preferred_exit_time: float | None = None,
    reason: str = "No higher-quality measured candidate was playable.",
    policy: AutomaticFallbackPolicy | None = None,
    render_context: TransitionRenderContext | None = None,
    verify_audio: bool = False,
) -> TransitionPlan:
    """Construct the final automatic rescue transition.

    Automatic delivery defaults to an audible multi-second filtered overlap.
    A literal hard cut is available only when DJAI_ALLOW_HARD_CUTS=1.  This
    makes the product contract match what a user means by "create a DJ set":
    failure of the learned/analytical policy may simplify the transition, but
    must not silently turn the junction into stop-A/start-B.
    """
    resolved = policy or AutomaticFallbackPolicy.from_environment()
    bpm_a = float(effective_bpm_a or track_a.bpm or 120.0)
    allow_hard_cut = _hard_cuts_allowed()

    if allow_hard_cut:
        transition_bars = 1
        transition_seconds = min(
            resolved.emergency_seconds,
            max(0.45, 4.0 * 60.0 / max(bpm_a, 1.0)),
        )
        technique = "quick_cut"
        source_b_fraction = 0.28
        rescue_kind = "hard_cut"
        rationale_action = "A short native-tempo cut was selected because hard cuts were explicitly enabled."
        choreography_reason = "An explicitly enabled structural cut avoids an unsafe long overlap."
    else:
        transition_bars, natural_profile = _rescue_bars(track_a, track_b)
        bar_seconds = max(0.5, float(track_a.time_signature or 4) * 60.0 / max(bpm_a, 1.0))
        transition_seconds = max(4.0, min(10.0, transition_bars * bar_seconds))
        technique = "filter_build"
        source_b_fraction = 1.0
        rescue_kind = "filtered_overlap"
        rationale_action = (
            "A compact phrase-aligned filtered handoff was selected because a longer overlap would sound forced."
            if natural_profile == "compact"
            else "A phrase-aligned filtered overlap was selected so automatic delivery remains a real DJ transition."
        )
        choreography_reason = "Use an audible filtered handoff rather than silently degrading to a splice."

    desired_source_a = transition_seconds * max(render_rate_a, 1e-6)
    latest_exit = max(0.0, track_a.duration_seconds - desired_source_a)
    remaining = max(0.0, track_a.duration_seconds - source_cursor)
    body_target = source_cursor + max(20.0, min(150.0, remaining * 0.58))
    natural_exit: TransitionCandidate | None = None
    natural_entry: TransitionCandidate | None = None
    if not allow_hard_cut:
        natural_exit, natural_entry = _natural_rescue_pair(
            track_a,
            track_b,
            source_cursor=source_cursor,
            latest_exit=latest_exit,
            body_target=body_target,
            transition_seconds=transition_seconds,
        )
    if preferred_exit_time is None:
        if natural_exit is not None:
            preferred_exit_time = natural_exit.time_seconds
        else:
            viable = [
                candidate.time_seconds
                for candidate in track_a.exit_candidates
                if candidate.time_seconds >= source_cursor + 0.25 and candidate.time_seconds <= latest_exit
            ]
            preferred_exit_time = (
                min(viable, key=lambda value: abs(value - body_target))
                if viable
                else body_target
            )
    exit_time = max(float(source_cursor), min(latest_exit, float(preferred_exit_time)))
    source_duration_a = min(
        desired_source_a, max(0.0, track_a.duration_seconds - exit_time)
    )

    entry_template = natural_entry or _fallback_entry_candidate(track_b)
    entry_time = float(entry_template.time_seconds if entry_template else 0.0)
    remaining_b = max(0.0, track_b.duration_seconds - entry_time)
    source_duration_b = min(
        remaining_b,
        max(0.12, transition_seconds * source_b_fraction),
    )

    # If a mature cue was too close to the end to sustain the rescue overlap,
    # fall back to an earlier structural cue before shortening the transition.
    if not allow_hard_cut and source_duration_b + 1e-6 < transition_seconds:
        viable_entries = [
            candidate for candidate in track_b.entry_candidates
            if candidate.time_seconds <= max(0.0, track_b.duration_seconds - transition_seconds)
        ]
        if viable_entries:
            entry_template = max(
                viable_entries,
                key=lambda candidate: (candidate.score, candidate.boundary_strength, candidate.phrase_strength),
            )
            entry_time = float(entry_template.time_seconds)
            remaining_b = max(0.0, track_b.duration_seconds - entry_time)
            source_duration_b = min(remaining_b, transition_seconds)

    exit_candidate = _fallback_candidate(track_a, kind="exit", time_seconds=exit_time)
    exit_candidate.time_seconds = round(exit_time, 6)
    exit_candidate.available_bars = max(1, transition_bars)
    entry_candidate = _fallback_candidate(track_b, kind="entry", time_seconds=entry_time)
    entry_candidate.time_seconds = round(entry_time, 6)
    entry_candidate.available_bars = max(1, entry_candidate.available_bars)
    breakdown = score_transition(track_a, track_b, exit_candidate, entry_candidate)

    assessment = TechniqueAssessment(
        technique=technique,
        intent="emergency_escape",
        feasible=True,
        score=max(0.18, min(0.62, breakdown.overall)),
        transition_bars=transition_bars,
        beatmatched=False,
        tempo_strategy="native_cut",
        transition_mode="emergency_cut",
        warnings=[reason],
        reasons=[choreography_reason],
    )
    plan = TransitionPlan(
        track_a=track_a.path,
        track_b=track_b.path,
        exit_time_a=round(exit_time, 6),
        entry_time_b=round(entry_time, 6),
        transition_bars=transition_bars,
        transition_seconds=round(transition_seconds, 6),
        target_bpm=round(bpm_a, 4),
        time_stretch_rate_b=1.0,
        time_stretch_applied=False,
        technique=technique,
        score=round(max(0.22, min(0.58, breakdown.overall)), 4),
        compatibility=breakdown,
        tempo_policy="cut",
        tempo_strategy="native_cut",
        transition_mode="emergency_cut",
        beatmatched=False,
        source_bpm_a=round(track_a.bpm, 4),
        source_bpm_b=round(track_b.bpm, 4),
        rendered_bpm_b=round(track_b.bpm, 4),
        warning=reason,
        forced=True,
        style="emergency",
        alternative_id="automatic_fallback",
        rationale=[
            "DJAI exhausted the preferred measured transitions.",
            rationale_action,
            "The fallback entry uses a mature structural cue instead of restarting the incoming track at time zero when possible.",
        ],
        exit_section=exit_candidate.section_label,
        entry_section=entry_candidate.section_label,
        cue_confidence=round(min(exit_candidate.score, entry_candidate.score), 4),
        requires_confirmation=False,
        quality="risky",
        genre_pair=(track_a.genre.label, track_b.genre.label),
        intent="emergency_escape",
        intent_score=assessment.score,
        technique_score=assessment.score,
        technique_assessments=[assessment],
        effective_bpm_a=round(bpm_a, 4),
        effective_bpm_b=round(track_b.bpm, 4),
        render_rate_a=round(render_rate_a, 6),
        render_rate_b=1.0,
        source_duration_a=round(source_duration_a, 6),
        source_duration_b=round(source_duration_b, 6),
        compiled_execution=True,
        analytical_score=round(breakdown.overall, 4),
        fallback_level=5,
        fallback_reason=reason,
        auto_approved=True,
        metadata={
            "automatic_fallback": True,
            "fallback_level": 5,
            "fallback_reason": reason,
            "rescue_transition_kind": rescue_kind,
            "continuous_mix_required": not allow_hard_cut,
            "transition_naturalness_profile": ("hard_cut" if allow_hard_cut else natural_profile),
            "rescue_duration_source": ("explicit_or_adaptive" if not allow_hard_cut else "hard_cut"),
        },
    )
    from .performance.deck import apply_deck_performance
    apply_deck_performance(plan, track_a, track_b, entry_candidate=entry_candidate)
    attach_performance_program(plan)
    seal_execution_plan(plan)
    if not verify_audio:
        plan.metadata = {
            **plan.metadata,
            "fallback_render_verified": False,
            "fallback_validation_pending": True,
        }
        return plan

    exact_context = render_context or TransitionRenderContext(
        sample_rate=QualityGateConfig.from_environment().sample_rate
    )
    try:
        verified = evaluate_transition_candidates(
            [plan],
            config=QualityGateConfig(
                enabled=True,
                top_k=1,
                sample_rate=exact_context.sample_rate,
                context_seconds=1.0,
                analytical_weight=1.0,
                measured_weight=0.0,
                minimum_measured_score=0.0,
                log_path=None,
            ),
            analytical_scores={plan.execution_signature: plan.score},
            context={
                "automatic_delivery": True,
                "fallback_level": 5,
                "technical_fallback_validation": True,
                "continuous_mix_required": not allow_hard_cut,
            },
            render_context=exact_context,
        ).selected
    except MeasuredTransitionError as exc:
        detail = (
            exc.best.measurement.hard_failures[0]
            if exc.best is not None
            and exc.best.measurement is not None
            and exc.best.measurement.hard_failures
            else str(exc)
        )
        raise MeasuredTransitionError(
            f"The deterministic last-resort transition failed rendered technical validation: {detail}",
            best=exc.best,
            evaluated=exc.evaluated,
            log_path=exc.log_path,
        ) from exc
    verified.requires_confirmation = False
    verified.auto_approved = True
    verified.fallback_level = 5
    verified.fallback_reason = reason
    verified.metadata = {
        **verified.metadata,
        "automatic_fallback": True,
        "fallback_render_verified": True,
        "fallback_validation_pending": False,
        "fallback_validation_sample_rate": exact_context.sample_rate,
        "rescue_transition_kind": rescue_kind,
        "continuous_mix_required": not allow_hard_cut,
        "transition_naturalness_profile": ("hard_cut" if allow_hard_cut else natural_profile),
    }
    return verified


def select_automatic_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    candidates: Sequence[TransitionPlan],
    *,
    analytical_scores: dict[str, float] | None = None,
    quality_config: QualityGateConfig | None = None,
    context: dict[str, Any] | None = None,
    source_cursor: float = 0.0,
    preferred_exit_time: float | None = None,
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    policy: AutomaticFallbackPolicy | None = None,
    render_context: TransitionRenderContext | None = None,
) -> TransitionPlan:
    resolved_policy = policy or AutomaticFallbackPolicy.from_environment()
    base_config = (quality_config or QualityGateConfig.from_environment()).normalized()
    exact_context = render_context or TransitionRenderContext(sample_rate=base_config.sample_rate)
    attempts = [
        (1, base_config),
        (
            2,
            QualityGateConfig(
                enabled=True,
                top_k=max(base_config.top_k, resolved_policy.expanded_top_k),
                sample_rate=base_config.sample_rate,
                context_seconds=base_config.context_seconds,
                analytical_weight=base_config.analytical_weight,
                measured_weight=base_config.measured_weight,
                minimum_measured_score=resolved_policy.relaxed_measured_score,
                log_path=base_config.log_path,
            ).normalized(),
        ),
    ]
    best_failed: TransitionPlan | None = None
    for level, config in attempts:
        try:
            selection = evaluate_transition_candidates(
                candidates,
                config=config,
                analytical_scores=analytical_scores,
                context={"automatic_delivery": True, "fallback_level": level, **dict(context or {})},
                render_context=exact_context,
            )
            selected = selection.selected
            selected.requires_confirmation = False
            selected.auto_approved = True
            selected.fallback_level = level - 1
            selected.fallback_reason = None if level == 1 else "A relaxed measured-quality pass was used."
            selected.metadata = {
                **selected.metadata,
                "automatic_delivery": True,
                "fallback_level": selected.fallback_level,
            }
            return selected
        except MeasuredTransitionError as exc:
            best_failed = exc.best or best_failed

    reason = (
        best_failed.warning
        if best_failed is not None and best_failed.warning
        else "All rendered candidates failed the automatic objective quality gate."
    )
    return build_last_resort_plan(
        track_a,
        track_b,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        source_cursor=source_cursor,
        preferred_exit_time=preferred_exit_time,
        reason=reason,
        policy=resolved_policy,
        render_context=exact_context,
        verify_audio=True,
    )
