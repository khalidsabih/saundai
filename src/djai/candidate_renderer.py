from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .audio import load_audio
from .execution import TransitionRenderContext, normalization_gain
from .decision import validate_execution_plan
from .measurement import DEFAULT_MEASUREMENT_POLICY, MeasurementPolicy, measure_transition_audio
from .judge.musical import measure_musical_execution
from .judge.render_embedding import embed_rendered_transition
from .measurement_store import append_measurement_records, measurement_record
from .models import TransitionMeasurement, TransitionPlan
from .naturalness import annotate_transition_naturalness, naturalness_blended_score
from .renderer import DEFAULT_RENDER_SAMPLE_RATE, render_transition_audio_from_loaded
from .version import MEASUREMENT_VERSION
from .stems import measure_stem_transition_safety
from .rehearsal_cache import (
    load_cached_rehearsal,
    rehearsal_cache_key,
    store_cached_rehearsal,
)


def _capture_render_embeddings(context: Mapping[str, Any] | None = None) -> bool:
    """Compute rendered embeddings only when a live decision or training needs them.

    Safety and musical measurements are always computed. In auto mode Taste Lab
    still captures the full rendered representation. Normal set creation captures
    it only when an authoritative deployed model has non-zero dependence on those
    features, avoiding expensive unused DSP on every finalist.
    """

    requested = os.environ.get("DJAI_CAPTURE_RENDER_EMBEDDINGS", "auto").strip().lower()
    if requested in {"1", "true", "yes", "on"}:
        return True
    if requested in {"0", "false", "no", "off"}:
        return False
    resolved_context = dict(context or {})
    if resolved_context.get("mode") == "taste_lab" or bool(resolved_context.get("capture_render_embeddings")):
        return True
    try:
        from .intelligence.neural_policy import scalar_head_uses_feature_prefix
        from .intelligence.transition_ranker import ranker_uses_render_embedding

        rendered_judge_needs_it = scalar_head_uses_feature_prefix(
            "rendered_dj_judge",
            "render_embedding_",
            require_authority=True,
        )
        ranker_needs_it = (
            os.environ.get("DJAI_TRANSITION_RANKER", "1").strip().lower()
            not in {"0", "false", "no", "off"}
            and ranker_uses_render_embedding()
        )
        return bool(rendered_judge_needs_it or ranker_needs_it)
    except (OSError, ValueError, RuntimeError):
        return False


@lru_cache(maxsize=2)
def _load_rehearsal_audio_cached(
    path: str,
    sample_rate: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
) -> tuple[Any, int]:
    """Keep only the current directional pair decoded across adaptive retries."""
    del size, mtime_ns, ctime_ns
    return load_audio(path, sample_rate=sample_rate, mono=False)


def _load_rehearsal_audio(path: str | Path, sample_rate: int) -> tuple[Any, int]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _load_rehearsal_audio_cached(
        str(resolved),
        int(sample_rate),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000))),
    )


class MeasuredTransitionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        best: TransitionPlan | None = None,
        evaluated: Sequence[TransitionPlan] | None = None,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.best = best
        self.evaluated = list(evaluated or ([] if best is None else [best]))
        self.log_path = log_path


@dataclass(frozen=True, slots=True)
class QualityGateConfig:
    enabled: bool = True
    top_k: int = 4
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE
    context_seconds: float = 2.0
    analytical_weight: float = 0.55
    measured_weight: float = 0.45
    minimum_measured_score: float = 0.42
    log_path: str | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        enabled: bool = True,
        top_k: int | None = None,
        sample_rate: int | None = None,
        context_seconds: float | None = None,
        log_path: str | None = None,
    ) -> "QualityGateConfig":
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

        env_enabled = os.environ.get("DJAI_MEASUREMENT_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        return cls(
            enabled=enabled and env_enabled,
            top_k=top_k if top_k is not None else _int("DJAI_MEASUREMENT_TOP_K", 4),
            sample_rate=(
                sample_rate
                if sample_rate is not None
                else _int("DJAI_MEASUREMENT_SAMPLE_RATE", DEFAULT_RENDER_SAMPLE_RATE)
            ),
            context_seconds=(
                context_seconds
                if context_seconds is not None
                else max(
                    _float("DJAI_MEASUREMENT_PRE_ROLL", 2.0),
                    _float("DJAI_MEASUREMENT_POST_ROLL", 2.0),
                )
            ),
            analytical_weight=_float("DJAI_MEASUREMENT_ANALYTICAL_WEIGHT", 0.55),
            measured_weight=_float("DJAI_MEASUREMENT_AUDIO_WEIGHT", 0.45),
            minimum_measured_score=_float("DJAI_MEASUREMENT_MINIMUM", 0.42),
            log_path=(
                log_path
                if log_path is not None
                else os.environ.get(
                    "DJAI_MEASUREMENT_LOG",
                    "ratings/automatic-measurements.jsonl",
                )
            ),
        ).normalized()

    def normalized(self) -> "QualityGateConfig":
        top_k = max(1, min(16, int(self.top_k)))
        sample_rate = max(8_000, min(DEFAULT_RENDER_SAMPLE_RATE, int(self.sample_rate)))
        context = max(0.5, min(8.0, float(self.context_seconds)))
        analytical = max(0.0, float(self.analytical_weight))
        measured = max(0.0, float(self.measured_weight))
        total = analytical + measured
        if total <= 1e-9:
            analytical, measured, total = 0.55, 0.45, 1.0
        return QualityGateConfig(
            enabled=bool(self.enabled),
            top_k=top_k,
            sample_rate=sample_rate,
            context_seconds=context,
            analytical_weight=analytical / total,
            measured_weight=measured / total,
            minimum_measured_score=max(0.0, min(1.0, float(self.minimum_measured_score))),
            log_path=self.log_path,
        )


@dataclass(slots=True)
class MeasuredSelection:
    selected: TransitionPlan
    evaluated: list[TransitionPlan]
    log_path: Path | None = None
    log_error: str | None = None
    rejected_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_execution_signature": self.selected.execution_signature,
            "selected_measured_score": self.selected.measured_score,
            "selected_final_score": self.selected.score,
            "rendered_candidates": len(self.evaluated),
            "rejected_candidates": self.rejected_count,
            "gate_passed": self.selected.measurement_passed,
            "log_path": str(self.log_path) if self.log_path else None,
            "log_error": self.log_error,
            **self.metadata,
        }



def _style_diverse_top_k(
    plans: Sequence[TransitionPlan],
    scores: Mapping[str, float],
    top_k: int,
) -> list[TransitionPlan]:
    ranked = sorted(
        plans,
        key=lambda plan: (
            float(scores.get(plan.execution_signature, plan.score)),
            plan.score,
            plan.cue_confidence,
            plan.execution_signature,
        ),
        reverse=True,
    )
    if top_k < 3:
        return ranked[:top_k]

    chosen: list[TransitionPlan] = []
    for style in ("safe", "creative", "emergency"):
        candidate = next((plan for plan in ranked if plan.style == style), None)
        if candidate is not None and candidate not in chosen:
            chosen.append(candidate)
    for plan in ranked:
        if plan not in chosen:
            chosen.append(plan)
        if len(chosen) >= top_k:
            break
    return chosen[:top_k]


def _rehearse_one_candidate(
    candidate: TransitionPlan,
    *,
    audio_a: Any,
    audio_b: Any,
    sample_rate: int,
    context_seconds: float,
    policy: MeasurementPolicy,
    render_context: TransitionRenderContext | None,
    capture_render_embeddings: bool,
) -> tuple[TransitionMeasurement, bool, str | None]:
    """Render/measure one signed candidate or reuse identical prior evidence."""

    # A cache hit never skips source provenance validation. If any master,
    # stem, layer or program changed, this rejects the plan before lookup.
    validate_execution_plan(candidate)
    cache_key = rehearsal_cache_key(
        candidate,
        sample_rate=sample_rate,
        context_seconds=context_seconds,
        policy=policy,
        render_context=render_context,
        capture_render_embedding=capture_render_embeddings,
    )
    cached = load_cached_rehearsal(cache_key)
    if cached is not None:
        measurement, cached_metadata = cached
        candidate.metadata = {
            **candidate.metadata,
            **cached_metadata,
            "rehearsal_cache_hit": True,
            "rehearsal_cache_key": cache_key,
        }
        return measurement, True, cache_key

    rendered = render_transition_audio_from_loaded(
        audio_a,
        audio_b,
        candidate,
        sample_rate=sample_rate,
        pre_roll_seconds=context_seconds,
        post_roll_seconds=context_seconds,
        render_context=render_context,
    )
    measurement = measure_transition_audio(
        pre=rendered.pre,
        transition=rendered.transition,
        post=rendered.post,
        sample_rate=sample_rate,
        plan=candidate,
        policy=policy,
    )
    musical = measure_musical_execution(
        pre=rendered.pre,
        transition=rendered.transition,
        post=rendered.post,
        sample_rate=sample_rate,
        plan=candidate,
    )
    measurement.rhythm_coherence = musical.rhythm_coherence
    measurement.transient_clarity = musical.transient_clarity
    measurement.groove_continuity = musical.groove_continuity
    measurement.harmonic_continuity = musical.harmonic_continuity
    measurement.energy_intent_fit = musical.energy_intent_fit
    measurement.drop_payoff = musical.drop_payoff
    measurement.musical_overall = musical.overall
    measurement.hard_failures.extend(musical.hard_failures)
    measurement.warnings.extend(musical.warnings)

    if capture_render_embeddings:
        rendered_embedding = embed_rendered_transition(rendered.preview, sample_rate)
        candidate.metadata = {
            **candidate.metadata,
            "rendered_transition_embedding": [
                round(float(value), 7) for value in rendered_embedding.vector
            ],
            "rendered_transition_embedding_backend": rendered_embedding.backend,
            "rendered_transition_embedding_model": rendered_embedding.model,
            "rendered_transition_embedding_fingerprint": rendered_embedding.fingerprint,
            "rendered_transition_embedding_captured": True,
        }
    else:
        candidate.metadata = {
            **candidate.metadata,
            "rendered_transition_embedding_captured": False,
            "rendered_transition_embedding_backend": "not_required",
        }

    stem_safety = measure_stem_transition_safety(candidate, sample_rate=sample_rate)
    if stem_safety is not None:
        measurement.vocal_cleanliness = float(stem_safety["vocal_cleanliness"])
        measurement.stem_bass_cleanliness = float(stem_safety["stem_bass_cleanliness"])
        measurement.stem_evidence = True
        measurement.overall = round(
            0.82 * measurement.overall
            + 0.09 * measurement.vocal_cleanliness
            + 0.09 * measurement.stem_bass_cleanliness,
            4,
        )
        if candidate.transition_bars >= 8 and measurement.vocal_cleanliness < 0.18:
            measurement.hard_failures.append(
                "Stem analysis found severe vocal-on-vocal overlap in a long blend."
            )
        if candidate.transition_bars >= 8 and measurement.stem_bass_cleanliness < 0.16:
            measurement.hard_failures.append(
                "Stem analysis found severe simultaneous bass activity in a long blend."
            )
        elif measurement.stem_bass_cleanliness < 0.35:
            measurement.warnings.append(
                "Stem analysis found elevated simultaneous bass activity."
            )

    cached_metadata = {
        key: value
        for key, value in candidate.metadata.items()
        if key.startswith("rendered_transition_embedding")
    }
    cache_path = store_cached_rehearsal(
        cache_key,
        execution_signature=candidate.execution_signature,
        measurement=measurement,
        metadata=cached_metadata,
    )
    candidate.metadata = {
        **candidate.metadata,
        "rehearsal_cache_hit": False,
        "rehearsal_cache_key": cache_key,
        "rehearsal_cache_path": str(cache_path) if cache_path else None,
    }
    return measurement, False, cache_key


def evaluate_transition_candidates(
    plans: Sequence[TransitionPlan],
    *,
    config: QualityGateConfig,
    analytical_scores: Mapping[str, float] | None = None,
    policy: MeasurementPolicy = DEFAULT_MEASUREMENT_POLICY,
    context: dict[str, Any] | None = None,
    render_context: TransitionRenderContext | None = None,
) -> MeasuredSelection:
    """Render top-ranked candidates, measure them, and keep the best result.

    This belongs at final pair/sequence selection time. The O(N²) graph stage
    remains analytical so the quality loop does not erase Phase 0 performance.
    """
    if not plans:
        raise MeasuredTransitionError("No transition candidates were provided to the quality gate.")
    normalized = config.normalized()
    resolved_render_context = render_context.normalized() if render_context is not None else None
    if resolved_render_context is not None and resolved_render_context.sample_rate != normalized.sample_rate:
        raise ValueError(
            "Quality-gate sample rate must match the exact transition render context."
        )
    if not normalized.enabled:
        selected = copy.deepcopy(max(plans, key=lambda plan: plan.score))
        return MeasuredSelection(selected=selected, evaluated=[selected])

    track_pairs = {(plan.track_a, plan.track_b) for plan in plans}
    if len(track_pairs) != 1:
        raise ValueError("Measured candidate selection requires one directional track pair.")
    score_map = dict(analytical_scores or {})
    candidates = _style_diverse_top_k(plans, score_map, normalized.top_k)
    track_a_path, track_b_path = next(iter(track_pairs))
    audio_a, sample_rate_a = _load_rehearsal_audio(track_a_path, normalized.sample_rate)
    audio_b, sample_rate_b = _load_rehearsal_audio(track_b_path, normalized.sample_rate)
    if sample_rate_a != normalized.sample_rate or sample_rate_b != normalized.sample_rate:
        raise RuntimeError("Unexpected sample-rate mismatch while rendering quality candidates.")

    evaluated: list[TransitionPlan] = []
    passing: list[TransitionPlan] = []
    capture_render_embeddings = _capture_render_embeddings(context)
    for plan in candidates:
        candidate = copy.deepcopy(plan)
        # `analytical_scores` is a search-order utility and may include cue or
        # set-position bonuses. The persisted analytical score must remain the
        # actual transition-plan score so reports and future learning data have
        # a stable meaning.
        base_score = max(0.0, min(1.0, float(plan.score)))
        try:
            measurement, _, _ = _rehearse_one_candidate(
                candidate,
                audio_a=audio_a,
                audio_b=audio_b,
                sample_rate=normalized.sample_rate,
                context_seconds=normalized.context_seconds,
                policy=policy,
                render_context=resolved_render_context,
                capture_render_embeddings=capture_render_embeddings,
            )
        except Exception as exc:
            measurement = TransitionMeasurement(
                version=MEASUREMENT_VERSION,
                sample_rate=normalized.sample_rate,
                duration_seconds=0.0,
                overall=0.0,
                hard_failures=[
                    f"Candidate rendering or measurement failed: {type(exc).__name__}: {exc}"
                ],
            )
        candidate.analytical_score = round(base_score, 4)
        candidate.measured_score = round(measurement.overall, 4)
        candidate.measurement = measurement
        candidate.quality_checked = True
        candidate.measurement_passed = (
            not measurement.hard_failures
            and measurement.overall >= normalized.minimum_measured_score
        )
        candidate.score = round(
            normalized.analytical_weight * base_score
            + normalized.measured_weight * measurement.overall,
            4,
        )
        naturalness = annotate_transition_naturalness(candidate)
        natural_selection_score = naturalness_blended_score(candidate.score, naturalness)
        candidate.metadata = {
            **candidate.metadata,
            "naturalness_selection_score": round(natural_selection_score, 6),
            "measurement_version": MEASUREMENT_VERSION,
            "measurement_sample_rate": normalized.sample_rate,
            "measurement_top_k": normalized.top_k,
            "measurement_candidate_count": len(candidates),
            "measurement_passed": candidate.measurement_passed,
            "render_context": (
                {
                    "sample_rate": resolved_render_context.sample_rate,
                    "gain_a": round(resolved_render_context.gain_a, 8),
                    "gain_b": (
                        round(resolved_render_context.gain_b, 8)
                        if resolved_render_context.gain_b is not None
                        else None
                    ),
                    "target_loudness_dbfs": resolved_render_context.target_loudness_dbfs,
                    "context_seconds": normalized.context_seconds,
                    "exact_set_context": True,
                }
                if resolved_render_context is not None
                else {
                    "sample_rate": normalized.sample_rate,
                    "context_seconds": normalized.context_seconds,
                    "exact_set_context": False,
                }
            ),
            "quality_loop": {
                "measurement_version": MEASUREMENT_VERSION,
                "analytical_score": candidate.analytical_score,
                "measured_score": candidate.measured_score,
                "combined_score": candidate.score,
                "naturalness_score": naturalness.score,
                "naturalness_selection_score": round(natural_selection_score, 6),
                "gate_passed": candidate.measurement_passed,
            },
        }
        candidate.rationale = list(candidate.rationale)
        candidate.rationale.append(
            f"Rendered-audio quality measured {measurement.overall:.0%} from loudness, spectrum, bass, phase, silence, and artifact checks."
        )
        candidate.rationale.append(
            f"Transition naturalness measured {naturalness.score:.0%} from phrase alignment, structural flow, duration, tempo strain, source cleanliness, and rendered musical continuity."
        )
        candidate.rationale.extend(measurement.warnings)
        if measurement.hard_failures:
            candidate.warning = measurement.hard_failures[0]
            candidate.quality = "poor"
            candidate.requires_confirmation = True
        elif candidate.measurement_passed:
            if candidate.transition_mode == "emergency_cut":
                candidate.quality = "risky"
            elif (
                candidate.score >= 0.72
                and measurement.overall >= 0.68
                and not candidate.requires_confirmation
            ):
                candidate.quality = "good"
            elif candidate.score >= 0.48:
                candidate.quality = "risky"
            else:
                candidate.quality = "poor"
            passing.append(candidate)
        else:
            candidate.quality = "poor" if measurement.overall < 0.32 else "risky"
            candidate.requires_confirmation = True
            candidate.warning = candidate.warning or (
                "Rendered transition did not pass the automatic quality threshold."
            )
        evaluated.append(candidate)

    evaluated.sort(
        key=lambda plan: (
            plan.measurement_passed,
            plan.score,
            plan.measured_score,
            plan.analytical_score,
            plan.execution_signature,
        ),
        reverse=True,
    )
    passing.sort(
        key=lambda plan: (
            float(plan.metadata.get("naturalness_selection_score", plan.score)),
            plan.score,
            plan.measured_score,
            plan.analytical_score,
            plan.execution_signature,
        ),
        reverse=True,
    )
    if passing:
        # DJAI 3.1 taste layer: rerank only candidates that already passed the
        # deterministic render/measure gate. Technical safety remains supreme.
        from .intelligence.transition_ranker import rank_transition_plans
        passing = rank_transition_plans(passing)
    selected = passing[0] if passing else evaluated[0]
    for rank, candidate in enumerate(evaluated, start=1):
        candidate.measurement_rank = rank
        candidate.metadata["measurement_rank"] = rank
        candidate.metadata["measurement_selected"] = (
            bool(passing) and candidate.execution_signature == selected.execution_signature
        )

    # SAUND.AI 1.1.1 Taste Learning Loop: persist the structured candidate
    # population from real rehearsals. This is deliberately best-effort and
    # audio-free; a telemetry filesystem problem must never block delivery.
    try:
        from .training.taste_evidence import capture_rehearsal_evidence

        evidence_path = capture_rehearsal_evidence(
            evaluated,
            selected_signature=(selected.execution_signature if passing else None),
            context=context,
        )
        if evidence_path is not None:
            selected.metadata["taste_evidence_path"] = str(evidence_path)
    except Exception as exc:
        selected.metadata["taste_evidence_error"] = f"{type(exc).__name__}: {exc}"

    log_destination: Path | None = None
    log_error: str | None = None
    if normalized.log_path:
        records = [
            measurement_record(
                candidate,
                accepted=bool(passing)
                and candidate.execution_signature == selected.execution_signature,
                rank=candidate.measurement_rank,
                candidate_count=len(evaluated),
                context=context,
            )
            for candidate in evaluated
        ]
        try:
            log_destination = append_measurement_records(normalized.log_path, records)
            selected.metadata["measurement_log_path"] = str(log_destination)
        except OSError as exc:
            # Measurement logging is valuable training telemetry, but a local
            # filesystem problem must not invalidate an otherwise usable mix.
            log_error = f"{type(exc).__name__}: {exc}"
            selected.metadata["measurement_log_error"] = log_error

    if not passing:
        details = "; ".join(selected.measurement.hard_failures if selected.measurement else [])
        raise MeasuredTransitionError(
            "None of the rendered transition candidates passed the audio quality gate."
            + (f" Best candidate: {details}" if details else ""),
            best=selected,
            evaluated=evaluated,
            log_path=log_destination,
        )

    return MeasuredSelection(
        selected=selected,
        evaluated=evaluated,
        log_path=log_destination,
        log_error=log_error,
        rejected_count=sum(not candidate.measurement_passed for candidate in evaluated),
        metadata={
            "measurement_version": MEASUREMENT_VERSION,
            "top_k": normalized.top_k,
            "sample_rate": normalized.sample_rate,
            "evaluated": len(evaluated),
        },
    )


# Backward-compatible Phase 1 names used by set planning and the API.
QualityLoopConfig = QualityGateConfig


def measure_and_select_transition(
    track_a: object,
    track_b: object,
    plans: Sequence[TransitionPlan],
    *,
    config: QualityGateConfig,
    context: dict[str, Any] | None = None,
    render_context: TransitionRenderContext | None = None,
) -> MeasuredSelection:
    if render_context is None and hasattr(track_a, "loudness_dbfs") and hasattr(track_b, "loudness_dbfs"):
        render_context = TransitionRenderContext(
            sample_rate=config.normalized().sample_rate,
            gain_a=normalization_gain(float(getattr(track_a, "loudness_dbfs"))),
            gain_b=normalization_gain(float(getattr(track_b, "loudness_dbfs"))),
        )
    scores = {
        plan.execution_signature: float(
            plan.metadata.get("selection_utility", plan.score)
        )
        for plan in plans
    }
    return evaluate_transition_candidates(
        plans,
        config=config,
        analytical_scores=scores,
        context={"mode": "set_sequence", **dict(context or {})},
        render_context=render_context,
    )
