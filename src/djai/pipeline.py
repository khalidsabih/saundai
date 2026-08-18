from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cache import analyze_track_cached
from .decision import validate_execution_plan
from .models import TrackAnalysis, TransitionPlan
from .learning import maybe_refit_scoring_model
from .planner import (
    plan_transition,
    plan_transition_alternatives,
    plan_transition_automatic,
    plan_transition_measured,
)
from .renderer import render_full_mix, render_transition
from .set_models import SetPlan
from .set_planner import plan_folder, plan_manual_paths, save_set_plan
from .set_renderer import prepare_render_sequence, render_set, render_set_previews
from .layers import enrich_analysis_with_performance_layers


@dataclass(slots=True)
class PairPipelineResult:
    track_a: TrackAnalysis
    track_b: TrackAnalysis
    plan: TransitionPlan
    preview_path: Path | None = None
    full_mix_path: Path | None = None


@dataclass(slots=True)
class SetPipelineResult:
    plan: SetPlan
    plan_path: Path | None = None
    preview_paths: list[Path] | None = None
    preview_report_path: Path | None = None
    audio_path: Path | None = None
    render_report_path: Path | None = None


def run_pair_pipeline(
    track_a_path: str | Path,
    track_b_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    tempo_policy: str = "safe",
    style: str = "safe",
    force: bool = False,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    render_preview: bool = True,
    render_full: bool = False,
    measure_quality: bool = True,
    measurement_top_k: int = 4,
    measurement_sample_rate: int = 22_050,
    measurement_log: str | None = None,
    automatic_delivery: bool = True,
) -> PairPipelineResult:
    track_a = analyze_track_cached(track_a_path, cache_dir=cache_dir, use_cache=use_cache).analysis
    track_b = analyze_track_cached(track_b_path, cache_dir=cache_dir, use_cache=use_cache).analysis
    enrich_analysis_with_performance_layers(track_a)
    enrich_analysis_with_performance_layers(track_b)
    if measure_quality and automatic_delivery:
        plan = plan_transition_automatic(
            track_a,
            track_b,
            tempo_policy=tempo_policy,
            measurement_top_k=measurement_top_k,
            measurement_sample_rate=measurement_sample_rate,
            measurement_log_path=measurement_log,
        )
    elif measure_quality:
        plan = plan_transition_measured(
            track_a,
            track_b,
            tempo_policy=tempo_policy,
            style=style,
            force=force,
            measurement_top_k=measurement_top_k,
            measurement_sample_rate=measurement_sample_rate,
            measurement_log_path=measurement_log,
        )
    else:
        plan = plan_transition(
            track_a,
            track_b,
            tempo_policy=tempo_policy,
            style=style,
            force=force,
        )
    validate_execution_plan(plan)
    result = PairPipelineResult(track_a=track_a, track_b=track_b, plan=plan)
    if output_dir is not None:
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if render_preview:
            result.preview_path = render_transition(plan, root / "transition-preview.wav")
        if render_full:
            result.full_mix_path = render_full_mix(plan, root / "two-track-mix.wav")
    return result


def run_pair_alternatives(
    track_a_path: str | Path,
    track_b_path: str | Path,
    *,
    tempo_policy: str = "safe",
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
) -> list[TransitionPlan]:
    track_a = analyze_track_cached(track_a_path, cache_dir=cache_dir, use_cache=use_cache).analysis
    track_b = analyze_track_cached(track_b_path, cache_dir=cache_dir, use_cache=use_cache).analysis
    enrich_analysis_with_performance_layers(track_a)
    enrich_analysis_with_performance_layers(track_b)
    plans = plan_transition_alternatives(
        track_a,
        track_b,
        force=True,
        tempo_policy=tempo_policy,
    )
    for plan in plans:
        validate_execution_plan(plan)
    return plans


def run_set_pipeline(
    source: str | Path | Iterable[str | Path],
    *,
    output_dir: str | Path,
    manual: bool = False,
    tempo_policy: str = "safe",
    preferred_style: str = "auto",
    journey: str = "balanced",
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    recursive: bool = False,
    minimum_body_seconds: float = 30.0,
    minimum_body_fraction: float = 0.35,
    render_previews: bool = True,
    render_audio: bool = True,
    analysis_workers: int | None = None,
    graph_workers: int | None = None,
    measure_transitions: bool = True,
    measurement_top_k: int = 4,
    measurement_log: str | None = "ratings/automatic-measurements.jsonl",
    automatic_delivery: bool = True,
    listener: str = "local",
) -> SetPipelineResult:
    if manual:
        if isinstance(source, (str, Path)):
            raise ValueError("Manual set mode expects an ordered iterable of track paths.")
        plan = plan_manual_paths(
            source,
            cache_dir=cache_dir,
            use_cache=use_cache,
            tempo_policy=tempo_policy,
            preferred_style=preferred_style,
            journey=journey,  # type: ignore[arg-type]
            minimum_body_seconds=minimum_body_seconds,
            minimum_body_fraction=minimum_body_fraction,
            analysis_workers=analysis_workers,
            graph_workers=graph_workers,
            measure_transitions=measure_transitions,
            measurement_top_k=measurement_top_k,
            measurement_log=measurement_log,
            automatic_delivery=automatic_delivery,
            listener=listener,
        )
    else:
        if not isinstance(source, (str, Path)):
            raise ValueError("Automatic set mode expects a folder path.")
        plan = plan_folder(
            source,
            cache_dir=cache_dir,
            use_cache=use_cache,
            recursive=recursive,
            tempo_policy=tempo_policy,
            preferred_style=preferred_style,
            journey=journey,  # type: ignore[arg-type]
            minimum_body_seconds=minimum_body_seconds,
            minimum_body_fraction=minimum_body_fraction,
            analysis_workers=analysis_workers,
            graph_workers=graph_workers,
            measure_transitions=measure_transitions,
            measurement_top_k=measurement_top_k,
            measurement_log=measurement_log,
            automatic_delivery=automatic_delivery,
            listener=listener,
        )
    if automatic_delivery and measurement_log:
        try:
            updated_model = maybe_refit_scoring_model(
                measurement_path=measurement_log,
            )
            if updated_model is not None:
                plan.metadata["learning_model_updated"] = str(updated_model)
        except Exception as exc:
            plan.metadata["learning_model_update_error"] = f"{type(exc).__name__}: {exc}"
    prepare_render_sequence(plan)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = SetPipelineResult(plan=plan)
    result.plan_path = save_set_plan(plan, root / "set-plan.json")
    if render_previews:
        result.preview_paths, result.preview_report_path = render_set_previews(
            plan, root / "previews"
        )
    if render_audio:
        result.audio_path, result.render_report_path = render_set(
            plan, root / "full-set.wav"
        )
    return result
