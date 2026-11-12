from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze_track
from .demo import generate_demo_pair, generate_demo_set
from .genre import train_from_manifest
from .learning import fit_scoring_model
from .live import play_live
from .models import TrackAnalysis, TransitionPlan
from .planner import (
    DifficultTransitionError,
    plan_transition,
    plan_transition_alternatives,
    plan_transition_measured,
)
from .ratings import append_listening_rating
from .renderer import render_full_mix, render_transition
from .set_models import SetPlan
from .set_planner import load_set_plan, plan_folder, plan_manual_paths, save_set_plan
from .set_renderer import render_set, render_set_previews
from .stems import separate_stems
from .layers import register_performance_bundle, resolve_performance_layers
from .intelligence.lab import create_listening_session
from .intelligence.transition_ranker import train_transition_ranker
from .intelligence.benchmark import benchmark_brain
from .intelligence.dataset import create_benchmark_manifest, benchmark_manifest_stats
from .intelligence.edm_cue import import_edm_cue_annotations, edm_cue_stats, edm_cue_source_manifest
from .intelligence.cue_detr import predict_cue_points, cue_model_fingerprint
from .training.provenance import audit_dataset_registry
from .training.curriculum import training_curriculum
from .training.synthetic_failures import generate_synthetic_failure_set
from .training.research_synthetic import (
    create_research_synthetic_snapshot,
    research_synthetic_status,
    train_research_synthetic_shadow,
    validate_research_synthetic_dataset,
)
from .phase16_validation import validate_real_music_library
from .training.expert_dataset import expert_dataset_stats
from .evaluation.arena import ArenaVote, append_arena_vote, arena_report
from .evaluation.human_release_gate import (
    DEFAULT_MANIFEST as HUMAN_RELEASE_DEFAULT_MANIFEST,
    HumanReleaseThresholds,
    create_human_release_session,
    human_release_report,
    lock_human_release_benchmark,
    record_human_release_vote,
)
from .intelligence.expert_policy import train_expert_policy
from .intelligence.model_registry import registry_status
from .intelligence.model_bootstrap import bootstrap_models
from .training.neural_heads import promote_model, train_neural_head
from .intelligence.governance import load_arena_evidence, load_human_release_evidence
from .training.lab_dataset import create_training_snapshot, train_shadow_ranker, training_lab_status
from .training.taste_learning import train_taste_learning_heads, taste_learning_status
from .training.real_taste import real_taste_status, train_real_taste_from_snapshot
from .training.source_catalog import fetch_source_asset, source_catalog
from .training.dataset_builder import (
    create_bootstrap_corpus,
    dataset_builder_status,
    generate_weak_labels,
    import_source_dataset,
    train_bootstrap_shadow_models,
)
from .training.teacher_consensus import build_consensus_file
from .training.teacher_governance import teacher_data_status
from .training.teacher_quality import (
    build_hard_negative_queue,
    build_human_review_queue,
    build_teacher_calibration,
    write_balanced_scenarios,
)
from .training.teacher_dataset import (
    TeacherScenario,
    append_teacher_judgment,
    load_scenarios,
    scenarios_from_rehearsal_evidence,
    teacher_prompt_payload,
    validate_teacher_judgment,
    write_scenarios,
)
from .environment import current_environment


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False)
    if output is None:
        print(text)
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Report: {path}", file=sys.stderr)


def _parse_time_value(value: str) -> float:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("time value cannot be empty")
    try:
        if ":" not in text:
            seconds = float(text)
        else:
            parts = text.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            seconds = 0.0
            for number in (float(part) for part in parts):
                if number < 0:
                    raise ValueError
                seconds = seconds * 60.0 + number
        if seconds < 0:
            raise ValueError
        return seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use seconds, MM:SS, or HH:MM:SS (for example 168, 2:48, or 1:02:03)"
        ) from exc


class _ConsoleProgress:
    def __init__(self) -> None:
        self.last: dict[str, int] = {}

    def __call__(self, stage: str, current: int, total: int, message: str) -> None:
        percent = 100 if total <= 0 else int(100 * current / total)
        bucket = percent // 10
        if self.last.get(stage) == bucket and current != total:
            return
        self.last[stage] = bucket
        print(f"[{stage}] {percent:3d}% {message}", file=sys.stderr)


def _add_pair_planning_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_style: bool = True,
) -> None:
    parser.add_argument("track_a", help="Outgoing track")
    parser.add_argument("track_b", help="Incoming track")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Confirm an unsafe score, weak beat grid, or experimental large stretch",
    )
    parser.add_argument(
        "--tempo-policy",
        choices=("safe", "stretch", "cut"),
        default="safe",
        help="safe=normal beatmatch/cut; stretch=explicit BPM adjustment; cut=native-tempo cut",
    )
    if include_style:
        parser.add_argument(
            "--style",
            choices=("safe", "creative", "emergency"),
            default="safe",
        )
    parser.add_argument("--bars", type=int, choices=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--exit-time", type=_parse_time_value)
    parser.add_argument("--entry-time", type=_parse_time_value)
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip top-K rendered-audio measurement for this pair",
    )
    parser.add_argument(
        "--measurement-top-k",
        type=int,
        default=4,
        choices=range(1, 17),
        metavar="1-16",
        help="Number of analytical candidates to render and measure",
    )
    parser.add_argument(
        "--measurement-log",
        default="ratings/automatic-measurements.jsonl",
        help="Append-only JSONL path for automatic transition measurements",
    )


def _add_transition_arguments(parser: argparse.ArgumentParser) -> None:
    _add_pair_planning_arguments(parser)
    parser.add_argument("-o", "--output", default="saund_full_mix.wav")
    parser.add_argument("--report")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--pre-roll", type=float, default=8.0)
    parser.add_argument("--post-roll", type=float, default=8.0)


def _add_set_planning_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tempo-policy", choices=("safe", "stretch", "cut"), default="safe")
    parser.add_argument(
        "--transition-style",
        choices=("auto", "safe", "creative", "emergency"),
        default="auto",
    )
    parser.add_argument(
        "--journey",
        choices=("balanced", "smooth", "build", "warmup", "peak", "sunset"),
        default="balanced",
    )
    parser.add_argument("--listener", default="local", help="Listener profile used for personalization")
    parser.add_argument(
        "--pro",
        action="store_true",
        help="Keep risk confirmation enabled instead of automatic fallback delivery",
    )
    parser.add_argument(
        "--min-play-seconds",
        type=float,
        default=30.0,
        help="Minimum audible body per normal-length track before its outgoing transition",
    )
    parser.add_argument(
        "--min-play-fraction",
        type=float,
        default=0.35,
        help="Minimum fraction of each normal-length track to preserve (0.10-0.90)",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=None,
        help="Analysis processes: 0/omitted=auto, 1=serial, 2+=explicit",
    )
    parser.add_argument(
        "--graph-workers",
        type=int,
        default=None,
        help="Graph-planning processes: 0/omitted=auto, 1=serial, 2+=explicit",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Skip top-K render-and-measure selection for final set junctions",
    )
    parser.add_argument(
        "--measurement-top-k",
        type=int,
        default=4,
        choices=range(1, 17),
        metavar="1-16",
        help="Number of analytically ranked candidates to render and measure per junction",
    )
    parser.add_argument(
        "--measurement-log",
        default="ratings/automatic-measurements.jsonl",
        help="Append-only JSONL path for automatic transition measurements",
    )


def _analyze_pair(args: argparse.Namespace) -> tuple[TrackAnalysis, TrackAnalysis]:
    print(f"Analyzing outgoing track: {args.track_a}", file=sys.stderr)
    track_a = analyze_track(args.track_a)
    print(f"Analyzing incoming track: {args.track_b}", file=sys.stderr)
    track_b = analyze_track(args.track_b)
    return track_a, track_b


def _analyze_and_plan(args: argparse.Namespace) -> tuple[TrackAnalysis, TrackAnalysis, TransitionPlan]:
    track_a, track_b = _analyze_pair(args)
    no_quality_check = bool(getattr(args, "no_quality_check", False))
    planner = plan_transition if no_quality_check else plan_transition_measured
    kwargs: dict[str, Any] = {
        "force": args.force,
        "requested_bars": args.bars,
        "exit_time_a": args.exit_time,
        "entry_time_b": args.entry_time,
        "tempo_policy": args.tempo_policy,
        "style": args.style,
    }
    if not no_quality_check:
        kwargs.update(
            measurement_top_k=int(getattr(args, "measurement_top_k", 4)),
            measurement_log_path=getattr(
                args,
                "measurement_log",
                "ratings/automatic-measurements.jsonl",
            ),
        )
    plan = planner(track_a, track_b, **kwargs)
    return track_a, track_b, plan


def _report_payload(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    output_wav: str | None = None,
    render_mode: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "djai_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "track_a": track_a.to_dict(include_beats=False),
        "track_b": track_b.to_dict(include_beats=False),
        "transition": plan.to_dict(),
    }
    if output_wav is not None:
        report["render"] = {"mode": render_mode, "output_wav": output_wav}
    return report


def _pair_id(track_a: TrackAnalysis, track_b: TrackAnalysis) -> str:
    content = f"{track_a.path}|{track_b.path}|{track_a.duration_seconds}|{track_b.duration_seconds}|{track_a.bpm}|{track_b.bpm}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _print_plan_summary(plan: TransitionPlan, *, render_mode: str | None = None) -> None:
    print(file=sys.stderr)
    print(f"Alternative: {plan.alternative_id} ({plan.style})", file=sys.stderr)
    print(f"Quality: {plan.quality}; score {plan.score:.0%}; cue confidence {plan.cue_confidence:.0%}", file=sys.stderr)
    print(f"Genre: {plan.genre_pair[0]} -> {plan.genre_pair[1]}", file=sys.stderr)
    print(f"Sections: {plan.exit_section} -> {plan.entry_section}", file=sys.stderr)
    print(f"Intent: {plan.intent} ({plan.intent_score:.0%})", file=sys.stderr)
    print(
        f"BPM: {plan.source_bpm_a:.2f} -> {plan.source_bpm_b:.2f} "
        f"(difference {plan.compatibility.tempo_difference_percent:.2f}%)",
        file=sys.stderr,
    )
    print(
        f"Grid {plan.compatibility.beat_grid:.0%}; boundary {plan.compatibility.boundary:.0%}; "
        f"section {plan.compatibility.section_fit:.0%}; genre {plan.compatibility.genre:.0%}",
        file=sys.stderr,
    )
    print(f"Tempo strategy: {plan.tempo_strategy}; beatmatched: {'yes' if plan.beatmatched else 'no'}", file=sys.stderr)
    if plan.time_stretch_applied:
        print(
            f"Incoming rate {plan.time_stretch_rate_b:.6f}; rendered at {plan.rendered_bpm_b:.2f} BPM",
            file=sys.stderr,
        )
    print(f"Technique: {plan.technique} ({plan.technique_score:.0%})", file=sys.stderr)
    print(
        f"A {plan.exit_time_a:.2f}s -> B {plan.entry_time_b:.2f}s over {plan.transition_bars} bar(s)",
        file=sys.stderr,
    )
    if render_mode:
        print(f"Render mode: {render_mode}", file=sys.stderr)
    if plan.quality_checked:
        print(
            f"Rendered quality: {plan.measured_score or 0.0:.0%} "
            f"(analytical {plan.analytical_score:.0%})",
            file=sys.stderr,
        )
    if plan.compiled_execution:
        print(f"Execution: {plan.execution_signature}", file=sys.stderr)
    for reason in plan.rationale:
        print(f"- {reason}", file=sys.stderr)
    if plan.warning:
        print(f"Warning: {plan.warning}", file=sys.stderr)


def _print_set_summary(plan: SetPlan) -> None:
    print("\nSAUND.AI SET PLAN", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    for position, track_index in enumerate(plan.order, start=1):
        track = plan.tracks[track_index]
        print(
            f"{position:02d}  {track.title[:38]:38} {track.bpm:6.1f} BPM  "
            f"{track.key.camelot:>3}  {track.genre.label}",
            file=sys.stderr,
        )
        if position <= len(plan.transitions):
            edge = plan.transitions[position - 1]
            print(
                f"    -> {edge.plan.intent}/{edge.plan.technique}  "
                f"{edge.quality}  {edge.plan.score:.0%}",
                file=sys.stderr,
            )
    print("-" * 72, file=sys.stderr)
    print(f"Average objective: {plan.optimized_score / max(1, len(plan.transitions)):.3f}", file=sys.stderr)
    print(f"Estimated duration: {plan.estimated_duration_seconds / 60.0:.1f} min", file=sys.stderr)
    print(f"Requires confirmation: {'yes' if plan.requires_confirmation else 'no'}", file=sys.stderr)
    if plan.confirmation_token:
        print(f"Confirmation token: {plan.confirmation_token}", file=sys.stderr)


def _handle_difficult_transition(exc: DifficultTransitionError) -> int:
    print(f"SAUND.AI stopped: {exc}", file=sys.stderr)
    return 2


def _run_plan(args: argparse.Namespace) -> int:
    try:
        track_a, track_b, plan = _analyze_and_plan(args)
    except DifficultTransitionError as exc:
        return _handle_difficult_transition(exc)
    _write_json(_report_payload(track_a, track_b, plan), args.output)
    _print_plan_summary(plan)
    return 0


def _run_transition(args: argparse.Namespace) -> int:
    try:
        track_a, track_b, plan = _analyze_and_plan(args)
    except DifficultTransitionError as exc:
        return _handle_difficult_transition(exc)
    if args.preview:
        output = render_transition(plan, args.output, pre_roll_seconds=args.pre_roll, post_roll_seconds=args.post_roll)
        mode = "preview"
    else:
        output = render_full_mix(plan, args.output)
        mode = "full_mix"
    report_path = args.report or str(Path(output).with_suffix(".json"))
    _write_json(_report_payload(track_a, track_b, plan, output_wav=str(output), render_mode=mode), report_path)
    _print_plan_summary(plan, render_mode=mode)
    print(f"Audio output: {output}", file=sys.stderr)
    return 0


def _run_alternatives(args: argparse.Namespace) -> int:
    try:
        track_a, track_b = _analyze_pair(args)
        plans = plan_transition_alternatives(
            track_a,
            track_b,
            force=args.force,
            requested_bars=args.bars,
            exit_time_a=args.exit_time,
            entry_time_b=args.entry_time,
            tempo_policy=args.tempo_policy,
        )
    except DifficultTransitionError as exc:
        return _handle_difficult_transition(exc)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for plan in plans:
        preview_path: str | None = None
        if not args.no_render:
            preview_path = str(
                render_transition(
                    plan,
                    output_dir / f"{plan.alternative_id}.wav",
                    pre_roll_seconds=args.pre_roll,
                    post_roll_seconds=args.post_roll,
                )
            )
        items.append({"id": plan.alternative_id, "style": plan.style, "preview_wav": preview_path, "plan": plan.to_dict()})
        _print_plan_summary(plan, render_mode="preview" if preview_path else None)
    report = {
        "djai_version": __version__,
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_id": _pair_id(track_a, track_b),
        "track_a": track_a.to_dict(include_beats=False),
        "track_b": track_b.to_dict(include_beats=False),
        "alternatives": items,
    }
    _write_json(report, args.report or str(output_dir / "alternatives.json"))
    return 0


def _run_set_plan(args: argparse.Namespace) -> int:
    plan = plan_folder(
        args.folder,
        use_cache=not args.no_cache,
        recursive=args.recursive,
        tempo_policy=args.tempo_policy,
        preferred_style=args.transition_style,
        journey=args.journey,
        start_track=args.start_track,
        end_track=args.end_track,
        minimum_body_seconds=args.min_play_seconds,
        minimum_body_fraction=args.min_play_fraction,
        progress=_ConsoleProgress(),
        analysis_workers=args.analysis_workers,
        graph_workers=args.graph_workers,
        measure_transitions=not args.no_quality_check,
        measurement_top_k=args.measurement_top_k,
        measurement_log=args.measurement_log,
        automatic_delivery=not args.pro,
        listener=args.listener,
    )
    output = save_set_plan(plan, args.output)
    _print_set_summary(plan)
    print(f"Set plan: {output}", file=sys.stderr)
    return 0


def _run_manual_plan(args: argparse.Namespace) -> int:
    plan = plan_manual_paths(
        args.tracks,
        use_cache=not args.no_cache,
        tempo_policy=args.tempo_policy,
        preferred_style=args.transition_style,
        journey=args.journey,
        minimum_body_seconds=args.min_play_seconds,
        minimum_body_fraction=args.min_play_fraction,
        progress=_ConsoleProgress(),
        analysis_workers=args.analysis_workers,
        graph_workers=args.graph_workers,
        measure_transitions=not args.no_quality_check,
        measurement_top_k=args.measurement_top_k,
        measurement_log=args.measurement_log,
        automatic_delivery=False,
        listener=args.listener,
    )
    output = save_set_plan(plan, args.output)
    _print_set_summary(plan)
    print(f"Manual plan: {output}", file=sys.stderr)
    return 2 if plan.requires_confirmation else 0


def _require_set_confirmation(plan: SetPlan, confirmed: bool) -> None:
    if plan.requires_confirmation and not confirmed:
        _print_set_summary(plan)
        raise RuntimeError(
            "This set contains risky transitions. Review the plan and run again with --confirm-risky."
        )
    if confirmed:
        plan.confirmed = True


def _run_set_render(args: argparse.Namespace) -> int:
    plan = load_set_plan(args.plan)
    _require_set_confirmation(plan, args.confirm_risky)
    output, report = render_set(plan, args.output, report_path=args.report, progress=_ConsoleProgress())
    print(f"Set audio: {output}", file=sys.stderr)
    print(f"Render report: {report}", file=sys.stderr)
    return 0


def _run_set(args: argparse.Namespace) -> int:
    plan = plan_folder(
        args.folder,
        use_cache=not args.no_cache,
        recursive=args.recursive,
        tempo_policy=args.tempo_policy,
        preferred_style=args.transition_style,
        journey=args.journey,
        start_track=args.start_track,
        end_track=args.end_track,
        minimum_body_seconds=args.min_play_seconds,
        minimum_body_fraction=args.min_play_fraction,
        progress=_ConsoleProgress(),
        analysis_workers=args.analysis_workers,
        graph_workers=args.graph_workers,
        measure_transitions=not args.no_quality_check,
        measurement_top_k=args.measurement_top_k,
        measurement_log=args.measurement_log,
        automatic_delivery=not args.pro,
        listener=args.listener,
    )
    plan_output = Path(args.plan_output).expanduser().resolve()
    save_set_plan(plan, plan_output)
    _require_set_confirmation(plan, args.confirm_risky)
    output, report = render_set(plan, args.output, report_path=args.report, progress=_ConsoleProgress())
    _print_set_summary(plan)
    print(f"Set plan: {plan_output}\nSet audio: {output}\nRender report: {report}", file=sys.stderr)
    return 0


def _run_manual_set(args: argparse.Namespace) -> int:
    plan = plan_manual_paths(
        args.tracks,
        use_cache=not args.no_cache,
        tempo_policy=args.tempo_policy,
        preferred_style=args.transition_style,
        journey=args.journey,
        minimum_body_seconds=args.min_play_seconds,
        minimum_body_fraction=args.min_play_fraction,
        progress=_ConsoleProgress(),
        analysis_workers=args.analysis_workers,
        graph_workers=args.graph_workers,
        measure_transitions=not args.no_quality_check,
        measurement_top_k=args.measurement_top_k,
        measurement_log=args.measurement_log,
        automatic_delivery=False,
        listener=args.listener,
    )
    save_set_plan(plan, args.plan_output)
    _require_set_confirmation(plan, args.confirm_risky)
    output, report = render_set(plan, args.output, report_path=args.report, progress=_ConsoleProgress())
    _print_set_summary(plan)
    print(f"Set audio: {output}\nRender report: {report}", file=sys.stderr)
    return 0


def _run_set_previews(args: argparse.Namespace) -> int:
    plan = load_set_plan(args.plan)
    outputs, report = render_set_previews(
        plan,
        args.output_dir,
        pre_roll_seconds=args.pre_roll,
        post_roll_seconds=args.post_roll,
    )
    print(f"Rendered {len(outputs)} junction previews. Index: {report}", file=sys.stderr)
    return 0


def _run_rate(args: argparse.Namespace) -> int:
    output = append_listening_rating(
        args.report,
        alternative_id=args.alternative,
        transition_index=args.transition_index,
        beat_alignment=args.beat,
        phrase_timing=args.phrase,
        bass_cleanliness=args.bass,
        energy_flow=args.energy,
        cue_quality=args.cue,
        overall=args.overall,
        notes=args.notes,
        listener=args.listener,
        ratings_path=args.ratings_file,
    )
    print(f"Rating saved: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    invoked = Path(sys.argv[0]).name
    program_name = invoked if invoked in {"saund", "djai"} else "saund"
    parser = argparse.ArgumentParser(
        prog=program_name,
        description="SAUND.AI local-first autonomous music performance engine",
    )
    parser.add_argument("--version", action="version", version=f"SAUND.AI {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one audio track")
    analyze.add_argument("track")
    analyze.add_argument("-o", "--output")
    analyze.add_argument("--full-beats", action="store_true")

    pair_plan = sub.add_parser("plan", help="Plan one track-to-track transition")
    _add_pair_planning_arguments(pair_plan)
    pair_plan.add_argument("-o", "--output")

    transition = sub.add_parser("transition", help="Render one pair transition or two-track mix")
    _add_transition_arguments(transition)

    alternatives = sub.add_parser("alternatives", help="Render safe, creative, and emergency alternatives")
    _add_pair_planning_arguments(alternatives, include_style=False)
    alternatives.add_argument("-o", "--output-dir", default="output/alternatives")
    alternatives.add_argument("--report")
    alternatives.add_argument("--pre-roll", type=float, default=10.0)
    alternatives.add_argument("--post-roll", type=float, default=10.0)
    alternatives.add_argument("--no-render", action="store_true")

    set_plan = sub.add_parser("set-plan", help="Optimize a folder and save a complete set plan")
    set_plan.add_argument("folder")
    set_plan.add_argument("-o", "--output", default="output/set-plan.json")
    set_plan.add_argument("--recursive", action="store_true")
    set_plan.add_argument("--start-track")
    set_plan.add_argument("--end-track")
    _add_set_planning_arguments(set_plan)

    set_render = sub.add_parser("set-render", help="Render a saved set plan")
    set_render.add_argument("plan")
    set_render.add_argument("-o", "--output", default="output/full-set.wav")
    set_render.add_argument("--report")
    set_render.add_argument("--confirm-risky", action="store_true")

    set_command = sub.add_parser("set", help="Folder in -> optimized continuous WAV and report")
    set_command.add_argument("folder")
    set_command.add_argument("-o", "--output", default="output/full-set.wav")
    set_command.add_argument("--report")
    set_command.add_argument("--plan-output", default="output/set-plan.json")
    set_command.add_argument("--recursive", action="store_true")
    set_command.add_argument("--start-track")
    set_command.add_argument("--end-track")
    set_command.add_argument("--confirm-risky", action="store_true")
    _add_set_planning_arguments(set_command)

    manual_plan = sub.add_parser("manual-plan", help="Validate a client-supplied track order")
    manual_plan.add_argument("tracks", nargs="+")
    manual_plan.add_argument("-o", "--output", default="output/manual-plan.json")
    _add_set_planning_arguments(manual_plan)

    manual_set = sub.add_parser("manual-set", help="Validate and render a client-supplied order")
    manual_set.add_argument("tracks", nargs="+")
    manual_set.add_argument("-o", "--output", default="output/manual-set.wav")
    manual_set.add_argument("--report")
    manual_set.add_argument("--plan-output", default="output/manual-plan.json")
    manual_set.add_argument("--confirm-risky", action="store_true")
    _add_set_planning_arguments(manual_set)

    previews = sub.add_parser("set-preview", help="Render every junction from a saved plan")
    previews.add_argument("plan")
    previews.add_argument("-o", "--output-dir", default="output/set-previews")
    previews.add_argument("--pre-roll", type=float, default=8.0)
    previews.add_argument("--post-roll", type=float, default=8.0)

    live = sub.add_parser("live", help="Play a saved plan through the host audio device")
    live.add_argument("plan")
    live.add_argument("--confirm-risky", action="store_true")
    live.add_argument("--block-frames", type=int, default=2048)

    stems = sub.add_parser("stems", help="Separate a track into vocals, drums, bass, and other")
    stems.add_argument("track")
    stems.add_argument("--model", default="htdemucs")
    stems.add_argument("--device", default="cpu")
    stems.add_argument("--force", action="store_true")

    layers = sub.add_parser("layers-register", help="Register supplied full/instrumental/vocal versions as one performance track")
    layers.add_argument("full")
    layers.add_argument("--instrumental")
    layers.add_argument("--vocals")

    genre_train = sub.add_parser("genre-train", help="Train the lightweight genre centroids from path,label CSV")
    genre_train.add_argument("manifest")
    genre_train.add_argument("-o", "--output", default="config/genre-model.json")

    refit = sub.add_parser("refit-scoring", help="Fit transition scoring weights from listening ratings")
    refit.add_argument("ratings", nargs="?", default="ratings/listening-tests.jsonl")
    refit.add_argument(
        "--measurements",
        default="ratings/automatic-measurements.jsonl",
        help="Automatic render-measure records used as lower-weight training examples",
    )
    refit.add_argument("-o", "--output", default="config/scoring-model.json")

    benchmark_manifest = sub.add_parser("benchmark-manifest", help="Create a stable train/validation/test manifest for legal benchmark audio")
    benchmark_manifest.add_argument("folder")
    benchmark_manifest.add_argument("-o", "--output", default="benchmarks/tracks.jsonl")
    benchmark_manifest.add_argument("--no-recursive", action="store_true")

    listening = sub.add_parser("listening-session", help="Create blind A/B transition previews for taste training")
    listening.add_argument("track_a")
    listening.add_argument("track_b")
    listening.add_argument("--candidates", type=int, default=4)
    listening.add_argument("-o", "--output-dir", default="output/listening-lab")

    brain_train = sub.add_parser("brain-train", help="Train the pairwise transition preference ranker")
    brain_train.add_argument("--preferences", default="ratings/transition-preferences.jsonl")
    brain_train.add_argument("-o", "--output", default="config/transition-ranker.json")

    brain_benchmark = sub.add_parser("brain-benchmark", help="Evaluate held-out transition-ranker quality")
    brain_benchmark.add_argument("--preferences", default="ratings/transition-preferences.jsonl")
    brain_benchmark.add_argument("--model", default="config/transition-ranker.json")

    edm_import = sub.add_parser("edm-cue-import", help="Normalize the public EDM-CUE expert cue annotations")
    edm_import.add_argument("source", help="Downloaded EDM-CUE CSV, JSON, or JSONL annotation file")
    edm_import.add_argument("-o", "--output", default="datasets/edm-cue/annotations.jsonl")

    edm_stats = sub.add_parser("edm-cue-stats", help="Summarize normalized EDM-CUE annotations")
    edm_stats.add_argument("annotations", nargs="?", default="datasets/edm-cue/annotations.jsonl")

    sub.add_parser("edm-cue-source", help="Show official EDM-CUE paper, project, and model sources")

    cue_predict = sub.add_parser("cue-predict", help="Predict expert-style EDM cue points with the published CUE-DETR checkpoint")
    cue_predict.add_argument("track")
    cue_predict.add_argument("--model", default="disco-eth/cue-detr")
    cue_predict.add_argument("--confidence", type=float, default=0.90)
    cue_predict.add_argument("--minimum-spacing-bars", type=int, default=8)
    cue_predict.add_argument("--device", default="auto")
    cue_predict.add_argument("--allow-download", action="store_true")

    dataset_sources = sub.add_parser("dataset-sources", help="Show SAUND.AI external bootstrap-data source catalog")

    dataset_fetch = sub.add_parser("dataset-fetch", help="Download a configured dataset asset with checksum verification")
    dataset_fetch.add_argument("source", choices=("edm-cue", "unmixdb", "fma", "listenbrainz", "medley2k"))
    dataset_fetch.add_argument("--asset")
    dataset_fetch.add_argument("--output-dir", default="datasets/downloads")
    dataset_fetch.add_argument("--allow-audio-download", action="store_true")
    dataset_fetch.add_argument("--overwrite", action="store_true")

    dataset_import = sub.add_parser("dataset-import", help="Normalize an acquired external dataset into the SAUND.AI bootstrap schema")
    dataset_import.add_argument("source", choices=("edm-cue", "unmixdb", "fma", "listenbrainz", "medley2k"))
    dataset_import.add_argument("path", help="Downloaded archive/file or extracted dataset directory")
    dataset_import.add_argument("-o", "--output")
    dataset_import.add_argument("--limit", type=int)
    dataset_import.add_argument("--rights-cleared", action="store_true", help="Only set after a source-specific production rights review")

    dataset_weak = sub.add_parser("dataset-weak-labels", help="Generate traceable synthetic/weak labels from normalized expert ground truth")
    dataset_weak.add_argument("records", help="Normalized records.jsonl")
    dataset_weak.add_argument("-o", "--output")

    sub.add_parser("dataset-builder-status", help="Show imported sources, weak-label counts, model readiness, and corpus history")

    dataset_build = sub.add_parser("dataset-build", help="Freeze an immutable bootstrap corpus from normalized external sources")
    dataset_build.add_argument("--name", default="bootstrap-v1")
    dataset_build.add_argument("--production", action="store_true", help="Include only records explicitly marked rights-cleared")

    dataset_train = sub.add_parser("dataset-train-bootstrap", help="Train research-only bootstrap shadow priors from an immutable corpus")
    dataset_train.add_argument("--corpus-id", help="Corpus ID; defaults to the latest immutable bootstrap corpus")

    dataset_audit = sub.add_parser("dataset-audit", help="Audit research/production training-data provenance")
    dataset_audit.add_argument("--registry", default="datasets/registry.json")
    dataset_audit.add_argument("--lane", choices=("research", "production"), default="production")

    sub.add_parser("training-curriculum", help="Show the staged professional-DJ-brain training curriculum")

    sub.add_parser("training-lab-status", help="Show first-train data, snapshot, and shadow-model readiness")

    training_snapshot = sub.add_parser("training-snapshot", help="Freeze an immutable model-ready Training Lab dataset")
    training_snapshot.add_argument("--name", default="first-train")

    training_shadow = sub.add_parser("training-shadow-train", help="Train the Global Taste Ranker shadow from the latest immutable snapshot")
    training_shadow.add_argument("--snapshot-id")
    training_shadow.add_argument("--ridge", type=float, default=0.18)

    taste_train = sub.add_parser("training-taste-train", help="Train Global Taste Ranker + Rendered DJ Judge as shadow heads")
    taste_train.add_argument("--snapshot-id")
    taste_train.add_argument("--ridge", type=float, default=0.18)
    sub.add_parser("training-taste-status", help="Show runtime evidence and both taste-learning shadow heads")

    real_taste_train = sub.add_parser("training-real-taste-train", help="Teacher pretrain -> human fine-tune -> held-out human benchmark")
    real_taste_train.add_argument("--snapshot-id")
    real_taste_train.add_argument("--ridge", type=float, default=0.18)
    sub.add_parser("training-real-taste-status", help="Show staged real-taste training and held-out human results")

    release_lock = sub.add_parser("human-release-lock", help="Lock a blind candidate-vs-incumbent human release benchmark")
    release_lock.add_argument("cases", help="JSONL with case_id plus incumbent_audio/candidate_audio")
    release_lock.add_argument("--incumbent", required=True)
    release_lock.add_argument("--candidate", required=True)
    release_lock.add_argument("-o", "--output", default=HUMAN_RELEASE_DEFAULT_MANIFEST)

    release_session = sub.add_parser("human-release-session", help="Open the next blind case from the locked release benchmark")
    release_session.add_argument("--evaluator", required=True)
    release_session.add_argument("--cohort", choices=("general_listener", "experienced_dj", "professional_dj"), required=True)

    release_vote = sub.add_parser("human-release-vote", help="Record one blind human release-gate vote")
    release_vote.add_argument("session_id")
    release_vote.add_argument("choice", choices=("a", "b", "tie", "neither"))
    release_vote.add_argument("--catastrophic-a", action="store_true")
    release_vote.add_argument("--catastrophic-b", action="store_true")
    release_vote.add_argument("--reason", action="append", default=[])

    release_report = sub.add_parser("human-release-report", help="Evaluate the locked human release gate")
    release_report.add_argument("--minimum-cases", type=int, default=150)
    release_report.add_argument("--minimum-covered", type=int, default=150)
    release_report.add_argument("--minimum-decisive", type=int, default=150)
    release_report.add_argument("--minimum-evaluators", type=int, default=3)
    release_report.add_argument("--minimum-professional-votes", type=int, default=20)
    release_report.add_argument("--target-win-rate", type=float, default=0.65)
    release_report.add_argument("--maximum-catastrophic-rate", type=float, default=0.01)
    release_report.add_argument("--maximum-new-catastrophes", type=int, default=0)
    release_report.add_argument("-o", "--output")

    teacher_status = sub.add_parser("teacher-status", help="Show provider-neutral LLM teacher-data status and governance")
    teacher_status.add_argument("--workspace")

    teacher_scenarios = sub.add_parser("teacher-scenarios", help="Generate score-blind LLM teacher scenarios from rehearsal evidence")
    teacher_scenarios.add_argument("evidence", help="Runtime rehearsal candidate-evidence JSONL")
    teacher_scenarios.add_argument("-o", "--output", default="datasets/generated/teacher/scenarios.jsonl")
    teacher_scenarios.add_argument("--pairs-per-event", type=int, default=3)

    teacher_prompt = sub.add_parser("teacher-prompt", help="Render a provider-neutral prompt payload for one teacher scenario")
    teacher_prompt.add_argument("scenarios", help="Teacher scenarios JSONL")
    teacher_prompt.add_argument("scenario_id")
    teacher_prompt.add_argument("-o", "--output")

    teacher_ingest = sub.add_parser("teacher-ingest", help="Validate and append one external LLM teacher judgment")
    teacher_ingest.add_argument("scenarios", help="Teacher scenarios JSONL")
    teacher_ingest.add_argument("scenario_id")
    teacher_ingest.add_argument("judgment", help="JSON file containing choice/ranking/confidence/reasons")
    teacher_ingest.add_argument("--provider", required=True)
    teacher_ingest.add_argument("--model", required=True)
    teacher_ingest.add_argument("--model-version", default="")
    teacher_ingest.add_argument("--temperature", type=float)
    teacher_ingest.add_argument("--seed")
    teacher_ingest.add_argument("-o", "--output", default="datasets/generated/teacher/judgments.jsonl")

    teacher_consensus = sub.add_parser("teacher-consensus", help="Build consensus weak labels from teacher judgments")
    teacher_consensus.add_argument("judgments", help="Teacher judgments JSONL")
    teacher_consensus.add_argument("-o", "--output", default="datasets/generated/teacher/consensus.jsonl")
    teacher_consensus.add_argument("--minimum-teachers", type=int, default=2)
    teacher_consensus.add_argument("--minimum-agreement", type=float, default=0.67)
    teacher_consensus.add_argument("--minimum-confidence", type=float, default=0.55)
    teacher_consensus.add_argument("--minimum-providers", type=int, default=2)
    teacher_consensus.add_argument("--minimum-models", type=int, default=2)
    teacher_consensus.add_argument("--minimum-families", type=int, default=2)
    teacher_consensus.add_argument("--calibration", help="Optional JSON calibration profile")

    teacher_balance = sub.add_parser("teacher-balance", help="Balance teacher scenarios across task/context/style/technique strata")
    teacher_balance.add_argument("scenarios", help="Teacher scenarios JSONL")
    teacher_balance.add_argument("-o", "--output", default="datasets/generated/teacher/scenarios-balanced.jsonl")
    teacher_balance.add_argument("--maximum-per-stratum", type=int, default=200)

    teacher_calibrate = sub.add_parser("teacher-calibrate", help="Calibrate teacher confidence against held-out human/reference labels")
    teacher_calibrate.add_argument("judgments", help="Teacher judgments JSONL")
    teacher_calibrate.add_argument("references", help="JSON object mapping scenario_id to reference choice")
    teacher_calibrate.add_argument("-o", "--output", default="datasets/generated/teacher/calibration.json")

    teacher_review = sub.add_parser("teacher-review-queue", help="Build active-learning queue for uncertain/diverse teacher disagreements")
    teacher_review.add_argument("consensus", help="Teacher consensus JSONL")
    teacher_review.add_argument("-o", "--output", default="datasets/generated/teacher/human-review-queue.jsonl")
    teacher_review.add_argument("--minimum-priority", type=float, default=0.10)

    teacher_hardneg = sub.add_parser("teacher-hard-negatives", help="Build research hard-negative queue from teacher consensus")
    teacher_hardneg.add_argument("consensus", help="Teacher consensus JSONL")
    teacher_hardneg.add_argument("-o", "--output", default="datasets/generated/teacher/hard-negatives.jsonl")

    research_validate = sub.add_parser(
        "research-synthetic-validate",
        help="Validate a research-backed synthetic corpus and its provenance contract",
    )
    research_validate.add_argument("source_dir")
    research_import = sub.add_parser(
        "research-synthetic-import",
        help="Freeze a validated research-synthetic corpus into model-ready grouped datasets",
    )
    research_import.add_argument("source_dir")
    research_import.add_argument("--name", default="research-synthetic-v1")
    sub.add_parser(
        "research-synthetic-status",
        help="Show the latest imported research-synthetic training snapshot",
    )
    research_train = sub.add_parser(
        "research-synthetic-train",
        help="Train a research-only shadow next-track head from the imported synthetic corpus",
    )
    research_train.add_argument("--dataset-id")

    phase16 = sub.add_parser(
        "phase16-validate-real",
        help="Validate the real SAUND set pipeline on a local music folder without granting learned authority",
    )
    phase16.add_argument("folder", help="Folder containing real music for validation")
    phase16.add_argument(
        "--journey",
        action="append",
        choices=("balanced", "smooth", "build", "warmup", "peak", "sunset"),
        help="Journey to validate; repeat for multiple journeys (default: balanced)",
    )
    phase16.add_argument("--recursive", action="store_true")
    phase16.add_argument("--render", action="store_true", help="Render each planned validation set")
    phase16.add_argument("--target-duration-seconds", type=float)
    phase16.add_argument("--peak-fraction", type=float, default=0.65)
    phase16.add_argument("--closing-window-seconds", type=float, default=600.0)
    phase16.add_argument("--ending-energy", type=float, default=0.45)
    phase16.add_argument("--measurement-top-k", type=int, default=4, choices=range(1, 17), metavar="1-16")
    phase16.add_argument("--analysis-workers", type=int, default=1, help="Analysis workers; Phase 16 defaults to serial for reliability")
    phase16.add_argument("--graph-workers", type=int, default=1, help="Graph workers; Phase 16 defaults to serial for reliability")
    phase16.add_argument("--no-cache", action="store_true")
    phase16.add_argument("--listener", default="local")
    phase16.add_argument(
        "--neural-retrieval",
        action="store_true",
        help="Use the latest research-synthetic MLP to filter next-track candidates; broad all-pairs fallback is disabled by default for this experiment",
    )
    phase16.add_argument(
        "--allow-broad-graph-fallback",
        action="store_true",
        help="Permit historical all-pairs graph fallback even when --neural-retrieval is enabled",
    )
    phase16.add_argument("-o", "--output-dir", default="output/phase16-real-music")

    synthetic = sub.add_parser("synthetic-failures", help="Generate labelled technical-failure audio from a rights-cleared render")
    synthetic.add_argument("source")
    synthetic.add_argument("-o", "--output-dir", default="datasets/generated/synthetic-failures")

    expert_stats_parser = sub.add_parser("expert-stats", help="Summarize an expert-decision or demonstration JSONL dataset")
    expert_stats_parser.add_argument("path", nargs="?", default="ratings/expert-decisions.jsonl")

    expert_train_parser = sub.add_parser("expert-train", help="Fit the bounded expert-imitation action prior")
    expert_train_parser.add_argument("path", nargs="?", default="ratings/expert-decisions.jsonl")
    expert_train_parser.add_argument("-o", "--output", default="config/expert-policy.json")

    arena_record = sub.add_parser("arena-record", help="Record a blind champion/challenger SAUND.AI comparison")
    arena_record.add_argument("champion")
    arena_record.add_argument("challenger")
    arena_record.add_argument("choice", choices=("champion", "challenger", "tie", "neither"))
    arena_record.add_argument("--evaluator", default="local")
    arena_record.add_argument("--context-id", required=True)
    arena_record.add_argument("--technical-regression", action="store_true")
    arena_record.add_argument("--notes", default="")
    arena_record.add_argument("--output", default="ratings/arena.jsonl")

    arena_report_parser = sub.add_parser("arena-report", help="Evaluate a challenger against the current SAUND.AI champion")
    arena_report_parser.add_argument("--path", default="ratings/arena.jsonl")
    arena_report_parser.add_argument("--champion")
    arena_report_parser.add_argument("--challenger")
    arena_report_parser.add_argument("--target-win-rate", type=float, default=0.60)
    arena_report_parser.add_argument("--minimum-decisive", type=int, default=50)

    models_status_parser = sub.add_parser("models-status", help="Show the governed neural model registry and checkpoint readiness")
    models_status_parser.add_argument("--registry", default=None)
    models_bootstrap_parser = sub.add_parser("models-bootstrap", help="Explicitly download/cache requested pretrained neural ears")
    models_bootstrap_parser.add_argument("--no-beat-this", action="store_true")
    models_bootstrap_parser.add_argument("--mert", action="store_true")
    neural_train_parser = sub.add_parser("neural-train", help="Train a governed structured policy/judge head")
    neural_train_parser.add_argument("task", choices=("transition_family","duration","stem_choreography","rendered_judge","next_track"))
    neural_train_parser.add_argument("dataset")
    neural_train_parser.add_argument("-o", "--output", required=True)
    neural_train_parser.add_argument("--allow-unreviewed-data", action="store_true")
    promote_parser = sub.add_parser("models-promote", help="Promote a validated DJ-specific head")
    promote_parser.add_argument("model_id", choices=("transition_family_policy","duration_policy","stem_choreography_policy","rendered_dj_judge","next_track_policy"))
    promote_parser.add_argument("authority", choices=("shadow","advisory","production"))
    promote_parser.add_argument("--registry", default=None)
    promote_parser.add_argument("--rights-reviewed", action="store_true")
    promote_parser.add_argument("--governance-approved", action="store_true")
    promote_parser.add_argument("--arena-report", default=None, help="Optional legacy blind Arena evidence")
    promote_parser.add_argument("--human-release-report", default=None, help="Phase 11 locked human release-gate JSON required for production promotion")

    rate = sub.add_parser("rate", help="Append a structured listening rating")
    rate.add_argument("report")
    rate.add_argument("--alternative", choices=("safe", "creative", "emergency"))
    rate.add_argument("--transition-index", type=int)
    for name, help_text in (
        ("beat", "Beat alignment 1-5"),
        ("phrase", "Phrase timing 1-5"),
        ("bass", "Bass cleanliness 1-5"),
        ("energy", "Energy flow 1-5"),
        ("cue", "Cue quality 1-5"),
        ("overall", "Overall quality 1-5"),
    ):
        rate.add_argument(f"--{name}", type=int, required=True, help=help_text)
    rate.add_argument("--notes", default="")
    rate.add_argument("--listener", default="local")
    rate.add_argument("--ratings-file", default="ratings/listening-tests.jsonl")

    serve = sub.add_parser("serve", help="Run the FastAPI service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    demo = sub.add_parser("demo", help="Generate two synthetic tracks")
    demo.add_argument("--output-dir", default="output/demo")
    demo.add_argument("--render", action="store_true")

    demo_set = sub.add_parser("demo-set", help="Generate, plan, preview, and optionally render a five-track synthetic set")
    demo_set.add_argument("--output-dir", default="output/demo-set")
    demo_set.add_argument("--bars", type=int, default=32)
    demo_set.add_argument("--render", action="store_true")
    demo_set.add_argument(
        "--journey",
        choices=("balanced", "smooth", "build", "warmup", "peak", "sunset"),
        default="build",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            analysis = analyze_track(args.track)
            _write_json(analysis.to_dict(include_beats=args.full_beats), args.output)
            raise SystemExit(0)
        if args.command == "plan":
            raise SystemExit(_run_plan(args))
        if args.command == "transition":
            raise SystemExit(_run_transition(args))
        if args.command == "alternatives":
            raise SystemExit(_run_alternatives(args))
        if args.command == "set-plan":
            raise SystemExit(_run_set_plan(args))
        if args.command == "set-render":
            raise SystemExit(_run_set_render(args))
        if args.command == "set":
            raise SystemExit(_run_set(args))
        if args.command == "manual-plan":
            raise SystemExit(_run_manual_plan(args))
        if args.command == "manual-set":
            raise SystemExit(_run_manual_set(args))
        if args.command == "set-preview":
            raise SystemExit(_run_set_previews(args))
        if args.command == "live":
            plan = load_set_plan(args.plan)
            _require_set_confirmation(plan, args.confirm_risky)
            play_live(plan, block_frames=args.block_frames)
            raise SystemExit(0)
        if args.command == "stems":
            mapping = separate_stems(args.track, model=args.model, device=args.device, force=args.force)
            bundle = resolve_performance_layers(args.track)
            _write_json(
                {
                    "track": str(Path(args.track).resolve()),
                    "stems": mapping,
                    "performance_layers": bundle.to_dict() if bundle is not None else None,
                },
                None,
            )
            raise SystemExit(0)
        if args.command == "layers-register":
            bundle = register_performance_bundle(
                args.full,
                instrumental=args.instrumental,
                vocals=args.vocals,
            )
            _write_json({"performance_layers": bundle.to_dict()}, None)
            raise SystemExit(0)
        if args.command == "genre-train":
            print(f"Genre model: {train_from_manifest(args.manifest, args.output)}")
            raise SystemExit(0)
        if args.command == "refit-scoring":
            measurement_path = Path(args.measurements) if args.measurements and Path(args.measurements).exists() else None
            print(
                f"Scoring model: {fit_scoring_model(args.ratings, args.output, measurement_path=measurement_path)}"
            )
            raise SystemExit(0)
        if args.command == "benchmark-manifest":
            output = create_benchmark_manifest(args.folder, args.output, recursive=not args.no_recursive)
            _write_json({"manifest": str(output), **benchmark_manifest_stats(output)}, None)
            raise SystemExit(0)
        if args.command == "listening-session":
            manifest = create_listening_session(
                args.track_a, args.track_b, output_root=args.output_dir, candidates=args.candidates
            )
            _write_json(manifest, None)
            raise SystemExit(0)
        if args.command == "brain-train":
            print(f"Transition ranker: {train_transition_ranker(args.preferences, args.output)}")
            raise SystemExit(0)
        if args.command == "brain-benchmark":
            _write_json(benchmark_brain(args.preferences, args.model).to_dict(), None)
            raise SystemExit(0)
        if args.command == "edm-cue-import":
            output = import_edm_cue_annotations(args.source, args.output)
            _write_json({"annotations": str(output), **edm_cue_stats(output)}, None)
            raise SystemExit(0)
        if args.command == "edm-cue-stats":
            _write_json(edm_cue_stats(args.annotations), None)
            raise SystemExit(0)
        if args.command == "edm-cue-source":
            _write_json(edm_cue_source_manifest(), None)
            raise SystemExit(0)
        if args.command == "cue-predict":
            predictions = predict_cue_points(
                args.track,
                model_id=args.model,
                confidence=args.confidence,
                minimum_spacing_bars=args.minimum_spacing_bars,
                allow_download=args.allow_download,
                device=args.device,
            )
            _write_json(
                {
                    "track": str(Path(args.track).expanduser().resolve()),
                    "model": args.model,
                    "model_fingerprint": cue_model_fingerprint(args.model),
                    "cue_points": [
                        {"time_seconds": item.time_seconds, "confidence": item.confidence}
                        for item in predictions
                    ],
                },
                None,
            )
            raise SystemExit(0)
        if args.command == "dataset-sources":
            _write_json(source_catalog(), None)
            raise SystemExit(0)
        if args.command == "dataset-fetch":
            output = fetch_source_asset(
                args.source,
                asset_name=args.asset,
                output_dir=args.output_dir,
                allow_audio_download=args.allow_audio_download,
                overwrite=args.overwrite,
            )
            _write_json({"source": args.source, "download": str(output)}, None)
            raise SystemExit(0)
        if args.command == "dataset-import":
            output = import_source_dataset(
                args.source,
                args.path,
                args.output,
                limit=args.limit,
                rights_cleared=args.rights_cleared,
            )
            _write_json({"source": args.source, "records": str(output), "manifest": str(output.with_name("manifest.json"))}, None)
            raise SystemExit(0)
        if args.command == "dataset-weak-labels":
            output = generate_weak_labels(args.records, args.output)
            _write_json({"weak_labels": str(output)}, None)
            raise SystemExit(0)
        if args.command == "dataset-builder-status":
            _write_json(dataset_builder_status(current_environment().workspace), None)
            raise SystemExit(0)
        if args.command == "dataset-build":
            corpus = create_bootstrap_corpus(
                current_environment().workspace,
                name=args.name,
                production=args.production,
            )
            manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
            _write_json({"corpus_path": str(corpus), "manifest": manifest}, None)
            raise SystemExit(0)
        if args.command == "dataset-train-bootstrap":
            _write_json(
                train_bootstrap_shadow_models(
                    current_environment().workspace,
                    corpus_id=args.corpus_id,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "dataset-audit":
            _write_json(audit_dataset_registry(args.registry, lane=args.lane), None)
            raise SystemExit(0)
        if args.command == "training-curriculum":
            _write_json({"curriculum": training_curriculum()}, None)
            raise SystemExit(0)
        if args.command == "training-lab-status":
            _write_json(training_lab_status(current_environment().workspace), None)
            raise SystemExit(0)
        if args.command == "training-snapshot":
            snapshot = create_training_snapshot(current_environment().workspace, name=args.name)
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            _write_json({"dataset_path": str(snapshot), "manifest": manifest}, None)
            raise SystemExit(0)
        if args.command == "training-shadow-train":
            _write_json(
                train_shadow_ranker(
                    current_environment().workspace,
                    snapshot_id=args.snapshot_id,
                    ridge=args.ridge,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "training-taste-train":
            _write_json(
                train_taste_learning_heads(
                    current_environment().workspace,
                    snapshot_id=args.snapshot_id,
                    ridge=args.ridge,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "training-taste-status":
            _write_json(taste_learning_status(current_environment().workspace), None)
            raise SystemExit(0)
        if args.command == "training-real-taste-train":
            _write_json(
                train_real_taste_from_snapshot(
                    current_environment().workspace,
                    snapshot_id=args.snapshot_id,
                    ridge=args.ridge,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "training-real-taste-status":
            _write_json(real_taste_status(current_environment().workspace), None)
            raise SystemExit(0)
        if args.command == "human-release-lock":
            locked = lock_human_release_benchmark(
                args.cases,
                args.output,
                incumbent=args.incumbent,
                candidate=args.candidate,
            )
            payload = json.loads(locked.read_text(encoding="utf-8"))
            _write_json(
                {
                    "benchmark_path": str(locked),
                    "benchmark_id": payload.get("benchmark_id"),
                    "locked_sha256": payload.get("locked_sha256"),
                    "case_count": payload.get("case_count"),
                },
                None,
            )
            raise SystemExit(0)
        if args.command == "human-release-session":
            _write_json(
                create_human_release_session(
                    current_environment().workspace,
                    evaluator=args.evaluator,
                    cohort=args.cohort,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "human-release-vote":
            output = record_human_release_vote(
                current_environment().workspace,
                session_id=args.session_id,
                choice=args.choice,
                catastrophic_a=args.catastrophic_a,
                catastrophic_b=args.catastrophic_b,
                reasons=args.reason,
            )
            _write_json(
                {
                    "votes_path": str(output),
                    "report": human_release_report(current_environment().workspace),
                },
                None,
            )
            raise SystemExit(0)
        if args.command == "human-release-report":
            thresholds = HumanReleaseThresholds(
                minimum_locked_cases=args.minimum_cases,
                minimum_covered_cases=args.minimum_covered,
                minimum_decisive_votes=args.minimum_decisive,
                minimum_unique_evaluators=args.minimum_evaluators,
                minimum_professional_dj_votes=args.minimum_professional_votes,
                target_candidate_win_rate=args.target_win_rate,
                maximum_candidate_catastrophic_rate=args.maximum_catastrophic_rate,
                maximum_new_catastrophic_cases=args.maximum_new_catastrophes,
            )
            _write_json(
                human_release_report(
                    current_environment().workspace,
                    thresholds=thresholds,
                ),
                args.output,
            )
            raise SystemExit(0)
        if args.command == "teacher-status":
            workspace = args.workspace or current_environment().workspace
            _write_json(teacher_data_status(workspace), None)
            raise SystemExit(0)
        if args.command == "teacher-scenarios":
            scenarios = scenarios_from_rehearsal_evidence(
                args.evidence,
                maximum_pairs_per_event=args.pairs_per_event,
            )
            output = write_scenarios(args.output, scenarios)
            _write_json({"scenarios": str(output), "scenario_count": len(scenarios)}, None)
            raise SystemExit(0)
        if args.command == "teacher-prompt":
            rows = load_scenarios(args.scenarios)
            selected = next((row for row in rows if str(row.get("scenario_id")) == args.scenario_id), None)
            if selected is None:
                raise ValueError(f"Teacher scenario not found: {args.scenario_id}")
            payload = teacher_prompt_payload(selected)
            _write_json(payload, args.output)
            raise SystemExit(0)
        if args.command == "teacher-ingest":
            rows = load_scenarios(args.scenarios)
            selected = next((row for row in rows if str(row.get("scenario_id")) == args.scenario_id), None)
            if selected is None:
                raise ValueError(f"Teacher scenario not found: {args.scenario_id}")
            response = json.loads(Path(args.judgment).expanduser().resolve().read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise ValueError("Teacher judgment JSON must contain an object.")
            judgment = validate_teacher_judgment(
                selected,
                response,
                provider=args.provider,
                model=args.model,
                model_version=args.model_version,
                temperature=args.temperature,
                seed=args.seed,
            )
            output = append_teacher_judgment(args.output, judgment)
            _write_json({"judgments": str(output), "judgment": judgment.to_dict()}, None)
            raise SystemExit(0)
        if args.command == "teacher-consensus":
            calibration = None
            if args.calibration:
                calibration = json.loads(Path(args.calibration).expanduser().resolve().read_text(encoding="utf-8"))
                if not isinstance(calibration, dict):
                    raise ValueError("Teacher calibration JSON must contain an object.")
            output = build_consensus_file(
                args.judgments,
                args.output,
                minimum_teachers=args.minimum_teachers,
                minimum_agreement=args.minimum_agreement,
                minimum_mean_confidence=args.minimum_confidence,
                minimum_distinct_providers=args.minimum_providers,
                minimum_distinct_models=args.minimum_models,
                minimum_distinct_families=args.minimum_families,
                calibration=calibration,
            )
            _write_json({"consensus": str(output)}, None)
            raise SystemExit(0)
        if args.command == "teacher-balance":
            scenarios = load_scenarios(args.scenarios)
            output = write_balanced_scenarios(
                scenarios,
                args.output,
                maximum_per_stratum=args.maximum_per_stratum,
            )
            _write_json({"balanced_scenarios": str(output)}, None)
            raise SystemExit(0)
        if args.command == "teacher-calibrate":
            judgments = []
            with Path(args.judgments).expanduser().resolve().open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        judgments.append(row)
            references = json.loads(Path(args.references).expanduser().resolve().read_text(encoding="utf-8"))
            if not isinstance(references, dict):
                raise ValueError("Reference labels JSON must contain a scenario_id -> choice object.")
            calibration = build_teacher_calibration(judgments, {str(k): str(v) for k, v in references.items()})
            _write_json(calibration, args.output)
            raise SystemExit(0)
        if args.command == "teacher-review-queue":
            output = build_human_review_queue(
                args.consensus,
                args.output,
                minimum_priority=args.minimum_priority,
            )
            _write_json({"human_review_queue": str(output)}, None)
            raise SystemExit(0)
        if args.command == "teacher-hard-negatives":
            output = build_hard_negative_queue(args.consensus, args.output)
            _write_json({"hard_negatives": str(output)}, None)
            raise SystemExit(0)
        if args.command == "research-synthetic-validate":
            _write_json(validate_research_synthetic_dataset(args.source_dir).to_dict(), None)
            raise SystemExit(0)
        if args.command == "research-synthetic-import":
            snapshot = create_research_synthetic_snapshot(
                current_environment().workspace,
                args.source_dir,
                name=args.name,
            )
            manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
            _write_json({"dataset_path": str(snapshot), "manifest": manifest}, None)
            raise SystemExit(0)
        if args.command == "research-synthetic-status":
            _write_json(research_synthetic_status(current_environment().workspace), None)
            raise SystemExit(0)
        if args.command == "research-synthetic-train":
            _write_json(
                train_research_synthetic_shadow(
                    current_environment().workspace,
                    dataset_id=args.dataset_id,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "phase16-validate-real":
            if args.neural_retrieval:
                os.environ["DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL"] = "1"
                os.environ["DJAI_ALLOW_BROAD_GRAPH_FALLBACK"] = (
                    "1" if args.allow_broad_graph_fallback else "0"
                )
            _write_json(
                validate_real_music_library(
                    current_environment().workspace,
                    args.folder,
                    output_dir=args.output_dir,
                    journeys=args.journey or ("balanced",),
                    recursive=args.recursive,
                    render=args.render,
                    target_duration_seconds=args.target_duration_seconds,
                    peak_fraction=args.peak_fraction,
                    closing_window_seconds=args.closing_window_seconds,
                    ending_energy=args.ending_energy,
                    measurement_top_k=args.measurement_top_k,
                    use_cache=not args.no_cache,
                    listener=args.listener,
                    analysis_workers=args.analysis_workers,
                    graph_workers=args.graph_workers,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "synthetic-failures":
            manifest = generate_synthetic_failure_set(args.source, args.output_dir)
            _write_json({"manifest": str(manifest)}, None)
            raise SystemExit(0)
        if args.command == "expert-stats":
            _write_json(expert_dataset_stats(args.path), None)
            raise SystemExit(0)
        if args.command == "expert-train":
            output = train_expert_policy(args.path, args.output)
            _write_json({"expert_policy": str(output), **expert_dataset_stats(args.path)}, None)
            raise SystemExit(0)
        if args.command == "arena-record":
            path = append_arena_vote(
                ArenaVote(
                    champion=args.champion, challenger=args.challenger, choice=args.choice,
                    evaluator=args.evaluator, context_id=args.context_id,
                    technical_regression=args.technical_regression, notes=args.notes,
                ),
                args.output,
            )
            _write_json({"arena_path": str(path)}, None)
            raise SystemExit(0)
        if args.command == "arena-report":
            _write_json(
                arena_report(
                    args.path, champion=args.champion, challenger=args.challenger,
                    target_win_rate=args.target_win_rate, minimum_decisive=args.minimum_decisive,
                ),
                None,
            )
            raise SystemExit(0)
        if args.command == "models-status":
            _write_json(registry_status(args.registry), None)
            raise SystemExit(0)
        if args.command == "models-bootstrap":
            _write_json(bootstrap_models(beat_this=not args.no_beat_this, mert=args.mert), None)
            raise SystemExit(0)
        if args.command == "neural-train":
            output=train_neural_head(args.task,args.dataset,args.output,require_rights_cleared=not args.allow_unreviewed_data)
            _write_json({"checkpoint":str(output)},None)
            raise SystemExit(0)
        if args.command == "models-promote":
            arena_payload = load_arena_evidence(args.arena_report) if args.arena_report else None
            release_payload = (
                load_human_release_evidence(args.human_release_report)
                if args.human_release_report
                else None
            )
            output = promote_model(
                args.model_id,
                authority=args.authority,
                registry_path=args.registry,
                rights_reviewed=args.rights_reviewed,
                governance_approved=args.governance_approved,
                arena_report_payload=arena_payload,
                human_release_gate_payload=release_payload,
            )
            _write_json({"registry":str(output),"model_id":args.model_id,"authority":args.authority},None)
            raise SystemExit(0)
        if args.command == "rate":
            raise SystemExit(_run_rate(args))
        if args.command == "serve":
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError("The API dependency is not installed.") from exc
            uvicorn.run("djai.api:app", host=args.host, port=args.port, reload=args.reload)
            raise SystemExit(0)
        if args.command == "demo-set":
            root = Path(args.output_dir).expanduser().resolve()
            tracks = generate_demo_set(root / "audio", bars=args.bars)
            plan = plan_folder(
                root / "audio",
                use_cache=False,
                tempo_policy="safe",
                preferred_style="auto",
                journey=args.journey,
                start_track="01_opening_122bpm_Aminor",
                end_track="05_close_125bpm_Dminor",
                progress=_ConsoleProgress(),
            )
            plan_path = save_set_plan(plan, root / "set-plan.json")
            preview_paths, preview_report = render_set_previews(plan, root / "previews", pre_roll_seconds=3.0, post_roll_seconds=3.0)
            print(f"Plan: {plan_path}")
            print(f"Previews: {preview_report} ({len(preview_paths)} files)")
            if args.render:
                output, report = render_set(plan, root / "full-set.wav", progress=_ConsoleProgress())
                print(f"Full set: {output}")
                print(f"Report: {report}")
            raise SystemExit(0)
        if args.command == "demo":
            track_a, track_b = generate_demo_pair(args.output_dir)
            print(f"Generated: {track_a}\nGenerated: {track_b}")
            if args.render:
                demo_args = argparse.Namespace(
                    track_a=str(track_a),
                    track_b=str(track_b),
                    output=str(Path(args.output_dir) / "demo_transition.wav"),
                    report=str(Path(args.output_dir) / "demo_transition.json"),
                    force=False,
                    tempo_policy="safe",
                    style="safe",
                    bars=8,
                    pre_roll=6.0,
                    post_roll=6.0,
                    preview=True,
                    exit_time=None,
                    entry_time=None,
                )
                raise SystemExit(_run_transition(demo_args))
            raise SystemExit(0)
        parser.error("Unknown command")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"SAUND.AI error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
