from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .brain.timing import SetTimingIntent
from .set_planner import plan_folder, save_set_plan
from .set_renderer import render_set

PHASE16_SCHEMA_VERSION = "16.1"


def _phase16_progress(stage: str, current: int, total: int, detail: str) -> None:
    total_value = max(1, int(total))
    current_value = max(0, int(current))
    percent = min(100.0, 100.0 * current_value / total_value)
    print(
        f"[phase16] {stage}: {current_value}/{total_value} ({percent:.0f}%) {detail}",
        flush=True,
    )


def _measurement_hard_failures(edge: Any) -> list[str]:
    measurement = getattr(edge.plan, "measurement", None)
    if measurement is None:
        return []
    if hasattr(measurement, "hard_failures"):
        return [str(item) for item in (getattr(measurement, "hard_failures", None) or [])]
    if hasattr(measurement, "to_dict"):
        payload = measurement.to_dict()
        return [str(item) for item in (payload.get("hard_failures", []) or [])]
    return []


def summarize_real_set_plan(plan: Any) -> dict[str, Any]:
    metadata = dict(getattr(plan, "metadata", {}) or {})
    quality = dict(metadata.get("quality_loop", {}) or {})
    unified = dict(metadata.get("unified_transition_intelligence", {}) or {})
    timing = dict(metadata.get("timing", {}) or {})
    timing_state = dict(timing.get("state", {}) or {})
    large_crate = dict(metadata.get("large_crate_planning", {}) or {})
    performance = dict(metadata.get("performance", {}) or {})
    planning_performance = dict(performance.get("planning", {}) or {})

    hard_failures: list[dict[str, Any]] = []
    measured_scores: list[float] = []
    fallback_count = 0
    for position, edge in enumerate(getattr(plan, "transitions", []) or []):
        failures = _measurement_hard_failures(edge)
        if failures:
            hard_failures.append(
                {
                    "transition_index": position,
                    "from_index": int(edge.from_index),
                    "to_index": int(edge.to_index),
                    "failures": failures,
                }
            )
        measured = getattr(edge.plan, "measured_score", None)
        if measured is not None:
            measured_scores.append(float(measured))
        fallback_count += int(int(getattr(edge.plan, "fallback_level", 0) or 0) > 0)

    checked = int(quality.get("checked_transitions", 0) or 0)
    passed = int(quality.get("passed_transitions", 0) or 0)
    transition_count = len(getattr(plan, "transitions", []) or [])
    track_count = len(getattr(plan, "tracks", []) or [])
    quality_pass_rate = None if checked == 0 else round(passed / checked, 6)
    mean_measured = None if not measured_scores else round(sum(measured_scores) / len(measured_scores), 6)

    planning_ready = (
        track_count >= 2
        and transition_count == max(0, track_count - 1)
        and not hard_failures
        and (checked == 0 or passed == checked)
    )
    full_set_benchmark = metadata.get("full_set_benchmark")
    production_safe = bool(
        isinstance(full_set_benchmark, dict)
        and full_set_benchmark.get("production_safe") is True
    )

    return {
        "plan_id": str(getattr(plan, "plan_id", "")),
        "journey": str(metadata.get("journey") or "unknown"),
        "tracks": track_count,
        "transitions": transition_count,
        "estimated_duration_seconds": round(float(getattr(plan, "estimated_duration_seconds", 0.0) or 0.0), 6),
        "warnings": list(getattr(plan, "warnings", []) or []),
        "warning_count": len(getattr(plan, "warnings", []) or []),
        "requires_confirmation": bool(getattr(plan, "requires_confirmation", False)),
        "fallback_transition_count": fallback_count,
        "fallback_rate": round(fallback_count / transition_count, 6) if transition_count else 0.0,
        "replan_count": int(unified.get("replan_count", 0) or 0),
        "quality_checked_transitions": checked,
        "quality_passed_transitions": passed,
        "quality_pass_rate": quality_pass_rate,
        "mean_measured_score": mean_measured,
        "hard_failure_transitions": hard_failures,
        "hard_failure_count": len(hard_failures),
        "timing": {
            "target_duration_seconds": timing_state.get("target_duration_seconds"),
            "projected_total_seconds": timing_state.get("projected_total_seconds"),
            "duration_error_seconds": timing_state.get("duration_error_seconds"),
            "planned_peak_seconds": timing_state.get("planned_peak_seconds"),
            "closing_window_seconds": timing_state.get("closing_window_seconds"),
            "ending_energy_target": timing_state.get("ending_energy_target"),
        },
        "full_set_benchmark": full_set_benchmark,
        "production_safe": production_safe,
        "structural_reset_rate": (
            float(full_set_benchmark.get("structural_reset_rate", 0.0))
            if isinstance(full_set_benchmark, dict)
            else None
        ),
        "emergency_rate": (
            float(full_set_benchmark.get("emergency_rate", 0.0))
            if isinstance(full_set_benchmark, dict)
            else None
        ),
        "large_crate_planning": large_crate,
        "planning_performance": planning_performance,
        "planning_ready": planning_ready,
    }


def validate_real_music_library(
    workspace: str | Path,
    folder: str | Path,
    *,
    output_dir: str | Path = "output/phase16-real-music",
    journeys: Iterable[str] = ("balanced",),
    recursive: bool = False,
    render: bool = False,
    target_duration_seconds: float | None = None,
    peak_fraction: float = 0.65,
    closing_window_seconds: float = 600.0,
    ending_energy: float = 0.45,
    measurement_top_k: int = 4,
    use_cache: bool = True,
    listener: str = "local",
    analysis_workers: int = 1,
    graph_workers: int = 1,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    music = Path(folder).expanduser().resolve()
    if not music.is_dir():
        raise ValueError(f"Real-music validation folder not found: {music}")

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    normalized_journeys: list[str] = []
    for value in journeys:
        journey = str(value).strip().lower()
        if journey and journey not in normalized_journeys:
            normalized_journeys.append(journey)
    if not normalized_journeys:
        normalized_journeys = ["balanced"]

    timing_intent = SetTimingIntent(
        target_duration_seconds=target_duration_seconds,
        peak_fraction=peak_fraction,
        closing_window_seconds=closing_window_seconds,
        ending_energy=ending_energy,
    )

    runs: list[dict[str, Any]] = []
    total_runs = len(normalized_journeys)
    print(
        f"[phase16] starting real-music validation: journeys={normalized_journeys}, "
        f"render={render}, analysis_workers={analysis_workers}, graph_workers={graph_workers}",
        flush=True,
    )
    for run_index, journey in enumerate(normalized_journeys, start=1):
        journey_started = time.monotonic()
        print(f"[phase16] [{run_index}/{total_runs}] {journey}: planning started", flush=True)
        run_dir = destination / journey
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            plan = plan_folder(
                music,
                recursive=recursive,
                journey=journey,
                measure_transitions=True,
                measurement_top_k=measurement_top_k,
                measurement_log=str(run_dir / "automatic-measurements.jsonl"),
                automatic_delivery=True,
                listener=listener,
                use_cache=use_cache,
                analysis_workers=analysis_workers,
                graph_workers=graph_workers,
                timing_intent=timing_intent,
                progress=_phase16_progress,
            )
        except Exception as exc:
            failure = {
                "journey": journey,
                "stage": "planning",
                "planning_ready": False,
                "render_requested": bool(render),
                "render_passed": False if render else None,
                "rehearsal_ready": False,
                "hard_failure_count": 1,
                "fallback_transition_count": 0,
                "warning_count": 0,
                "warnings": [],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "planner_diagnostics": (
                    dict(getattr(exc, "diagnostics"))
                    if isinstance(getattr(exc, "diagnostics", None), dict)
                    else None
                ),
            }
            (run_dir / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            runs.append(failure)
            print(
                f"[phase16] [{run_index}/{total_runs}] {journey}: planning FAILED after "
                f"{time.monotonic() - journey_started:.1f}s: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        planning_elapsed = time.monotonic() - journey_started
        print(
            f"[phase16] [{run_index}/{total_runs}] {journey}: planning complete in "
            f"{planning_elapsed:.1f}s; tracks={len(getattr(plan, 'tracks', []) or [])}, "
            f"transitions={len(getattr(plan, 'transitions', []) or [])}",
            flush=True,
        )

        plan_path = save_set_plan(plan, run_dir / "set-plan.json")
        summary = summarize_real_set_plan(plan)
        summary["plan_path"] = str(plan_path.relative_to(root)) if plan_path.is_relative_to(root) else str(plan_path)
        summary["render_requested"] = bool(render)
        summary["render_passed"] = None
        summary["render_output"] = None
        summary["render_report"] = None

        if render:
            render_started = time.monotonic()
            print(f"[phase16] [{run_index}/{total_runs}] {journey}: full-set render started", flush=True)
            try:
                audio_path, report_path = render_set(
                    plan,
                    run_dir / "full-set.wav",
                    report_path=run_dir / "render-report.json",
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                verification = dict(dict(report.get("render", {})).get("final_transition_verification", {}) or {})
                summary["render_passed"] = bool(verification.get("passed", False))
                summary["render_output"] = str(audio_path.relative_to(root)) if audio_path.is_relative_to(root) else str(audio_path)
                summary["render_report"] = str(report_path.relative_to(root)) if report_path.is_relative_to(root) else str(report_path)
                print(
                    f"[phase16] [{run_index}/{total_runs}] {journey}: render complete in "
                    f"{time.monotonic() - render_started:.1f}s; passed={summary['render_passed']}",
                    flush=True,
                )
            except Exception as exc:
                summary["render_passed"] = False
                summary["render_error_type"] = type(exc).__name__
                summary["render_error"] = str(exc)
                print(
                    f"[phase16] [{run_index}/{total_runs}] {journey}: render FAILED after "
                    f"{time.monotonic() - render_started:.1f}s: {type(exc).__name__}: {exc}",
                    flush=True,
                )

        summary["rehearsal_ready"] = bool(
            summary["planning_ready"]
            and summary["production_safe"]
            and (summary["render_passed"] is True if render else True)
        )
        runs.append(summary)
        print(
            f"[phase16] [{run_index}/{total_runs}] {journey}: done; "
            f"planning_ready={summary['planning_ready']}, rehearsal_ready={summary['rehearsal_ready']}, "
            f"elapsed={time.monotonic() - journey_started:.1f}s",
            flush=True,
        )

    total_hard_failures = sum(int(item["hard_failure_count"]) for item in runs)
    total_fallbacks = sum(int(item["fallback_transition_count"]) for item in runs)
    all_planning_ready = all(bool(item["planning_ready"]) for item in runs)
    all_production_safe = all(bool(item.get("production_safe", False)) for item in runs)
    all_rehearsal_ready = all(bool(item["rehearsal_ready"]) for item in runs)

    report = {
        "schema_version": PHASE16_SCHEMA_VERSION,
        "phase": 16,
        "name": "Real-Music Validation",
        "music_folder": str(music),
        "output_dir": str(destination),
        "journeys": normalized_journeys,
        "render_requested": bool(render),
        "worker_policy": {
            "analysis_workers": int(analysis_workers),
            "graph_workers": int(graph_workers),
            "default_safe_serial": analysis_workers == 1 and graph_workers == 1,
        },
        "timing_intent": timing_intent.normalized().to_dict(),
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "planning_ready_runs": sum(int(bool(item["planning_ready"])) for item in runs),
            "production_safe_runs": sum(int(bool(item.get("production_safe", False))) for item in runs),
            "rehearsal_ready_runs": sum(int(bool(item["rehearsal_ready"])) for item in runs),
            "hard_failure_count": total_hard_failures,
            "fallback_transition_count": total_fallbacks,
            "all_planning_ready": all_planning_ready,
            "all_production_safe": all_production_safe,
            "all_rehearsal_ready": all_rehearsal_ready,
            "phase16_passed": all_rehearsal_ready,
        },
        "authority": {
            "learned_model_changes_live_decisions": False,
            "synthetic_mlp_authority": 0.0,
            "purpose": "real-library validation only",
        },
    }
    report_path = destination / "phase16-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.relative_to(root)) if report_path.is_relative_to(root) else str(report_path)
    print(
        f"[phase16] complete: passed={report['summary']['phase16_passed']}, "
        f"planning_ready={report['summary']['planning_ready_runs']}/{report['summary']['run_count']}, "
        f"rehearsal_ready={report['summary']['rehearsal_ready_runs']}/{report['summary']['run_count']}",
        flush=True,
    )
    return report
