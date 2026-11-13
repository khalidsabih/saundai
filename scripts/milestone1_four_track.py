#!/usr/bin/env python3
"""Milestone 1: four real tracks, three auditable DJ transitions.

Run from the repository root:
    python scripts/milestone1_four_track.py \
        audio/uploads/one.wav audio/uploads/two.wav \
        audio/uploads/three.wav audio/uploads/four.wav

Runs the existing Phase 16 planner and renderer unchanged. Generates a
separate musical-performance diagnostic; a safe render is NOT automatically
certified as a professional mix.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _transition_kind(edge: dict[str, Any]) -> str:
    plan = dict(edge.get("plan") or {})
    metadata = dict(plan.get("metadata") or {})
    kind = str(metadata.get("rescue_transition_kind") or "")
    reason = str(plan.get("fallback_reason") or "")
    emergency = bool(
        metadata.get("unified_transition_final_escape")
        or metadata.get("deterministic_final_escape")
        or "All alternate track routes" in reason
        or "No evidence matrix remained" in reason
    )
    if emergency:
        return "emergency_escape"
    if kind == "structural_reset" or str(plan.get("technique") or "") in {
        "echo_out_drop_in", "quick_cut", "backspin",
    } or str(plan.get("transition_mode") or "") == "emergency_cut":
        return "structural_reset"
    if int(plan.get("fallback_level") or 0) > 0:
        return "other_fallback"
    return "normal_transition"


def summarize_four_track(
    phase_report: dict[str, Any],
    set_plan: dict[str, Any] | None,
    phase_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    runs = list(phase_report.get("runs") or [])
    run = dict(runs[0]) if runs else {}
    plan = dict(set_plan or {})
    tracks = list(plan.get("tracks") or [])
    edges = list(plan.get("transitions") or [])
    detailed: list[dict[str, Any]] = []
    types: dict[str, int] = {
        "normal_transition": 0,
        "structural_reset": 0,
        "other_fallback": 0,
        "emergency_escape": 0,
    }
    for index, edge in enumerate(edges):
        kind = _transition_kind(edge)
        types[kind] += 1
        p = dict(edge.get("plan") or {})
        meta = dict(p.get("metadata") or {})
        mixing = dict(meta.get("professional_mix") or {})
        contract = dict(meta.get("professional_mix_contract") or {})
        a = int(edge.get("from_index") or 0)
        b = int(edge.get("to_index") or 0)
        title_a = str(tracks[a].get("title") or tracks[a].get("path")) if 0 <= a < len(tracks) else "unknown"
        title_b = str(tracks[b].get("title") or tracks[b].get("path")) if 0 <= b < len(tracks) else "unknown"
        detailed.append({
            "number": index + 1,
            "from": title_a,
            "to": title_b,
            "classification": kind,
            "technique": p.get("technique"),
            "transition_bars": p.get("transition_bars"),
            "beatmatched": p.get("beatmatched"),
            "fallback_level": p.get("fallback_level"),
            "fallback_reason": p.get("fallback_reason"),
            "measured_score": p.get("measured_score"),
            "measurement_passed": p.get("measurement_passed"),
            "mix_family": mixing.get("family"),
            "mix_contract_passed": contract.get("passed"),
            "dual_deck_overlap_fraction": contract.get("dual_deck_overlap_fraction"),
            "warnings": edge.get("warnings") or [],
        })
    n = len(detailed)
    complete = len(tracks) == 4 and n == 3
    return {
        "schema": "saund-milestone1-four-track-v1",
        "purpose": "Measure actual transition technique and safety; professional taste requires human review.",
        "tracks": len(tracks) if set_plan else int(run.get("tracks") or 0),
        "transitions": n,
        "complete_four_track_set": complete,
        "phase16_passed": bool(phase_report.get("summary", {}).get("phase16_passed")),
        "render_passed": run.get("render_passed"),
        "production_safe": run.get("production_safe"),
        "rehearsal_ready": run.get("rehearsal_ready"),
        "planning_seconds": (run.get("planning_performance") or {}).get("executive_seconds"),
        "story_fit": (run.get("full_set_benchmark") or {}).get("story_fit"),
        "mean_measured_score": run.get("mean_measured_score"),
        "transition_classifications": types,
        "normal_transition_rate": round(types["normal_transition"] / n, 6) if n else None,
        "emergency_escape_count": types["emergency_escape"],
        "hard_failure_count": run.get("hard_failure_count"),
        "phase_rejections": phase_trace,
        "per_transition": detailed,
        "musically_certified": False,
        "missing_evidence": [
            "No blinded listener or professional DJ preference evaluation was performed.",
            "A normal transition is not necessarily an audible, sustained continuous blend.",
            "The phase trace lists rejected candidates only; no acceptance rate can be inferred.",
        ],
    }


def prepare_four_track_crate(selected: list[Path], destination: Path) -> None:
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError("Provide four distinct source audio files.")
    destination.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for track in selected:
        source = track.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        if source.suffix.lower() not in {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg"}:
            raise ValueError(f"Unsupported audio extension: {source.name}")
        if source.name in used:
            raise ValueError(f"Duplicate audio filename: {source.name}")
        used.add(source.name)
        target = destination / source.name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path, nargs=4, help="Four real source audio files")
    parser.add_argument("--journey", choices=("balanced", "smooth", "build", "warmup", "peak", "sunset"), default="balanced")
    parser.add_argument("--output-dir", type=Path, default=Path("output/milestone1-four-track"))
    parser.add_argument("--no-render", action="store_true", help="Diagnostic-only; cannot establish render readiness")
    args = parser.parse_args(argv)

    root = ROOT
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "phase-rejections.jsonl"
    # A failed re-run must never inherit a prior successful plan or report.
    for stale in (
        output / args.journey / "set-plan.json",
        output / args.journey / "render-report.json",
        output / args.journey / "full-set.wav",
        output / "phase16-report.json",
        output / "milestone1-report.json",
    ):
        stale.unlink(missing_ok=True)
    # Start with a clean rejection trace, without changing phase safety rules.
    trace_path.write_text("", encoding="utf-8")
    previous_trace = os.environ.get("DJAI_PHASE_LOCK_TRACE_PATH")
    os.environ["DJAI_PHASE_LOCK_TRACE_PATH"] = str(trace_path)
    os.environ.setdefault("DJAI_BEAT_GAP_AWARE", "0")

    from scripts.summarize_phase_lock_trace import summarize
    from djai.phase16_validation import validate_real_music_library

    try:
        audio_dir = root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="milestone1-crate-", dir=audio_dir) as staged:
            prepare_four_track_crate(args.tracks, Path(staged))
            phase = validate_real_music_library(
                root, staged,
                output_dir=output,
                journeys=(args.journey,),
                render=not args.no_render,
                analysis_workers=1,
                graph_workers=1,
            )
    finally:
        if previous_trace is None:
            os.environ.pop("DJAI_PHASE_LOCK_TRACE_PATH", None)
        else:
            os.environ["DJAI_PHASE_LOCK_TRACE_PATH"] = previous_trace

    plan_path = output / args.journey / "set-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    trace_summary = summarize(trace_path)
    report = summarize_four_track(phase, plan, trace_summary)
    report["beat_gap_aware"] = os.environ.get("DJAI_BEAT_GAP_AWARE") == "1"
    report["phase16_report"] = str(output / "phase16-report.json")
    report["render_path"] = str(output / args.journey / "full-set.wav") if not args.no_render else None
    report_path = output / "milestone1-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"milestone1_report": str(report_path), **report}, indent=2, ensure_ascii=False))
    return 0 if (report["complete_four_track_set"] and report["phase16_passed"] and report["render_passed"] is True) else 1


if __name__ == "__main__":
    # Running as a script adds scripts/ to sys.path; install or expose src/.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
