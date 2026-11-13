#!/usr/bin/env python3
from __future__ import annotations

"""Capture/compare Phase 4 planner execution identities on fixed real audio.

Run capture against the same audio directory in two separate Git worktrees.
This script lives on the comparison branch but imports djai from PYTHONPATH.
JSON report fields intentionally exclude generated plan IDs, timestamps, and
machine-specific paths. Audio comparison hashes decoded PCM, not WAV headers.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf


def _pcm_digest(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with sf.SoundFile(path) as audio:
        info = {
            "sample_rate": audio.samplerate,
            "channels": audio.channels,
            "frames": audio.frames,
        }
        while True:
            block = audio.read(65536, dtype="float32", always_2d=True)
            if not len(block):
                break
            digest.update(block.tobytes())
    return {**info, "pcm_sha256": digest.hexdigest()}


def capture(args: argparse.Namespace) -> int:
    from djai.cache import analyze_tracks, discover_tracks
    from djai.decision import validate_execution_plan
    from djai.set_planner import make_set_plan
    from djai.set_renderer import render_set

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    paths = discover_tracks(audio_dir, recursive=False)
    if len(paths) != 3:
        raise SystemExit(
            f"Expected exactly 3 fixed WAV files in {audio_dir}; found {len(paths)}"
        )

    fingerprints = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    started = time.perf_counter()
    analyses = analyze_tracks(
        paths,
        workers=1,
        use_cache=bool(args.analysis_cache_dir),
        cache_dir=args.analysis_cache_dir or ".djai-cache/tracks",
    )
    analysis_seconds = time.perf_counter() - started
    # Prove the two planner runs consumed identical analyzed features.
    # Source hashes alone do not establish perceptual evidence identity.
    analysis_digests = [
        hashlib.sha256(
            json.dumps(
                item.to_dict(include_beats=True),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            ).encode("utf-8")
        ).hexdigest()
        for item in analyses
    ]

    if args.manual_order:
        forced = [int(value) for value in args.manual_order.split(",")]
        if sorted(forced) != list(range(len(paths))):
            raise SystemExit(
                f"--manual-order must be a permutation of 0..{len(paths) - 1}"
            )

    selection_trace: list[dict[str, object]] = []
    if args.trace_selection:
        import djai.set_planner as planner_module

        original_select = planner_module._select_alternative
        original_exit = planner_module._sequence_exit_candidates
        original_entry = planner_module._sequence_entry_candidates

        def trace_exit(track, **kwargs):
            result = original_exit(track, **kwargs)
            selection_trace.append({
                "stage": "exit_cues",
                "track": track.title,
                "source_cursor": kwargs.get("source_cursor"),
                "required_source": kwargs.get("required_source"),
                "cues": [
                    {
                        "time": cue.time_seconds,
                        "bar": cue.bar_index,
                        "section": cue.section_label,
                        "score": cue.score,
                    }
                    for cue in result[0]
                ],
            })
            return result

        def trace_entry(track, **kwargs):
            result = original_entry(track, **kwargs)
            selection_trace.append({
                "stage": "entry_cues",
                "track": track.title,
                "cues": [
                    {
                        "time": cue.time_seconds,
                        "bar": cue.bar_index,
                        "section": cue.section_label,
                        "score": cue.score,
                    }
                    for cue in result
                ],
            })
            return result

        def trace_select(plans, track_a, track_b, **kwargs):
            selected = original_select(plans, track_a, track_b, **kwargs)
            selection_trace.append({
                "stage": "select_alternative",
                "from": track_a.title,
                "to": track_b.title,
                "plans": [
                    {
                        "style": candidate.style,
                        "technique": candidate.technique,
                        "score": candidate.score,
                        "selection_utility": candidate.metadata.get("selection_utility"),
                        "exit": candidate.exit_time_a,
                        "entry": candidate.entry_time_b,
                        "signature": candidate.execution_signature,
                    }
                    for candidate in plans
                ],
                "winner": selected[0].execution_signature,
            })
            return selected

        planner_module._sequence_exit_candidates = trace_exit
        planner_module._sequence_entry_candidates = trace_entry
        planner_module._select_alternative = trace_select

        # Capture the candidate pool BEFORE downstream DJMove ranking, metadata
        # sealing and final edge selection. The pre-extraction planner imports
        # this callback in set_planner; the refactor imports it in its helper.
        import dataclasses
        import djai.planner as transition_planner

        def evidence_digest(matrix):
            if matrix is None:
                return None
            payload = json.dumps(
                dataclasses.asdict(matrix),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        def trace_candidate_pool(original):
            def traced(track_a, track_b, **kwargs):
                candidates = original(track_a, track_b, **kwargs)
                matrix = kwargs.get("evidence_matrix")
                selection_trace.append({
                    "stage": "candidate_pool",
                    "from": track_a.title,
                    "to": track_b.title,
                    "matrix_digest": evidence_digest(matrix),
                    "matrix_pairs": len(matrix.pairs) if matrix is not None else None,
                    "styles": kwargs.get("styles"),
                    "per_style": kwargs.get("per_style"),
                    "effective_bpm_a": kwargs.get("effective_bpm_a"),
                    "render_rate_a": kwargs.get("render_rate_a"),
                    "stems_available": kwargs.get("stems_available"),
                    "candidates": [
                        {
                            "style": plan.style,
                            "technique": plan.technique,
                            "score": plan.score,
                            "exit": plan.exit_time_a,
                            "entry": plan.entry_time_b,
                            "signature": plan.execution_signature,
                            "program_fingerprint": plan.performance_program_fingerprint,
                        }
                        for plan in candidates
                    ],
                })
                return candidates
            return traced

        if hasattr(planner_module, "plan_transition_candidate_pool"):
            original_pool = planner_module.plan_transition_candidate_pool
            planner_module.plan_transition_candidate_pool = trace_candidate_pool(original_pool)
        else:
            import djai.planner_sequence_candidates as sequence_candidates
            original_pool = sequence_candidates.plan_transition_candidate_pool
            sequence_candidates.plan_transition_candidate_pool = trace_candidate_pool(original_pool)

    started = time.perf_counter()
    plan = make_set_plan(
        analyses,
        source="phase4-equivalence",
        manual_order=(
            [int(value) for value in args.manual_order.split(",")]
            if args.manual_order
            else None
        ),
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        minimum_body_seconds=8.0,
        minimum_body_fraction=0.10,
        graph_workers=1,
        quality_loop=None,
        automatic_delivery=False,
    )
    planning_seconds = time.perf_counter() - started

    transitions = []
    for edge in plan.transitions:
        validate_execution_plan(edge.plan)
        candidate = edge.plan
        transitions.append({
            "from": edge.from_index,
            "to": edge.to_index,
            "execution_signature": candidate.execution_signature,
            "program_fingerprint": candidate.performance_program_fingerprint,
            "technique": candidate.technique,
            "intent": candidate.intent,
            "exit_time_a": candidate.exit_time_a,
            "entry_time_b": candidate.entry_time_b,
            "render_rate_a": candidate.render_rate_a,
            "render_rate_b": candidate.render_rate_b,
            "source_end_b": candidate.post_transition_source_end_b,
            "alternative_signatures": [
                value.execution_signature for value in edge.alternatives
            ],
        })

    rendered = None
    render_seconds = None
    if args.render:
        started = time.perf_counter()
        render_path, _report_path = render_set(
            plan,
            Path(args.output).with_suffix(".wav"),
        )
        render_seconds = time.perf_counter() - started
        rendered = _pcm_digest(render_path)

    report = {
        "schema": "phase4-equivalence-v1",
        "audio_fingerprints": fingerprints,
        "analysis_digests": analysis_digests,
        "order": plan.order,
        "transitions": transitions,
        **({"selection_trace": selection_trace} if args.trace_selection else {}),
        "rendered_pcm": rendered,
        "timing_seconds": {
            "analysis": round(analysis_seconds, 4),
            "planning": round(planning_seconds, 4),
            "render": round(render_seconds, 4) if render_seconds is not None else None,
        },
        "context": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "graph_workers": 1,
            "quality_loop": False,
            "automatic_delivery": False,
            "manual_order": args.manual_order,
            "analysis_cache_dir": args.analysis_cache_dir,
            "environment": {
                key: os.getenv(key)
                for key in (
                    "DJAI_PLANNER_MODE",
                    "DJAI_PROFESSIONAL_BRAIN",
                    "DJAI_GRAPH_WORKERS",
                    "DJAI_LARGE_SET_THRESHOLD",
                )
            },
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Captured {len(transitions)} edges: {output}")
    return 0


def compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    keys = ("schema", "audio_fingerprints", "analysis_digests", "order", "transitions", "rendered_pcm")
    mismatches = [key for key in keys if baseline.get(key) != current.get(key)]
    baseline_times = baseline.get("timing_seconds", {})
    current_times = current.get("timing_seconds", {})
    print("Phase 4 comparison:")
    for key in keys:
        if key == "rendered_pcm" and baseline.get(key) is None and current.get(key) is None:
            print("  rendered_pcm: NOT CAPTURED (no audio equivalence evidence)")
        else:
            print(f"  {key}: {'MATCH' if key not in mismatches else 'DIFFERENT'}")
    if baseline.get("order") != current.get("order"):
        print(f"  baseline order: {baseline.get('order')}")
        print(f"  current order:  {current.get('order')}")
    if baseline.get("transitions") != current.get("transitions"):
        left = baseline.get("transitions") or []
        right = current.get("transitions") or []
        print(f"  baseline edge count: {len(left)}; current edge count: {len(right)}")
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else {}
            b = right[index] if index < len(right) else {}
            if a == b:
                continue
            print(f"  edge {index + 1} differences:")
            for field in sorted(set(a) | set(b)):
                if a.get(field) != b.get(field):
                    print(f"    {field}: baseline={a.get(field)!r} current={b.get(field)!r}")
    if baseline.get("selection_trace") is not None and current.get("selection_trace") is not None:
        a_trace = baseline["selection_trace"]
        b_trace = current["selection_trace"]
        print(f"  selection trace entries: baseline={len(a_trace)} current={len(b_trace)}")
        for index in range(max(len(a_trace), len(b_trace))):
            a = a_trace[index] if index < len(a_trace) else None
            b = b_trace[index] if index < len(b_trace) else None
            if a != b:
                print(f"  first selection trace mismatch at step {index + 1}:")
                print(f"    baseline: {json.dumps(a, sort_keys=True)}")
                print(f"    current:  {json.dumps(b, sort_keys=True)}")
                break
    if baseline.get("context") != current.get("context"):
        print(f"  baseline context: {baseline.get('context')!r}")
        print(f"  current context:  {current.get('context')!r}")
    for key in ("analysis", "planning", "render"):
        old = baseline_times.get(key)
        new = current_times.get(key)
        if old is not None and new is not None:
            print(f"  {key}: baseline={old:.3f}s current={new:.3f}s ratio={new / max(old, 1e-9):.3f}x")
    if mismatches:
        print("FAILED equivalence fields: " + ", ".join(mismatches), file=sys.stderr)
        return 1
    if baseline.get("rendered_pcm") is None:
        print("WARNING: rendering was not captured; exact-render equivalence is unverified.")
    print("Execution identity equivalence passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--audio-dir", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--render", action="store_true")
    capture_parser.add_argument("--trace-selection", action="store_true")
    capture_parser.add_argument(
        "--manual-order",
        help="Comma-separated track indices, e.g. 1,0,2; isolate compiler from route search",
    )
    capture_parser.add_argument(
        "--analysis-cache-dir",
        help="Shared absolute cache path; use the same analyzed evidence on both branches",
    )
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("current")
    args = parser.parse_args()
    return capture(args) if args.command == "capture" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
