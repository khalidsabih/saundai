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
    analyses = analyze_tracks(paths, workers=1, use_cache=False)
    analysis_seconds = time.perf_counter() - started

    started = time.perf_counter()
    plan = make_set_plan(
        analyses,
        source="phase4-equivalence",
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
        "order": plan.order,
        "transitions": transitions,
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
    keys = ("schema", "audio_fingerprints", "order", "transitions", "rendered_pcm")
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
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("baseline")
    compare_parser.add_argument("current")
    args = parser.parse_args()
    return capture(args) if args.command == "capture" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
