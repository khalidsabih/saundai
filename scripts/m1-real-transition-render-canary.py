#!/usr/bin/env python3
"""Real production-planner/renderer M1 canary on privately licensed music pairs.

OFFLINE ONLY. This tests SAUND's currently selected rhythm grids, NOT the
frozen 0.50 shadow challenger. No production policy is modified. The operator
must separately approve these files for local derived-audio rendering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Runtime model downloads / adaptive model writes are disabled before imports.
os.environ["DJAI_NEURAL_BEATS"] = "1"
os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"

MAX_ABSOLUTE_SAMPLE_PEAK = 0.995
MAX_SILENT_FRACTION = 0.50
SILENCE_THRESHOLD = 0.0005
MAX_BOUNDARY_JUMP = 0.80
MAX_BPM_GRID_DIFFERENCE = 2.0


def is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def grid_difference(track):
    import statistics
    beats = track.beats_seconds
    if len(beats) < 4 or any(b <= a for a, b in zip(beats, beats[1:])):
        return None
    bpm = 60 / statistics.median(b - a for a, b in zip(beats, beats[1:]))
    return abs(bpm - track.bpm)


def audio_metrics(rendered):
    import numpy as np
    y = rendered.preview
    if y.ndim != 2 or y.shape[0] not in (1, 2) or y.shape[-1] < 2:
        raise ValueError("renderer returned invalid audio shape")
    if not np.isfinite(y).all():
        raise ValueError("renderer returned nonfinite samples")
    peak = float(np.max(np.abs(y)))
    silent_fraction = float(np.mean(np.max(np.abs(y), axis=0) <= SILENCE_THRESHOLD))
    window_edges = [
        rendered.pre.shape[-1],
        rendered.pre.shape[-1] + rendered.transition.shape[-1],
    ]
    jumps = [
        float(np.max(np.abs(y[:, t] - y[:, t-1])))
        for t in window_edges if 0 < t < y.shape[-1]
    ]
    return {
        "sample_rate": rendered.sample_rate,
        "channels": int(y.shape[0]),
        "duration_seconds": round(y.shape[-1] / rendered.sample_rate, 4),
        "peak_absolute": round(peak, 6),
        "silent_frame_fraction": round(silent_fraction, 6),
        "max_component_boundary_sample_jump": round(max(jumps, default=0), 6),
        "passed": peak <= MAX_ABSOLUTE_SAMPLE_PEAK
                  and silent_fraction <= MAX_SILENT_FRACTION
                  and max(jumps, default=0) <= MAX_BOUNDARY_JUMP,
    }


def validate_manifest(manifest, audio_root):
    if manifest.get("schema_version") != "saund-m1-render-canary-v1":
        raise ValueError("invalid render manifest schema")
    if manifest.get("rights_confirmed") is not True or manifest.get("render_permission_confirmed") is not True:
        raise ValueError("explicit local rendering rights approval is required")
    if manifest.get("split") != "development":
        raise ValueError("only non-test development music is allowed in render canary")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 8:
        raise ValueError("requires 1-8 approved track pairs")
    seen = set()
    files = []
    root = audio_root.expanduser().resolve()
    for item in pairs:
        identity = item.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", identity):
            raise ValueError("invalid pair ID")
        if identity in seen:
            raise ValueError("duplicate pair ID")
        seen.add(identity)
        pair = []
        for field in ("track_a", "track_b"):
            rel = item.get(field)
            if not isinstance(rel, str) or Path(rel).is_absolute():
                raise ValueError("only relative audio paths under --audio-root are allowed")
            file = (root / rel).resolve()
            if not file.is_relative_to(root) or not file.is_file() or file.suffix.lower() not in (".wav", ".flac", ".mp3", ".aiff"):
                raise ValueError(f"missing or disallowed audio: {field}")
            pair.append(file)
        if pair[0] == pair[1]:
            raise ValueError("pair cannot mix a track into itself")
        files.append((identity, *pair))
    return files


def evaluate_pair(identity, path_a, path_b, output_root):
    import numpy as np
    from djai.analysis import analyze_track
    from djai.planner import DifficultTransitionError, plan_transition
    from djai.renderer import render_transition_audio
    from djai.audio import write_audio

    a = analyze_track(str(path_a))
    b = analyze_track(str(path_b))
    differences = [grid_difference(a), grid_difference(b)]
    report = {
        "id": identity,
        "selected_grid_bpm_differences": [
            round(v, 5) if v is not None else None for v in differences
        ],
        "status": "blocked",
        "rendered": False,
    }
    if any(v is None or v > MAX_BPM_GRID_DIFFERENCE for v in differences):
        report["block_reason"] = "selected_bpm_vs_beat_grid_disagreement"
        return report
    try:
        # Never force unsafe planning. This uses the real signed execution plan.
        plan = plan_transition(a, b, style="safe", tempo_policy="safe", force=False)
    except DifficultTransitionError as exc:
        report["block_reason"] = f"planner_abstained: {exc}"
        return report
    if plan.requires_confirmation or plan.forced:
        report["block_reason"] = "plan_requires_confirmation"
        return report
    rendered = render_transition_audio(plan, pre_roll_seconds=4.0, post_roll_seconds=4.0)
    report["metrics"] = audio_metrics(rendered)
    report["rendered"] = True
    report["plan"] = {
        "technique": str(plan.technique),
        "transition_bars": plan.transition_bars,
        "transition_seconds": plan.transition_seconds,
        "exit_time_a": plan.exit_time_a,
        "entry_time_b": plan.entry_time_b,
        "source_bpm_a": plan.source_bpm_a,
        "source_bpm_b": plan.source_bpm_b,
        "target_bpm": plan.target_bpm,
        "quality": str(plan.quality),
        "requires_confirmation": plan.requires_confirmation,
    }
    wav_path = output_root / (identity + ".wav")
    if wav_path.exists() or wav_path.is_symlink():
        raise ValueError(f"refusing to overwrite rendered audio: {wav_path}")
    write_audio(wav_path, rendered.preview, rendered.sample_rate)
    report["private_preview_filename"] = wav_path.name
    report["status"] = "technical_check_passed" if report["metrics"]["passed"] else "technical_check_failed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-render-rights", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_render_rights:
        raise ValueError("explicit --confirm-render-rights is required")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("output directory already exists: no overwrite allowed")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    pairs = validate_manifest(manifest, args.audio_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for identity, a, b in pairs:
        try:
            results.append(evaluate_pair(identity, a, b, args.output_dir))
        except Exception as exc:
            # Report failure but do not silently mark a failed render successful.
            results.append({"id": identity, "status": "error", "rendered": False,
                            "reason": f"{type(exc).__name__}: {exc}"})
    summary = {
        "schema_version": "saund-m1-real-render-canary-v1",
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "pair_count": len(pairs),
        "actual_renders": sum(r.get("rendered", False) for r in results),
        "technical_checks_passed": sum(r["status"] == "technical_check_passed" for r in results),
        "abstained_or_blocked": sum(r["status"] == "blocked" for r in results),
        "errors": sum(r["status"] == "error" for r in results),
        "pairs": results,
        "production_changed": False,
        "challenger_grid_applied": False,
        "human_listening_review_completed": False,
        "disclaimer": "Runs the real planner and renderer on CURRENT selected grids. Peak, silence and sample-jump metrics are technical canary checks, not proof of audible mix quality.",
    }
    with (args.output_dir / "canary-report.json").open("x") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({k: summary[k] for k in (
        "pair_count", "actual_renders", "technical_checks_passed",
        "abstained_or_blocked", "errors", "production_changed",
        "challenger_grid_applied", "human_listening_review_completed"
    )}, indent=2))
    print(f"Private report: {args.output_dir / 'canary-report.json'}")
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 real render canary failed: {exc}") from exc
