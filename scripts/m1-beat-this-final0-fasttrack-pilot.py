#!/usr/bin/env python3
"""Fast-track offline, research-only final0 vs deterministic SAUND audio pilot.

Input is a separately rights-reviewed local Ballroom development/validation
manifest already supported by scripts/m1-rhythm-benchmark.py. Never sees or
scores a test/sealed split. Reference events remain in the parent process;
inference child processes receive ONLY WAV path and offline model checkpoint.

No auto-downloads, registry edits, promotion, planner calls, or renders. The
checkpoint must exist locally (absolute path), and all outputs are private.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKS = 8
SAUND_CODE = (
    "import json,sys;"
    "from djai.analysis import analyze_track;"
    "a=analyze_track(sys.argv[1]);"
    "print(json.dumps({'beats_seconds':a.beats_seconds,"
    "'downbeats_seconds':a.downbeats_seconds,'bpm':a.bpm,"
    "'neural_used':a.neural_evidence.get('beat_downbeat',{}).get('used')}))"
)


def load_script(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / name)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing required benchmark: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_inputs(manifest, audio_root, checkpoint, cli, output):
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise ValueError("explicit absolute local final0 checkpoint required; no downloads")
    if not cli.is_absolute() or not cli.is_file():
        raise ValueError("explicit absolute beat_this executable required")
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite output directory")
    if manifest.get("dataset_id") != "ballroom":
        raise ValueError("this pilot uses independently annotated Ballroom only")
    if manifest.get("split") not in ("development", "validation"):
        raise ValueError("only development/validation; test and sealed forbidden")
    benchmark = load_script("m1-rhythm-benchmark.py", "m1_fasttrack_manifest")
    items = benchmark.validate_manifest(manifest, audio_root)
    if not (1 <= len(items) <= MAX_TRACKS):
        raise ValueError(f"pilot requires 1 to {MAX_TRACKS} tracks")
    if any(item.get("beats_seconds") is None or
           item.get("downbeats_seconds") is None for item in items):
        raise ValueError("independent beat and downbeat labels required")
    return items, benchmark


def _safe_id(raw):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", raw) or raw in {".", ".."}:
        raise ValueError("unsafe track ID")
    return raw


def independent_scores(ref, baseline, challenger, scorer):
    beat_ref = ref["beats_seconds"]
    down_ref = ref["downbeats_seconds"]
    b_beat = scorer.evaluate_events(baseline["beats_seconds"], beat_ref)
    c_beat = scorer.evaluate_events(challenger["beats"], beat_ref)
    b_down = scorer.evaluate_events(baseline["downbeats_seconds"], down_ref)
    c_down = scorer.evaluate_events(challenger["downbeats"], down_ref)
    return {
        "baseline_beats": b_beat,
        "final0_beats": c_beat,
        "baseline_downbeats": b_down,
        "final0_downbeats": c_down,
        "beat_f1_difference": round(c_beat["f1"] - b_beat["f1"], 4),
        "downbeat_f1_difference": round(c_down["f1"] - b_down["f1"], 4),
        "beat_regression": c_beat["f1"] < b_beat["f1"],
        "downbeat_regression": c_down["f1"] < b_down["f1"],
    }


def command_env():
    env = os.environ.copy()
    env.update({
        "DJAI_NEURAL_BEATS": "off",
        "DJAI_MODEL_ALLOW_DOWNLOAD": "0",
        "DJAI_BEAT_THIS_ALLOW_DOWNLOAD": "0",
        "DJAI_CUE_MODEL_ENABLED": "0",
        "DJAI_EMBEDDINGS_ENABLED": "0",
        "DJAI_TEMPO_CONSISTENCY": "0",
        "PYTHONPATH": str(ROOT / "src"),
    })
    return env


def _execute(command, *, env, timeout_seconds):
    begun = time.perf_counter()
    proc = subprocess.run(command, cwd=str(ROOT), env=env, text=True,
                          capture_output=True, timeout=timeout_seconds, check=False)
    elapsed = time.perf_counter() - begun
    if proc.returncode:
        raise RuntimeError(
            f"child failed (exit {proc.returncode}), stderr tail: {proc.stderr[-800:]}"
        )
    return proc.stdout, round(elapsed, 3)


def run(manifest_path, audio_root, cli, checkpoint, output, *, timeout_seconds=300):
    if not 5 <= timeout_seconds <= 1800:
        raise ValueError("timeout must be 5..1800 seconds")
    if not manifest_path.is_file():
        raise ValueError("missing reviewed manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items, benchmark = check_inputs(manifest, audio_root, checkpoint, cli, output)
    parser = load_script("m1-beat-this-benchmark.py", "m1_fasttrack_saved_tsv")
    scorer = load_script("m1-beat-this-independent-regression.py", "m1_fasttrack_scoring")
    checkpoint_hash = sha256_file(checkpoint)
    manifest_hash = sha256_file(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    predictions = output / "predictions"
    predictions.mkdir(exist_ok=False)
    rows = []
    for item in items:
        tid = _safe_id(item["id"])
        wav = (audio_root / item["audio"]).resolve()
        tsv = predictions / f"{tid}.beats"
        # Absolute existing path cannot be interpreted by Beat This's loader
        # as a downloadable model name or URL. No annotation data in process.
        _, neural_elapsed = _execute(
            [str(cli), str(wav), "-o", str(tsv), "--gpu=-1",
             "--model", str(checkpoint)],
            env=command_env(), timeout_seconds=timeout_seconds,
        )
        if not tsv.is_file():
            raise RuntimeError(f"missing external inference output: {tid}")
        prediction = parser.read_beat_tsv(tsv)
        # Fallback baseline runs in separate process with neural fully disabled.
        raw, baseline_elapsed = _execute(
            [sys.executable, "-c", SAUND_CODE, str(wav)],
            env=command_env(), timeout_seconds=timeout_seconds,
        )
        try:
            baseline = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"SAUND returned malformed JSON for {tid}") from exc
        if baseline.get("neural_used") is True:
            raise RuntimeError("deterministic baseline unexpectedly used neural grid")
        scoring = independent_scores(item, baseline, prediction, scorer)
        rows.append({
            "id": tid,
            "genre": item.get("genre"),
            "neural_seconds": neural_elapsed,
            "deterministic_seconds": baseline_elapsed,
            "baseline_bpm": baseline.get("bpm"),
            "final0_predicted_beat_count": len(prediction["beats"]),
            "final0_predicted_downbeat_count": len(prediction["downbeats"]),
            **scoring,
        })
    scores = {
        "mean_baseline_beat_f1": round(statistics.mean(
            x["baseline_beats"]["f1"] for x in rows), 4),
        "mean_final0_beat_f1": round(statistics.mean(
            x["final0_beats"]["f1"] for x in rows), 4),
        "mean_baseline_downbeat_f1": round(statistics.mean(
            x["baseline_downbeats"]["f1"] for x in rows), 4),
        "mean_final0_downbeat_f1": round(statistics.mean(
            x["final0_downbeats"]["f1"] for x in rows), 4),
        "beat_regression_count": sum(x["beat_regression"] for x in rows),
        "downbeat_regression_count": sum(x["downbeat_regression"] for x in rows),
        "total_neural_seconds": round(sum(x["neural_seconds"] for x in rows), 3),
        "total_baseline_seconds": round(sum(x["deterministic_seconds"] for x in rows), 3),
    }
    report = {
        "schema": "saund-m1-final0-fasttrack-offline-pilot-v1",
        "input_manifest_sha256": manifest_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_identifier": "operator-provided local final0; verify separately",
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "track_count": len(rows),
        "research_use_rights_attested_by_manifest": True,
        "commercial_rights_approved": False,
        "model_checkpoint_loaded_by_neural_ears_adapter": False,
        "neural_model_used_for_saund_baseline": False,
        "model_downloaded": False,
        "production_changed": False,
        "registry_changed": False,
        "planner_25ms_gate_tested": False,
        "transition_rendered": False,
        "research_70ms_tolerance_not_planner_gate": True,
        "aggregate": scores,
        "tracks": rows,
    }
    with (output / "report.json").open("x", encoding="utf-8") as dest:
        json.dump(report, dest, indent=2, sort_keys=True)
        dest.write("\n")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, required=True)
    p.add_argument("--beat-this-cli", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timeout-seconds", type=int, default=300)
    args = p.parse_args(argv)
    report = run(
        args.manifest.expanduser().resolve(),
        args.audio_root.expanduser().resolve(),
        args.beat_this_cli.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({
        "report": str(args.output_dir.resolve() / "report.json"),
        "track_count": report["track_count"],
        "split": report["split"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "aggregate": report["aggregate"],
        "per_track": [
            {"id": r["id"], "genre": r["genre"],
             "beat_f1": [r["baseline_beats"]["f1"], r["final0_beats"]["f1"]],
             "downbeat_f1": [r["baseline_downbeats"]["f1"], r["final0_downbeats"]["f1"]],
             "beat_regression": r["beat_regression"],
             "downbeat_regression": r["downbeat_regression"]}
            for r in report["tracks"]
        ],
        "production_changed": False,
        "planner_gate_tested": False,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"M1 final0 fasttrack pilot failed: {exc}") from exc
