#!/usr/bin/env python3
"""Offline, source-isolated acoustic phase *proxy* for a private M1 render.

This is intentionally NOT independent 25 ms beat-phase certification:
low-frequency transients are not human beat annotations, and the isolated
deck signals are reconstructed upstream of mix processing. The verifier
checks that a newly compiled, signed plan reproduces the provided WAV,
then measures relative pulse timing in isolated deck audio. It fails closed
on inadequate or ambiguous pulse coverage and never changes a safety gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Match the M1 canary's offline research configuration before djai imports.
os.environ["DJAI_NEURAL_BEATS"] = "1"
os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"


def acoustic_pulses(y, sr):
    """Find robust bass-band envelope peaks; diagnostic proxy, not beat labels."""
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, find_peaks, sosfilt

    mono = np.mean(np.asarray(y, dtype=np.float64), axis=0)
    if mono.size < sr or not np.isfinite(mono).all():
        return []
    sos = butter(3, [35, min(165, 0.4 * sr)], btype="bandpass", fs=sr, output="sos")
    filtered = sosfilt(sos, mono)
    envelope = np.sqrt(np.maximum(
        0.0, uniform_filter1d(filtered * filtered, size=max(3, int(sr * 0.022)))
    ))
    hop = max(1, int(sr * 0.005))
    env = envelope[::hop]
    floor, high = np.percentile(env, [25, 95])
    contrast = float(high - floor)
    if contrast < 1e-6:
        return []
    indices, _ = find_peaks(
        env,
        distance=max(1, int(0.31 * sr / hop)),
        prominence=0.16 * contrast,
        height=float(floor + 0.22 * contrast),
    )
    return [round(float(i * hop / sr), 6) for i in indices]


def compare_pulse_tracks(a_times, b_times, duration, *, max_pair_delta=0.20):
    """One-to-one chronologically ordered pulse pairs, subtract initial phase.

    Nearest matches are diagnostic only: different drum timbres and accented
    subdivisions can bias the estimate. Insufficient coverage => inconclusive.
    """
    import numpy as np
    a = np.asarray(a_times, dtype=float)
    b = np.asarray(b_times, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite acoustic event times")
    if np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
        raise ValueError("pulse times must increase")
    pairs = []
    i = j = 0
    while i < len(a) and j < len(b):
        delta = float(b[j] - a[i])
        if abs(delta) <= max_pair_delta:
            pairs.append((float(a[i]), delta))
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    if not pairs:
        return {"status": "inconclusive", "reason": "no paired acoustic pulses", "matched": 0}
    times = np.array([x[0] for x in pairs])
    raw = np.array([x[1] * 1000 for x in pairs])
    # Use only the first two seconds for reference: centering all results at
    # the median of the whole window would obscure progressive drift.
    initial = raw[times <= min(2.0, duration / 5.0)]
    if len(initial) < 2:
        return {"status": "inconclusive", "reason": "insufficient initial pulses", "matched": len(pairs)}
    anchor = float(np.median(initial))
    residuals = raw - anchor
    coverage = float((times[-1] - times[0]) / duration) if len(times) > 1 else 0.0
    max_gap = float(max(np.diff(times))) if len(times) > 1 else float(duration)
    data = {
        "matched": len(pairs),
        "a_pulses": len(a),
        "b_pulses": len(b),
        "coverage_fraction": round(coverage, 5),
        "max_pair_gap_seconds": round(max_gap, 5),
        "initial_relative_offset_ms": round(anchor, 3),
        "max_absolute_relative_pulse_drift_ms": round(float(np.max(abs(residuals))), 3),
        "p95_absolute_relative_pulse_drift_ms": round(float(np.percentile(abs(residuals), 95)), 3),
        "points": [
            {"time_seconds": round(t, 4), "raw_offset_ms": round(x, 3),
             "relative_drift_ms": round(d, 3)}
            for t, x, d in zip(times.tolist(), raw.tolist(), residuals.tolist())
        ],
    }
    # A pulse match is only informative if it covers the start, middle and
    # tail. The 25 ms comparison is NOT proof of the interdeck phase limit.
    reliable = (
        len(pairs) >= 12 and len(pairs) >= 0.60 * min(len(a), len(b))
        and coverage >= 0.70 and max_gap <= 1.6
        and times[0] <= 1.5 and times[-1] >= duration - 1.5
    )
    data["status"] = "acoustic_proxy_measured" if reliable else "inconclusive"
    if not reliable:
        data["reason"] = "insufficient start/end, density or contiguous pulse coverage"
    data["proxy_within_25ms"] = (
        bool(data["max_absolute_relative_pulse_drift_ms"] <= 25.0)
        if reliable else None
    )
    return data


def verify_private_render(manifest, audio_root, canary_report, wav, samples):
    import importlib.util
    import numpy as np
    import soundfile as sf

    from djai.analysis import analyze_track
    from djai.audio import load_audio
    from djai.planner import plan_transition
    from djai.renderer import (
        _extract_with_padding, _render_segment, _time_stretch_phase_warp,
        render_transition_audio_from_loaded,
    )
    from djai.performance.program import transition_program_from_dict

    spec = importlib.util.spec_from_file_location(
        "m1_canary", ROOT / "scripts" / "m1-real-transition-render-canary.py"
    )
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)
    root = audio_root.expanduser().resolve()
    raw = manifest.read_bytes()
    pairs = canary.validate_manifest(json.loads(raw), root)
    if len(pairs) != 1:
        raise ValueError("this verifier currently supports exactly one approved pair")
    identity, path_a, path_b = pairs[0]
    supplied = json.loads(canary_report.read_text())
    if supplied.get("source_manifest_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("canary manifest SHA256 does not match")
    row = next((x for x in supplied["pairs"] if x["id"] == identity), None)
    if not row or row.get("status") != "technical_check_passed":
        raise ValueError("canary must contain a technically passing pair")
    plan_summary = row["plan"]
    os.environ["DJAI_DECK_PHASE_WARP_SAMPLES"] = str(samples)
    a = analyze_track(str(path_a))
    b = analyze_track(str(path_b))
    plan = plan_transition(a, b, style="safe", tempo_policy="safe", force=False)
    for key, value in (
        ("technique", plan.technique),
        ("transition_bars", plan.transition_bars),
        ("transition_seconds", plan.transition_seconds),
        ("exit_time_a", plan.exit_time_a),
        ("entry_time_b", plan.entry_time_b),
    ):
        if plan_summary.get(key) != value:
            raise ValueError(f"recompiled plan does not match rendered canary: {key}")
    phase = plan.metadata.get("deck_performance", {}).get("phase_integrity", {})
    if plan.forced or plan.requires_confirmation or phase.get("safe") is not True:
        raise ValueError("recompiled plan is not automatically safe")
    program = transition_program_from_dict(dict(plan.performance_program or {}))
    if program.source_mode != "master" or program.annotations.get("incoming_alignment", "full") != "full":
        raise ValueError("isolated proxy only supports full-alignment master transitions")
    sr = 44100
    audio_a, _ = load_audio(path_a, sample_rate=sr, mono=False)
    audio_b, _ = load_audio(path_b, sample_rate=sr, mono=False)
    rendered = render_transition_audio_from_loaded(
        audio_a, audio_b, plan, sample_rate=sr, pre_roll_seconds=4, post_roll_seconds=4
    )
    disk, disk_sr = sf.read(wav, dtype="float32", always_2d=True)
    disk = disk.T
    if disk_sr != sr or disk.shape != rendered.preview.shape:
        raise ValueError("private WAV sample rate or dimensions differ from fresh render")
    max_abs_difference = float(np.max(np.abs(disk - rendered.preview)))
    if max_abs_difference > 0.0005:
        raise ValueError(
            f"private WAV does not reproduce the signed plan (max diff {max_abs_difference:.6f})"
        )
    length = int(round(plan.transition_seconds * sr))
    deck_a = _render_segment(
        audio_a, sr, plan.exit_time_a, plan.source_end_a,
        rate=plan.render_rate_a, gain=1.0, target_length=length
    )
    if not phase.get("enabled") or not phase.get("points"):
        raise ValueError("no enabled phase-warp points in signed plan")
    source_b = _extract_with_padding(
        audio_b, sr, plan.source_start_b, plan.source_duration_b
    )
    deck_b = _time_stretch_phase_warp(
        source_b, sample_rate=sr, base_rate=plan.render_rate_b,
        target_length=length, correction_points=phase["points"]
    )
    acoustic = compare_pulse_tracks(
        acoustic_pulses(deck_a, sr), acoustic_pulses(deck_b, sr),
        plan.transition_seconds,
    )
    return {
        "schema_version": "saund-m1-rendered-phase-acoustic-proxy-v1",
        "status": acoustic["status"],
        "qualification": (
            "Acoustic-proxy-only; reconstructed isolated pre-mix deck signals, "
            "not independent annotated beats or final-mix 25ms certification."
        ),
        "rendered_wav_reproduced": True,
        "rendered_wav_max_sample_difference": round(max_abs_difference, 8),
        "identity": identity,
        "plan": {
            "technique": plan.technique, "exit_time_a": plan.exit_time_a,
            "entry_time_b": plan.entry_time_b, "transition_seconds": plan.transition_seconds,
            "bars": plan.transition_bars, "phase_sample_count": samples,
        },
        "planner_predicted_residual_ms": phase["max_residual_ms"],
        "acoustic": acoustic,
        "independent_beat_annotations_present": False,
        "independent_25ms_phase_certified": False,
        "production_changed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--canary-report", required=True, type=Path)
    parser.add_argument("--rendered-wav", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase-samples", type=int, choices=(9, 17), default=17)
    parser.add_argument("--confirm-render-rights", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_render_rights:
        parser.error("explicit --confirm-render-rights required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists: refusing overwrite")
    if os.environ.get("DJAI_TEMPO_CONSISTENCY") != "1":
        parser.error("set DJAI_TEMPO_CONSISTENCY=1 to match this M1 render")
    result = verify_private_render(
        args.manifest, args.audio_root, args.canary_report,
        args.rendered_wav, args.phase_samples,
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "status": result["status"],
        "rendered_wav_reproduced": result["rendered_wav_reproduced"],
        "planner_predicted_residual_ms": result["planner_predicted_residual_ms"],
        "acoustic": {k:v for k,v in result["acoustic"].items() if k != "points"},
        "independent_25ms_phase_certified": False,
        "report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
