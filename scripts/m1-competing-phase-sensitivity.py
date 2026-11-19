#!/usr/bin/env python3
"""M1 research-only weak-phase sensitivity and false-ambiguity controls.

Synthetic fixture truth only constructs WAVs and names experiments. The
audio-only periodicity estimator, peak finder, and phase comparator receive
ONLY samples/sample-rate or detected event timestamps. The report never
identifies the correct beat phase, corrects an analysis, or authorizes render.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Fixed BEFORE observing this experiment; do not select a best threshold.
KICK_AMPLITUDES = (.02, .06, .12, .25, .5)
NOISE_STDS = (0., .001, .005)
CONTROL_VARIANTS = ("stable_pulses", "strong_offbeat_only", "noise_only")
DURATION_SECONDS = 18.
SAMPLE_RATE = 16000


def load_script(filename, name):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synth_waveform(*, kick_amplitude, noise_std, variant="two_phase",
                   seed=431):
    """Fixture's known pulse positions are intentionally NOT returned."""
    if variant not in ("two_phase",) + CONTROL_VARIANTS:
        raise ValueError("unknown fixture variant")
    if kick_amplitude < 0 or noise_std < 0:
        raise ValueError("amplitudes must be nonnegative")
    rate = SAMPLE_RATE
    size = round(DURATION_SECONDS * rate)
    wave = np.zeros(size, dtype=np.float64)
    t = np.arange(round(.095 * rate)) / rate
    kick = np.sin(2 * np.pi * 87 * t) * np.exp(-38. * t)
    bass = np.sin(2 * np.pi * 118 * t) * np.exp(-45. * t)

    def place(at, signal, gain):
        index = round(at * rate)
        if index + len(signal) <= size:
            wave[index:index + len(signal)] += gain * signal

    for when in np.arange(.5, DURATION_SECONDS - .2, .5):
        if variant in {"two_phase", "stable_pulses"}:
            place(float(when), kick, float(kick_amplitude))
        if variant in {"two_phase", "strong_offbeat_only"}:
            place(float(when + .25), bass, 1.1)
    if noise_std:
        wave += np.random.default_rng(seed).normal(0., noise_std, size)
    return wave, rate


def summarize_case(wave, sample_rate, *, adaptive, locator, phase):
    result = phase.assess_audio(wave, sample_rate, adaptive=adaptive, locator=locator)
    return {
        "audio_periodicity": result["audio_periodicity"],
        "phase_evidence": result["phase_evidence"],
    }


def run():
    adaptive = load_script("m1-audio-adaptive-event-spacing.py", "m1_sens_period")
    locator = load_script("m1-unconditioned-transient-localization.py", "m1_sens_locator")
    phase = load_script("m1-competing-acoustic-phase-trains.py", "m1_sens_phase")
    rows = []
    for kick in KICK_AMPLITUDES:
        for noise in NOISE_STDS:
            wave, sr = synth_waveform(kick_amplitude=kick, noise_std=noise)
            rows.append({
                "variant": "two_phase",
                "kick_amplitude": kick,
                "offbeat_bass_amplitude": 1.1,
                "noise_std": noise,
                **summarize_case(wave, sr, adaptive=adaptive, locator=locator, phase=phase),
            })
    for variant in CONTROL_VARIANTS:
        for noise in NOISE_STDS:
            wave, sr = synth_waveform(
                kick_amplitude=.12, noise_std=noise, variant=variant
            )
            rows.append({
                "variant": variant,
                "kick_amplitude": .12 if variant == "stable_pulses" else 0.,
                "offbeat_bass_amplitude": 1.1 if variant == "strong_offbeat_only" else 0.,
                "noise_std": noise,
                **summarize_case(wave, sr, adaptive=adaptive, locator=locator, phase=phase),
            })
    return {
        "schema": "saund-m1-competing-phase-sensitivity-v1",
        "cases": rows,
        "configured_kick_amplitudes": list(KICK_AMPLITUDES),
        "configured_noise_stds": list(NOISE_STDS),
        "configured_negative_controls": list(CONTROL_VARIANTS),
        "selected_configuration": None,
        "generator_reference_used_in_detector": False,
        "beat_grid_used": False,
        "correct_phase_identified": False,
        "production_changed": False,
        "beat_grid_modified": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": (
            "Low-band competing-phase heuristic is an abstention-only research "
            "diagnostic. A single detected phase or a false ambiguity is not "
            "evidence that a track is safe or unsafe. Synthetic controls "
            "do not represent rights-cleared real music."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"M1 phase-sensitivity audit failed: {exc}") from exc
