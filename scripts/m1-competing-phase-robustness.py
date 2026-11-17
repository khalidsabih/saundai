#!/usr/bin/env python3
"""M1 shadow robustness: held-out waveform noise, timbre, jitter and seeds.

Research only. This matrix is predeclared; NO detector parameters are tuned.
Fixture beat timings are used only to synthesize waveforms and are never
passed to the waveform-only ACF, event locator or phase comparator.
A competing-phase result is evidence of acoustic ambiguity, not an unsafe
musical beat. The absence of competing evidence is not approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (101, 907)
NOISE_TYPES = ("white", "lowpass")
KICK_TIMBRES = ("decay_87hz", "chirp_65_to_125hz")
JITTER_MS = (0., 14.)
MIXED_AMPLITUDES = (.06, .12)
NOISE_STD = .002
SAMPLE_RATE = 16000
DURATION_SECONDS = 12.
# 2 seeds x 2 noise x 2 timbre x 2 jitter x 3 signal categories
# (2 mixed kick amplitudes, 1 pulse-only) = 48, plus 4 noise-only.
EXPECTED_CASES = 52


def load_script(filename, label):
    import importlib.util
    source = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(label, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import research dependency {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_noise(rng, count, noise_type):
    raw = rng.normal(size=count)
    if noise_type == "white":
        shaped = raw
    elif noise_type == "lowpass":
        shaped = lfilter([.15], [1., -.85], raw)
    else:
        raise ValueError("unknown noise type")
    std = float(np.std(shaped))
    if std <= 0:
        raise ValueError("degenerate noise")
    return shaped / std


def make_waveform(*, seed, noise_type, timbre, jitter_ms,
                  variant, kick_amplitude=.12):
    """Return audio + sample rate only; never return the fixture reference grid."""
    if variant not in {"two_phase", "pulse_only", "noise_only"}:
        raise ValueError("unknown variant")
    if timbre not in KICK_TIMBRES:
        raise ValueError("unknown timbre")
    if jitter_ms < 0 or jitter_ms > 40 or kick_amplitude < 0:
        raise ValueError("invalid fixture amplitude or jitter")
    sr = SAMPLE_RATE
    size = int(DURATION_SECONDS * sr)
    rng = np.random.default_rng(seed)
    data = np.zeros(size, dtype=np.float64)
    t = np.arange(round(.1 * sr), dtype=float) / sr
    if timbre == "decay_87hz":
        kick = np.sin(2 * np.pi * 87 * t) * np.exp(-t * 38)
    else:
        cycles = 65 * t + .5 * (125 - 65) / .1 * t * t
        kick = np.sin(2 * np.pi * cycles) * np.exp(-t * 42)
    bass = np.sin(2 * np.pi * 118 * t) * np.exp(-t * 45)

    def add(at, waveform, gain):
        index = round(float(at) * sr)
        if index >= 0 and index + len(waveform) <= size:
            data[index:index + len(waveform)] += gain * waveform

    if variant != "noise_only":
        for expected in np.arange(.5, DURATION_SECONDS - .2, .5):
            # The two phase trains have independent, deterministic timing
            # jitter. Their nominal offbeat relationship stays .25 seconds.
            shift_kick = rng.uniform(-jitter_ms, jitter_ms) / 1000
            add(expected + shift_kick, kick, kick_amplitude)
            if variant == "two_phase":
                shift_bass = rng.uniform(-jitter_ms, jitter_ms) / 1000
                add(expected + .25 + shift_bass, bass, 1.1)
    data += NOISE_STD * normalized_noise(rng, size, noise_type)
    return data, sr


def assess_waveform(audio, sr, *, adaptive, locator, phase):
    evidence = phase.assess_audio(audio, sr, adaptive=adaptive, locator=locator)
    return {
        "periodicity": evidence["audio_periodicity"],
        "phase_evidence": evidence["phase_evidence"],
    }


def run(*, assessor=None):
    if assessor is None:
        adaptive = load_script("m1-audio-adaptive-event-spacing.py", "m1_robust_acf")
        locator = load_script("m1-unconditioned-transient-localization.py", "m1_robust_locator")
        phase = load_script("m1-competing-acoustic-phase-trains.py", "m1_robust_phase")

        def assessor(audio, sr):
            return assess_waveform(
                audio, sr, adaptive=adaptive, locator=locator, phase=phase
            )

    cases = []
    for seed in SEEDS:
        for noise_type in NOISE_TYPES:
            for timbre in KICK_TIMBRES:
                for jitter in JITTER_MS:
                    for variant, amp in (
                        ("two_phase", .06), ("two_phase", .12),
                        ("pulse_only", .12),
                    ):
                        audio, sr = make_waveform(
                            seed=seed, noise_type=noise_type, timbre=timbre,
                            jitter_ms=jitter, variant=variant,
                            kick_amplitude=amp,
                        )
                        cases.append({
                            "seed": seed, "noise_type": noise_type,
                            "timbre": timbre, "jitter_ms": jitter,
                            "variant": variant, "kick_amplitude": amp,
                            **assessor(audio, sr),
                        })
    for seed in SEEDS:
        for noise_type in NOISE_TYPES:
            audio, sr = make_waveform(
                seed=seed, noise_type=noise_type,
                timbre=KICK_TIMBRES[0], jitter_ms=0.,
                variant="noise_only", kick_amplitude=0.,
            )
            cases.append({
                "seed": seed, "noise_type": noise_type,
                "timbre": KICK_TIMBRES[0], "jitter_ms": 0.,
                "variant": "noise_only", "kick_amplitude": 0.,
                **assessor(audio, sr),
            })

    if len(cases) != EXPECTED_CASES:
        raise AssertionError("incomplete fixed control matrix")
    totals = {}
    for variant in ("two_phase", "pulse_only", "noise_only"):
        rows = [r for r in cases if r["variant"] == variant]
        totals[variant] = {
            "cases": len(rows),
            "competing_phase_count": sum(
                r["phase_evidence"].get("status") == "competing_acoustic_phase_trains"
                for r in rows
            ),
            "periodicity_abstain_count": sum(
                r["periodicity"].get("status") != "supported_audio_periodicity"
                for r in rows
            ),
        }
    return {
        "schema": "saund-m1-competing-phase-robustness-v1",
        "cases": cases, "summary_by_fixture": totals,
        "predeclared_seeds": list(SEEDS),
        "predeclared_noise_types": list(NOISE_TYPES),
        "predeclared_timbres": list(KICK_TIMBRES),
        "predeclared_jitter_ms": list(JITTER_MS),
        "mixed_kick_amplitudes": list(MIXED_AMPLITUDES),
        "noise_std": NOISE_STD,
        "selected_configuration": None,
        "detector_thresholds_modified": False,
        "generator_reference_used_in_detector": False,
        "beat_grid_used": False,
        "correct_phase_identified": False,
        "production_changed": False,
        "beat_grid_modified": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": (
            "Detection of rival phases is not evidence of an unsafe grid. "
            "Failure to detect rivals is not a safe-phase certification. "
            "Synthetic fixture family, seeds and timbres remain narrow."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite report")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"M1 shadow robustness audit failed: {exc}") from exc
