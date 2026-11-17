#!/usr/bin/env python3
"""Research-only adversarial tests of audio ACF tempo and adaptive event spacing.

Creates independently generated WAV controls with half/double-time accents,
tempo ramps and step changes, then passes ONLY the waveform into the existing
ACF estimator and event locator. Ground-truth pulse schedules enter post-hoc
scoring only. Any supported ACF period is a periodicity hypothesis, not a
verified musical beat or an endorsement of a phase correction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "halftime_accents_120",
    "doubletime_bass_120",
    "halfbeat_bass_120",
    "tempo_step_120_to_200",
    "tempo_ramp_100_to_220",
    "three_beat_accents",
    "six_beat_accents",
    "low_kick_high_bass",
)


def load_script(name, label):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pulse_times(name, seconds=18.0):
    """Controlled schedule for scoring; NOT supplied to periodicity estimator."""
    if name in {"tempo_step_120_to_200", "tempo_ramp_100_to_220"}:
        t = .5
        out = []
        while t < seconds - .2:
            out.append(t)
            if name == "tempo_step_120_to_200":
                period = .5 if t < 9.0 else .3
            else:
                bpm = 100.0 + 120.0 * min(t / seconds, 1.0)
                period = 60.0 / bpm
            t += period
        return out
    return [float(t) for t in np.arange(.5, seconds - .2, .5)]


def generate_case(name, *, sample_rate=16000, seconds=18.0):
    if name not in CASES:
        raise ValueError("unsupported adversarial control")
    size = round(seconds * sample_rate)
    audio = np.zeros(size, dtype=np.float64)
    t = np.arange(round(.095 * sample_rate)) / sample_rate
    kick = np.sin(2 * np.pi * 87 * t) * np.exp(-t * 38)
    bass = np.sin(2 * np.pi * 118 * t) * np.exp(-t * 45)
    true_beats = pulse_times(name, seconds)
    distractors = []

    def put(when, wave, amplitude):
        start = round(when * sample_rate)
        if 0 <= start and start + len(wave) <= size:
            audio[start:start + len(wave)] += amplitude * wave

    for idx, when in enumerate(true_beats):
        if name == "halftime_accents_120":
            amp = 1.0 if idx % 2 == 0 else .20
        elif name == "low_kick_high_bass":
            amp = .12
        elif name in {"three_beat_accents", "six_beat_accents"}:
            meter = 3 if name == "three_beat_accents" else 6
            amp = 1.0 if idx % meter == 0 else .35
        else:
            amp = 1.0
        put(when, kick, amp)
        if name == "doubletime_bass_120":
            secondary = when + .25
            if secondary + .095 < seconds:
                put(secondary, bass, .85)
                distractors.append(float(secondary))
        elif name == "halfbeat_bass_120" and idx % 2 == 0:
            secondary = when + .25
            if secondary + .095 < seconds:
                put(secondary, bass, 1.2)
                distractors.append(float(secondary))
        elif name == "low_kick_high_bass":
            secondary = when + .25
            if secondary + .095 < seconds:
                put(secondary, bass, 1.1)
                distractors.append(float(secondary))
    return audio, sample_rate, true_beats, distractors


def assess_case(name, *, adaptive, locator, scorer):
    audio, sr, true_beats, distractors = generate_case(name)
    result = adaptive.evaluate_case(
        audio, sr, true_beats, distractors, locator=locator, scoring=scorer
    )
    estimate = result["audio_periodicity"]
    periods = np.diff(true_beats)
    expected_median = float(np.median(periods))
    estimated = estimate.get("period_seconds")
    # Separate recoverability of actual events from *metrical* aliasing.
    # This post-hoc comparison is not an input to the estimator.
    estimated_ratio = (float(estimated) / expected_median
                       if estimated is not None else None)
    return {
        "name": name,
        "reference_event_count": len(true_beats),
        "distractor_reference_count": len(distractors),
        "reference_median_pulse_period_seconds": round(expected_median, 6),
        "reference_period_range_seconds": [
            round(float(np.min(periods)), 6),
            round(float(np.max(periods)), 6),
        ],
        "hypothetical_period_to_reference_ratio_posthoc": (
            round(estimated_ratio, 3) if estimated_ratio is not None else None
        ),
        "potential_half_or_double_time_alias_posthoc": bool(
            estimated_ratio is not None and
            (0.43 <= estimated_ratio <= .57 or 1.8 <= estimated_ratio <= 2.2)
        ),
        **result,
    }


def run():
    adaptive = load_script("m1-audio-adaptive-event-spacing.py", "m1_adaptive_adversarial")
    locator = load_script("m1-unconditioned-transient-localization.py", "m1_locator_adversarial")
    scorer = load_script("m1-low-band-negative-controls.py", "m1_scoring_adversarial")
    cases = [
        assess_case(name, adaptive=adaptive, locator=locator, scorer=scorer)
        for name in CASES
    ]
    return {
        "schema": "saund-m1-audio-adaptive-adversarial-controls-v1",
        "cases": cases,
        "selected_configuration": None,
        "generator_reference_used_for_period": False,
        "beat_grid_used_for_period": False,
        "reference_used_only_for_posthoc_scoring": True,
        "production_changed": False,
        "beat_grid_modified": False,
        "transition_rendered": False,
        "warning": (
            "Post-hoc reference period ratio detects plausible ACF aliases "
            "in this synthetic fixture only. The estimator cannot know these "
            "reference labels. No result authorizes beat changes or rendering."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing report")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"M1 adversarial ACF research failed: {exc}") from exc
