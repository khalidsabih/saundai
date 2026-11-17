#!/usr/bin/env python3
"""Research-only independent low-band event detection negative controls.

The waveform event detector sees ONLY audio, never fixture pulse positions.
Known synthetic event positions are used afterwards to score recovery.
No production tuning, beat repairs, or render authorization occurs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ((85.0, 0.12), (150.0, 0.12), (250.0, 0.12),
         (250.0, 0.30), (350.0, 0.30))
CASES = ("stable_120", "fast_200", "accelerating",
         "three_pulse_accents", "weak_kicks", "syncopated_bass", "silence")


def load_locator():
    path = ROOT / "scripts" / "m1-unconditioned-transient-localization.py"
    spec = importlib.util.spec_from_file_location("m1_global_events_negative_controls", path)
    if spec is None or spec.loader is None:
        raise ValueError("global transient locator unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_control(name, *, sample_rate=16000, seconds=12):
    """Return (waveform, sample_rate, primary_event_times, distractor_times)."""
    length = int(sample_rate * seconds)
    audio = np.zeros(length, dtype=np.float64)
    rng = np.random.default_rng(77)
    distractors = []
    if name == "stable_120":
        primary = np.arange(.5, seconds - .3, .5)
    elif name == "fast_200":
        primary = np.arange(.5, seconds - .3, .3)
    elif name == "accelerating":
        times = []
        t = .5
        while t < seconds - .3:
            times.append(t)
            t += max(.27, .52 - t * .019)
        primary = np.asarray(times)
    elif name == "three_pulse_accents":
        primary = np.arange(.5, seconds - .3, .48)
    elif name == "weak_kicks":
        primary = np.arange(.5, seconds - .3, .5)
    elif name == "syncopated_bass":
        primary = np.arange(.5, seconds - .3, .5)
        distractors = [float(t + .24) for t in primary if t + .24 < seconds - .15]
    elif name == "silence":
        return audio, sample_rate, [], []
    else:
        raise ValueError("unknown control name")
    t = np.arange(round(.11 * sample_rate)) / sample_rate
    attack = np.sin(2 * np.pi * 85 * t) * np.exp(-t * 35)
    for i, when in enumerate(primary):
        start = round(float(when) * sample_rate)
        gain = (.02 if name == "weak_kicks"
                else .4 if name == "three_pulse_accents" and i % 3
                else 1.)
        audio[start:start + len(attack)] += gain * attack
    for when in distractors:
        start = round(when * sample_rate)
        audio[start:start + len(attack)] += .65 * attack
    if name == "weak_kicks":
        audio += rng.normal(0, .004, length)
    return audio, sample_rate, [float(x) for x in primary], distractors


def match_recovery(events, reference_times, *, tolerance_ms=55.0):
    """One-to-one greedy time matching for post-hoc measurement only."""
    candidates = sorted(float(x["time_seconds"]) for x in events)
    reference = sorted(float(t) for t in reference_times)
    possible = []
    for i, expected in enumerate(reference):
        for j, observed in enumerate(candidates):
            error = abs(observed - expected) * 1000
            if error <= tolerance_ms:
                possible.append((error, i, j))
    matched_reference = set()
    matched_events = set()
    for _, i, j in sorted(possible):
        if i not in matched_reference and j not in matched_events:
            matched_reference.add(i)
            matched_events.add(j)
    return {
        "expected_count": len(reference),
        "matched_count": len(matched_reference),
        "recall": round(len(matched_reference) / len(reference), 3) if reference else None,
        "unmatched_detected_event_count": len(candidates) - len(matched_events),
    }


def run_controls(*, locator=None):
    if locator is None:
        locator = load_locator()
    cases = []
    for name in CASES:
        audio, rate, reference, distractors = generate_control(name)
        measurements = []
        for spacing_ms, threshold_fraction in SWEEP:
            events = locator.locate_events(
                audio, rate, band="low",
                minimum_spacing_ms=spacing_ms,
                threshold_fraction=threshold_fraction,
            )
            measurements.append({
                "minimum_spacing_ms": spacing_ms,
                "threshold_fraction": threshold_fraction,
                "detected_event_count": len(events),
                "primary": match_recovery(events, reference),
                "distractor": match_recovery(events, distractors),
            })
        cases.append({
            "name": name,
            "primary_reference_count": len(reference),
            "distractor_reference_count": len(distractors),
            "settings": measurements,
        })
    return {
        "schema": "saund-m1-global-peak-negative-controls-v1",
        "cases": cases,
        "selected_configuration": None,
        "generator_reference_used_in_detector": False,
        "reference_used_for_posthoc_scoring": True,
        "production_changed": False,
        "beat_grid_modified": False,
        "transition_rendered": False,
        "warning": "Synthetic controls expose failure modes; they do not validate real-world beat identity or promote any setting.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    result = run_controls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"M1 low-band control evaluation failed: {exc}") from exc
