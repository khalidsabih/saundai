#!/usr/bin/env python3
"""M1 shadow-only competing acoustic phase trains; never a beat-phase authority.

The audio-only ACF supplies a period hypothesis, and a globally operating
low-band energy-rise detector supplies timestamped acoustic peaks. Neither
the generator pulse schedule nor TrackAnalysis beat grid enters these steps.
This script asks whether *two* event trains exist about half a period apart.
It NEVER labels either train as a true kick or safe beat. Ground-truth events
on synthetic controls are used only for the separate prior scoring reports.
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
    "stable_120", "fast_200", "syncopated_bass",
    "halftime_accents_120", "doubletime_bass_120",
    "halfbeat_bass_120", "tempo_step_120_to_200",
    "tempo_ramp_100_to_220", "three_beat_accents",
    "six_beat_accents", "low_kick_high_bass", "silence",
)


def load_script(filename, name):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_phase_trains(events, period_seconds, duration_seconds, *,
                         radius_ms=42.0, min_cycles=12, min_coverage=.60,
                         min_secondary_to_primary=.006):
    """Measure repeated opposite-phase audio attacks without deciding identity.

    An anchor is inferred from the strongest peaks' modulo-period clustering.
    Every cycle is then checked for a separate peak near that anchor and near
    its half-period opposite. This deliberately refuses 'correct phase':
    even one apparently dominant event train is not proof of kick identity.
    Missing/weak events and ambiguous periodicity always yield abstention.
    """
    if (period_seconds is None or not math.isfinite(float(period_seconds))
            or not .20 <= float(period_seconds) <= .82):
        return {"status": "insufficient_evidence", "reason": "unsupported_period"}
    period = float(period_seconds)
    if (not math.isfinite(float(duration_seconds)) or duration_seconds < 6
            or not 0 < radius_ms < period * 250):
        return {"status": "insufficient_evidence", "reason": "invalid_geometry"}
    clean = []
    for item in events:
        t = float(item.get("time_seconds", float("nan")))
        strength = float(item.get("rise_energy", float("nan")))
        if not math.isfinite(t) or not math.isfinite(strength) or strength < 0:
            return {"status": "insufficient_evidence", "reason": "invalid_acoustic_events"}
        if 0 <= t < duration_seconds and strength > 0:
            clean.append((t, strength))
    if len(clean) < 2 * min_cycles:
        return {"status": "insufficient_evidence", "reason": "too_few_localized_events",
                "localized_event_count": len(clean)}
    clean.sort()
    times = np.array([p[0] for p in clean])
    weights = np.array([p[1] for p in clean])
    # Dominant-event circular mean: no known generator pulse times or grid.
    # Top quartile suppresses weak extra peaks when finding the anchor.
    cut = float(np.quantile(weights, .75))
    top = weights >= cut
    angles = (2 * np.pi / period) * times[top]
    vector = np.sum(weights[top] * np.exp(1j * angles))
    if abs(vector) <= 1e-12:
        return {"status": "insufficient_evidence", "reason": "unclear_phase_anchor"}
    origin = (float(np.angle(vector)) % (2 * np.pi)) * period / (2 * np.pi)
    radius = radius_ms / 1000.
    # Ignore edges: all compared slots must fall inside the actual WAV.
    cycles = np.arange(math.ceil((radius - origin) / period),
                       math.floor((duration_seconds - radius - period / 2 - origin) / period) + 1)
    if len(cycles) < min_cycles:
        return {"status": "insufficient_evidence", "reason": "too_few_complete_cycles"}
    strengths = []
    seen_indices = set()
    for shift in (0., period / 2):
        values = []
        for index in cycles:
            center = origin + float(index) * period + shift
            left = int(np.searchsorted(times, center - radius, side="left"))
            right = int(np.searchsorted(times, center + radius, side="right"))
            if left == right:
                values.append(0.)
            else:
                best = left + int(np.argmax(weights[left:right]))
                values.append(float(weights[best]))
                seen_indices.add(best)
        strengths.append(np.array(values, dtype=float))
    primary, secondary = strengths
    count = int(len(cycles))
    first_hits = int(np.count_nonzero(primary))
    second_hits = int(np.count_nonzero(secondary))
    cover_first = first_hits / count
    cover_second = second_hits / count
    median_first = float(np.median(primary[primary > 0])) if first_hits else 0.
    median_second = float(np.median(secondary[secondary > 0])) if second_hits else 0.
    relative = min(median_first, median_second) / max(median_first, median_second, 1e-15)
    competing = (cover_first >= min_coverage and cover_second >= min_coverage
                 and relative >= min_secondary_to_primary)
    return {
        "status": "competing_acoustic_phase_trains" if competing else "insufficient_evidence",
        "reason": ("two_sustained_opposite_phase_peak_trains" if competing
                   else "not_enough_independent_opposite_phase_support"),
        "period_seconds": round(period, 6),
        "anchor_phase_seconds": round(origin, 6),
        "complete_cycles": count,
        "anchor_event_coverage": round(cover_first, 3),
        "opposite_event_coverage": round(cover_second, 3),
        "weaker_to_stronger_median_rise_ratio": round(relative, 5),
        "events_used_by_either_phase": len(seen_indices),
        "localized_event_count": len(clean),
        "cannot_identify_correct_beat_phase": True,
        "authority": "shadow_abstain_only",
    }


def assess_audio(audio, sample_rate, *, adaptive, locator):
    periodicity = adaptive.estimate_pulse_period(audio, sample_rate)
    if periodicity.get("status") != "supported_audio_periodicity":
        return {
            "audio_periodicity": periodicity,
            "phase_evidence": {
                "status": "insufficient_evidence",
                "reason": "periodicity_abstained",
                "authority": "shadow_abstain_only",
            },
        }
    # Low threshold and 85ms distance retain secondary candidate events.
    # These are NOT kick classifications; the candidate itself can be false.
    peaks = locator.locate_events(
        audio, sample_rate, band="low",
        minimum_spacing_ms=85., threshold_fraction=.02,
    )
    return {
        "audio_periodicity": periodicity,
        "phase_evidence": compare_phase_trains(
            peaks, periodicity["period_seconds"], len(audio) / sample_rate,
        ),
    }


def run():
    controls = load_script("m1-low-band-negative-controls.py", "m1_competing_old_controls")
    adversarial = load_script("m1-audio-adaptive-adversarial-controls.py", "m1_competing_adversarial")
    adaptive = load_script("m1-audio-adaptive-event-spacing.py", "m1_competing_adaptive")
    locator = load_script("m1-unconditioned-transient-localization.py", "m1_competing_locator")
    cases = []
    for name in CASES:
        if name in controls.CASES:
            audio, sample_rate, _, _ = controls.generate_control(name)
        else:
            audio, sample_rate, _, _ = adversarial.generate_case(name)
        cases.append({
            "name": name,
            **assess_audio(audio, sample_rate, adaptive=adaptive, locator=locator),
        })
    return {
        "schema": "saund-m1-competing-acoustic-phase-trains-v1",
        "cases": cases,
        "generator_reference_used": False,
        "beat_grid_used": False,
        "true_beat_phase_identified": False,
        "production_changed": False,
        "beat_grid_modified": False,
        "plan_generated": False,
        "transition_rendered": False,
        "selected_configuration": None,
        "warning": (
            "Two phase trains do not identify the kick train. A negative result "
            "also does not establish a safe grid. All decisions are shadow-only."
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
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"M1 competing phase research failed: {exc}") from exc
