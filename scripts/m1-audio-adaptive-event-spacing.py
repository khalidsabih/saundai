#!/usr/bin/env python3
"""M1: research-only audio-derived adaptive event-spacing versus fixed 250 ms.

Tempo candidates are estimated from the full low-frequency WAV envelope,
independently of TrackAnalysis, generator BPM, or annotated pulse times.
The synthetic oracle is used ONLY after event localization to score controls.
No parameter is selected for production. Weak or conflicting evidence abstains.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt

ROOT = Path(__file__).resolve().parents[1]
FIXED_SPACING_MS = 250.0
THRESHOLD_FRACTION = 0.30
WINDOW_SECONDS = 6.0
HOP_SECONDS = 3.0
TEMPO_RANGE_BPM = (75.0, 280.0)


def load_module(filename, label):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def estimate_pulse_period(audio, sample_rate):
    """Return an audio-only pulse estimate with explicit confidence checks.

    A low-band positive frame-energy change drives normalized autocorrelation.
    Distinct peaks in the supported tempo range must agree across local
    analysis windows; ambiguous/weak windows abstain. The estimate is a
    *periodicity hypothesis*, NOT a correct metrical beat duration.
    """
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim != 1 or signal.size < sample_rate * 4 or not np.all(np.isfinite(signal)):
        return {"status": "insufficient_evidence", "reason": "short_or_invalid_audio"}
    if not 16000 <= sample_rate <= 192000:
        return {"status": "insufficient_evidence", "reason": "unsupported_sample_rate"}
    filtered = sosfilt(
        butter(3, [35.0, 155.0], btype="bandpass", fs=sample_rate, output="sos"),
        signal,
    )
    hop = max(1, round(sample_rate * .005))
    n = len(filtered) // hop
    if n < 400:
        return {"status": "insufficient_evidence", "reason": "insufficient_frames"}
    energy = np.mean((filtered[:n * hop].reshape(n, hop)) ** 2, axis=1)
    novelty = np.maximum(0., np.diff(energy, prepend=energy[0]))
    if np.max(novelty) < 1e-10:
        return {"status": "insufficient_evidence", "reason": "weak_low_band"}
    frame_rate = sample_rate / hop
    window_frames = min(n, round(WINDOW_SECONDS * frame_rate))
    stride = max(1, round(HOP_SECONDS * frame_rate))
    estimates = []
    strengths = []
    start_positions = list(range(0, n - window_frames + 1, stride))
    if not start_positions:
        start_positions = [0]
    for start in start_positions:
        segment = novelty[start:start + window_frames]
        if np.max(segment) < 1e-10:
            continue
        centered = segment - np.mean(segment)
        if np.std(centered) < 1e-11:
            continue
        size = 1 << (2 * len(centered) - 1).bit_length()
        fft = np.fft.rfft(centered, n=size)
        acf = np.fft.irfft(fft * np.conj(fft), n=size)[:len(centered)]
        acf /= np.arange(len(centered), 0, -1)
        if acf[0] <= 1e-14:
            continue
        min_lag = max(2, round(frame_rate * 60 / TEMPO_RANGE_BPM[1]))
        max_lag = min(len(acf) - 2, round(frame_rate * 60 / TEMPO_RANGE_BPM[0]))
        if max_lag <= min_lag:
            continue
        correlations = acf[min_lag:max_lag + 1] / acf[0]
        peaks, _ = find_peaks(correlations)
        if not len(peaks):
            continue
        ranked = sorted(
            (float(correlations[index]), min_lag + int(index))
            for index in peaks
        )[::-1]
        best_strength, best_lag = ranked[0]
        if best_strength < .22:
            continue
        # A strong non-harmonic alternative makes the period ambiguous.
        competing = [
            strength for strength, lag in ranked[1:]
            if abs(lag / best_lag - 2.0) > .12
            and abs(best_lag / lag - 2.0) > .12
            and abs(lag / best_lag - 1.0) > .08
        ]
        if competing and max(competing) >= .90 * best_strength:
            continue
        estimates.append(best_lag / frame_rate)
        strengths.append(best_strength)
    if not estimates:
        return {"status": "insufficient_evidence", "reason": "weak_or_ambiguous_acf"}
    median_period = float(np.median(estimates))
    # Inconsistent windows indicate tempo change or competing periodicities:
    # this prototype abstains rather than pretending global periodicity.
    if len(estimates) < max(1, len(start_positions) // 2) or (
        max(abs(x / median_period - 1) for x in estimates) > .09
    ):
        return {
            "status": "insufficient_evidence",
            "reason": "variable_or_conflicting_audio_periodicity",
            "supported_windows": len(estimates),
            "total_windows": len(start_positions),
        }
    return {
        "status": "supported_audio_periodicity",
        "period_seconds": round(median_period, 6),
        "hypothetical_bpm": round(60.0 / median_period, 3),
        "supported_windows": len(estimates),
        "total_windows": len(start_positions),
        "median_acf_strength": round(float(np.median(strengths)), 3),
    }


def adaptive_spacing_ms(period_report):
    if period_report.get("status") != "supported_audio_periodicity":
        return None
    period = float(period_report["period_seconds"])
    if not np.isfinite(period) or period <= 0:
        return None
    # Research hypothesis only; shorter than pulse period so true rapid
    # successive attacks are not forcibly deleted as in fixed 350 ms.
    return round(float(np.clip(.55 * period * 1000., 90., 320.)), 2)


def evaluate_case(audio, sample_rate, reference, distractors, *, locator, scoring):
    period = estimate_pulse_period(audio, sample_rate)
    spacing = adaptive_spacing_ms(period)
    strategies = []
    for label, chosen in (("fixed_250", FIXED_SPACING_MS),
                          ("audio_adaptive", spacing)):
        if chosen is None:
            strategies.append({
                "name": label, "status": "abstain",
                "reason": period.get("reason", "unsupported_period"),
                "spacing_ms": None, "primary": None, "distractor": None,
            })
            continue
        detected = locator.locate_events(
            audio, sample_rate, band="low",
            minimum_spacing_ms=chosen, threshold_fraction=THRESHOLD_FRACTION,
        )
        strategies.append({
            "name": label, "status": "measured",
            "spacing_ms": chosen,
            "detected_event_count": len(detected),
            "primary": scoring.match_recovery(detected, reference),
            "distractor": scoring.match_recovery(detected, distractors),
        })
    return {"audio_periodicity": period, "strategies": strategies}


def extra_control(name, *, sample_rate=16000, seconds=12):
    """Independent fixed fixtures; truth is excluded from period estimation."""
    if name not in {"very_fast_260", "kick_omissions", "alternating_bass"}:
        raise ValueError("unknown additional negative control")
    t = np.arange(round(sample_rate * .1)) / sample_rate
    kick = np.sin(2 * np.pi * 85 * t) * np.exp(-t * 35.)
    duration = int(sample_rate * seconds)
    audio = np.zeros(duration, dtype=float)
    times = (np.arange(.5, seconds - .2, 60 / 260)
             if name == "very_fast_260" else np.arange(.5, seconds - .2, .5))
    primary = []
    distractors = []
    for i, when in enumerate(times):
        if name == "kick_omissions" and i % 5 == 2:
            continue
        index = round(float(when) * sample_rate)
        audio[index:index + len(kick)] += kick
        primary.append(float(when))
        if name == "alternating_bass" and i % 2 == 0:
            other = float(when + .23)
            if other < seconds - .1:
                p = round(other * sample_rate)
                audio[p:p + len(kick)] += .9 * kick
                distractors.append(other)
    return audio, sample_rate, primary, distractors


def run():
    scoring = load_module("m1-low-band-negative-controls.py", "m1_controls_adaptive")
    locator = load_module("m1-unconditioned-transient-localization.py", "m1_events_adaptive")
    cases = []
    for name in tuple(scoring.CASES) + (
        "very_fast_260", "kick_omissions", "alternating_bass"
    ):
        audio, sr, reference, distractors = (
            scoring.generate_control(name) if name in scoring.CASES
            else extra_control(name)
        )
        cases.append({
            "name": name,
            "reference_event_count": len(reference),
            "distractor_reference_count": len(distractors),
            **evaluate_case(audio, sr, reference, distractors,
                            locator=locator, scoring=scoring),
        })
    return {
        "schema": "saund-m1-audio-period-adaptive-peak-spacing-v1",
        "cases": cases,
        "selected_configuration": None,
        "beat_grid_used_for_period": False,
        "generator_reference_used_for_period": False,
        "reference_used_only_for_posthoc_scoring": True,
        "production_changed": False,
        "beat_grid_modified": False,
        "transition_rendered": False,
        "warning": "Audio ACF is rhythm ambiguity-prone. No threshold, beat repair, or render approval is promoted.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite report")
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 adaptive spacing study failed: {exc}") from exc
