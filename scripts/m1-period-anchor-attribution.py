#!/usr/bin/env python3
"""M1 research-only 2x2 period/anchor attribution on overlapping waveform windows.

For every *same* set of window-localized low-band events, compare:
  global audio period + local acoustic anchor (GL)
  global audio period + held global acoustic anchor (GG)
  local audio period  + local acoustic anchor (LL)
  local audio period  + held global acoustic anchor (LG)

The held anchor is inferred from the FULL waveform's audio peaks at the
corresponding period, then translated into each window's local coordinates.
Neither reference beat times nor generated BPM enter any estimate. Both
period and anchor interventions are observational diagnostics only; the
phase comparator retains all original detection thresholds. No configuration
is chosen or authorized for production, beat repair or render.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SECONDS = 8.
HOP_SECONDS = 4.
COMPETING = "competing_acoustic_phase_trains"
SUPPORTED = "supported_audio_periodicity"
ARMS = ("global_period_local_anchor", "global_period_held_anchor",
        "local_period_local_anchor", "local_period_held_anchor")


def load_script(filename, label):
    import importlib.util
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing M1 dependency: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def global_anchor_for_period(global_events, period, duration, *, comparator):
    """Infer a global acoustic anchor at a given AUDIO-derived period."""
    report = comparator.compare_phase_trains(global_events, period, duration)
    value = report.get("anchor_phase_seconds")
    return float(value) if value is not None else None


def evaluate_window_events(events, window_duration, start_seconds,
                           global_period, local_period, global_events,
                           global_duration, *, comparator):
    """All four arms share the exact same window event timestamps/strengths."""
    answers = {}
    for name, period, locked in (
        (ARMS[0], global_period, False),
        (ARMS[1], global_period, True),
        (ARMS[2], local_period, False),
        (ARMS[3], local_period, True),
    ):
        if period is None:
            answers[name] = {
                "status": "insufficient_evidence",
                "reason": "audio_periodicity_abstained",
            }
            continue
        if locked:
            global_anchor = global_anchor_for_period(
                global_events, period, global_duration, comparator=comparator
            )
            if global_anchor is None:
                answers[name] = {
                    "status": "insufficient_evidence",
                    "reason": "global_acoustic_anchor_unavailable",
                }
                continue
            # modulo relative to the WINDOW start, NOT the full-clip origin.
            local_anchor = (global_anchor - start_seconds) % period
        else:
            local_anchor = None
        report = comparator.compare_phase_trains(
            events, period, window_duration,
            anchor_phase_seconds=local_anchor,
        )
        answers[name] = report
    return answers


def compare_waveform(audio, sample_rate, *, adaptive, locator, comparator):
    samples = np.asarray(audio, dtype=np.float64)
    if (samples.ndim != 1 or not np.all(np.isfinite(samples))
            or sample_rate <= 0
            or len(samples) < round(WINDOW_SECONDS * sample_rate)):
        raise ValueError("invalid or short waveform")
    duration = len(samples) / sample_rate
    global_period_report = adaptive.estimate_pulse_period(samples, sample_rate)
    global_period = (float(global_period_report["period_seconds"])
                     if global_period_report.get("status") == SUPPORTED else None)
    # Whole-track peaks are used ONLY to infer the held global anchor. They are
    # never used to replace peaks inside a window or to decide kick identity.
    global_events = locator.locate_events(
        samples, sample_rate, band="low",
        minimum_spacing_ms=85., threshold_fraction=.02,
    ) if global_period is not None else []

    size = round(WINDOW_SECONDS * sample_rate)
    hop = round(HOP_SECONDS * sample_rate)
    windows = []
    for start in range(0, len(samples) - size + 1, hop):
        chunk = samples[start:start + size]
        offset = start / sample_rate
        local_report = adaptive.estimate_pulse_period(chunk, sample_rate)
        local_period = (float(local_report["period_seconds"])
                        if local_report.get("status") == SUPPORTED else None)
        local_events = locator.locate_events(
            chunk, sample_rate, band="low",
            minimum_spacing_ms=85., threshold_fraction=.02,
        ) if global_period is not None or local_period is not None else []
        arms = evaluate_window_events(
            local_events, len(chunk) / sample_rate, offset,
            global_period, local_period, global_events, duration,
            comparator=comparator,
        )
        windows.append({
            "start_seconds": round(offset, 3),
            "end_seconds": round((start + size) / sample_rate, 3),
            "local_audio_periodicity": local_report,
            "local_event_count": len(local_events),
            "period_delta_local_minus_global_ms": (
                round((local_period - global_period) * 1000., 3)
                if local_period is not None and global_period is not None
                else None
            ),
            "arms": arms,
        })
    return {
        "global_audio_periodicity": global_period_report,
        "global_localization_event_count": len(global_events),
        "windows": windows,
        "windows_overlap": True,
        "correct_phase_identified": False,
    }


def run(*, assessor=None):
    controls = load_script(
        "m1-competing-phase-robustness.py", "m1_attribution_controls"
    )
    if assessor is None:
        adaptive = load_script(
            "m1-audio-adaptive-event-spacing.py", "m1_attribution_acf"
        )
        locator = load_script(
            "m1-unconditioned-transient-localization.py", "m1_attribution_events"
        )
        comparator = load_script(
            "m1-competing-acoustic-phase-trains.py", "m1_attribution_comparator"
        )

        def assessor(wave, sr):
            return compare_waveform(
                wave, sr, adaptive=adaptive,
                locator=locator, comparator=comparator,
            )

    rows = []
    for seed in controls.SEEDS:
        for noise in controls.NOISE_TYPES:
            for timbre in controls.KICK_TIMBRES:
                for jitter in controls.JITTER_MS:
                    for variant, amp in (
                        ("two_phase", .06), ("two_phase", .12),
                        ("pulse_only", .12),
                    ):
                        wav, sr = controls.make_waveform(
                            seed=seed, noise_type=noise, timbre=timbre,
                            jitter_ms=jitter, variant=variant,
                            kick_amplitude=amp,
                        )
                        rows.append({
                            "seed": seed, "noise_type": noise,
                            "timbre": timbre, "jitter_ms": jitter,
                            "variant": variant, "kick_amplitude": amp,
                            **assessor(wav, sr),
                        })
    for seed in controls.SEEDS:
        for noise in controls.NOISE_TYPES:
            wav, sr = controls.make_waveform(
                seed=seed, noise_type=noise,
                timbre=controls.KICK_TIMBRES[0], jitter_ms=0.,
                variant="noise_only", kick_amplitude=0.,
            )
            rows.append({
                "seed": seed, "noise_type": noise,
                "timbre": controls.KICK_TIMBRES[0],
                "jitter_ms": 0., "variant": "noise_only",
                "kick_amplitude": 0., **assessor(wav, sr),
            })
    if len(rows) != controls.EXPECTED_CASES:
        raise AssertionError("original fixture matrix changed")

    summary = {}
    for variant in ("two_phase", "pulse_only", "noise_only"):
        for jitter in controls.JITTER_MS:
            subset = [x for x in rows
                      if x["variant"] == variant and x["jitter_ms"] == jitter]
            if not subset:
                continue
            summary[f"{variant}_jitter_{int(jitter)}ms"] = {
                "cases": len(subset),
                "arms": {
                    name: {
                        "any_window_competing_count": sum(
                            any(w["arms"][name].get("status") == COMPETING
                                for w in case["windows"])
                            for case in subset
                        ),
                        "both_windows_competing_count": sum(
                            len(case["windows"]) == 2 and all(
                                w["arms"][name].get("status") == COMPETING
                                for w in case["windows"]
                            )
                            for case in subset
                        ),
                    } for name in ARMS
                },
            }
    return {
        "schema": "saund-m1-period-anchor-attribution-v1",
        "cases": rows, "summary_by_fixture_jitter": summary,
        "arms": list(ARMS),
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "windows_overlap": True,
        "source_matrix_case_count": controls.EXPECTED_CASES,
        "detector_thresholds_modified": False,
        "selected_configuration": None,
        "generator_reference_used_in_detector": False,
        "beat_grid_used": False,
        "correct_phase_identified": False,
        "production_changed": False,
        "beat_grid_modified": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": (
            "Global anchors are inferred from audio-only peaks, not truth. "
            "All comparisons are observational and overlapping. No arm "
            "certifies musical beat identity or authorizes a transition."
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
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 period-anchor attribution failed: {exc}") from exc
