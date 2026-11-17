#!/usr/bin/env python3
"""M1: shadow-only local-versus-global acoustic phase stability experiment.

Run the SAME audio-only ACF + event locator + competing-phase comparator on
the entire waveform and on overlapping 8 s windows. No detector threshold,
minimum-cycle count, peak radius, phase coverage or rival-strength criterion
is modified. Locality changes the evidence interval, NOT beat authority.

The generated fixtures are passed ONLY as samples/sample rate to the
detector. Synthetic schedules, known nominal BPM and track beat grids never
enter the periodicity or phase decision. Positive results show two acoustic
trains, never which one is musically correct. The windows overlap and MUST
NOT be counted as independent validation trials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SECONDS = 8.
HOP_SECONDS = 4.
COMPETING = "competing_acoustic_phase_trains"
SUPPORTED = "supported_audio_periodicity"


def load_script(filename, name):
    import importlib.util
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"research dependency unavailable: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_local_vs_global(audio, sample_rate, *, assessor):
    """Call waveform-only assessor on full clip and each raw window.

    assessor only sees (samples, rate). Its result has audio_periodicity and
    phase_evidence keys exactly like phase.assess_audio; no oracle enters.
    """
    samples = np.asarray(audio, dtype=np.float64)
    if samples.ndim != 1 or sample_rate <= 0 or not np.all(np.isfinite(samples)):
        raise ValueError("invalid waveform")
    total = len(samples) / sample_rate
    if total < WINDOW_SECONDS:
        raise ValueError("audio too short for fixed 8 second windows")
    global_report = assessor(samples, sample_rate)
    window_size = int(round(WINDOW_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    starts = list(range(0, len(samples) - window_size + 1, hop))
    window_reports = []
    for offset in starts:
        segment = samples[offset:offset + window_size]
        evidence = assessor(segment, sample_rate)
        window_reports.append({
            "start_seconds": round(offset / sample_rate, 3),
            "end_seconds": round((offset + window_size) / sample_rate, 3),
            "audio_periodicity": evidence["audio_periodicity"],
            "phase_evidence": evidence["phase_evidence"],
        })
    supported_periods = [
        float(row["audio_periodicity"]["period_seconds"])
        for row in window_reports
        if row["audio_periodicity"].get("status") == SUPPORTED
    ]
    period_range_ms = (
        round((max(supported_periods) - min(supported_periods)) * 1000., 3)
        if len(supported_periods) >= 2 else None
    )
    n_competing = sum(
        row["phase_evidence"].get("status") == COMPETING
        for row in window_reports
    )
    # BOTH overlapping windows must independently produce competing evidence
    # as an analysis computation. Their audio is shared, so this is not
    # independent validation and cannot approve/veto a production transition.
    recurrent = n_competing >= 2
    return {
        "global": global_report,
        "local_windows": window_reports,
        "local_window_count": len(window_reports),
        "supported_period_window_count": len(supported_periods),
        "local_period_range_ms": period_range_ms,
        "local_competing_window_count": n_competing,
        "local_status": (
            "recurrent_competing_acoustic_evidence_shadow_only" if recurrent
            else "insufficient_evidence"
        ),
        "windows_overlap": True,
        "correct_phase_identified": False,
    }


def run(*, assessor=None):
    robustness = load_script(
        "m1-competing-phase-robustness.py", "m1_local_robustness_controls"
    )
    if assessor is None:
        phase = load_script(
            "m1-competing-acoustic-phase-trains.py", "m1_local_phase"
        )
        adaptive = load_script(
            "m1-audio-adaptive-event-spacing.py", "m1_local_adaptive"
        )
        locator = load_script(
            "m1-unconditioned-transient-localization.py", "m1_local_events"
        )

        def assessor(wave, sr):
            return phase.assess_audio(
                wave, sr, adaptive=adaptive, locator=locator
            )

    cases = []
    # Reuse exactly the original predeclared 52-case robustness matrix:
    # 32 two-phase, 16 single-phase, four noise-only.
    configurations = []
    for seed in robustness.SEEDS:
        for noise in robustness.NOISE_TYPES:
            for timbre in robustness.KICK_TIMBRES:
                for jitter in robustness.JITTER_MS:
                    configurations.extend([
                        (seed, noise, timbre, jitter, "two_phase", .06),
                        (seed, noise, timbre, jitter, "two_phase", .12),
                        (seed, noise, timbre, jitter, "pulse_only", .12),
                    ])
    for seed in robustness.SEEDS:
        for noise in robustness.NOISE_TYPES:
            configurations.append((
                seed, noise, robustness.KICK_TIMBRES[0], 0.,
                "noise_only", 0.,
            ))
    if len(configurations) != robustness.EXPECTED_CASES:
        raise AssertionError("original 52-case matrix mismatch")

    for seed, noise, timbre, jitter, variant, amp in configurations:
        wave, sr = robustness.make_waveform(
            seed=seed, noise_type=noise, timbre=timbre,
            jitter_ms=jitter, variant=variant, kick_amplitude=amp,
        )
        results = measure_local_vs_global(wave, sr, assessor=assessor)
        cases.append({
            "seed": seed, "noise_type": noise, "timbre": timbre,
            "jitter_ms": jitter, "variant": variant,
            "kick_amplitude": amp, **results,
        })

    summary = {}
    for variant in ("two_phase", "pulse_only", "noise_only"):
        for jitter in robustness.JITTER_MS:
            subset = [r for r in cases
                      if r["variant"] == variant and r["jitter_ms"] == jitter]
            if not subset:
                continue
            key = f"{variant}_jitter_{int(jitter)}ms"
            summary[key] = {
                "cases": len(subset),
                "global_competing_count": sum(
                    r["global"]["phase_evidence"].get("status") == COMPETING
                    for r in subset
                ),
                "recurrent_local_competing_count": sum(
                    r["local_status"] ==
                    "recurrent_competing_acoustic_evidence_shadow_only"
                    for r in subset
                ),
                "some_local_competing_count": sum(
                    r["local_competing_window_count"] > 0 for r in subset
                ),
                "both_local_periods_supported_count": sum(
                    r["supported_period_window_count"] ==
                    r["local_window_count"]
                    for r in subset
                ),
            }
    return {
        "schema": "saund-m1-window-local-competing-phase-v1",
        "cases": cases,
        "summary_by_fixture_jitter": summary,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": HOP_SECONDS,
        "local_windows_overlap": True,
        "fixed_source_matrix_case_count": robustness.EXPECTED_CASES,
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
            "Overlapping windows are NOT independent validation. "
            "Any competing-phase report indicates acoustic ambiguity only; "
            "the absence of evidence is NOT proof of a correct beat. "
            "No metric authorizes production veto, beat correction or render."
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
        raise SystemExit(f"M1 local phase research failed: {exc}") from exc
