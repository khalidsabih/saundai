#!/usr/bin/env python3
"""Research-only robustness sweep for independent low-band transient events.

The detector localizes audio events globally, before looking at any beat grid.
This script compares how low-band association changes under a predeclared
spacing/threshold sweep; it never chooses the 'best' setting per track or
modifies a beat grid. Reporting windows identify known synthetic regions,
but are not passed into event detection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FILENAMES = ("01_opening_122bpm_Aminor.wav", "02_groove_124bpm_Eminor.wav")
# Fixed exploratory sweep, NOT validated production hyperparameters.
SWEEP = (
    (85.0, 0.12),
    (150.0, 0.12),
    (250.0, 0.12),
    (250.0, 0.30),
    (350.0, 0.30),
)


def source_module():
    source = ROOT / "scripts" / "m1-unconditioned-transient-localization.py"
    spec = importlib.util.spec_from_file_location("m1_global_events", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import independent event detector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sweep_track(audio, sample_rate, grid, regions, *, locator=None):
    """Summarize all predeclared settings, without selecting one."""
    if locator is None:
        locator = source_module()
    original = np.asarray(grid, dtype=float).copy()
    output = []
    for spacing, fraction in SWEEP:
        events = locator.locate_events(
            audio, sample_rate, band="low",
            minimum_spacing_ms=spacing, threshold_fraction=fraction,
        )
        compared = locator.compare_events(grid, events, windows=regions)
        entries = []
        for row in compared:
            n = row["compared_beat_midpoint_pairs"]
            entries.append({
                "region": row["region"],
                "pairs": n,
                "beat_matches": row["beat_within_tolerance_count"],
                "midpoint_matches": row["midpoint_within_tolerance_count"],
                "beat_match_fraction": round(row["beat_within_tolerance_count"] / n, 3) if n else None,
                "midpoint_match_fraction": round(row["midpoint_within_tolerance_count"] / n, 3) if n else None,
                "unique_beat_events_matched": row["unique_beat_events_matched"],
                "unique_midpoint_events_matched": row["unique_midpoint_events_matched"],
                "shared_event_count": row["shared_event_count"],
                "median_absolute_beat_error_ms": row["median_absolute_beat_error_ms"],
                "median_absolute_midpoint_error_ms": row["median_absolute_midpoint_error_ms"],
                "insufficient_event_evidence": row["insufficient_event_evidence"],
            })
        output.append({
            "minimum_spacing_ms": spacing,
            "threshold_fraction": fraction,
            "global_low_event_count": len(events),
            "regions": entries,
        })
    if not np.array_equal(original, np.asarray(grid, dtype=float)):
        raise AssertionError("beat grid mutated in shadow experiment")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite output")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track

    locator = source_module()
    tracks = []
    for name in FILENAMES:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing rights-free demo WAV: {name}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(np.mean(audio, axis=1), dtype=np.float64)
        analysis = analyze_track(path)
        # These are reporting ranges from earlier investigation, not detector
        # hints. No known generator BPM or pulse timing is used.
        windows = (
            ("outgoing_start", None, 8.823583),
            ("outgoing_middle", 9.636281, 58.305306),
            ("outgoing_end", 59.535964, None),
        ) if name == FILENAMES[0] else (("incoming", None, None),)
        tracks.append({
            "filename": name,
            "beat_count": len(analysis.beats_seconds),
            "low_event_selection_sweep": sweep_track(
                mono, rate, analysis.beats_seconds, windows, locator=locator
            ),
        })
    report = {
        "schema": "saund-m1-low-band-event-robustness-v1",
        "tracks": tracks,
        "predeclared_settings": [
            {"minimum_spacing_ms": space, "threshold_fraction": frac}
            for space, frac in SWEEP
        ],
        "selected_configuration": None,
        "generator_reference_used": False,
        "beat_grid_modified": False,
        "production_changed": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": "Research sweep only; no setting selected or validated. Peak matches are not kick identity or render approval.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"M1 peak sensitivity sweep failed: {exc}") from exc
