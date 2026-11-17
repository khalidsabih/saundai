#!/usr/bin/env python3
"""Independent audio-transient event localization for M1 rhythm research.

An event finder searches the entire filtered waveform without using the
detected grid to choose windows. Only AFTER localizing peaks does it compare
the resulting event times to detected beats and adjacent-beat midpoints.

No generator BPM, instrument template, or hidden oracle is used. Results are
diagnostic, NOT a production beat correction, a source of musical beat truth,
or permission to render unsafe transitions.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def locate_events(samples, sample_rate, *, band, frame_ms=5.0,
                  hop_ms=2.0, minimum_spacing_ms=85.0,
                  threshold_fraction=0.12):
    """Global positive high/low-band energy-rise peaks, no beat-grid input.

    The threshold is relative to the 90th percentile of positive frame rises,
    so this detector can produce spurious events on noise/soft sections.
    Report event counts and require positive, separated controls. Event
    presence does not establish instrument type or true beat/downbeat identity.
    """
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 1 or audio.size < sample_rate or not np.all(np.isfinite(audio)):
        raise ValueError("expected finite mono waveform at least one second long")
    if not 16000 <= sample_rate <= 192000:
        raise ValueError("unsupported sample rate")
    if band not in {"low", "high"}:
        raise ValueError("invalid frequency band")
    if not (0 < threshold_fraction < 1):
        raise ValueError("invalid threshold")
    low, high = ((35., 155.) if band == "low"
                 else (2500., min(9000., sample_rate * .43)))
    filtered = sosfilt(butter(3, [low, high], btype="bandpass",
                              fs=sample_rate, output="sos"), audio)
    width = max(2, round(frame_ms * sample_rate / 1000))
    hop = max(1, round(hop_ms * sample_rate / 1000))
    if len(filtered) < width + hop:
        return []
    squared = filtered * filtered
    integral = np.r_[0., np.cumsum(squared)]
    positions = np.arange(0, len(filtered) - width + 1, hop, dtype=int)
    energy = (integral[positions + width] - integral[positions]) / width
    # A first difference measures only increases in energy; frame timestamps
    # are at the starts of windows, with bounded frame-width uncertainty.
    rise = np.maximum(0., np.diff(energy, prepend=energy[0]))
    positives = rise[rise > 0]
    if positives.size < 3 or np.max(positives) <= 1e-10:
        return []
    floor = max(1e-10, float(np.percentile(positives, 90)) * threshold_fraction)
    spacing = max(1, round(minimum_spacing_ms / hop_ms))
    peaks, properties = find_peaks(rise, height=floor, distance=spacing,
                                  prominence=floor * .5)
    return [
        {"time_seconds": round(float(positions[i]) / sample_rate, 6),
         "rise_energy": round(float(rise[i]), 10)}
        for i in peaks
    ]


def compare_events(grid_seconds, events, *, tolerance_ms=80.0,
                   windows=None):
    """Evaluate detected grid / grid midpoints only after global localization.

    Duplicate nearest-event matches are counted, never described as separate
    independently observed events. Both positions can match one real event;
    such comparisons must not be interpreted as evidence of correct phase.
    """
    grid = np.asarray(grid_seconds, dtype=float)
    if grid.ndim != 1 or len(grid) < 16 or not np.all(np.isfinite(grid)) or (
        np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("invalid beat grid")
    if tolerance_ms <= 0:
        raise ValueError("invalid tolerance")
    times = sorted(float(e["time_seconds"]) for e in events)
    if any(not np.isfinite(t) for t in times):
        raise ValueError("nonfinite event timestamps")
    if windows is None:
        windows = (("full_track", None, None),)
    mids = .5 * (grid[:-1] + grid[1:])
    result = []
    for label, start, end in windows:
        mask = ((grid[:-1] >= start) if start is not None
                else np.ones(len(mids), dtype=bool))
        if end is not None:
            mask &= grid[:-1] < end
        beats = grid[:-1][mask]
        halves = mids[mask]
        beat_errors = []
        mid_errors = []
        beat_matches = []
        half_matches = []
        for candidate_beats, errors, matches in (
            (beats, beat_errors, beat_matches),
            (halves, mid_errors, half_matches)
        ):
            for t in candidate_beats:
                if not times:
                    continue
                j = bisect.bisect_left(times, float(t))
                choices = [k for k in (j - 1, j) if 0 <= k < len(times)]
                idx = min(choices, key=lambda k: abs(times[k] - t))
                residual = (float(t) - times[idx]) * 1000.
                errors.append(residual)
                if abs(residual) <= tolerance_ms:
                    matches.append(idx)
        paired_count = min(len(beat_errors), len(mid_errors))
        result.append({
            "region": label,
            "compared_beat_midpoint_pairs": len(beats),
            "localized_event_count": len(times),
            "beat_within_tolerance_count": len(beat_matches),
            "midpoint_within_tolerance_count": len(half_matches),
            "unique_beat_events_matched": len(set(beat_matches)),
            "unique_midpoint_events_matched": len(set(half_matches)),
            "shared_event_count": len(set(beat_matches) & set(half_matches)),
            "median_absolute_beat_error_ms": (
                round(float(np.median(np.abs(beat_errors))), 3)
                if beat_errors else None
            ),
            "median_absolute_midpoint_error_ms": (
                round(float(np.median(np.abs(mid_errors))), 3)
                if mid_errors else None
            ),
            "insufficient_event_evidence": paired_count == 0,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an existing report")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track
    names = ("01_opening_122bpm_Aminor.wav", "02_groove_124bpm_Eminor.wav")
    tracks = []
    for name in names:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing rights-free demo WAV: {name}")
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = np.mean(audio, axis=1, dtype=np.float64)
        analysis = analyze_track(path)
        sections = (
            ("outgoing_start", None, 8.823583),
            ("outgoing_middle", 9.636281, 58.305306),
            ("outgoing_end", 59.535964, None),
        ) if name == names[0] else (("incoming", None, None),)
        # Sections are *reporting* boundaries from prior experiments; neither
        # the global event finder nor the grid analyzer uses them for scoring.
        bands = {}
        for band in ("low", "high"):
            events = locate_events(mono, sr, band=band)
            bands[band] = {
                "localized_global_events": len(events),
                "regions": compare_events(analysis.beats_seconds, events, windows=sections),
                "first_events": events[:6],
            }
        tracks.append({"filename": name, "grid_count": len(analysis.beats_seconds),
                       "neural_grid_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
                       "bands": bands})
    report = {
        "schema": "saund-m1-unconditioned-transient-localization-v1",
        "tracks": tracks, "generator_reference_used": False,
        "beat_grid_modified": False, "production_changed": False,
        "transition_rendered": False, "plan_generated": False,
        "warning": "Global high/low band peaks can include non-drum sounds; no correctness inference or render authorization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"M1 event localization audit failed: {exc}") from exc
