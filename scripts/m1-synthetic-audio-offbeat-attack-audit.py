#!/usr/bin/env python3
"""Read-only high-frequency offbeat transient audit on SAUND generated demo pair.

Search windows use known generator 122/124 BPM, but the locations of any
attacks are found from high-frequency WAV energy, independently from the
analysis beat grid and low-frequency kick detector. This is not applicable
to arbitrary copyrighted music and never changes beat positions or plans.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)
OUTGOING_REGIONS = (
    ("before_first_compression", None, 8.823583),
    ("first_compression", 8.823583, 9.636281),
    ("between_compressions", 9.636281, 58.305306),
    ("second_compression", 58.305306, 59.535964),
    ("after_second_compression", 59.535964, None),
)


def _median(values):
    return round(float(statistics.median(values)), 3) if values else None


def detect_offbeat_transients(audio, sample_rate, bpm, *, search_ms=60.0,
                              window_ms=6.0, min_rise_ratio=2.0):
    """Measure broadband-hat high-band attacks near generator half-beats.

    The expected timestamps select search ranges only. Unsupported attack
    measurements are abstentions, not synthesized onsets.
    """
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim != 1 or data.size < sample_rate or not np.all(np.isfinite(data)):
        raise ValueError("expected finite mono audio at least one second long")
    if not 16000 <= sample_rate <= 192000 or not 60 <= bpm <= 200:
        raise ValueError("unsupported sample rate or BPM")
    if search_ms <= 0 or window_ms <= 0 or min_rise_ratio <= 1:
        raise ValueError("invalid transient parameters")
    upper = min(9000.0, sample_rate * .43)
    sos = butter(3, [2500.0, upper], btype="bandpass", fs=sample_rate, output="sos")
    high = sosfilt(sos, data)
    power = high * high
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    width = max(1, int(round(sample_rate * window_ms / 1000.0)))
    radius = int(round(sample_rate * search_ms / 1000.0))
    beat_seconds = 60.0 / bpm
    count = int(math.floor((data.size / sample_rate - .01) / beat_seconds))
    events = []
    for index in range(count):
        expected = (index + .5) * beat_seconds
        center = int(round(expected * sample_rate))
        starts = np.arange(max(width, center-radius),
                           min(data.size-width, center+radius)+1)
        if not len(starts):
            events.append({"index": index, "status": "out_of_audio_bounds"})
            continue
        preceding = (cumulative[starts] - cumulative[starts-width]) / width
        following = (cumulative[starts+width] - cumulative[starts]) / width
        rise = np.maximum(following - preceding, 0)
        best = int(np.argmax(rise))
        peak = float(rise[best])
        floor = float(np.median(preceding))
        ratio = peak / max(floor, 1e-12)
        if peak <= 1e-10 or ratio < min_rise_ratio or float(np.max(following)) < 1e-9:
            events.append({"index": index, "status": "insufficient_high_frequency_attack",
                           "attack_strength": round(ratio, 3)})
            continue
        seconds = starts[best] / sample_rate
        events.append({
            "index": index, "status": "measured",
            "attack_seconds": round(seconds, 6),
            "offset_from_synthetic_halfbeat_ms": round((seconds-expected)*1000, 3),
            "attack_strength": round(ratio, 3),
        })
    return events


def regional_hat_alignment(beats, events, regions, *, tolerance_ms=80.0):
    """Compare each unchanged beat timestamp to independently measured hats."""
    if len(beats) < 2 or any(not math.isfinite(float(t)) for t in beats) or any(
        right <= left for left, right in zip(beats, beats[1:])
    ):
        raise ValueError("invalid beat grid")
    measured = sorted(float(e["attack_seconds"]) for e in events
                      if e.get("status") == "measured")
    output = []
    for name, start, end in regions:
        subset = [float(t) for t in beats
                  if (start is None or t >= start) and (end is None or t < end)]
        errors = []
        examples = []
        for beat in subset:
            if not measured:
                continue
            idx = bisect.bisect_left(measured, beat)
            candidates = [measured[j] for j in (idx-1, idx)
                          if 0 <= j < len(measured)]
            closest = min(candidates, key=lambda t: abs(t-beat))
            error = 1000.0*(beat-closest)
            errors.append(error)
            if len(examples) < 5:
                examples.append({"beat_seconds": round(beat, 6),
                                 "measured_offbeat_seconds": round(closest, 6),
                                 "signed_error_ms": round(error, 3)})
        output.append({
            "region": name, "beat_count": len(subset),
            "comparable_count": len(errors),
            "within_80ms_count": sum(abs(e) <= tolerance_ms for e in errors),
            "median_absolute_attack_error_ms": _median([abs(e) for e in errors]),
            "first_examples": examples,
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track
    tracks = []
    for name, bpm in FIXTURES:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing known synthetic demo track: {name}")
        audio, sr = sf.read(path, always_2d=True, dtype="float32")
        events = detect_offbeat_transients(audio[:, 0], sr, bpm)
        analysis = analyze_track(path)
        regions = OUTGOING_REGIONS if name == FIXTURES[0][0] else (
            ("entire_incoming", None, None),)
        tracks.append({
            "filename": name, "generator_bpm": bpm,
            "expected_offbeat_events": len(events),
            "measured_offbeat_events": sum(e.get("status") == "measured" for e in events),
            "median_attack_offset_from_synthetic_halfbeat_ms": _median([
                e["offset_from_synthetic_halfbeat_ms"]
                for e in events if e.get("status") == "measured"
            ]),
            "regional_measured_offbeat_alignment": regional_hat_alignment(
                analysis.beats_seconds, events, regions
            ),
            "first_attack_examples": events[:6],
            "neural_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
        })
    report = {
        "schema": "saund-m1-synthetic-audio-offbeat-attack-v1",
        "tracks": tracks,
        "production_changed": False,
        "beat_grid_corrected": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": "Generator-window-guided high-band attacks are synthetic-only. Observed offbeat proximity does not prove tracker causality or qualify a transition.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 offbeat audio audit failed: {exc}") from exc
