#!/usr/bin/env python3
"""Independent kick-onset measurement for the known rights-free synthetic M1 pair.

Localizes low-frequency energy attacks in the *audio waveform*, using the
generator's exact kick schedule only to select search windows. The analysis
beat grid is never used to select an onset, and the template is not imported
from the demo generator. This is a synthetic-oracle aided audit, NOT a
standalone onset detector or permission to correct production beat timestamps.

Do not use on real music: its pulse reference is specific to generate_demo_set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = (
    ("01_opening_122bpm_Aminor.wav", 122.0),
    ("02_groove_124bpm_Eminor.wav", 124.0),
)


def _median(values):
    return round(float(statistics.median(values)), 3) if values else None


def extract_kick_attacks(
    samples: np.ndarray, sample_rate: int, bpm: float, *,
    search_ms: float = 85.0, window_ms: float = 12.0,
    min_attack_ratio: float = 1.8,
) -> list[dict]:
    """Detect audible low-frequency energy rises near each reference pulse.

    Compares 12-ms energy just after each candidate sample to energy in the
    preceding 12 ms. A minimum peak-to-background ratio is required. Searches
    around *known synthetic pulse times*; cannot infer pulses in arbitrary
    music. At t=0, the previous window is zero-padded, not assumed observed.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim != 1 or len(samples) < sample_rate:
        raise ValueError("expected mono audio of at least one second")
    if sample_rate < 8000 or not (60.0 <= bpm <= 200.0):
        raise ValueError("unsupported sample rate or generator BPM")
    if not np.all(np.isfinite(samples)):
        raise ValueError("audio must be finite")
    if search_ms <= 0 or window_ms <= 0 or min_attack_ratio <= 1:
        raise ValueError("invalid attack search parameters")
    # Audible kick energy is concentrated below ~160 Hz. Unlike the system's
    # librosa beat tracker, this uses only a fixed bandpass + energy change.
    sos = butter(3, [30.0, 155.0], btype="bandpass", fs=sample_rate, output="sos")
    filtered = sosfilt(sos, np.asarray(samples, dtype=np.float64))
    power = filtered * filtered
    integral = np.concatenate(([0.0], np.cumsum(power)))
    win = max(1, int(round(window_ms / 1000.0 * sample_rate)))
    radius = int(round(search_ms / 1000.0 * sample_rate))
    period = 60.0 / bpm
    count = int(math.floor((len(samples) / sample_rate - 0.01) / period)) + 1
    events = []
    for index in range(count):
        expected = index * period
        center = int(round(expected * sample_rate))
        starts = np.arange(max(win, center - radius), min(len(samples) - win, center + radius) + 1)
        if not len(starts):
            events.append({"index": index, "generator_seconds": round(expected, 6),
                           "status": "out_of_audio_bounds"})
            continue
        before = (integral[starts] - integral[starts - win]) / win
        after = (integral[starts + win] - integral[starts]) / win
        change = np.maximum(after - before, 0.0)
        peak = int(np.argmax(change))
        onset = int(starts[peak])
        maximum = float(change[peak])
        # Use the local power floor as a robust scale; a silent segment must
        # not be "detected" just because we looked at the expected timestamp.
        local_floor = float(np.median(before))
        local_power = float(np.max(after))
        strength = maximum / max(local_floor, 1e-12)
        if maximum <= 1e-10 or strength < min_attack_ratio or local_power < 1e-9:
            events.append({"index": index, "generator_seconds": round(expected, 6),
                           "status": "insufficient_low_frequency_attack",
                           "attack_strength": round(strength, 3)})
            continue
        events.append({
            "index": index,
            "generator_seconds": round(expected, 6),
            "audio_attack_seconds": round(onset / sample_rate, 6),
            "attack_minus_generator_ms": round(1000.0 * (onset / sample_rate - expected), 3),
            "attack_strength": round(strength, 3),
            "status": "measured",
        })
    return events


def align_grid_to_attacks(grid, events, *, tolerance_ms: float = 80.0) -> dict:
    """Match estimated grid positions to independently measured attacks.

    Generator index is used only for synthetic event association. Attacks that
    were not measured are excluded; a missing attack can never be synthesized
    from the expected generator pulse.
    """
    if not isinstance(grid, (list, tuple)) or any(
        not isinstance(t, (float, int)) or not math.isfinite(float(t)) for t in grid
    ) or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("invalid detected beat grid")
    measured = [e for e in events if e.get("status") == "measured"]
    if not measured:
        return {
            "status": "insufficient_independent_audio_evidence",
            "measured_audio_events": 0,
            "compared_beats": 0,
        }
    points = sorted((e["audio_attack_seconds"], e["index"]) for e in measured)
    import bisect
    timestamps = [x[0] for x in points]
    deviations = []
    examples = []
    for idx, beat in enumerate(grid):
        j = bisect.bisect_left(timestamps, float(beat))
        candidates = [p for k in (j - 1, j) if 0 <= k < len(points) for p in [points[k]]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda item: abs(item[0] - float(beat)))
        error_ms = 1000.0 * (float(beat) - nearest[0])
        # Large alignment errors remain visible and must NOT be treated as
        # matched merely because the closest attack exists elsewhere in audio.
        matched = abs(error_ms) <= tolerance_ms
        deviations.append((error_ms, matched))
        if len(examples) < 12 and (not matched or idx < 3):
            examples.append({
                "detected_beat_index": idx,
                "detected_seconds": round(float(beat), 6),
                "closest_audio_kick_index": nearest[1],
                "closest_audio_kick_seconds": round(nearest[0], 6),
                "detected_minus_audio_ms": round(error_ms, 3),
                "within_tolerance": matched,
            })
    matched_errors = [error for error, matched in deviations if matched]
    all_errors = [abs(error) for error, _ in deviations]
    return {
        "status": "measured",
        "measured_audio_events": len(measured),
        "compared_beats": len(deviations),
        "beats_outside_audio_attack_tolerance": sum(not m for _, m in deviations),
        "tolerance_ms": tolerance_ms,
        "median_absolute_nearest_attack_error_ms": _median(all_errors),
        "median_signed_error_ms_for_matched_beats": _median(matched_errors),
        "examples": examples,
    }



# Fixed source-time windows diagnosed in the earlier synthetic beat-index audit.
OUTGOING_REGIONS = (
    ("before_first_compression", None, 8.823583),
    ("first_compression", 8.823583, 9.636281),
    ("between_compressions", 9.636281, 58.305306),
    ("second_compression", 58.305306, 59.535964),
    ("after_second_compression", 59.535964, None),
)


def regional_alignment(grid, events, regions, *, tolerance_ms=80.0):
    """Report nearest measured acoustic kick errors by predefined source window.

    Nearest proximity does not prove the generator beat index was preserved.
    """
    import bisect

    if len(grid) < 2 or any(not math.isfinite(float(v)) for v in grid) or any(
        b <= a for a, b in zip(grid, grid[1:])
    ):
        raise ValueError("invalid beat grid")
    measured = sorted(
        (float(e["audio_attack_seconds"]), int(e["index"]))
        for e in events if e.get("status") == "measured"
    )
    positions = [pair[0] for pair in measured]
    result = []
    for name, start, end in regions:
        subset = [float(beat) for beat in grid
                  if (start is None or beat >= start) and (end is None or beat < end)]
        residuals = []
        mismatches = []
        for beat in subset:
            if not measured:
                continue
            j = bisect.bisect_left(positions, beat)
            candidates = [measured[k] for k in (j - 1, j) if 0 <= k < len(measured)]
            kick_seconds, kick_index = min(candidates, key=lambda p: abs(beat - p[0]))
            offset_ms = (beat - kick_seconds) * 1000.0
            residuals.append(offset_ms)
            if abs(offset_ms) > tolerance_ms and len(mismatches) < 4:
                mismatches.append({
                    "beat_seconds": round(beat, 6),
                    "nearest_audio_kick_index": kick_index,
                    "signed_error_ms": round(offset_ms, 3),
                })
        result.append({
            "region": name,
            "start_seconds": start,
            "end_seconds": end,
            "beat_count": len(subset),
            "measured_nearest_count": len(residuals),
            "outside_tolerance_count": (
                sum(abs(offset) > tolerance_ms for offset in residuals)
                + len(subset) - len(residuals)
            ),
            "tolerance_ms": tolerance_ms,
            "median_absolute_nearest_attack_error_ms": _median([abs(x) for x in residuals]),
            "median_signed_nearest_attack_error_ms": _median(residuals),
            "first_mismatches": mismatches,
        })
    return result



def index_association_regions(grid, events, regions, *, tolerance_ms=80.0):
    """Diagnose synthetic pulse-index associations without changing any beat.

    The generator index is available only for these two explicitly generated
    fixtures. Nearest-event association may be ambiguous near the midpoint
    between kicks and does not establish an actual missing musical beat.
    """
    import bisect
    from collections import Counter

    if not isinstance(grid, (list, tuple)) or len(grid) < 2 or any(
        not math.isfinite(float(beat)) for beat in grid
    ) or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("invalid detected beat grid")
    measured = sorted(
        (float(e["audio_attack_seconds"]), int(e["index"]))
        for e in events if e.get("status") == "measured"
    )
    times = [pair[0] for pair in measured]
    result = []
    for name, start, end in regions:
        associations = []
        for detected_index, beat in enumerate(grid):
            if (start is not None and beat < start) or (end is not None and beat >= end):
                continue
            if not measured:
                continue
            j = bisect.bisect_left(times, beat)
            candidates = [measured[k] for k in (j - 1, j) if 0 <= k < len(measured)]
            attack, pulse_index = min(candidates, key=lambda pair: abs(pair[0] - beat))
            signed_error = 1000.0 * (float(beat) - attack)
            associations.append({
                "beat_index": detected_index,
                "kick_index": pulse_index,
                "index_offset": pulse_index - detected_index,
                "signed_error_ms": round(signed_error, 3),
                "within_tolerance": abs(signed_error) <= tolerance_ms,
            })
        reliable = [x for x in associations if x["within_tolerance"]]
        counts = Counter(x["index_offset"] for x in reliable)
        nearest_indices = [x["kick_index"] for x in associations]
        result.append({
            "region": name,
            "beat_count": sum(
                (start is None or b >= start) and (end is None or b < end)
                for b in grid
            ),
            "associated_count": len(associations),
            "within_tolerance_count": len(reliable),
            "out_of_tolerance_count": len(associations) - len(reliable),
            "reliable_index_offset_histogram": {
                str(k): counts[k] for k in sorted(counts)
            },
            "nearest_kick_duplicate_steps": sum(
                y == x for x, y in zip(nearest_indices, nearest_indices[1:])
            ),
            "nearest_kick_reverse_steps": sum(
                y < x for x, y in zip(nearest_indices, nearest_indices[1:])
            ),
            "first_associations": associations[:4],
            "last_associations": associations[-4:],
            "warning": "Synthetic kick indices; out-of-tolerance associations do not establish beat identity.",
        })
    return result


def audit(directory: Path, analyze) -> dict:
    tracks = []
    for name, bpm in FIXTURES:
        path = (directory / name).resolve()
        if not path.is_file():
            raise ValueError(f"missing known synthetic fixture: {name}")
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        samples = np.asarray(data[:, 0], dtype=np.float64)
        attacks = extract_kick_attacks(samples, sr, bpm)
        analysis = analyze(path)
        aligned = align_grid_to_attacks(analysis.beats_seconds, attacks)
        regional = regional_alignment(
            analysis.beats_seconds, attacks, OUTGOING_REGIONS
        ) if name == FIXTURES[0][0] else None
        index_associations = index_association_regions(
            analysis.beats_seconds, attacks, OUTGOING_REGIONS
        ) if name == FIXTURES[0][0] else None
        confident = [e for e in attacks if e["status"] == "measured"]
        tracks.append({
            "filename": name,
            "generator_bpm": bpm,
            "estimated_scalar_bpm": analysis.bpm,
            "detected_beat_count": len(analysis.beats_seconds),
            "expected_audio_pulses": len(attacks),
            "measured_audio_pulses": len(confident),
            "missing_audio_attack_evidence": len(attacks) - len(confident),
            "median_audio_attack_minus_generator_ms": _median(
                [e["attack_minus_generator_ms"] for e in confident]
            ),
            "grid_alignment_to_independent_audio": aligned,
            "outgoing_region_alignment": regional,
            "outgoing_region_index_associations": index_associations,
            "first_audio_attack_examples": attacks[:8],
            "neural_grid_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
        })
    return {
        "schema": "saund-m1-synthetic-audio-kick-audit-v1",
        "tracks": tracks,
        "production_changed": False,
        "grid_corrected": False,
        "plan_generated": False,
        "transition_rendered": False,
        "warning": "Search locations are synthetic generator pulse windows. Attacks are extracted from WAV independently of analysis beats; no reference is production evidence.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing report")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track
    report = audit(args.audio_root, analyze_track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 audio kick audit failed: {exc}") from exc
