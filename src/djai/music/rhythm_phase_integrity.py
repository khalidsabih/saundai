"""Shadow-only, generator-independent acoustic beat-phase integrity evidence.

This module intentionally does not modify or veto TrackAnalysis: it measures
whether independent low-frequency transient energy or high-frequency attack
energy is concentrated at an *existing* beat grid versus the inter-beat
midpoints. Midpoints come ONLY from the detected grid, never a known BPM
or annotated pulse schedule. It is a diagnostic, not a beat corrector.

Two-band contrast is insufficient to infer beat identity on music with weak
kicks, syncopation, mixed accents, halftime/doubletime or changing meter.
Abstain when contrasts are weak; even strong signals require validation before
an opt-in policy may use them.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt


def _sample_attack_energy(power, sample_rate, timestamps, *, radius_ms=22.0, width_ms=13.0):
    """Peak short-window band energy close to a proposed event coordinate."""
    width = max(1, round(sample_rate * width_ms / 1000))
    radius = max(1, round(sample_rate * radius_ms / 1000))
    integral = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    values = []
    for time in timestamps:
        center = round(float(time) * sample_rate)
        left = max(0, center - radius)
        right = min(len(power) - width, center + radius)
        if right < left:
            values.append(float("nan"))
            continue
        starts = np.arange(left, right + 1, max(1, width // 4), dtype=int)
        if starts[-1] != right:
            starts = np.append(starts, right)
        energy = (integral[starts + width] - integral[starts]) / width
        values.append(float(max(0.0, np.max(energy))))
    return np.asarray(values, dtype=float)



def _sample_attack_rise(power, sample_rate, timestamps, *, radius_ms=22.0, width_ms=6.0):
    """Peak positive before/after energy rise near each detected-grid point.

    Unlike a peak-energy sampler, this penalizes sustained sound extending
    across a candidate beat. The value is diagnostic, not a classifier vote.
    """
    width = max(1, round(sample_rate * width_ms / 1000))
    radius = max(1, round(sample_rate * radius_ms / 1000))
    integral = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    out = []
    for time in timestamps:
        center = round(float(time) * sample_rate)
        left = max(width, center - radius)
        right = min(len(power) - width, center + radius)
        if right < left:
            out.append(float("nan"))
            continue
        starts = np.arange(left, right + 1, max(1, width // 4), dtype=int)
        if starts[-1] != right:
            starts = np.append(starts, right)
        previous = (integral[starts] - integral[starts - width]) / width
        following = (integral[starts + width] - integral[starts]) / width
        out.append(float(max(0.0, np.max(following - previous))))
    return np.asarray(out, dtype=float)


def phase_evidence_features(audio, sample_rate, beat_seconds):
    """Return sample-aligned band powers on original grid and halfbeat.

    Assumes NO time signature, tempo annotation or generator time reference.
    Beat timestamps must be strictly increasing and spaced far enough to
    distinguish the tested phases. An invalid grid is rejected rather than
    silently interpolated.
    """
    data = np.asarray(audio, dtype=np.float64)
    grid = np.asarray(beat_seconds, dtype=np.float64)
    if data.ndim != 1 or data.size < sample_rate or not np.all(np.isfinite(data)):
        raise ValueError("audio must be finite mono and at least one second long")
    if sample_rate < 16000:
        raise ValueError("sample rate too low for independent high band")
    if grid.ndim != 1 or len(grid) < 16 or not np.all(np.isfinite(grid)):
        raise ValueError("insufficient valid beat grid")
    if np.any(np.diff(grid) < .24) or np.any(np.diff(grid) > 1.4):
        raise ValueError("beat intervals outside shadow detector support")
    if grid[0] < 0 or grid[-1] > (len(data) - 1) / sample_rate:
        raise ValueError("beat grid exceeds audio bounds")
    midpoints = (grid[:-1] + grid[1:]) * .5
    beats = grid[:-1]
    upper = min(9000.0, sample_rate * .43)
    low_filter = butter(3, [35.0, 155.0], btype="bandpass", fs=sample_rate, output="sos")
    high_filter = butter(3, [2500.0, upper], btype="bandpass", fs=sample_rate, output="sos")
    low = sosfilt(low_filter, data)
    high = sosfilt(high_filter, data)
    return {
        "times": beats,
        "low_at_beat": _sample_attack_energy(low * low, sample_rate, beats),
        "low_at_midpoint": _sample_attack_energy(low * low, sample_rate, midpoints),
        "high_at_beat": _sample_attack_energy(high * high, sample_rate, beats),
        "high_at_midpoint": _sample_attack_energy(high * high, sample_rate, midpoints),
        "high_rise_at_beat": _sample_attack_rise(high * high, sample_rate, beats),
        "high_rise_at_midpoint": _sample_attack_rise(high * high, sample_rate, midpoints),
        # Wider spatial evidence is strictly observational. Earlier independent
        # WAV experiments measured median offbeat attack offsets near 32 ms,
        # beyond the initial +/-22 ms search window. No classification vote
        # below uses either of these fields.
        "high_rise_wide_at_beat": _sample_attack_rise(
            high * high, sample_rate, beats, radius_ms=55.0
        ),
        "high_rise_wide_at_midpoint": _sample_attack_rise(
            high * high, sample_rate, midpoints, radius_ms=55.0
        ),
    }


def classify_phase_window(features, *, start, count=16):
    """Classify only with independent opposite-band contrasts in this window.

    Median and 75%-agreement tests make one kick drop or isolated syncopation
    insufficient. Window labels are not permission to repair any beat.
    """
    end = start + count
    if count < 12:
        raise ValueError("need >=12 beat observations")
    required = ("low_at_beat", "low_at_midpoint", "high_at_beat", "high_at_midpoint")
    arrays = [np.asarray(features[name], dtype=np.float64)[start:end] for name in required]
    if any(len(a) != count or not np.all(np.isfinite(a)) or np.any(a < 0) for a in arrays):
        return {"status": "insufficient_evidence", "reason": "nonfinite_or_missing_band_energy"}
    lb, lm, hb, hm = arrays
    # Silence and weak bands must never turn tiny numerical ratios into
    # authoritative rhythmic phase evidence.
    if max(float(np.median(lb)), float(np.median(lm))) < 1e-8:
        return {"status": "insufficient_evidence", "reason": "weak_low_band"}
    if max(float(np.median(hb)), float(np.median(hm))) < 1e-9:
        return {"status": "insufficient_evidence", "reason": "weak_high_band"}
    floor_l = max(1e-12, float(np.median(np.concatenate([lb, lm]))) * .02)
    floor_h = max(1e-13, float(np.median(np.concatenate([hb, hm]))) * .02)
    low_kick_ratio = (lb + floor_l) / (lm + floor_l)
    high_hat_ratio = (hm + floor_h) / (hb + floor_h)
    kick_support = (low_kick_ratio >= 2.5) & (high_hat_ratio >= 2.0)
    offbeat_support = (low_kick_ratio <= 1 / 2.5) & (high_hat_ratio <= .5)
    kick_count = int(np.count_nonzero(kick_support))
    offbeat_count = int(np.count_nonzero(offbeat_support))
    needed = math.ceil(count * .75)
    status = ("kick_aligned" if kick_count >= needed else
              "suspected_offbeat_lock" if offbeat_count >= needed else
              "insufficient_evidence")
    result = {
        "status": status,
        "beat_observations": count,
        "kick_support_count": kick_count,
        "offbeat_support_count": offbeat_count,
        "median_low_beat_to_mid_ratio": round(float(np.median(low_kick_ratio)), 3),
        "median_high_mid_to_beat_ratio": round(float(np.median(high_hat_ratio)), 3),
        "reason": "two_band_sustained_contrast" if status != "insufficient_evidence"
                  else "mixed_or_insufficient_phase_contrast",
    }
    # Additional independent attack-rise evidence is observational only:
    # preserve existing classification and thresholds until validated.
    if "high_rise_at_beat" in features and "high_rise_at_midpoint" in features:
        rise_beat = np.asarray(features["high_rise_at_beat"], dtype=float)[start:end]
        rise_mid = np.asarray(features["high_rise_at_midpoint"], dtype=float)[start:end]
        if (len(rise_beat) == count and len(rise_mid) == count
                and np.all(np.isfinite(rise_beat)) and np.all(np.isfinite(rise_mid))
                and np.all(rise_beat >= 0) and np.all(rise_mid >= 0)):
            rise_floor = max(1e-12, float(np.median(np.r_[rise_beat, rise_mid])) * .02)
            rise_ratio = (rise_mid + rise_floor) / (rise_beat + rise_floor)
            result["median_high_rise_mid_to_beat_ratio"] = round(float(np.median(rise_ratio)), 3)
            result["high_rise_mid_dominant_count"] = int(np.count_nonzero(rise_ratio >= 2.0))
            result["high_rise_beat_dominant_count"] = int(np.count_nonzero(rise_ratio <= .5))
            result["high_rise_diagnostic_only"] = True
        else:
            result["high_rise_diagnostic_unavailable"] = True
    if "high_rise_wide_at_beat" in features and "high_rise_wide_at_midpoint" in features:
        wide_beat = np.asarray(features["high_rise_wide_at_beat"], dtype=float)[start:end]
        wide_mid = np.asarray(features["high_rise_wide_at_midpoint"], dtype=float)[start:end]
        if (len(wide_beat) == count and len(wide_mid) == count
                and np.all(np.isfinite(wide_beat)) and np.all(np.isfinite(wide_mid))
                and np.all(wide_beat >= 0) and np.all(wide_mid >= 0)):
            absolute_floor = 1e-10
            has_beat = wide_beat > absolute_floor
            has_mid = wide_mid > absolute_floor
            both_absent = ~(has_beat | has_mid)
            wide_floor = max(absolute_floor, float(np.median(np.r_[wide_beat, wide_mid])) * .02)
            wide_ratio = (wide_mid + wide_floor) / (wide_beat + wide_floor)
            valid = ~both_absent
            result["wide_high_rise_beat_evidence_count"] = int(np.count_nonzero(has_beat))
            result["wide_high_rise_mid_evidence_count"] = int(np.count_nonzero(has_mid))
            result["wide_high_rise_both_absent_count"] = int(np.count_nonzero(both_absent))
            result["wide_high_rise_mid_dominant_evidence_count"] = int(
                np.count_nonzero((wide_ratio >= 2) & valid)
            )
            result["wide_high_rise_beat_dominant_evidence_count"] = int(
                np.count_nonzero((wide_ratio <= .5) & valid)
            )
            result["wide_high_rise_median_ratio_on_evidence"] = (
                round(float(np.median(wide_ratio[valid])), 3) if np.any(valid) else None
            )
            result["wide_high_rise_diagnostic_only"] = True
        else:
            result["wide_high_rise_diagnostic_unavailable"] = True
    return result


def summarize_phase_integrity(features, *, window_beats=16, stride_beats=8,
                              sustained_windows=3):
    """Report confirmed long runs, while keeping individual windows advisory."""
    if window_beats < 12 or stride_beats < 1 or sustained_windows < 2:
        raise ValueError("invalid shadow window parameters")
    times = np.asarray(features["times"], dtype=float)
    n = len(times)
    if n < window_beats:
        return {"decision": "insufficient_evidence", "windows": [],
                "sustained_offbeat_runs": [], "sustained_kick_runs": []}
    windows = []
    for start in range(0, n - window_beats + 1, stride_beats):
        item = classify_phase_window(features, start=start, count=window_beats)
        windows.append({
            "start_seconds": round(float(times[start]), 4),
            "end_seconds": round(float(times[start + window_beats - 1]), 4),
            **item,
        })

    def runs(label):
        output = []
        i = 0
        while i < len(windows):
            if windows[i]["status"] != label:
                i += 1
                continue
            j = i + 1
            while j < len(windows) and windows[j]["status"] == label:
                j += 1
            if j - i >= sustained_windows:
                output.append({
                    "first_window": i,
                    "last_window": j - 1,
                    "window_count": j - i,
                    "start_seconds": windows[i]["start_seconds"],
                    "end_seconds": windows[j - 1]["end_seconds"],
                })
            i = j
        return output

    offbeat = runs("suspected_offbeat_lock")
    stable = runs("kick_aligned")
    return {
        "decision": "sustained_offbeat_phase_inconsistency" if offbeat
                    else "kick_aligned_with_support" if stable
                    else "insufficient_evidence",
        "windows": windows,
        "sustained_offbeat_runs": offbeat,
        "sustained_kick_runs": stable,
        "generator_reference_used": False,
        "beat_grid_modified": False,
        "authority": "shadow_only",
        "warning": "Band-phase contrast is not a general kick/hat classifier; no beat correction or rendering permission.",
    }


def inspect_waveform(audio, sample_rate, beat_seconds):
    return summarize_phase_integrity(
        phase_evidence_features(audio, sample_rate, beat_seconds)
    )
