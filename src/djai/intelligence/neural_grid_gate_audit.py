"""Read-only reproduction of the live neural grid *gate* with real fallback grids.

Replicates the deterministic pre-neural path in analyze_track, then invokes
predict_beats with the same fallback_beats and fallback_downbeats arguments.
Never invokes analyze_track or writes track data, cues or live configuration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def deterministic_fallback_grid(path: Path) -> dict:
    """Mirror the pre-neural beat/downbeat calculation in analyze_track."""
    import librosa
    import numpy as np
    from ..analysis import (
        ANALYSIS_SAMPLE_RATE, HOP_LENGTH, BEATS_PER_BAR,
        _beat_grid_diagnostics, _estimate_time_signature, _estimate_downbeats,
    )
    from ..audio import load_audio

    source_audio, sr = load_audio(path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
    y = librosa.to_mono(source_audio)
    duration = float(librosa.get_duration(y=y, sr=sr))
    if duration < 4.0:
        raise ValueError("Track is too short. DJAI requires at least 4 seconds of audio.")

    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sr, hop_length=HOP_LENGTH,
        units="frames", sparse=True,
    )
    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)
    if bpm <= 0.0:
        bpm = 120.0
    grid_confidence, _ = _beat_grid_diagnostics(beat_times, duration)
    time_signature, _ = _estimate_time_signature(
        beat_frames, onset_envelope, grid_confidence,
    )
    beats_per_bar = time_signature if time_signature in {3, 4, 6} else BEATS_PER_BAR
    downbeat_frames, _ = _estimate_downbeats(
        beat_frames, onset_envelope, beats_per_bar=beats_per_bar,
    )
    downbeat_times = librosa.frames_to_time(
        downbeat_frames, sr=sr, hop_length=HOP_LENGTH,
    )
    return {
        "bpm": bpm,
        "beat_times": tuple(float(value) for value in beat_times),
        "downbeat_times": tuple(float(value) for value in downbeat_times),
        "duration_seconds": duration,
        "time_signature": int(time_signature),
    }


def audit_grid_gate(path: Path) -> dict:
    """Read-only comparison, never applying neural results to DJ structures."""
    from .neural_ears import predict_beats, tempo_only_adoption
    from .neural_grid_safety_screen import screen_full_grid_candidate

    baseline = deterministic_fallback_grid(path)
    # This is the actual live call signature, but not a live DJ analysis run.
    neural = predict_beats(
        path,
        fallback_beats=baseline["beat_times"],
        fallback_downbeats=baseline["downbeat_times"],
    )
    tempo_allowed, tempo_reason = tempo_only_adoption(neural, baseline["bpm"])
    bpm = float(neural.bpm) if neural.available and math.isfinite(neural.bpm) else None
    fallback_bpm = float(baseline["bpm"])
    drift = (
        round(abs(bpm - fallback_bpm), 4)
        if bpm is not None else None
    )
    safety_screen = screen_full_grid_candidate(
        baseline_bpm=fallback_bpm,
        neural_bpm=bpm,
        fallback_beat_count=len(baseline["beat_times"]),
        fallback_downbeat_count=len(baseline["downbeat_times"]),
        neural_beat_count=len(neural.beats_seconds),
        neural_downbeat_count=len(neural.downbeats_seconds),
        fallback_beat_agreement=neural.agreement if neural.available else None,
        neural_available=bool(neural.available),
        existing_gate_would_pass=bool(neural.used),
    )
    return {
        "filename": path.name,
        "mode": "read_only_live_signature_reproduction",
        "live_analysis_executed": False,
        "live_grid_adoption_observed": False,
        "changes_dj_decisions": False,
        "baseline_bpm": round(fallback_bpm, 4),
        "baseline_beats": len(baseline["beat_times"]),
        "baseline_downbeats": len(baseline["downbeat_times"]),
        "baseline_time_signature": baseline["time_signature"],
        "neural_available": bool(neural.available),
        "neural_bpm": bpm,
        "bpm_difference": drift,
        "neural_beat_count": len(neural.beats_seconds),
        "neural_downbeat_count": len(neural.downbeats_seconds),
        "neural_grid_confidence": neural.confidence,
        "neural_fallback_beat_agreement": neural.agreement,
        "neural_grid_gate_would_pass": bool(neural.used),
        "counterfactual_safety_screen": safety_screen,
        "neural_gate_reason": neural.reason,
        "tempo_only_gate_would_pass": bool(tempo_allowed),
        "tempo_only_gate_reason": tempo_reason,
        "requires_human_review": bool(neural.used and bpm is not None and
            abs(bpm - fallback_bpm) / max(fallback_bpm, 1e-9) > 0.02),
        "warning": (
            "Live-signature gate reproduction only; does not establish beat/downbeat "
            "musical correctness, does not execute analyze_track, and never replaces a grid."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only neural full-grid adoption gate audit")
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--filename", action="append", required=True,
                        help="Exact basename, repeat for each test track")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON report (must not already exist)")
    args = parser.parse_args()
    if not args.audio_root.is_dir():
        parser.error(f"Audio root not found: {args.audio_root}")
    # Fail closed on ambiguous names to avoid auditing a different copy.
    paths = [p for p in args.audio_root.rglob("*") if p.is_file()]
    selected = []
    for filename in args.filename:
        matches = [p for p in paths if p.name == filename]
        if len(matches) != 1:
            parser.error(f"{filename!r}: found {len(matches)} copies; require exactly one")
        selected.append(matches[0])
    results = [audit_grid_gate(path) for path in selected]
    report = {
        "mode": "read_only_live_signature_reproduction",
        "tracks": results,
        "live_analysis_executed": False,
        "live_adoptions": 0,
        "limitations": [
            "This invokes the same neural gate with the same fallback arguments as live analysis, not the running application.",
            "Neither BPM correctness nor beat/downbeat alignment can be established without independent annotations.",
            "A gate passing is a risk flag for review, not permission to apply it.",
            "Counterfactual safety-screen thresholds are provisional, not calibrated on independently labeled audio.",
        ],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        except FileExistsError:
            parser.error(f"Refusing to overwrite report: {args.output}")


if __name__ == "__main__":
    main()
