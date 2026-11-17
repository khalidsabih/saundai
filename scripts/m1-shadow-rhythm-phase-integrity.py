#!/usr/bin/env python3
"""Shadow-only, generator-independent rhythmic phase-integrity report for WAVs.

The two filenames are rights-free fixture inputs, NOT tempo/meter references.
All candidate timing is derived from actual TrackAnalysis.beats_seconds and
independent filter-band energy in the waveform. Does not change analysis,
planning, tempo reconciliation or rendering. Reports are never authorization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_report(paths, analyze):
    from djai.music.rhythm_phase_integrity import inspect_waveform

    tracks = []
    for path in paths:
        analysis = analyze(path)
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = np.asarray(audio.mean(axis=1), dtype=np.float64)
        try:
            evidence = inspect_waveform(mono, sr, analysis.beats_seconds)
        except ValueError as exc:
            evidence = {
                "decision": "insufficient_evidence",
                "reason": "unsupported_audio_or_grid",
                "detail": str(exc),
                "windows": [],
                "sustained_offbeat_runs": [],
                "sustained_kick_runs": [],
                "generator_reference_used": False,
                "beat_grid_modified": False,
                "authority": "shadow_only",
            }
        tracks.append({
            "filename": path.name,
            "scalar_bpm": analysis.bpm,
            "beat_count": len(analysis.beats_seconds),
            "predicted_meter": analysis.time_signature,
            "neural_grid_used": (analysis.neural_evidence.get("beat_downbeat") or {}).get("used"),
            "phase_integrity": evidence,
        })
    return {
        "schema": "saund-m1-shadow-rhythm-phase-integrity-v1",
        "tracks": tracks,
        "generator_reference_used": False,
        "beat_grid_corrected": False,
        "plan_generated": False,
        "production_changed": False,
        "transition_rendered": False,
        "warning": (
            "Shadow-only contrast of low/high frequency energy sampled on detected beats "
            "and their midpoints. It does not determine correct beat identity or "
            "authorize modifying a transition plan."
        ),
    }


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
    paths = [args.audio_root.expanduser().resolve() / name for name in names]
    if any(not path.is_file() for path in paths):
        raise ValueError("missing known rights-free generated demo WAV")
    result = build_report(paths, analyze_track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 shadow phase integrity failed: {exc}") from exc
