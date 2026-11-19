#!/usr/bin/env python3
"""Extract independent Essentia beat candidates in isolated NumPy 1.x venv.

Output is SHA-bound to two local source recordings. No DJAI dependencies.
No audio is uploaded, copied, rendered or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(path):
    import essentia.standard as es
    signal = es.MonoLoader(filename=str(path), sampleRate=44100)()
    bpm, ticks, confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(signal)
    beats = [float(v) for v in ticks]
    if not all(math.isfinite(v) for v in beats) or any(b <= a for a, b in zip(beats, beats[1:])):
        raise ValueError(f"invalid Essentia beats for {path.name}")
    if not math.isfinite(float(bpm)) or not math.isfinite(float(confidence)):
        raise ValueError(f"invalid Essentia tempo or confidence for {path.name}")
    return {
        "audio_sha256": _hash(path),
        "method": "essentia_rhythmextractor2013_multifeature",
        "beats_seconds": [round(x, 7) for x in beats],
        "bpm": round(float(bpm), 6),
        "confidence": round(float(confidence), 6),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-a", type=Path, required=True)
    parser.add_argument("--audio-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-analysis-rights", action="store_true")
    args = parser.parse_args()
    if not args.confirm_analysis_rights:
        parser.error("explicit --confirm-analysis-rights is required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; refusing overwrite")
    a, b = args.audio_a.expanduser().resolve(), args.audio_b.expanduser().resolve()
    if a == b or not a.is_file() or not b.is_file():
        parser.error("two distinct existing source audio files required")
    import essentia
    import numpy as np
    result = {
        "schema_version": "saund-m1-essentia-sidecar-v1",
        "method": "essentia_rhythmextractor2013_multifeature",
        "environment": {"numpy": np.__version__, "essentia": getattr(essentia, "__version__", "unknown")},
        "tracks": {"A": extract(a), "B": extract(b)},
        "production_changed": False,
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps({"output": str(args.output), "counts": {
        deck: len(x["beats_seconds"]) for deck, x in result["tracks"].items()
    }, "numpy": np.__version__}, indent=2))


if __name__ == "__main__":
    main()
