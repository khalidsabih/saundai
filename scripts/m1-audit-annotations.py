#!/usr/bin/env python3
"""M1 no-audio annotation audit for local GiantSteps Tempo and Ballroom repositories.

Read-only, no network, no audio access, no rights assertion, no model calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("m1_reference_importer", HERE / "m1-import-references.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scan(dataset: str, root: Path, version: str) -> dict:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("annotation root is not a directory")
    if dataset == "giantsteps_tempo":
        directory = root / ("annotations_v2/tempo" if version == "v2" else "annotations/tempo")
        paths = sorted(directory.glob("*.bpm"))
    elif dataset == "ballroom":
        paths = sorted(root.rglob("*.beats"))
    else:
        raise ValueError("unsupported dataset")
    if not paths:
        raise ValueError("no matching reference annotations")
    stats = Counter()
    names = set()
    fingerprint = hashlib.sha256()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe annotation path")
        rel = path.relative_to(root).as_posix()
        if path.stem in names:
            raise ValueError("duplicate annotation ID")
        names.add(path.stem)
        raw = path.read_bytes()
        fingerprint.update(rel.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(len(raw).to_bytes(8, "big"))
        fingerprint.update(raw)
        if dataset == "giantsteps_tempo":
            bpm = MODULE.giantsteps_bpm(path)
            stats["bpm_present" if bpm is not None else "bpm_missing"] += 1
        else:
            beats, downbeats = MODULE.ballroom_beats(path)
            if beats is None:
                stats["beats_missing"] += 1
            else:
                stats["beats_present"] += 1
                stats["beat_events"] += len(beats)
                stats["downbeat_events"] += len(downbeats or [])
                if downbeats:
                    stats["downbeat_tracks"] += 1
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = None
    return {
        "dataset": dataset, "annotation_version": version,
        "annotation_file_count": len(paths), "source_revision": revision,
        "annotations_sha256": fingerprint.hexdigest(),
        "counts": dict(sorted(stats.items())),
        "audio_acquired": False, "audio_use_rights_confirmed": False,
        "benchmark_ready": False, "note": "Reference annotation audit only; audio pairing, licenses and independent development/test split still required.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("giantsteps_tempo", "ballroom"), required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--giantsteps-version", choices=("v1", "v2"), default="v2")
    p.add_argument("--output", type=Path, help="optional NEW private JSON file; refuses overwrite")
    args = p.parse_args(argv)
    version = args.giantsteps_version if args.dataset == "giantsteps_tempo" else "bar_positions_v1"
    result = scan(args.dataset, args.annotations, version)
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("refusing to overwrite an existing output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"M1 annotation audit failed: {exc}", file=sys.stderr)
        sys.exit(2)
