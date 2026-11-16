#!/usr/bin/env python3
"""Import locally acquired GiantSteps Tempo / Ballroom reference annotations.

Metadata-only, no network and no audio writes. Default split is development and
rights_confirmed is always false. The human owner must review rights/splits
before the separate M1 evaluator will accept the resulting manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg")


def _number(value: str, *, context: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric annotation {context}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric annotation {context}")
    return parsed


def giantsteps_bpm(path: Path) -> float | None:
    """Single BPM entry in annotations_v2/tempo/*.LOFI.bpm (0 = absent)."""
    values = path.read_text(encoding="utf-8").strip().split()
    if len(values) != 1:
        raise ValueError(f"expected one GiantSteps BPM value: {path.name}")
    bpm = _number(values[0], context=path.name)
    if bpm < 0:
        raise ValueError(f"negative GiantSteps BPM: {path.name}")
    return bpm if bpm > 0 else None


def ballroom_beats(path: Path) -> tuple[list[float] | None, list[float] | None]:
    """Each valid .beats row: timestamp seconds, ordinal beat-in-bar.

    The documented -1.0 first timestamp is a missing-annotation marker.
    Only beat position 1 yields an independently annotated downbeat.
    """
    beats: list[float] = []
    downs: list[float] = []
    first = True
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        words = line.split()
        if not words:
            continue
        if len(words) != 2:
            raise ValueError(f"expected timestamp and bar position: {path.name}:{lineno}")
        time = _number(words[0], context=f"{path.name}:{lineno}")
        if first and time == -1.0:
            return None, None
        first = False
        if time < 0 or (beats and time <= beats[-1]):
            raise ValueError(f"unsorted or invalid beat timestamp: {path.name}:{lineno}")
        if not re.fullmatch(r"[1-9][0-9]*", words[1]):
            raise ValueError(f"invalid beat bar-position: {path.name}:{lineno}")
        beats.append(time)
        if int(words[1]) == 1:
            downs.append(time)
    return (beats if beats else None), (downs if downs else None)


def ballroom_tempo(path: Path) -> float | None:
    """Optional separate single scalar tempo annotation, not inferred from beats."""
    values = path.read_text(encoding="utf-8").strip().splitlines()
    if len(values) != 1:
        raise ValueError(f"expected single Ballroom tempo: {path.name}")
    bpm = _number(values[0].split(",")[0].strip(), context=path.name)
    if bpm <= 0:
        raise ValueError(f"invalid Ballroom tempo: {path.name}")
    return bpm


def find_audio(root: Path, stem: str) -> str | None:
    # Reject ambiguous filenames rather than pairing wrong audio. Compare
    # full basenames independent of extension, case-insensitively.
    candidates = [
        p for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
        and p.suffix.lower() in AUDIO_EXTENSIONS and p.stem.casefold() == stem.casefold()
    ]
    if len(candidates) > 1:
        raise ValueError(f"ambiguous audio basename for {stem}: {len(candidates)} candidates")
    if not candidates:
        return None
    return candidates[0].relative_to(root).as_posix()


def import_sources(dataset: str, annotation_root: Path, audio_root: Path, *, version: str):
    root = annotation_root.resolve(strict=True)
    audio = audio_root.resolve(strict=True)
    if not root.is_dir() or not audio.is_dir():
        raise ValueError("annotation and audio roots must be directories")
    items = []
    counts = Counter()
    if dataset == "giantsteps_tempo":
        label_dir = root / ("annotations_v2/tempo" if version == "v2" else "annotations/tempo")
        annotations = sorted(label_dir.glob("*.bpm"))
    else:
        label_dir = root
        annotations = sorted(root.rglob("*.beats"))
    if not annotations:
        raise ValueError(f"no expected {dataset} annotation files in {label_dir}")
    for path in annotations:
        if path.is_symlink():
            raise ValueError(f"symlink annotation prohibited: {path}")
        if dataset == "giantsteps_tempo":
            track_id = path.stem  # 1030011.LOFI
            bpm = giantsteps_bpm(path)
            beat_times = None
            downbeat_times = None
        else:
            track_id = path.stem
            bpm = None
            beat_times, downbeat_times = ballroom_beats(path)
        if bpm is None and beat_times is None and downbeat_times is None:
            counts["missing_ground_truth"] += 1
            continue
        audio_rel = find_audio(audio, track_id)
        if audio_rel is None:
            counts["missing_audio"] += 1
            continue
        source = path.relative_to(root).as_posix()
        items.append({
            "id": track_id,
            "audio": audio_rel,
            "annotation_source": f"{dataset}:{version}:{source}",
            "bpm": bpm,
            "beats_seconds": beat_times,
            "downbeats_seconds": downbeat_times,
        })
        counts["included"] += 1
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate annotation IDs: resolve ambiguous files manually")
    if not items:
        raise ValueError("no annotated audio matched; check file paths and basename conventions")
    return items, counts, len(annotations)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("giantsteps_tempo", "ballroom"))
    parser.add_argument("--annotations", required=True, type=Path, help="local extracted annotation folder")
    parser.add_argument("--audio-root", required=True, type=Path, help="local manually acquired audio folder")
    parser.add_argument("--output", required=True, type=Path, help="new private unapproved manifest")
    parser.add_argument("--giantsteps-version", choices=("v1", "v2"), default="v2")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    if not args.annotations.is_dir() or not args.audio_root.is_dir():
        raise ValueError("annotations and audio root must exist")
    version = args.giantsteps_version if args.dataset == "giantsteps_tempo" else "bar_positions_v1"
    items, counts, scanned = import_sources(args.dataset, args.annotations, args.audio_root, version=version)
    manifest = {
        "schema_version": "saund-m1-rhythm-v1",
        "dataset_id": args.dataset,
        "split": "development",
        "rights_confirmed": False,
        "rights_review_note": "Owner must review recording/audio rights before enabling M1 evaluator.",
        "split_review_note": "Do not treat this provisional development-only manifest as held-out test data.",
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "dataset": args.dataset, "annotations_scanned": scanned,
        "included": len(items), "skipped_missing_audio": counts["missing_audio"],
        "skipped_missing_ground_truth": counts["missing_ground_truth"],
        "rights_confirmed": False, "split": "development", "ready_for_evaluation": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"M1 reference import failed: {exc}", file=sys.stderr)
        sys.exit(2)
