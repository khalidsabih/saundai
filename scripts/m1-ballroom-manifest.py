#!/usr/bin/env python3
"""Build a rights-unapproved Ballroom evaluation manifest from original annotations.

Reads local B_1.0/annotations/{beats,tempo} and B_1.0/audio/<genre>/*.wav.
Excludes prior pilot IDs, selects seeded genre-round-robin records, and never
infers annotations from SAUND. Evaluation-only; no training or audio copying.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import random
import wave
from pathlib import Path


def read_beats(path):
    """Ballroom .beats rows are '<time_seconds> <beat_position>'; position 1 = downbeat."""
    beats, downbeats = [], []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path.name}:{line_no}: expected time and beat index")
        try:
            time = float(fields[0])
            position = int(fields[1])
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid annotation") from exc
        if not math.isfinite(time) or time < 0 or (beats and time <= beats[-1]):
            raise ValueError(f"{path.name}:{line_no}: nonmonotonic timestamp")
        if position < 1:
            raise ValueError(f"{path.name}:{line_no}: invalid beat index")
        beats.append(time)
        if position == 1:
            downbeats.append(time)
    if len(beats) < 4 or len(downbeats) < 2:
        raise ValueError(f"{path.name}: insufficient labeled beats/downbeats")
    return beats, downbeats


def read_bpm(path):
    try:
        bpm = float(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"{path.name}: invalid reference BPM") from exc
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError(f"{path.name}: invalid positive reference BPM")
    return bpm


def read_duration(path):
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as exc:
        raise ValueError(f"{path.name}: unreadable WAV duration") from exc


def build(source_root, exclude_ids, *, count=80, seed=20261117, min_seconds=8.0, split="development"):
    if split not in {"development", "validation", "test"}:\n        raise ValueError("invalid split")\n    if count < 1 or min_seconds < 4:
        raise ValueError("count must be positive; minimum duration must be >=4 seconds")
    annotations = source_root / "annotations" / "beats"
    tempos = source_root / "annotations" / "tempo"
    audio = source_root / "audio"
    if not annotations.is_dir() or not tempos.is_dir() or not audio.is_dir():
        raise ValueError("expected B_1.0 annotations/beats, annotations/tempo, and audio roots")
    groups = defaultdict(list)
    counts = Counter()
    seen = set()
    for beat_file in sorted(annotations.glob("*.beats")):
        identity = beat_file.stem
        if identity in seen:
            raise ValueError(f"duplicate basename {identity}")
        seen.add(identity)
        if identity in exclude_ids:
            counts["excluded_prior_pilot"] += 1
            continue
        tempo_file = tempos / (identity + ".bpm")
        matches = sorted(audio.glob("*/" + identity + ".wav"))
        if not tempo_file.is_file() or len(matches) != 1:
            counts["unmatched_sources"] += 1
            continue
        wav = matches[0]
        try:
            duration = read_duration(wav)
        except ValueError:
            counts["unreadable_wav"] += 1
            continue
        if duration < min_seconds:
            counts["too_short"] += 1
            continue
        beats, downbeats = read_beats(beat_file)
        bpm = read_bpm(tempo_file)
        genre = wav.parent.name
        groups[genre].append({
            "id": identity,
            "audio": wav.relative_to(audio).as_posix(),
            "annotation_source": beat_file.relative_to(source_root).as_posix(),
            "bpm": bpm,
            "beats_seconds": beats,
            "downbeats_seconds": downbeats,
            "genre": genre,
        })
        counts["eligible"] += 1
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    selected = []
    while len(selected) < count:
        changed = False
        for genre in sorted(groups):
            if groups[genre] and len(selected) < count:
                selected.append(groups[genre].pop())
                changed = True
        if not changed:
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} tracks; only {len(selected)} eligible after exclusions")
    return {
        "schema_version": "saund-m1-rhythm-v1",
        "dataset_id": "ballroom",
        "split": split,
        "rights_confirmed": False,
        "items": selected,
    }, {
        "selected": len(selected),
        "genres": dict(sorted(Counter(i["genre"] for i in selected).items())),
        "source_counts": dict(sorted(counts.items())),
        "seed": seed,
        "excluded_id_count": len(exclude_ids),
        "rights_confirmed": False,
        "evaluation_only": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="existing local Ballroom B_1.0 source root")
    parser.add_argument("--exclude-manifest", type=Path, required=True, action="append",
                        help="previous pilot; its IDs may not appear in new sample")
    parser.add_argument("--output", type=Path, required=True,
                        help="new private unapproved manifest; no overwrite")
    parser.add_argument("--count", type=int, default=80)\n    parser.add_argument("--split", choices=("development", "validation", "test"), default="development")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--min-seconds", type=float, default=8)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    old = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
    if old.get("dataset_id") != "ballroom" or not isinstance(old.get("items"), list):
        raise ValueError("expected Ballroom exclusion manifest")
    exclude = {i["id"] for i in old["items"]}
    manifest, audit = build(args.source_root.resolve(), exclude, count=args.count,
                            seed=args.seed, min_seconds=args.min_seconds)
    assert not (exclude & {i["id"] for i in manifest["items"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({"manifest": str(args.output),
                      "audio_root": str(args.source_root.resolve() / "audio"),
                      **audit}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser_message = f"M1 Ballroom manifest failed: {exc}"
        raise SystemExit(parser_message) from exc
