#!/usr/bin/env python3
"""Audit official Groove MIDI metadata and prepare default-deny M1 tempo references.

No network, audio extraction, note-on->beat inference or rights attestation.
Official train/validation/test splits are preserved (train becomes development).
The MIDI-only source provides BPM but NO audio or independently marked beats.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

OFFICIAL_MIDI_SHA256 = "651cbc524ffb891be1a3e46d89dc82a1cecb09a57c748c7b45b844c4841dcc1e"
OFFICIAL_FULL_SHA256 = "21559feb2f1c96ca53988fd4d7060b1f2afe1d854fb2a8dcea5ff95cf3cce7e9"
REQUIRED = {"drummer", "session", "id", "style", "bpm", "beat_type", "time_signature", "midi_filename", "audio_filename", "duration", "split"}
SPLITS = {"train": "development", "validation": "validation", "test": "test"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(raw: str) -> str:
    if not raw or "\\" in raw:
        raise ValueError("missing or unsafe source file path")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or not all(path.parts):
        raise ValueError("unsafe source file path")
    return str(path)


def read_metadata(source: Path, *, verify_official: bool) -> tuple[list[dict[str, str]], str]:
    if source.suffix.lower() == ".zip":
        digest = sha256(source)
        if verify_official and digest not in (OFFICIAL_MIDI_SHA256, OFFICIAL_FULL_SHA256):
            raise ValueError("archive SHA256 differs from official Groove v1.0.0 downloads")
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if PurePosixPath(n).name == "info.csv" and not n.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError("expected exactly one info.csv in official archive")
            with archive.open(names[0]) as raw:
                records = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")))
        return records, digest
    if source.name.lower() != "info.csv":
        raise ValueError("metadata source must be official zip or local info.csv")
    if verify_official:
        raise ValueError("official SHA256 check requires original archive, not extracted info.csv")
    return list(csv.DictReader(source.open(encoding="utf-8-sig", newline=""))), sha256(source)


def audit(rows: list[dict[str, str]]) -> dict:
    if not rows:
        raise ValueError("empty Groove metadata")
    if not REQUIRED.issubset(rows[0]):
        raise ValueError("missing official Groove info.csv columns")
    counts = Counter()
    unique_ids = set()
    session_splits: dict[str, set[str]] = defaultdict(set)
    selected = []
    for row in rows:
        item_id = row["id"].strip()
        split = row["split"].strip().lower()
        if not item_id or item_id in unique_ids:
            raise ValueError("missing or duplicate Groove ID")
        unique_ids.add(item_id)
        if split not in SPLITS:
            raise ValueError(f"unrecognized official split for {item_id}")
        bpm = float(row["bpm"])
        duration = float(row["duration"])
        if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid BPM/duration for {item_id}")
        midi = safe_path(row["midi_filename"])
        # The MIDI-only distribution can still include a filename reference to
        # unavailable WAV audio. Do not treat metadata filename as actual audio.
        if row["audio_filename"].strip():
            safe_path(row["audio_filename"])
        session = row["drummer"].strip() + "/" + row["session"].strip()
        if not row["drummer"].strip() or not row["session"].strip():
            raise ValueError("missing group identifiers")
        session_splits[session].add(split)
        counts["split_" + split] += 1
        counts["type_" + row["beat_type"].strip().lower()] += 1
        counts["style_" + row["style"].split("/")[0].strip().lower()] += 1
        selected.append({
            "id": item_id,
            "official_split": split,
            "group": session,
            "bpm": bpm,
            "duration_seconds": duration,
            "style": row["style"],
            "beat_type": row["beat_type"],
            "time_signature": row["time_signature"],
            "midi_filename": midi,
            "audio_filename": row["audio_filename"],
        })
    overlap = {k: sorted(v) for k, v in session_splits.items() if len(v) > 1}
    return {"total": len(rows), "counts": dict(sorted(counts.items())),
            "session_cross_split_count": len(overlap),
            "session_cross_split_examples": dict(list(sorted(overlap.items()))[:12]),
            "entries": selected}


def build_manifest(data: dict, *, split: str, audio_root: Path) -> tuple[dict, dict]:
    if split not in SPLITS:
        raise ValueError("invalid official split")
    included = []
    no_audio = 0
    for row in data["entries"]:
        if row["official_split"] != split:
            continue
        rel = row["audio_filename"].strip()
        if not rel:
            no_audio += 1
            continue
        target = (audio_root / rel).resolve()
        if not target.is_relative_to(audio_root.resolve()) or not target.is_file():
            no_audio += 1
            continue
        included.append({
            "id": row["id"], "audio": rel,
            "annotation_source": "magenta:groove-v1.0.0:info.csv:bpm",
            "bpm": row["bpm"], "beats_seconds": None, "downbeats_seconds": None,
        })
    if not included:
        raise ValueError("no matching original Groove WAVs in selected split; MIDI-only archive is insufficient for audio evaluation")
    manifest = {
        "schema_version": "saund-m1-rhythm-v1", "dataset_id": "groove_midi",
        "split": SPLITS[split], "rights_confirmed": False,
        "rights_review_note": "Official dataset CC BY 4.0. Confirm attribution/use and recording provenance before evaluation.",
        "split_review_note": "Official performance split preserved; inspect drummer/session overlap before claiming artist-disjoint holdout.",
        "items": included,
    }
    return manifest, {"included": len(included), "missing_audio": no_audio}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="official groove-v1.0.0-midionly.zip, full archive, or extracted info.csv")
    p.add_argument("--allow-unverified-csv", action="store_true", help="permit local CSV fixture/metadata without official zip checksum")
    p.add_argument("--output", type=Path, help="NEW private audit JSON; never overwritten")
    p.add_argument("--manifest-output", type=Path, help="NEW unapproved M1 tempo-only manifest; requires matching WAV audio")
    p.add_argument("--audio-root", type=Path, help="locally acquired exact Groove WAV files")
    p.add_argument("--official-split", choices=tuple(SPLITS), default="train")
    args = p.parse_args(argv)
    if args.manifest_output and not args.audio_root:
        raise ValueError("--manifest-output requires --audio-root")
    outputs = [x for x in [args.output, args.manifest_output] if x]
    if any(x.exists() or x.is_symlink() for x in outputs):
        raise ValueError("output already exists; refuse overwrite")
    rows, checksum = read_metadata(args.source, verify_official=not args.allow_unverified_csv)
    inspected = audit(rows)
    result = {
        "source_sha256": checksum, "official_archive_verified": not args.allow_unverified_csv,
        "source": "Magenta Groove MIDI v1.0.0", "total": inspected["total"],
        "counts": inspected["counts"], "session_cross_split_count": inspected["session_cross_split_count"],
        "session_cross_split_examples": inspected["session_cross_split_examples"],
        "audio_acquired": False, "rights_confirmed": False,
        "beat_downbeat_ground_truth": False, "benchmark_ready": False,
    }
    manifest = None
    if args.manifest_output:
        manifest, coverage = build_manifest(inspected, split=args.official_split, audio_root=args.audio_root)
        result["paired_audio"] = coverage
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_output.open("x", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
        print(f"M1 Groove metadata failed: {exc}", file=sys.stderr)
        sys.exit(2)
