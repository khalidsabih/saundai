#!/usr/bin/env python3
"""Prepare a real Ballroom beat/downbeat benchmark manifest via mirdata.

No downloading unless --download AND --acknowledge-noncommercial-restrictions
are supplied. Uses mirdata's dataset index to pair audio and beat annotations.
Each annotation's documented beat-in-bar index 1 is exported as downbeats.
A deterministic audio-file SHA256 split keeps *exact* audio duplicates in
one partition. Near-duplicate recordings are not identified. Nothing is
committed, redistributed, or represented as a completed audio benchmark.

Ballroom is a non-EDM transfer benchmark. License noted by mirdata:
CC BY-NC-SA 4.0; determine independent rights to original audio before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

LICENSE = "Ballroom annotations: CC BY-NC-SA 4.0 as listed by mirdata; audio rights must be verified separately."
DATASET = "Ballroom Rhythm Dataset (beat/bar annotations)"
DEFAULT_HOME = Path("data/beat-benchmark/ballroom")
DEFAULT_MANIFEST = Path("data/beat-benchmark/manifest.json")


def split_for_hash(sha256: str) -> str:
    # Deterministic split by exact audio bytes, avoiding exact recording
    # duplicates in train/validation/test. 80% / 10% / 10%.
    number = int(sha256[:12], 16) % 10
    return "train" if number < 8 else ("validation" if number == 8 else "test")


def audio_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_ballroom_beats(path: Path) -> tuple[list[float], list[float]]:
    """Parse documented 'time in seconds <space/tab> beat-in-bar ID' lines."""
    beats: list[float] = []
    downbeats: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}: expected timestamp and beat-in-bar ID: {line!r}")
        time = float(fields[0])
        position = int(fields[1])
        if not math.isfinite(time) or time < 0:
            raise ValueError(f"{path}: invalid time {time!r}")
        if beats and time <= beats[-1]:
            raise ValueError(f"{path}: non-increasing timestamps")
        if position < 1 or position > 6:
            raise ValueError(f"{path}: unexpected beat-in-bar index {position!r}")
        beats.append(time)
        if position == 1:
            downbeats.append(time)
    if len(beats) < 8:
        raise ValueError(f"{path}: too few beat labels")
    return beats, downbeats


def format_labels(times: list[float]) -> str:
    return "".join(f"{t:.9f}\n" for t in times)


def generate_manifest(
    tracks: dict[str, Any], *, output: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert indexed local audio + annotations into benchmark-ready inputs.

    File paths are absolute because the manifest output and the mirdata data
    root need not share a parent directory. Only normalized annotations are
    written, under an explicitly local, user-owned output directory.
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    target = output.parent.resolve()
    normalized = target / "normalized-ballroom"
    rows = []
    missing_audio = 0
    missing_annotations = 0
    invalid_annotations = 0
    for track_id, track in sorted(tracks.items()):
        audio = Path(str(getattr(track, "audio_path", "") or "")).expanduser()
        beats_file = Path(str(getattr(track, "beats_path", "") or "")).expanduser()
        if not audio.is_file():
            missing_audio += 1
            continue
        if not beats_file.is_file():
            missing_annotations += 1
            continue
        try:
            beats, downbeats = parse_ballroom_beats(beats_file)
        except (ValueError, OSError):
            invalid_annotations += 1
            continue
        sha = audio_sha256(audio)
        key = hashlib.sha256(str(track_id).encode("utf-8")).hexdigest()[:12]
        bpath = normalized / f"{key}.beats"
        dpath = normalized / f"{key}.downbeats"
        # Write as normalized external labels only after both sources exist;
        # no output is a fabricated "successful" track.
        normalized.mkdir(parents=True, exist_ok=True)
        bpath.write_text(format_labels(beats), encoding="utf-8")
        row: dict[str, Any] = {
            "id": str(track_id),
            "split": split_for_hash(sha),
            "audio": str(audio.resolve()),
            "beats": str(bpath.resolve()),
            "audio_sha256": sha,
        }
        if downbeats:
            dpath.write_text(format_labels(downbeats), encoding="utf-8")
            row["downbeats"] = str(dpath.resolve())
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    manifest = {
        "dataset": DATASET,
        "license_note": LICENSE,
        "source": "mirdata.ballroom tracks audio_path and beats_path",
        "split_method": "first 12 hex digits of original audio SHA256 modulo 10 (80/10/10)",
        "validation_scope": "annotation-path and audio-file existence only; no music analysis run",
        "tracks": rows,
    }
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return {
        "manifest_written": bool(rows),
        "manifest": str(output.resolve()) if rows else None,
        "included_tracks": len(rows),
        "splits": {split: sum(t["split"] == split for t in rows)
                   for split in ("train", "validation", "test")},
        "skipped_missing_audio": missing_audio,
        "skipped_missing_annotations": missing_annotations,
        "skipped_invalid_annotations": invalid_annotations,
        "warning": (
            "No paired audio and beat annotations found; manifest was not created."
            if not rows else
            "Manifest ready; benchmark has not run. Verify licenses and audio-label alignment."
        ),
    }


def setup(
    *, data_home: Path, manifest: Path, download: bool,
    license_acknowledged: bool, limit: int | None = None,
) -> dict[str, Any]:
    if download and not license_acknowledged:
        raise ValueError(
            "Ballroom annotations are listed as CC BY-NC-SA 4.0. "
            "Review dataset and original audio rights, then pass "
            "--acknowledge-noncommercial-restrictions with --download."
        )
    try:
        import mirdata
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency mirdata. Install with: "
            "python -m pip install 'mirdata>=0.3.8,<2'"
        ) from exc
    dataset = mirdata.initialize("ballroom", data_home=str(data_home.resolve()))
    if download:
        # mirdata manages upstream URLs, checksums and unpacking. The audio
        # archive is approximately 1.5 GB; no audio is copied into GitHub.
        dataset.download()
    try:
        tracks = dataset.load_tracks()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            "Ballroom index/audio/annotations unavailable. If your rights "
            "permit research downloading, rerun with --download and the "
            "--acknowledge-noncommercial-restrictions flag."
        ) from exc
    return generate_manifest(tracks, output=manifest, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--download", action="store_true",
                        help="Explicit opt-in to download external dataset (about 1.5 GB audio)")
    parser.add_argument("--acknowledge-noncommercial-restrictions", action="store_true",
                        help="Confirm review of dataset license and original audio access rights")
    parser.add_argument("--limit", type=int, help="Optional cap on paired tracks for an initial trial")
    args = parser.parse_args()
    try:
        report = setup(
            data_home=args.data_home, manifest=args.manifest,
            download=args.download,
            license_acknowledged=args.acknowledge_noncommercial_restrictions,
            limit=args.limit,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    if not report["manifest_written"]:
        parser.exit(2, "No valid track pairs found. Nothing was benchmarked.\n")


if __name__ == "__main__":
    main()
