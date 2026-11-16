#!/usr/bin/env python3
"""Read-only M1 audio-pairing audit for officially acquired local files/archives.

No network, extraction, mutation or rights inference. Inspect exact basename
matches and documented Ballroom recording replicas before holding out tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import sys
from collections import defaultdict
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
BALLROOM_MD5 = "2872a3e52070bc342a4510a95e2fa0b8"
BALLROOM_REPLICA_PAIRS = (
    ("Albums-AnaBelen_Veneo-11", "Albums-Chrisanne2-12"),
    ("Albums-Fire-08", "Albums-Fire-09"),
    ("Albums-Latin_Jam2-05", "Albums-Latin_Jam2-13"),
    ("Albums-Secret_Garden-01", "Media-104705"),
    ("Albums-AnaBelen_Veneo-03", "Albums-AnaBelen_Veneo-15"),
    ("Albums-Ballroom_Magic-03", "Albums-Ballroom_Magic-18"),
    ("Albums-Latin_Jam-04", "Albums-Latin_Jam-13"),
    ("Albums-Latin_Jam-08", "Albums-Latin_Jam-14"),
    ("Albums-Latin_Jam-06", "Albums-Latin_Jam-15"),
    ("Albums-Latin_Jam2-02", "Albums-Latin_Jam2-14"),
    ("Albums-Latin_Jam2-07", "Albums-Latin_Jam2-15"),
    ("Albums-Latin_Jam3-02", "Media-103414"),
    ("Media-103402", "Media-103415"),
)


def md5(path: Path) -> str:
    h = hashlib.md5()  # nosec - published archive integrity checksum, not for security
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_index(*, root: Path | None = None, archive: Path | None = None) -> dict[str, list[str]]:
    if (root is None) == (archive is None):
        raise ValueError("choose exactly one of --audio-root or --archive")
    index: dict[str, list[str]] = defaultdict(list)
    if root is not None:
        base = root.resolve(strict=True)
        if not base.is_dir():
            raise ValueError("audio root must be a directory")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                continue
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                index[path.stem.casefold()].append(path.relative_to(base).as_posix())
    else:
        assert archive is not None
        with tarfile.open(archive, mode="r:*") as tar:
            for member in tar:
                path = Path(member.name)
                if member.issym() or member.islnk():
                    raise ValueError("archive contains links; refuse unsafe archive")
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("archive contains unsafe path")
                if member.isfile() and path.suffix.lower() in AUDIO_SUFFIXES:
                    index[path.stem.casefold()].append(path.as_posix())
    return dict(index)


def annotation_ids(root: Path, dataset: str, version: str) -> list[str]:
    root = root.resolve(strict=True)
    if dataset == "ballroom":
        paths = sorted(root.rglob("*.beats"))
    elif dataset == "giantsteps_tempo":
        paths = sorted((root / ("annotations_v2/tempo" if version == "v2" else "annotations/tempo")).glob("*.bpm"))
    else:
        raise ValueError("unsupported dataset")
    if not paths:
        raise ValueError("annotation source contains no reference files")
    names = [p.stem for p in paths]
    if len(names) != len({x.casefold() for x in names}):
        raise ValueError("ambiguous annotation identifiers")
    return names


def audit(ids: list[str], index: dict[str, list[str]], dataset: str, archive_hash: str | None) -> dict:
    matched = []
    missing = []
    ambiguous = []
    for name in ids:
        files = index.get(name.casefold(), [])
        if not files:
            missing.append(name)
        elif len(files) > 1:
            ambiguous.append({"id": name, "files": files})
        else:
            matched.append(name)
    replica_groups = []
    if dataset == "ballroom":
        for a, b in BALLROOM_REPLICA_PAIRS:
            replica_groups.append({
                "members": [a, b],
                "both_matched": a in matched and b in matched,
                "rule": "keep in same split; never mix test and development",
            })
    return {
        "dataset": dataset,
        "reference_count": len(ids),
        "matched_unique_audio": len(matched),
        "missing_audio": len(missing),
        "ambiguous_audio": len(ambiguous),
        "missing_ids": missing[:30],
        "ambiguous_ids": ambiguous[:30],
        "known_ballroom_replica_groups": replica_groups,
        "source_archive_md5": archive_hash,
        "audio_rights_confirmed": False,
        "benchmark_ready": False,
        "note": "Read-only matching audit. Match counts are not license approvals, independent metadata labels, approved splits or real accuracy results.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("ballroom", "giantsteps_tempo"))
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--giantsteps-version", choices=("v1", "v2"), default="v2")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio-root", type=Path)
    group.add_argument("--archive", type=Path, help="local .tar.gz only; inspected in place, never extracted")
    p.add_argument("--output", type=Path, help="optional NEW private JSON; refuses overwrite")
    args = p.parse_args(argv)
    if args.output and (args.output.exists() or args.output.is_symlink()):
        raise ValueError("output already exists; refusing overwrite")
    names = annotation_ids(args.annotations, args.dataset, args.giantsteps_version)
    archive_hash = None
    if args.archive is not None:
        if args.dataset != "ballroom":
            raise ValueError("GiantSteps audio is not published as a verified tar in this workflow")
        archive_hash = md5(args.archive)
        if archive_hash != BALLROOM_MD5:
            raise ValueError("Ballroom archive does not match official published MD5 checksum")
    result = audit(names, audio_index(root=args.audio_root, archive=args.archive), args.dataset, archive_hash)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, tarfile.TarError) as exc:
        print(f"M1 audio readiness failed: {exc}", file=sys.stderr)
        sys.exit(2)
