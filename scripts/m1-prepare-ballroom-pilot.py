#!/usr/bin/env python3
"""Prepare a small, local, unapproved Ballroom research pilot from official archive.

The user supplies the archive explicitly. No network/download, no model calls,
no automatic rights approval, no train/test split. Verifies original checksum,
filters exact annotation IDs, excludes documented recording replicas, and
extracts only a bounded number of regular WAV files to a new private directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READY_SCRIPT = ROOT / "scripts" / "m1-audio-readiness.py"
IMPORT_SCRIPT = ROOT / "scripts" / "m1-import-references.py"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(archive: Path, annotations: Path, *, max_tracks: int = 8):
    ready = load_script(READY_SCRIPT, "m1_audio_readiness_for_pilot")
    reference = load_script(IMPORT_SCRIPT, "m1_reference_import_for_pilot")
    if ready.md5(archive) != ready.BALLROOM_MD5:
        raise ValueError("archive does not match published Ballroom MD5; nothing extracted")
    annotation_files = {p.stem.casefold(): p for p in annotations.rglob("*.beats") if p.is_file() and not p.is_symlink()}
    if not annotation_files:
        raise ValueError("no Ballroom beat annotations")
    replica_stems = {x.casefold() for pair in ready.BALLROOM_REPLICA_PAIRS for x in pair}
    candidates = []
    count_by_stem = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("unsafe archive member")
            if not member.isfile() or path.suffix.lower() != ".wav":
                continue
            stem = path.stem.casefold()
            count_by_stem[stem] = count_by_stem.get(stem, 0) + 1
            if stem not in annotation_files or stem in replica_stems:
                continue
            if member.size > 64 * 1024 * 1024 or member.size < 1024:
                continue
            beats, downs = reference.ballroom_beats(annotation_files[stem])
            if not beats or not downs:
                continue
            genre = path.parent.name
            candidates.append((genre.casefold(), stem, member.name, member.size, annotation_files[stem]))
    safe = [c for c in candidates if count_by_stem[c[1]] == 1]
    if not safe:
        raise ValueError("no exact, unambiguous non-replica annotated WAV matches in archive")
    # Deterministic genre-diverse first pass; stable within each genre.
    genres = {}
    for c in sorted(safe):
        genres.setdefault(c[0], []).append(c)
    chosen = []
    while len(chosen) < max_tracks and any(genres.values()):
        for genre in sorted(genres):
            if genres[genre] and len(chosen) < max_tracks:
                chosen.append(genres[genre].pop(0))
    return chosen


def execute(archive: Path, annotations: Path, destination: Path, chosen, *, acknowledgement: bool):
    if not acknowledgement:
        raise ValueError("explicit --research-only-ack required; no rights automatically granted")
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; refuse to replace any files")
    destination.mkdir(parents=True)
    audio_root = destination / "audio"
    audio_root.mkdir()
    reference = load_script(IMPORT_SCRIPT, "m1_reference_import_for_extract")
    manifest = {
        "schema_version": "saund-m1-rhythm-v1",
        "dataset_id": "ballroom",
        "split": "development",
        "rights_confirmed": False,
        "rights_review_note": "Audio prep is not use-rights approval; intended evaluation use and attribution must be reviewed before running the M1 evaluator.",
        "source_archive_md5": "2872a3e52070bc342a4510a95e2fa0b8",
        "items": [],
    }
    with tarfile.open(archive, "r:gz") as handle:
        for genre, stem, member_name, size, annotation in chosen:
            member = handle.getmember(member_name)
            if not member.isfile() or member.size != size:
                raise ValueError("archive member changed between plan and extraction")
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError("missing archive audio")
            # Explicit flat ID path avoids trusting archive filenames for output.
            outfile = audio_root / (annotation.stem + ".wav")
            if outfile.exists():
                raise ValueError("duplicate extracted output name")
            with outfile.open("xb") as output:
                copied = shutil.copyfileobj(stream, output)
            if outfile.stat().st_size != size:
                raise ValueError("extracted WAV size mismatch")
            beats, downs = reference.ballroom_beats(annotation)
            manifest["items"].append({
                "id": annotation.stem,
                "audio": outfile.name,
                "annotation_source": "ballroom:bar_positions_v1:" + annotation.name,
                "bpm": None,
                "beats_seconds": beats,
                "downbeats_seconds": downs,
                "recording_sha256": hashlib.sha256(outfile.read_bytes()).hexdigest(),
            })
    out_manifest = destination / "ballroom-development-DRAFT.json"
    with out_manifest.open("x", encoding="utf-8") as writer:
        json.dump(manifest, writer, indent=2)
        writer.write("\n")
    return {"manifest": str(out_manifest), "audio_root": str(audio_root), "tracks": len(chosen),
            "rights_confirmed": False, "benchmark_ready": False,
            "note": "Exact-match audio prepared; no metrics executed, no license approval or held-out test."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="official original data1.tar.gz already on disk")
    parser.add_argument("--annotations", type=Path, required=True, help="pinned CPJKU BallroomAnnotations clone")
    parser.add_argument("--destination", type=Path, required=True, help="new private output directory")
    parser.add_argument("--max-tracks", type=int, default=8)
    parser.add_argument("--research-only-ack", action="store_true",
                        help="acknowledge local research pilot only, no redistribution, training or commercial-use assumption")
    parser.add_argument("--inspect-only", action="store_true", help="check archive and planned matches; extract nothing")
    args = parser.parse_args(argv)
    if not 1 <= args.max_tracks <= 16:
        raise ValueError("pilot track count must be 1..16")
    chosen = plan(args.archive, args.annotations, max_tracks=args.max_tracks)
    if args.inspect_only:
        print(json.dumps({"source_integrity_passed": True, "candidate_ids": [x[1] for x in chosen],
                          "sample_genres": sorted({x[0] for x in chosen}), "extracted": False,
                          "rights_confirmed": False}, indent=2))
        return 0
    print(json.dumps(execute(args.archive, args.annotations, args.destination, chosen,
                             acknowledgement=args.research_only_ack), indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, tarfile.TarError) as exc:
        print(f"M1 Ballroom pilot prep failed: {exc}", file=sys.stderr)
        sys.exit(2)
