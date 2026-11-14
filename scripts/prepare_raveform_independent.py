#!/usr/bin/env python3
"""Stage Raveform human beat/downbeat labels for the offline independent benchmark.

Reads ONLY structures/beats/*.beat.csv and tracks.jsonl. The derived labels are
reference data, never inference inputs. Produces a candidate index even when
no audio is available. A benchmark manifest is only emitted with explicit,
operator-attested, checksum-verified audio matching and alignment.

No production models, runtime beat grids, cache, phase gates, or DJ engine
files are read or modified. No audio downloads are attempted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "saund-raveform-preparation-v1"


def track_id_from_filename(path: Path) -> str:
    suffix = ".beat.csv"
    if not path.name.endswith(suffix):
        raise ValueError(f"not an annotation CSV: {path.name}")
    stem = path.name[:-len(suffix)]
    parts = stem.split(".", 1)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1]:
        raise ValueError(f"unexpected annotation filename: {path.name}")
    return parts[1]


def load_metadata(root: Path) -> dict[str, dict]:
    path = root / "tracks.jsonl"
    tracks = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            ident = item.get("id")
            if not isinstance(ident, str) or not ident or ident in tracks:
                raise ValueError(f"invalid or duplicate track ID at {path}:{lineno}")
            tracks[ident] = item
    return tracks


def parse_annotation(path: Path) -> tuple[list[float], list[float]]:
    beats, downbeats = [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"time", "downbeat"}.issubset(reader.fieldnames):
            raise ValueError(f"missing time/downbeat columns: {path}")
        for row_number, row in enumerate(reader, 2):
            try:
                timestamp = float(row["time"])
                position = int(row["downbeat"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"bad annotation row {path}:{row_number}") from exc
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"invalid timestamp at {path}:{row_number}")
            if beats and timestamp <= beats[-1]:
                raise ValueError(f"non-increasing beat times at {path}:{row_number}")
            if position < 1:
                raise ValueError(f"invalid bar position at {path}:{row_number}")
            beats.append(timestamp)
            if position == 1:
                downbeats.append(timestamp)
    if not beats or not downbeats:
        raise ValueError(f"no beats or downbeats in {path}")
    return beats, downbeats


def read_verified_map(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ValueError("verified audio map requires a tracks array")
    return data["tracks"]


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify_audio(item: dict, candidates: dict, audio_root: Path) -> tuple[str, Path]:
    ident = item.get("id")
    if ident not in candidates:
        raise ValueError(f"audio-map track ID has no human annotation: {ident}")
    if item.get("recording_verified") is not True:
        raise ValueError(f"{ident}: recording_verified must be true")
    if item.get("alignment_verified") is not True:
        raise ValueError(f"{ident}: alignment_verified must be true")
    source = item.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{ident}: documented lawful audio source required")
    audio = item.get("audio")
    if not isinstance(audio, str) or not audio:
        raise ValueError(f"{ident}: audio path required")
    path = (audio_root / audio).resolve()
    if not path.is_file():
        raise ValueError(f"{ident}: audio file missing: {path}")
    expected = item.get("sha256")
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in "0123456789abcdef" for c in expected.lower())):
        raise ValueError(f"{ident}: valid sha256 required")
    if hash_file(path) != expected.lower():
        raise ValueError(f"{ident}: SHA-256 mismatch")
    return ident, path


def prepare(root: Path, output: Path, audio_root: Path | None = None,
            verified_map: Path | None = None) -> dict:
    root = root.resolve()
    output = output.resolve()
    metadata = load_metadata(root)
    paths = sorted((root / "structures" / "beats").glob("*.beat.csv"))
    if not paths:
        raise ValueError("no structures/beats/*.beat.csv found")
    candidates = {}
    # Validate the entire annotation collection before writing anything.
    for path in paths:
        ident = track_id_from_filename(path)
        if ident not in metadata or ident in candidates:
            raise ValueError(f"unmatched or duplicate annotation ID: {ident}")
        beats, downbeats = parse_annotation(path)
        candidates[ident] = {
            "id": ident, "title": metadata[ident].get("title"),
            "annotation": str(path), "beat_count": len(beats),
            "downbeat_count": len(downbeats),
            "last_beat_seconds": beats[-1],
            "duration_metadata_seconds": metadata[ident].get("duration"),
            "_beats": beats, "_downbeats": downbeats,
        }
    verified = []
    if verified_map is not None:
        if audio_root is None:
            raise ValueError("--audio-root required with --verified-map")
        entries = read_verified_map(verified_map)
        if not entries:
            raise ValueError("verified audio map has no tracks")
        seen = set()
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError("audio map entries must be objects")
            ident, path = verify_audio(item, candidates, audio_root)
            if ident in seen:
                raise ValueError(f"duplicate verified track: {ident}")
            seen.add(ident)
            verified.append((ident, path, item))

    # A previous verified run must never leave a misleading runnable manifest\n    # behind when a later labels-only preparation has no verified audio.\n    stale_manifest = output / "raveform-independent-manifest.json"\n    if not verified and stale_manifest.exists():\n        stale_manifest.unlink()\n\n    labels = output / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    public_candidates = []
    for ident in sorted(candidates):
        row = candidates[ident]
        beat_path = labels / f"{ident}.beats"
        downbeat_path = labels / f"{ident}.downbeats"
        beat_path.write_text("".join(f"{t:.9f}\n" for t in row["_beats"]), encoding="utf-8")
        downbeat_path.write_text(
            "".join(f"{t:.9f}\n" for t in row["_downbeats"]), encoding="utf-8"
        )
        public_candidates.append({k: v for k, v in row.items() if not k.startswith("_")})
    index = {
        "schema": SCHEMA, "dataset": "Raveform",
        "annotation_source": str(root),
        "total_metadata_tracks": len(metadata),
        "annotated_tracks": len(public_candidates),
        "benchmark_ready": bool(verified),
        "warning": "Human reference labels only; audio is not included in Raveform ZIP.",
        "tracks": public_candidates,
    }
    index_path = output / "raveform-candidates.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    report = {"candidate_index": str(index_path), "annotated": len(public_candidates),
              "verified_audio": len(verified), "manifest": None}
    if verified:
        manifest = {
            "dataset": "Raveform independent EDM holdout",
            "provenance": (
                "Raveform structures/beats human annotations; externally sourced "
                "matching recordings, individually attested and SHA-256 checked; "
                "operator must ensure no overlap with prior evaluation."
            ),
            "tracks": [{
                "id": ident, "split": "independent", "audio": str(path),
                "beats": str(labels / f"{ident}.beats"),
                "downbeats": str(labels / f"{ident}.downbeats"),
            } for ident, path, _ in verified],
            "audio_provenance": [{
                "id": ident, "source": entry["source"],
                "sha256": entry["sha256"].lower(),
                "recording_verified": True, "alignment_verified": True,
                "verification_notes": entry.get("verification_notes", ""),
            } for ident, _, entry in verified],
            "limitations": [
                "Recording version and annotation alignment were attested by the operator, not automatically proven.",
                "Do not tune on this independent holdout or include Ballroom evaluation audio.",
                "No reference data enters phase inference. No 25ms production safety-gate changes.",
            ],
        }
        manifest_path = output / "raveform-independent-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        report["manifest"] = str(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raveform-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/raveform/benchmark"))
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--verified-map", type=Path,
                        help="Explicit operator-attested JSON mapping, not auto-discovered audio")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raveform_root, args.output_dir,
                             args.audio_root, args.verified_map), indent=2))


if __name__ == "__main__":
    main()
