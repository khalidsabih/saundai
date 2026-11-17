#!/usr/bin/env python3
"""M0-only, read-only music inventory and conservative scenario manifest validation.

No decoding, inference, download, modification to audio, or automatic genre labels.
Writes only the explicitly selected manifest file. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".aac", ".ogg"})
SCENARIOS = {
    "S01": "House -> House",
    "S02": "Techno -> Melodic Techno",
    "S03": "House -> Afro House",
    "S04": "Afro House -> Pop",
    "S05": "House -> Hip-Hop",
    "S06": "R&B -> vocal Pop",
    "S07": "Hip-Hop / Rap -> Pop",
    "S08": "Amapiano / Dancehall / Reggaeton",
    "S09": "Jazz / Rock / live drums",
    "S10": "Ambient / orchestral / sparse intro",
    "S11": "Odd meter / meter shift",
    "S12": "Multi-genre complete set",
}
SOURCE_SHA = "1aef9a8d8fbfaac90e2b65d4f9e70120618785bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(audio_root: Path) -> dict:
    if not audio_root.is_dir():
        raise ValueError(f"audio root missing: {audio_root}")
    tracks = []
    failures = []
    for current_root, dirs, names in os.walk(audio_root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(current_root) / d).is_symlink())
        for name in sorted(names):
            path = Path(current_root) / name
            if path.suffix.lower() not in AUDIO_EXTENSIONS or path.is_symlink():
                continue
            relative = path.relative_to(audio_root).as_posix()
            try:
                stat = path.stat()
                if not path.is_file() or stat.st_size <= 0:
                    failures.append({"path": relative, "error": "missing_or_empty"})
                    continue
                digest = sha256(path)
                tracks.append({
                    "id": digest[:16],
                    "path": relative,
                    "sha256": digest,
                    "bytes": stat.st_size,
                    "artist": None,
                    "genre": None,
                    "rights_status": "unreviewed",
                    "split": "unassigned",
                    "recording_group": None,
                    "notes": "",
                })
            except OSError as exc:
                failures.append({"path": relative, "error": type(exc).__name__})
    tracks.sort(key=lambda item: item["path"].lower())
    return {
        "schema": "saund-m0-crate-v1",
        "baseline_commit": SOURCE_SHA,
        "audio_root_hint": str(audio_root),
        "tracks": tracks,
        "scenarios": [
            {"id": ident, "description": description, "track_ids_in_order": [], "status": "unassigned", "notes": ""}
            for ident, description in SCENARIOS.items()
        ],
        "scan_failures": failures,
        "freeze_status": "draft",
        "important": "This is an inventory, not independent musical ground truth or a rights approval.",
    }


def check(payload: dict, strict: bool) -> tuple[list[str], dict]:
    errors: list[str] = []
    tracks = payload.get("tracks")
    scenarios = payload.get("scenarios")
    if payload.get("schema") != "saund-m0-crate-v1" or not isinstance(tracks, list) or not isinstance(scenarios, list):
        return ["invalid_manifest_schema"], {}
    ids = [t.get("id") for t in tracks]
    paths = [t.get("path") for t in tracks]
    if len(set(ids)) != len(ids):
        errors.append("duplicate_track_ids_or_identical_audio_hashes: remove duplicates or choose one canonical file")
    if len(set(paths)) != len(paths):
        errors.append("duplicate_file_paths")
    by_id = {t["id"]: t for t in tracks if isinstance(t.get("id"), str)}
    for t in tracks:
        if not isinstance(t.get("sha256"), str) or len(t["sha256"]) != 64:
            errors.append(f"missing_sha256:{t.get('path')}")
        if t.get("split") not in {"unassigned", "development", "test"}:
            errors.append(f"invalid_split:{t.get('path')}")
        if t.get("rights_status") not in {"unreviewed", "approved_local_research", "not_allowed"}:
            errors.append(f"invalid_rights_status:{t.get('path')}")
    scenario_ids = [s.get("id") for s in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)) or set(scenario_ids) != set(SCENARIOS):
        errors.append("scenario_ids_must_match_S01_through_S12")
    usage = []
    for s in scenarios:
        chosen = s.get("track_ids_in_order", [])
        if not isinstance(chosen, list):
            errors.append(f"invalid_selection:{s.get('id')}")
            continue
        unknown = [ident for ident in chosen if ident not in by_id]
        if unknown:
            errors.append(f"unknown_track_ids:{s.get('id')}:{unknown}")
        usage.extend(ident for ident in chosen if ident in by_id)
        if s.get("status") not in {"unassigned", "ready", "unavailable"}:
            errors.append(f"invalid_scenario_status:{s.get('id')}")
        if strict and s.get("status") == "ready" and len(chosen) < 2:
            errors.append(f"ready_scenario_requires_two_tracks:{s.get('id')}")
    groups: dict[str, set[str]] = {}
    for t in tracks:
        if t.get("split") in {"development", "test"}:
            # Artist and recording group both protect against leakage if provided.
            for kind in ("artist", "recording_group"):
                value = str(t.get(kind) or "").strip().casefold()
                if value:
                    groups.setdefault(f"{kind}:{value}", set()).add(t["split"])
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"development_test_leakage:{group}")
    if strict:
        if payload.get("scan_failures"):
            errors.append("scan_failures_not_resolved")
        if not tracks:
            errors.append("empty_track_inventory")
        for ident in usage:
            t = by_id[ident]
            if t.get("rights_status") != "approved_local_research":
                errors.append(f"unapproved_rights:{t['path']}")
            if not t.get("genre") or not t.get("artist") or not t.get("recording_group"):
                errors.append(f"incomplete_human_metadata:{t['path']}")
            if t.get("split") not in {"development", "test"}:
                errors.append(f"unassigned_split:{t['path']}")
        for s in scenarios:
            if s.get("status") == "unassigned":
                errors.append(f"scenario_unresolved:{s.get('id')}")
            if s.get("status") == "ready":
                splits = {by_id[ident].get("split") for ident in s["track_ids_in_order"] if ident in by_id}
                if len(splits) > 1:
                    errors.append(f"mixed_split_in_scenario:{s.get('id')}")
        if not any(t.get("split") == "development" for t in tracks) or not any(t.get("split") == "test" for t in tracks):
            errors.append("both_development_and_test_tracks_required")
    stats = {
        "track_count": len(tracks),
        "scenario_ready": sum(s.get("status") == "ready" for s in scenarios),
        "scenario_unavailable": sum(s.get("status") == "unavailable" for s in scenarios),
        "unassigned_scenarios": sum(s.get("status") == "unassigned" for s in scenarios),
        "split_counts": dict(Counter(t.get("split", "missing") for t in tracks)),
        "rights_counts": dict(Counter(t.get("rights_status", "missing") for t in tracks)),
        "scan_failure_count": len(payload.get("scan_failures", [])),
    }
    return sorted(set(errors)), stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="hash local audio and create an unreviewed manifest")
    inv.add_argument("--audio-root", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    val = sub.add_parser("validate", help="review a manifest without editing or scanning audio")
    val.add_argument("--manifest", type=Path, required=True)
    val.add_argument("--strict", action="store_true", help="check readiness for an M0 owner signoff")
    args = parser.parse_args()
    if args.command == "inventory":
        root = args.audio_root.expanduser().resolve(strict=True)
        output = args.output.expanduser().absolute()
        if output.is_symlink() or output.exists():
            parser.error("output already exists: refusing to overwrite")
        if output.is_relative_to(root):
            parser.error("output must be outside the audio root")
        payload = inventory(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        errors, stats = check(payload, strict=False)
        print(json.dumps({"manifest": str(output), **stats, "validation_errors": errors}, indent=2))
        return 1 if errors or payload["scan_failures"] else 0
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors, stats = check(payload, strict=args.strict)
    print(json.dumps({**stats, "ready_for_owner_review": not errors and args.strict, "validation_errors": errors}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"M0 inventory error: {exc}", file=sys.stderr)
        sys.exit(2)
