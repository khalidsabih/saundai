"""Local, read-only workflow for gathering independent BPM reference annotations.

No BPMs, genres, verification claims or model predictions are invented.
Audio filenames stay on the user's machine; only an explicit local CSV/JSON
is written, never the GitHub repository or application runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .tempo_validation import validate_references

EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif"}
HEADERS = (
    "filename", "reference_bpm", "reference_quality",
    "verification_method", "source", "genre", "notes",
)


def _audio_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Audio directory not found: {root}")
    return sorted(
        (p for p in root.rglob("*")
         if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: str(p.relative_to(root)).casefold(),
    )


def select_for_annotation(root: Path, *, count: int = 24, seed: str = "saund-bpm-v1") -> list[str]:
    """Deterministic balanced draw across *folders*, not presumed genres.

    Folder diversity is a heuristic only; real style/genre is annotated by
    the listener. Duplicate basenames are excluded because the benchmark
    resolves filenames by basename, not by full relative path.
    """
    if count <= 0:
        raise ValueError("Requested sample count must be positive")
    paths = _audio_paths(root)
    basename_counts = Counter(p.name for p in paths)
    unique = [p for p in paths if basename_counts[p.name] == 1]
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in unique:
        # Parent directory is merely a grouping key, never a genre label.
        groups[str(p.parent.relative_to(root))].append(p)
    def rank(value: str) -> str:
        return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    for group in groups.values():
        group.sort(key=lambda p: rank(str(p.relative_to(root))))
    ordered_groups = sorted(groups, key=rank)
    chosen: list[str] = []
    while len(chosen) < count and any(groups.values()):
        for key in ordered_groups:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop(0).name)
    return chosen


def prepare_csv(root: Path, output: Path, *, count: int = 24, seed: str = "saund-bpm-v1") -> dict:
    """Write empty human-label fields; never overwrite an existing annotation."""
    filenames = select_for_annotation(root, count=count, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        for filename in filenames:
            writer.writerow({"filename": filename})
    paths = _audio_paths(root)
    duplicates = sorted(name for name, n in Counter(p.name for p in paths).items() if n > 1)
    return {
        "selected": len(filenames), "requested": count,
        "scanned_audio_files": len(paths),
        "duplicate_basenames_excluded": duplicates,
        "note": "Genre, BPM and verification method must be filled by humans; no BPMs were inferred.",
    }


def export_annotations(csv_file: Path, output: Path) -> dict:
    """Export only fully annotated rows; reject unsafe partially labeled rows."""
    with csv_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(HEADERS).issubset(reader.fieldnames):
            raise ValueError("CSV is missing one or more annotation columns")
        entries = list(reader)
    completed = []
    pending = []
    for lineno, entry in enumerate(entries, start=2):
        name = (entry.get("filename") or "").strip()
        if not name:
            raise ValueError(f"CSV row {lineno}: filename is required")
        bpm = (entry.get("reference_bpm") or "").strip()
        quality = (entry.get("reference_quality") or "").strip()
        method = (entry.get("verification_method") or "").strip()
        if not bpm and not quality and not method:
            pending.append(name)
            continue
        if not bpm or not quality:
            raise ValueError(f"CSV row {lineno}: partially annotated BPM/quality")
        try:
            numeric_bpm = float(bpm)
        except ValueError as exc:
            raise ValueError(f"CSV row {lineno}: invalid BPM {bpm!r}") from exc
        item = {
            "filename": name,
            "reference_bpm": numeric_bpm,
            "reference_quality": quality,
            "source": (entry.get("source") or "").strip() or "local annotation",
        }
        if method:
            item["verification_method"] = method
        for optional in ("genre", "notes"):
            value = (entry.get(optional) or "").strip()
            if value:
                item[optional] = value
        completed.append(item)
    validated = validate_references(completed)
    # Protect already-curated output from accidental replacement.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return {
        "annotated_exported": len(validated), "pending": len(pending),
        "verified": sum(r["reference_quality"] == "independently_verified" for r in validated),
        "approximate": sum(r["reference_quality"] == "approximate" for r in validated),
        "pending_filenames": pending,
        "warning": "Verification methods are supplied by annotators, not independently audited by SAUND.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and export manual BPM reference annotations")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Sample local tracks into an unlabelled annotation CSV")
    prepare.add_argument("--audio-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, default=24)
    prepare.add_argument("--seed", default="saund-bpm-v1")
    export = commands.add_parser("export", help="Validate completed annotations to BPM JSON")
    export.add_argument("--csv", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_csv(args.audio_root, args.output, count=args.count, seed=args.seed)
        else:
            result = export_annotations(args.csv, args.output)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
