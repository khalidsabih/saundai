from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, TextIO

from ..intelligence.edm_cue import import_edm_cue_annotations
from .source_catalog import get_source, source_catalog

BUILDER_SCHEMA_VERSION = "1.3"
NORMALIZED_RECORD_SCHEMA_VERSION = "1.0"
BOOTSTRAP_CORPUS_SCHEMA_VERSION = "1.3"

RecordKind = Literal[
    "track_metadata",
    "cue_annotation",
    "mix_segment",
    "transition_annotation",
    "sequence_edge",
    "weak_example",
]


@dataclass(slots=True)
class NormalizedRecord:
    record_id: str
    source_dataset: str
    source_record_id: str
    kind: RecordKind
    split: str
    group: str
    label_source: str
    label_confidence: float
    rights_cleared: bool
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = NORMALIZED_RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*values: object, prefix: str = "rec") -> str:
    text = "|".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _normalize_split(value: Any, group: str) -> str:
    text = str(value or "").strip().lower()
    aliases = {"training": "train", "val": "validation", "valid": "validation", "testing": "test"}
    text = aliases.get(text, text)
    return text if text in {"train", "validation", "test"} else _stable_split(group)


def _manifest_path(output_path: Path) -> Path:
    return output_path.with_name("manifest.json")


def _write_import_manifest(
    output_path: Path,
    *,
    source_id: str,
    source_path: Path,
    counts: Counter[str],
    rights_cleared: bool,
    notes: list[str] | None = None,
) -> Path:
    manifest = {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "source_dataset": source_id,
        "source_path": str(source_path),
        "normalized_records": str(output_path),
        "created_at_utc": _now(),
        "counts": dict(sorted(counts.items())),
        "rights_cleared": bool(rights_cleared),
        "raw_audio_bundled": False,
        "notes": list(notes or []),
    }
    path = _manifest_path(output_path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _open_fma_tracks_csv(source: Path) -> TextIO:
    if source.is_dir():
        candidates = list(source.rglob("tracks.csv"))
        if not candidates:
            raise FileNotFoundError("tracks.csv was not found in the FMA metadata directory.")
        return candidates[0].open("r", encoding="utf-8-sig", newline="")
    if source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        names = [name for name in archive.namelist() if name.endswith("tracks.csv")]
        if not names:
            archive.close()
            raise FileNotFoundError("tracks.csv was not found in the FMA metadata archive.")
        raw = archive.open(names[0], "r")
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        original_close = text.close

        def close() -> None:
            original_close()
            archive.close()

        text.close = close  # type: ignore[method-assign]
        return text
    if source.name == "tracks.csv":
        return source.open("r", encoding="utf-8-sig", newline="")
    raise ValueError("FMA importer expects fma_metadata.zip, an extracted metadata directory, or tracks.csv.")


def _fma_headers(reader: csv.reader) -> tuple[list[str], list[str]]:
    first = next(reader)
    second = next(reader)
    width = max(len(first), len(second))
    first += [""] * (width - len(first))
    second += [""] * (width - len(second))
    names: list[str] = []
    for index, (group, field_name) in enumerate(zip(first, second, strict=True)):
        group = group.strip()
        field_name = field_name.strip()
        if index == 0:
            names.append("track_id")
        elif group and field_name:
            names.append(f"{group}.{field_name}")
        else:
            names.append(field_name or group or f"column_{index}")
    return names, second


def import_fma_metadata(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    rights_cleared: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    handle = _open_fma_tracks_csv(source)
    try:
        reader = csv.reader(handle)
        headers, _ = _fma_headers(reader)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            if not row:
                continue
            row += [""] * (len(headers) - len(row))
            data = dict(zip(headers, row[: len(headers)], strict=True))
            track_id = str(data.get("track_id") or "").strip()
            if not track_id:
                continue
            group = f"fma:{track_id}"
            split = _normalize_split(data.get("set.split"), group)
            license_value = str(data.get("track.license") or data.get("track.license_url") or "").strip()
            payload = {
                "track_id": track_id,
                "title": str(data.get("track.title") or "").strip() or None,
                "artist_name": str(data.get("artist.name") or "").strip() or None,
                "album_title": str(data.get("album.title") or "").strip() or None,
                "subset": str(data.get("set.subset") or "").strip() or None,
                "top_genre": str(data.get("track.genre_top") or "").strip() or None,
                "genres": str(data.get("track.genres_all") or data.get("track.genres") or "").strip() or None,
                "duration_seconds": _safe_float(data.get("track.duration")),
                "license": license_value or None,
                "license_review_required": not rights_cleared,
                "source_audio_ref": f"{int(track_id):06d}.mp3" if track_id.isdigit() else None,
            }
            record = NormalizedRecord(
                record_id=_stable_id("fma", track_id, prefix="track"),
                source_dataset="fma",
                source_record_id=track_id,
                kind="track_metadata",
                split=split,
                group=group,
                label_source="dataset_metadata",
                label_confidence=1.0,
                rights_cleared=bool(rights_cleared),
                payload=payload,
            )
            records.append(record.to_dict())
            counts[record.kind] += 1
    finally:
        handle.close()
    _write_jsonl(destination, records)
    _write_import_manifest(
        destination,
        source_id="fma",
        source_path=source,
        counts=counts,
        rights_cleared=rights_cleared,
        notes=["Per-track FMA licenses are preserved; the importer never infers commercial training rights from the dataset as a whole."],
    )
    return destination

def _iter_listen_files(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() in {".zst", ".xz"} or source.name.endswith(".tar.zst"):
            raise ValueError(
                "Extract the ListenBrainz dump first; then point SAUND.AI at the extracted listens directory."
            )
        return [source]

    # Official core-listens dumps use *.listens. Prefer those files when
    # present so metadata/cache JSONL files are not accidentally imported.
    listens = sorted(
        path for path in source.rglob("*.listens")
        if path.is_file()
    )

    if listens:
        return listens

    files = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jsonl", ".json", ".gz"}
    ]

    return sorted(files)


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".gz":
        handle: TextIO = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        text_mode_json = path.suffix.lower() == ".json"
        if text_mode_json:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                return
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(payload, dict):
                listens = payload.get("payload", {}).get("listens") if isinstance(payload.get("payload"), dict) else None
                if isinstance(listens, list):
                    for item in listens:
                        if isinstance(item, dict):
                            yield item
                else:
                    yield payload
            return
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _listen_timestamp(payload: dict[str, Any]) -> int | None:
    # The historical ListenBrainz dump schema used ``listened_at`` while
    # current core-listens incremental dumps expose the Unix timestamp as
    # ``timestamp``. Accept both so the importer remains compatible with
    # archived and current dumps.
    for key in ("listened_at", "timestamp"):
        value = _safe_int(payload.get(key))
        if value is not None:
            return value
    return None


def _listen_identity(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    metadata = payload.get("track_metadata") if isinstance(payload.get("track_metadata"), dict) else {}
    info = metadata.get("additional_info") if isinstance(metadata.get("additional_info"), dict) else {}
    recording = str(
        payload.get("recording_mbid")
        or payload.get("recording_msid")
        or info.get("recording_mbid")
        or info.get("recording_msid")
        or ""
    ).strip()
    artist = str(metadata.get("artist_name") or "").strip()
    title = str(metadata.get("track_name") or "").strip()
    identity = recording or _stable_id(artist.lower(), title.lower(), prefix="listen-track")
    return identity, artist, title or None


def import_listenbrainz_sequences(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    max_gap_seconds: int = 2 * 60 * 60,
    rights_cleared: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    # RC11.6B: stage compact listen identities on disk instead of retaining a
    # potentially multi-million-listen history for every user in RAM. The
    # primary key also deduplicates overlapping ListenBrainz incrementals.
    seen = 0
    unique_listens = 0
    counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix=".listenbrainz-", dir=destination.parent) as temp_dir:
        database = Path(temp_dir) / "listens.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE listens ("
                "user_hash TEXT NOT NULL, ts INTEGER NOT NULL, track_id TEXT NOT NULL, "
                "artist TEXT NOT NULL, title TEXT, "
                "PRIMARY KEY (user_hash, ts, track_id)"
                ") WITHOUT ROWID"
            )

            batch: list[tuple[str, int, str, str, str | None]] = []

            def flush_batch() -> None:
                nonlocal unique_listens
                if not batch:
                    return
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO listens(user_hash, ts, track_id, artist, title) "
                    "VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                unique_listens += int(connection.total_changes - before)
                connection.commit()
                batch.clear()

            stop = False
            for listen_path in _iter_listen_files(source):
                for payload in _iter_json_lines(listen_path):
                    timestamp = _listen_timestamp(payload)
                    if timestamp is None:
                        continue
                    user_raw = str(
                        payload.get("user_id")
                        or payload.get("user_name")
                        or payload.get("musicbrainz_id")
                        or "unknown"
                    )
                    user_hash = hashlib.sha256(user_raw.encode("utf-8")).hexdigest()[:20]
                    track_id, artist, title = _listen_identity(payload)
                    batch.append((user_hash, timestamp, track_id, artist, title))
                    seen += 1
                    if len(batch) >= 50_000:
                        flush_batch()
                    if limit is not None and seen >= limit:
                        stop = True
                        break
                if stop:
                    break
            flush_batch()

            temporary = destination.with_suffix(destination.suffix + ".tmp")
            edge_count = 0
            previous: tuple[str, int, str, str, str | None] | None = None
            with temporary.open("w", encoding="utf-8") as handle:
                cursor = connection.execute(
                    "SELECT user_hash, ts, track_id, artist, title "
                    "FROM listens ORDER BY user_hash ASC, ts ASC, track_id ASC"
                )
                for user_hash, timestamp, track_id, artist, title in cursor:
                    current = (str(user_hash), int(timestamp), str(track_id), str(artist), title)
                    if previous is not None and previous[0] == current[0]:
                        _, ts_a, track_a, artist_a, title_a = previous
                        _, ts_b, track_b, artist_b, title_b = current
                        gap = ts_b - ts_a
                        if 0 <= gap <= max_gap_seconds and track_a != track_b:
                            group = f"lb-user:{current[0]}"
                            record = NormalizedRecord(
                                record_id=_stable_id(
                                    "listenbrainz", current[0], ts_a, ts_b, track_a, track_b, prefix="edge"
                                ),
                                source_dataset="listenbrainz",
                                source_record_id=f"{current[0]}:{ts_a}:{ts_b}",
                                kind="sequence_edge",
                                split=_stable_split(group),
                                group=group,
                                label_source="behavioral_sequence",
                                label_confidence=0.35,
                                rights_cleared=bool(rights_cleared),
                                payload={
                                    "user_hash": current[0],
                                    "from_track_id": track_a,
                                    "to_track_id": track_b,
                                    "from_artist": artist_a or None,
                                    "to_artist": artist_b or None,
                                    "from_title": title_a,
                                    "to_title": title_b,
                                    "gap_seconds": gap,
                                    "interpretation": "listening adjacency only; not evidence that the pair is technically mixable",
                                },
                            )
                            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                            edge_count += 1
                    previous = current
            temporary.replace(destination)
            counts["sequence_edge"] = edge_count
        finally:
            connection.close()

    _write_import_manifest(
        destination,
        source_id="listenbrainz",
        source_path=source,
        counts=counts,
        rights_cleared=rights_cleared,
        notes=[
            "Raw usernames are not stored; user identifiers are one-way hashed.",
            "Historical listened_at and current timestamp ListenBrainz core-listens schemas are both accepted.",
            "Sequence edges are weak taste/adjacency signals, not DJ-transition labels.",
            f"RC11.6B disk-backed staging processed {seen} valid listens and retained {unique_listens} unique user/timestamp/track rows.",
            "Overlapping incremental dumps are deduplicated before adjacent edges are constructed.",
        ],
    )
    return destination


def _label_rows_from_text(text: str, source_name: str) -> Iterator[tuple[float, float, str, str]]:
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\t+|\s{2,}", stripped, maxsplit=2)
        if len(parts) < 3:
            parts = stripped.split(None, 2)
        if len(parts) < 3:
            continue
        start = _safe_float(parts[0])
        end = _safe_float(parts[1])
        if start is None or end is None or end < start:
            continue
        yield start, end, parts[2].strip(), f"{source_name}:{line_number}"


def _iter_unmix_label_sources(source: Path) -> Iterator[tuple[str, str]]:
    suffix = source.suffix.lower()
    if source.is_file() and suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if Path(name).suffix.lower() not in {".txt", ".lab", ".labels", ".tsv"}:
                    continue
                try:
                    text = archive.read(name).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                yield name, text
        return
    if source.is_file():
        yield source.name, source.read_text(encoding="utf-8", errors="replace")
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".lab", ".labels", ".tsv"}:
            yield str(path.relative_to(source)), path.read_text(encoding="utf-8", errors="replace")


def _unmix_track_ordinal(label: str) -> int | None:
    match = re.match(r"\s*([123])(?:\D|$)", label)
    return int(match.group(1)) if match else None


def import_unmixdb_ground_truth(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    rights_cleared: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    rows_seen = 0
    for file_name, text in _iter_unmix_label_sources(source):
        group = f"unmixdb:{file_name}"
        labels: list[tuple[float, float, str, str]] = []
        for row in _label_rows_from_text(text, file_name):
            labels.append(row)
            start, end, label, source_record_id = row
            lower = label.lower()
            bpm_match = re.search(r"\bbpm\D*([0-9]+(?:\.[0-9]+)?)", lower)
            speed_match = re.search(r"\b(?:speed|factor)\D*([0-9]+(?:\.[0-9]+)?)", lower)
            kind = "mix_segment"
            payload = {
                "file": file_name,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": max(0.0, end - start),
                "label": label,
                "track_ordinal": _unmix_track_ordinal(label),
                "is_cue": "cue" in lower,
                "is_cue_in": "cue" in lower and ("in" in lower or "fadein" in lower or "fade-in" in lower),
                "is_cue_out": "cue" in lower and ("out" in lower or "fadeout" in lower or "fade-out" in lower),
                "bpm": _safe_float(bpm_match.group(1)) if bpm_match else None,
                "speed_factor": _safe_float(speed_match.group(1)) if speed_match else None,
            }
            record = NormalizedRecord(
                record_id=_stable_id("unmixdb", source_record_id, start, end, label, prefix="segment"),
                source_dataset="unmixdb",
                source_record_id=source_record_id,
                kind=kind,
                split=_stable_split(group),
                group=group,
                label_source="ground_truth",
                label_confidence=1.0,
                rights_cleared=bool(rights_cleared),
                payload=payload,
            )
            records.append(record.to_dict())
            counts[record.kind] += 1
            rows_seen += 1
            if limit is not None and rows_seen >= limit:
                break
        # Derive transition windows when two different source-track labels overlap.
        track_rows = [row for row in labels if _unmix_track_ordinal(row[2]) is not None]
        for index, first in enumerate(track_rows):
            a_start, a_end, a_label, a_id = first
            a_track = _unmix_track_ordinal(a_label)
            for second in track_rows[index + 1 :]:
                b_start, b_end, b_label, b_id = second
                b_track = _unmix_track_ordinal(b_label)
                if a_track == b_track:
                    continue
                overlap_start = max(a_start, b_start)
                overlap_end = min(a_end, b_end)
                if overlap_end <= overlap_start:
                    continue
                source_record_id = f"{a_id}|{b_id}"
                record = NormalizedRecord(
                    record_id=_stable_id("unmixdb-transition", source_record_id, overlap_start, overlap_end, prefix="transition"),
                    source_dataset="unmixdb",
                    source_record_id=source_record_id,
                    kind="transition_annotation",
                    split=_stable_split(group),
                    group=group,
                    label_source="ground_truth_overlap",
                    label_confidence=0.95,
                    rights_cleared=bool(rights_cleared),
                    payload={
                        "file": file_name,
                        "from_track_ordinal": a_track,
                        "to_track_ordinal": b_track,
                        "start_seconds": overlap_start,
                        "end_seconds": overlap_end,
                        "duration_seconds": overlap_end - overlap_start,
                        "source_labels": [a_label, b_label],
                    },
                )
                records.append(record.to_dict())
                counts[record.kind] += 1
        if limit is not None and rows_seen >= limit:
            break
    _write_jsonl(destination, records)
    _write_import_manifest(
        destination,
        source_id="unmixdb",
        source_path=source,
        counts=counts,
        rights_cleared=rights_cleared,
        notes=["Transition windows are derived from overlapping source-track ground-truth labels; exact source labels are retained for audit."],
    )
    return destination


def _iter_tabular_records(source: Path) -> Iterator[dict[str, Any]]:
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield dict(row)
        return
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in text.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for key in ("transitions", "data", "records", "items"):
            values = payload.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def _first(payload: dict[str, Any], *keys: str) -> Any:
    lower = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def import_medley2k_annotations(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    rights_cleared: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source.is_dir():
        candidates = [path for path in sorted(source.rglob("*")) if path.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}]
    else:
        candidates = [source]
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen = 0
    for path in candidates:
        try:
            iterator = _iter_tabular_records(path)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for payload in iterator:
            medley_id = str(_first(payload, "medley_id", "medley", "set_id", "piece_id") or path.stem)
            transition_id = str(_first(payload, "transition_id", "id", "index") or seen)
            group = f"medley2k:{medley_id}"
            normalized = {
                "medley_id": medley_id,
                "transition_id": transition_id,
                "from_song": _first(payload, "from_song", "source_song", "song_a", "track_a"),
                "to_song": _first(payload, "to_song", "target_song", "song_b", "track_b"),
                "start": _safe_float(_first(payload, "start", "start_time", "transition_start")),
                "end": _safe_float(_first(payload, "end", "end_time", "transition_end")),
                "duration": _safe_float(_first(payload, "duration", "transition_duration")),
                "raw": payload,
                "interpretation": "human-curated medley transition structure; not automatically equivalent to a DJ transition preference",
            }
            record = NormalizedRecord(
                record_id=_stable_id("medley2k", medley_id, transition_id, prefix="transition"),
                source_dataset="medley2k",
                source_record_id=f"{medley_id}:{transition_id}",
                kind="transition_annotation",
                split=_stable_split(group),
                group=group,
                label_source="human_created_transition",
                label_confidence=0.55,
                rights_cleared=bool(rights_cleared),
                payload=normalized,
            )
            records.append(record.to_dict())
            counts[record.kind] += 1
            seen += 1
            if limit is not None and seen >= limit:
                break
        if limit is not None and seen >= limit:
            break
    _write_jsonl(destination, records)
    _write_import_manifest(
        destination,
        source_id="medley2k",
        source_path=source,
        counts=counts,
        rights_cleared=rights_cleared,
        notes=["Medley2K is used as structural evidence only; source license review remains required for production training."],
    )
    return destination


def import_edm_cue_normalized(
    source_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    rights_cleared: bool = False,
) -> Path:
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="saund-edm-cue-") as tmp:
        normalized = import_edm_cue_annotations(source, Path(tmp) / "annotations.jsonl")
        records: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for index, payload in enumerate(_read_jsonl(normalized)):
            if limit is not None and index >= limit:
                break
            track_id = str(payload.get("track_id"))
            group = f"edm-cue:{track_id}"
            source_split = _normalize_split(payload.get("split"), group)
            grouped_split = _stable_split(group)
            for cue_index, seconds in enumerate(payload.get("cue_points_seconds", [])):
                record = NormalizedRecord(
                    record_id=_stable_id("edm-cue", track_id, cue_index, seconds, prefix="cue"),
                    source_dataset="edm-cue",
                    source_record_id=f"{track_id}:{cue_index}",
                    kind="cue_annotation",
                    # RC11.1 deliberately re-splits labeled EDM-CUE tracks by
                    # track id. The public test partition does not provide
                    # usable cue labels in our normalized export, so preserving
                    # only the upstream train/validation split would leave the
                    # research benchmark without a labeled test set.
                    split=grouped_split,
                    group=group,
                    label_source="expert_annotation",
                    label_confidence=1.0,
                    rights_cleared=bool(rights_cleared),
                    payload={
                        "track_id": track_id,
                        "cue_seconds": float(seconds),
                        "bpm": _safe_float(payload.get("bpm")),
                        "grid_offset_seconds": _safe_float(payload.get("grid_offset_seconds")),
                        "initial_beat_number": _safe_int(payload.get("initial_beat_number")),
                        "artist": payload.get("artist"),
                        "title": payload.get("title"),
                        "source_split": source_split,
                    },
                )
                records.append(record.to_dict())
                counts[record.kind] += 1
    _write_jsonl(destination, records)
    _write_import_manifest(
        destination,
        source_id="edm-cue",
        source_path=source,
        counts=counts,
        rights_cleared=rights_cleared,
        notes=["Only annotation metadata is normalized; linked commercial audio is not acquired."],
    )
    return destination


def import_source_dataset(
    source_id: str,
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    limit: int | None = None,
    rights_cleared: bool = False,
) -> Path:
    source = get_source(source_id)
    destination = Path(output_path or f"datasets/external/{source.source_id}/records.jsonl").expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.source_id == "fma":
        return import_fma_metadata(source_path, destination, limit=limit, rights_cleared=rights_cleared)
    if source.source_id == "listenbrainz":
        return import_listenbrainz_sequences(source_path, destination, limit=limit, rights_cleared=rights_cleared)
    if source.source_id == "unmixdb":
        return import_unmixdb_ground_truth(source_path, destination, limit=limit, rights_cleared=rights_cleared)
    if source.source_id == "medley2k":
        return import_medley2k_annotations(source_path, destination, limit=limit, rights_cleared=rights_cleared)
    if source.source_id == "edm-cue":
        return import_edm_cue_normalized(source_path, destination, limit=limit, rights_cleared=rights_cleared)
    raise ValueError(f"No importer is implemented for {source.source_id}.")


def _cue_structural_features(payload: dict[str, Any], candidate: float) -> dict[str, float]:
    bpm = _safe_float(payload.get("bpm")) or 0.0
    grid_offset = _safe_float(payload.get("grid_offset_seconds")) or 0.0
    initial_beat = _safe_int(payload.get("initial_beat_number")) or 1
    if bpm > 0:
        beat_position = ((candidate - grid_offset) * bpm / 60.0) + float(initial_beat)
        bar_position = beat_position / 4.0
        bar_phase = ((beat_position - 1.0) % 4.0) / 4.0
        phrase_8bar_phase = (bar_position % 8.0) / 8.0
        phrase_16bar_phase = (bar_position % 16.0) / 16.0
    else:
        beat_position = 0.0
        bar_position = 0.0
        bar_phase = 0.0
        phrase_8bar_phase = 0.0
        phrase_16bar_phase = 0.0
    tau = 2.0 * math.pi
    return {
        "candidate_time_seconds": float(candidate),
        "candidate_log_time": float(math.log1p(max(0.0, candidate))),
        "bpm": float(bpm),
        "candidate_beat_position": float(beat_position),
        "candidate_bar_position": float(bar_position),
        "candidate_bar_phase": float(bar_phase),
        "candidate_bar_phase_sin": float(math.sin(tau * bar_phase)),
        "candidate_bar_phase_cos": float(math.cos(tau * bar_phase)),
        "candidate_8bar_phase": float(phrase_8bar_phase),
        "candidate_8bar_phase_sin": float(math.sin(tau * phrase_8bar_phase)),
        "candidate_8bar_phase_cos": float(math.cos(tau * phrase_8bar_phase)),
        "candidate_16bar_phase": float(phrase_16bar_phase),
        "candidate_16bar_phase_sin": float(math.sin(tau * phrase_16bar_phase)),
        "candidate_16bar_phase_cos": float(math.cos(tau * phrase_16bar_phase)),
    }


def _weak_cue_examples(
    record: dict[str, Any],
    *,
    known_track_cues: Iterable[float] = (),
) -> Iterator[dict[str, Any]]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    cue = _safe_float(payload.get("cue_seconds"))
    bpm = _safe_float(payload.get("bpm"))
    if cue is None:
        return
    ranking_group = str(record.get("record_id") or record.get("source_record_id") or record.get("group"))
    base = {
        "task": "cue_point_prior",
        "source_record_id": record.get("record_id"),
        "group": record.get("group"),
        "ranking_group": ranking_group,
        "split": record.get("split"),
        "rights_cleared": bool(record.get("rights_cleared", False)),
        "feature_space": "external-cue-structural-v3-ranking",
    }
    yield {
        **base,
        "id": _stable_id(record.get("record_id"), "positive", prefix="weak"),
        "label_source": str(record.get("label_source") or "expert_annotation"),
        "weight": float(record.get("label_confidence", 1.0)),
        "target": 1.0,
        "candidate_role": "expert_positive",
        "features": _cue_structural_features(payload, cue),
    }

    # RC11 uses deliberately plausible, beat-grid-aligned alternatives instead
    # of mostly easy off-grid corruptions. Negatives that collide with another
    # expert cue from the same track are skipped rather than mislabeled.
    beat_seconds = 60.0 / bpm if bpm and bpm > 0 else 0.5
    tolerance = max(0.08, beat_seconds * 0.45)
    expert_cues = [float(value) for value in known_track_cues]
    seen_candidates: set[float] = {round(cue, 6)}
    for beats in (4, 8, 16, 32, 64):
        offset = beat_seconds * beats
        for sign in (-1, 1):
            candidate = cue + sign * offset
            if candidate < 0.0:
                continue
            rounded = round(candidate, 6)
            if rounded in seen_candidates:
                continue
            if any(abs(candidate - expert) <= tolerance for expert in expert_cues if abs(expert - cue) > 1e-6):
                continue
            seen_candidates.add(rounded)
            distance_bars = beats / 4.0
            yield {
                **base,
                "id": _stable_id(record.get("record_id"), "hard-cue", beats, sign, prefix="weak"),
                "label_source": "synthetic_corruption",
                "weight": 0.62 if beats <= 16 else 0.52,
                "target": 0.0,
                "candidate_role": "hard_grid_aligned_negative",
                "corruption": {"offset_beats": beats * sign, "offset_bars": distance_bars * sign},
                "features": _cue_structural_features(payload, candidate),
            }


def _transition_structural_features(candidate_duration: float) -> dict[str, float]:
    duration = max(float(candidate_duration), 1e-6)
    return {
        "duration_seconds": duration,
        "duration_log_seconds": float(math.log(duration)),
        "duration_under_4s": 1.0 if duration < 4.0 else 0.0,
        "duration_over_32s": 1.0 if duration > 32.0 else 0.0,
    }


def _weak_transition_examples(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    duration = _safe_float(payload.get("duration_seconds") or payload.get("duration"))
    if duration is None or duration <= 0:
        return
    ranking_group = str(record.get("record_id") or record.get("source_record_id") or record.get("group"))
    base = {
        "task": "transition_duration_structure",
        "source_record_id": record.get("record_id"),
        "group": record.get("group"),
        "ranking_group": ranking_group,
        "split": record.get("split"),
        "rights_cleared": bool(record.get("rights_cleared", False)),
        "feature_space": "external-transition-duration-v4-ranking",
    }
    yield {
        **base,
        "id": _stable_id(record.get("record_id"), "duration-positive", prefix="weak"),
        "label_source": str(record.get("label_source") or "ground_truth"),
        "weight": float(record.get("label_confidence", 0.7)),
        "target": 1.0,
        "candidate_role": "observed_positive",
        "features": _transition_structural_features(duration),
    }
    # RC11.1 treats duration selection as ranking, so alternatives are
    # deliberately close enough to remain plausible rather than trivial
    # extremes. The observed transition duration is the preferred candidate
    # within this group; these are weak negatives, never human preference.
    for ratio in (0.75, 0.875, 1.125, 1.25):
        candidate_duration = duration * ratio
        yield {
            **base,
            "id": _stable_id(record.get("record_id"), "duration-hard", ratio, prefix="weak"),
            "label_source": "synthetic_corruption",
            "weight": 0.60 if ratio in {0.875, 1.125} else 0.56,
            "target": 0.0,
            "candidate_role": "plausible_duration_negative",
            "corruption": {"duration_ratio": ratio},
            "features": _transition_structural_features(candidate_duration),
        }


def generate_weak_labels(
    normalized_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(normalized_path).expanduser().resolve()
    destination = Path(output_path or source.with_name("weak-labels.jsonl")).expanduser().resolve()

    # Fast-fail for large datasets that cannot produce any of the structural
    # weak-label tasks. ListenBrainz can contain millions of sequence edges;
    # scanning all of them merely to emit an empty weak-label file is wasteful.
    source_manifest: dict[str, Any] = {}
    manifest_path = source.with_name("manifest.json")
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                source_manifest = payload
        except json.JSONDecodeError:
            source_manifest = {}
    manifest_counts = source_manifest.get("counts") if isinstance(source_manifest.get("counts"), dict) else None
    if manifest_counts is not None:
        relevant = int(manifest_counts.get("cue_annotation", 0) or 0) + int(manifest_counts.get("transition_annotation", 0) or 0)
        if relevant == 0:
            _write_jsonl(destination, ())
            summary = {
                "schema_version": BUILDER_SCHEMA_VERSION,
                "generator": "rc11.3-streaming-grouped-ranking",
                "source": str(source),
                "output": str(destination),
                "created_at_utc": _now(),
                "examples": 0,
                "tasks": {},
                "label_sources": {},
                "candidate_roles": {},
                "raw_audio_bundled": False,
                "streaming": True,
            }
            destination.with_name("weak-labels-manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            return destination

    # Cue negatives need to know the other expert cues on the same track so we
    # do one compact first pass containing only group -> cue-time lists. The
    # normalized records themselves are never materialized as a giant list.
    cue_times_by_group: dict[str, list[float]] = defaultdict(list)
    if manifest_counts is None or int(manifest_counts.get("cue_annotation", 0) or 0) > 0:
        for record in _read_jsonl(source):
            if str(record.get("kind")) != "cue_annotation":
                continue
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            cue = _safe_float(payload.get("cue_seconds"))
            if cue is not None:
                cue_times_by_group[str(record.get("group"))].append(cue)

    tasks: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    candidate_roles: Counter[str] = Counter()

    def iter_rows() -> Iterator[dict[str, Any]]:
        for record in _read_jsonl(source):
            kind = str(record.get("kind"))
            generated: Iterable[dict[str, Any]]
            if kind == "cue_annotation":
                generated = _weak_cue_examples(
                    record,
                    known_track_cues=cue_times_by_group.get(str(record.get("group")), []),
                ) or ()
            elif kind == "transition_annotation":
                generated = _weak_transition_examples(record) or ()
            else:
                continue
            for row in generated:
                tasks[str(row.get("task"))] += 1
                label_sources[str(row.get("label_source"))] += 1
                candidate_roles[str(row.get("candidate_role", "unknown"))] += 1
                yield row

    example_count = _write_jsonl(destination, iter_rows())
    summary = {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "generator": "rc11.3-streaming-grouped-ranking",
        "source": str(source),
        "output": str(destination),
        "created_at_utc": _now(),
        "examples": example_count,
        "tasks": dict(tasks),
        "label_sources": dict(label_sources),
        "candidate_roles": dict(candidate_roles),
        "raw_audio_bundled": False,
        "streaming": True,
    }
    destination.with_name("weak-labels-manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return destination

def _external_record_files(workspace: Path) -> list[Path]:
    root = workspace / "datasets" / "external"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/records.jsonl") if path.is_file())


def _weak_record_files(workspace: Path) -> list[Path]:
    root = workspace / "datasets" / "external"
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*/weak-labels.jsonl") if path.is_file())


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalized_summary(path: Path) -> tuple[dict[str, Any], int, Counter[str]]:
    manifest = _read_json_object(path.with_name("manifest.json"))
    raw_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else None
    counts: Counter[str] = Counter()
    if raw_counts is not None:
        for kind, value in raw_counts.items():
            try:
                amount = int(value)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                counts[str(kind)] += amount
    # Import manifests are authoritative and let status remain O(number of
    # source files) instead of O(number of rows). Fall back to a streaming
    # count for hand-authored/legacy datasets whose manifests are absent.
    if counts or path.stat().st_size == 0:
        return manifest, int(sum(counts.values())), counts
    for row in _read_jsonl(path):
        counts[str(row.get("kind", "unknown"))] += 1
    return manifest, int(sum(counts.values())), counts


def _weak_summary(path: Path) -> tuple[int, Counter[str]]:
    manifest = _read_json_object(path.with_name("weak-labels-manifest.json"))
    raw_tasks = manifest.get("tasks") if isinstance(manifest.get("tasks"), dict) else None
    if raw_tasks is not None and manifest.get("examples") is not None:
        tasks: Counter[str] = Counter()
        for task, value in raw_tasks.items():
            try:
                amount = int(value)
            except (TypeError, ValueError):
                continue
            if amount > 0:
                tasks[str(task)] += amount
        try:
            total = int(manifest.get("examples", sum(tasks.values())))
        except (TypeError, ValueError):
            total = int(sum(tasks.values()))
        return total, tasks
    tasks = Counter()
    total = 0
    for row in _read_jsonl(path):
        total += 1
        tasks[str(row.get("task", "unknown"))] += 1
    return total, tasks


def dataset_builder_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    sources: list[dict[str, Any]] = []
    total_records = 0
    kinds: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    weak_tasks: Counter[str] = Counter()
    weak_total = 0
    for path in _external_record_files(root):
        manifest, record_count, kind_counts = _normalized_summary(path)
        total_records += record_count
        kinds.update(kind_counts)
        source_id = str(manifest.get("source_dataset") or path.parent.name)
        source_counts[source_id] += record_count
        sources.append({
            "source_id": path.parent.name,
            "records": record_count,
            "path": str(path.relative_to(root)),
            "rights_cleared": bool(manifest.get("rights_cleared", False)),
            "counts": dict(kind_counts),
        })
    for path in _weak_record_files(root):
        count, tasks = _weak_summary(path)
        weak_total += count
        weak_tasks.update(tasks)

    corpora_root = root / "datasets" / "bootstrap-corpora"
    corpora: list[dict[str, Any]] = []
    if corpora_root.exists():
        for manifest_path in sorted(corpora_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            payload = _read_json_object(manifest_path)
            if not payload:
                continue
            corpora.append({
                "corpus_id": payload.get("corpus_id", manifest_path.parent.name),
                "created_at_utc": payload.get("created_at_utc"),
                "records": payload.get("counts", {}).get("records", 0),
                "weak_examples": payload.get("counts", {}).get("weak_examples", 0),
                "production_ready": bool(payload.get("production_ready", False)),
                "path": str(manifest_path.parent.relative_to(root)),
            })
    model_runs: list[dict[str, Any]] = []
    latest_benchmark: dict[str, Any] | None = None
    model_root = root / "config" / "bootstrap-models"
    if model_root.exists():
        for report_path in sorted(model_root.glob("*/training-report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            report = _read_json_object(report_path)
            if not report:
                continue
            models = report.get("models") if isinstance(report.get("models"), list) else []
            summary = {
                "corpus_id": report.get("corpus_id", report_path.parent.name),
                "trained_at_utc": report.get("trained_at_utc"),
                "trained_models": sum(1 for item in models if isinstance(item, dict) and item.get("status") == "trained"),
                "model_count": len(models),
                "live_models_changed": bool(report.get("live_models_changed", False)),
                "path": str(report_path.parent.relative_to(root)),
                "models": models,
            }
            model_runs.append(summary)
            if latest_benchmark is None:
                latest_benchmark = {
                    "schema_version": report.get("schema_version", "bootstrap-training-report/1.5"),
                    "corpus_id": summary["corpus_id"],
                    "trained_at_utc": summary["trained_at_utc"],
                    "shadow_only": bool(report.get("shadow_only", True)),
                    "live_models_changed": summary["live_models_changed"],
                    "models": models,
                }

    return {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "catalog": source_catalog(),
        "imports": sources,
        "counts": {
            "normalized_records": total_records,
            "weak_examples": weak_total,
            "kinds": dict(kinds),
            "sources": dict(source_counts),
            "weak_tasks": dict(weak_tasks),
        },
        "corpora": corpora[:12],
        "bootstrap_models": model_runs[:12],
        "latest_benchmark": latest_benchmark,
        "model_readiness": {
            "cue_point_prior": {"examples": weak_tasks.get("cue_point_prior", 0), "source": "EDM-CUE + cue corruptions"},
            "transition_duration_structure": {"examples": weak_tasks.get("transition_duration_structure", 0), "source": "UnmixDB/Medley structural transitions + corruptions"},
            "next_track_prior": {"examples": kinds.get("sequence_edge", 0), "source": "ListenBrainz adjacency"},
            "global_taste_ranker": {"examples": 0, "source": "requires runtime candidate features or human/expert A/B data"},
            "rendered_dj_judge": {"examples": 0, "source": "requires rights-cleared rendered candidates + synthetic technical failures"},
        },
        "policy": {
            "external_sources_default_research": True,
            "raw_audio_bundled_in_normalized_data": False,
            "synthetic_labels_equal_human_preferences": False,
            "production_default_deny": True,
        },
    }


def create_bootstrap_corpus(
    workspace: str | Path,
    *,
    name: str = "bootstrap-v1",
    production: bool = False,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    record_files = _external_record_files(root)
    if not record_files:
        raise ValueError("No normalized external datasets are available. Import at least one source first.")

    corpora_root = root / "datasets" / "bootstrap-corpora"
    corpora_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".building-", dir=corpora_root))
    tasks_dir = staging / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_handles: dict[str, TextIO] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    dataset_hash = hashlib.sha256()
    record_count = 0
    weak_count = 0
    kind_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_sources: Counter[str] = Counter()
    weak_label_sources: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    all_included_rights_cleared = True

    def canonical_line(row: dict[str, Any]) -> str:
        return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    try:
        records_output = staging / "records.jsonl"
        weak_output = staging / "weak-examples.jsonl"
        with records_output.open("w", encoding="utf-8") as records_handle, weak_output.open("w", encoding="utf-8") as weak_handle:
            # Stream every normalized source exactly once. No giant record list
            # and no global sort are needed; source paths and importer output
            # order are deterministic, so source-order hashing is stable.
            for path in record_files:
                manifest = _read_json_object(path.with_name("manifest.json"))
                source_manifests[path.parent.name] = manifest
                for row in _read_jsonl(path):
                    rights = bool(row.get("rights_cleared", False))
                    if production and not rights:
                        continue
                    line = canonical_line(row)
                    records_handle.write(line)
                    dataset_hash.update(b"record\0")
                    dataset_hash.update(line.encode("utf-8"))
                    record_count += 1
                    kind_counts[str(row.get("kind", "unknown"))] += 1
                    source_counts[str(row.get("source_dataset", path.parent.name))] += 1
                    split_counts[str(row.get("split", "unknown"))] += 1
                    label_sources[str(row.get("label_source", "unknown"))] += 1
                    all_included_rights_cleared = all_included_rights_cleared and rights

                weak_path = path.with_name("weak-labels.jsonl")
                if not weak_path.exists():
                    raw_counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else None
                    needs_weak = raw_counts is None or int(raw_counts.get("cue_annotation", 0) or 0) > 0 or int(raw_counts.get("transition_annotation", 0) or 0) > 0
                    if needs_weak:
                        generate_weak_labels(path, weak_path)
                if not weak_path.exists():
                    continue
                for row in _read_jsonl(weak_path):
                    rights = bool(row.get("rights_cleared", False))
                    if production and not rights:
                        continue
                    line = canonical_line(row)
                    weak_handle.write(line)
                    dataset_hash.update(b"weak\0")
                    dataset_hash.update(line.encode("utf-8"))
                    weak_count += 1
                    weak_label_sources[str(row.get("label_source", "unknown"))] += 1
                    task = str(row.get("task", "unknown"))
                    task_counts[task] += 1
                    all_included_rights_cleared = all_included_rights_cleared and rights
                    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "-", task).strip("-") or "unknown"
                    task_handle = task_handles.get(safe_task)
                    if task_handle is None:
                        task_handle = (tasks_dir / f"{safe_task}.jsonl").open("w", encoding="utf-8")
                        task_handles[safe_task] = task_handle
                    task_handle.write(line)

        for handle in task_handles.values():
            handle.close()
        task_handles.clear()

        if production and record_count == 0:
            raise ValueError("No rights-cleared normalized records are available for a production corpus.")

        digest = dataset_hash.hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "bootstrap"
        corpus_id = f"corpus-{safe_name}-{digest[:12]}"
        destination = corpora_root / corpus_id
        if destination.exists():
            existing = destination / "manifest.json"
            if existing.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return destination
            raise FileExistsError(destination)

        production_ready = bool(production and record_count > 0 and all_included_rights_cleared)
        manifest = {
            "schema_version": BOOTSTRAP_CORPUS_SCHEMA_VERSION,
            "corpus_id": corpus_id,
            "name": safe_name,
            "created_at_utc": _now(),
            "immutable": True,
            "production_requested": production,
            "production_ready": production_ready,
            "raw_audio_bundled": False,
            "content_sha256": digest,
            "content_hash_mode": "streamed-source-order-v1",
            "builder_memory_mode": "streaming-bounded",
            "counts": {
                "records": record_count,
                "weak_examples": weak_count,
                "by_kind": dict(kind_counts),
                "by_source": dict(source_counts),
                "by_split": dict(split_counts),
                "by_label_source": dict(label_sources),
                "weak_by_label_source": dict(weak_label_sources),
                "tasks": dict(sorted(task_counts.items())),
            },
            "source_manifests": {
                source_id: {
                    "rights_cleared": bool(payload.get("rights_cleared", False)),
                    "source_path": payload.get("source_path"),
                    "counts": payload.get("counts", {}),
                }
                for source_id, payload in sorted(source_manifests.items())
            },
            "policy": {
                "human_expert_labels_are_distinct_from_synthetic": True,
                "weak_labels_must_keep_label_source": True,
                "research_corpus_cannot_be_promoted_to_production": not production_ready,
                "runtime_model_promotion_requires_separate_feature_schema_and_rights_gate": True,
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        for item in staging.rglob("*"):
            if item.is_file():
                try:
                    item.chmod(0o444)
                except OSError:
                    pass
        staging.replace(destination)
        return destination
    except Exception:
        for handle in task_handles.values():
            try:
                handle.close()
            except OSError:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _latest_corpus(root: Path) -> Path:
    corpora = root / "datasets" / "bootstrap-corpora"
    if not corpora.exists():
        raise ValueError("No bootstrap corpus exists. Run dataset-build first.")
    candidates = [path.parent for path in corpora.glob("*/manifest.json")]
    if not candidates:
        raise ValueError("No bootstrap corpus exists. Run dataset-build first.")
    return max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime_ns)


def _binary_auc(target: "Any", probability: "Any") -> tuple[float, float]:
    import numpy as np

    truth = np.asarray(target, dtype=bool)
    scores = np.asarray(probability, dtype=float)
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives == 0 or negatives == 0:
        return 0.5, float(positives / max(1, len(truth)))
    if float(scores.max() - scores.min()) < 1e-12:
        prevalence = positives / len(truth)
        return 0.5, float(prevalence)

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs(scores[order[j]] - scores[order[i]]) < 1e-12:
            j += 1
        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j
    rank_sum = float(ranks[truth].sum())
    roc_auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    descending = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[descending]
    cumulative_tp = np.cumsum(sorted_truth, dtype=float)
    precision = cumulative_tp / np.arange(1, len(sorted_truth) + 1, dtype=float)
    average_precision = float(precision[sorted_truth].sum() / positives)
    return float(roc_auc), average_precision


def _classification_metrics(target: "Any", probability: "Any") -> dict[str, float | int]:
    import numpy as np

    truth = np.asarray(target, dtype=float) >= 0.5
    probability = np.asarray(probability, dtype=float)
    if len(truth) == 0:
        return {
            "examples": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "mae": 0.0,
            "positive_rate": 0.0,
        }
    predicted = probability >= 0.5
    tp = int(np.sum(predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    roc_auc, pr_auc = _binary_auc(truth, probability)
    return {
        "examples": int(len(truth)),
        "accuracy": round(float(np.mean(predicted == truth)), 6),
        "balanced_accuracy": round(float((tpr + tnr) / 2.0), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "pr_auc": round(float(pr_auc), 6),
        "mae": round(float(np.mean(np.abs(probability - truth.astype(float)))), 6),
        "positive_rate": round(float(np.mean(truth)), 6),
    }


def _fit_shadow_binary(rows: list[dict[str, Any]], *, iterations: int = 700, learning_rate: float = 0.04, l2: float = 0.002) -> dict[str, Any]:
    if len(rows) < 12:
        raise ValueError("Need at least 12 examples for a bootstrap binary shadow model.")
    feature_names = sorted({str(name) for row in rows for name in (row.get("features") or {})})
    if not feature_names:
        raise ValueError("Bootstrap examples contain no numeric features.")
    x = []
    y = []
    weights = []
    splits = []
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        x.append([float(features.get(name, 0.0) or 0.0) for name in feature_names])
        y.append(float(row.get("target", 0.0)))
        weights.append(max(0.05, min(10.0, float(row.get("weight", 1.0) or 1.0))))
        splits.append(str(row.get("split") or "train"))
    import numpy as np

    matrix = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    train = np.asarray([index for index, split in enumerate(splits) if split == "train"], dtype=int)
    valid = np.asarray([index for index, split in enumerate(splits) if split == "validation"], dtype=int)
    test = np.asarray([index for index, split in enumerate(splits) if split == "test"], dtype=int)
    if len(train) < 6:
        train = np.arange(max(1, int(len(rows) * 0.7)))
        valid = np.arange(len(train), max(len(train) + 1, int(len(rows) * 0.85)))
        test = np.arange(max(len(train) + len(valid), int(len(rows) * 0.85)), len(rows))
    mean = matrix[train].mean(axis=0)
    scale = matrix[train].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = (matrix - mean) / scale
    coef = np.zeros(len(feature_names), dtype=float)
    bias = 0.0
    for _ in range(max(80, int(iterations))):
        logits = np.clip(z[train] @ coef + bias, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = (probability - target[train]) * sample_weight[train]
        denom = max(float(sample_weight[train].sum()), 1e-9)
        coef -= learning_rate * ((z[train].T @ error) / denom + l2 * coef)
        bias -= learning_rate * float(error.sum() / denom)

    def metrics(indices: "np.ndarray") -> dict[str, float | int]:
        probability = 1.0 / (1.0 + np.exp(-np.clip(z[indices] @ coef + bias, -40.0, 40.0))) if len(indices) else np.asarray([], dtype=float)
        return _classification_metrics(target[indices], probability)

    train_positive_rate = float(np.average(target[train], weights=sample_weight[train])) if len(train) else 0.0
    baseline_class = 1.0 if train_positive_rate >= 0.5 else 0.0

    def baseline_metrics(indices: "np.ndarray") -> dict[str, float | int]:
        probability = np.full(len(indices), baseline_class, dtype=float)
        return _classification_metrics(target[indices], probability)

    model_metrics = {"train": metrics(train), "validation": metrics(valid), "test": metrics(test)}
    baseline = {"train": baseline_metrics(train), "validation": baseline_metrics(valid), "test": baseline_metrics(test)}
    improvement = {
        split: round(float(model_metrics[split]["accuracy"]) - float(baseline[split]["accuracy"]), 6)
        for split in ("train", "validation", "test")
    }
    balanced_improvement = {
        split: round(float(model_metrics[split]["balanced_accuracy"]) - float(baseline[split]["balanced_accuracy"]), 6)
        for split in ("train", "validation", "test")
    }

    return {
        "evaluation_kind": "binary_classification",
        "primary_metric": "balanced_accuracy",
        "feature_names": feature_names,
        "feature_mean": {name: float(value) for name, value in zip(feature_names, mean, strict=True)},
        "feature_scale": {name: float(value) for name, value in zip(feature_names, scale, strict=True)},
        "weights": [float(value) for value in coef],
        "bias": float(bias),
        "metrics": model_metrics,
        "baseline": {"kind": "weighted_train_majority_class", "class": int(baseline_class), "metrics": baseline},
        "accuracy_improvement_vs_baseline": improvement,
        "primary_improvement_vs_baseline": balanced_improvement,
    }


def _fit_shadow_ranking(rows: list[dict[str, Any]], *, iterations: int = 900, learning_rate: float = 0.05, l2: float = 0.003) -> dict[str, Any]:
    if len(rows) < 12:
        raise ValueError("Need at least 12 examples for a bootstrap ranking shadow model.")
    feature_names = sorted({str(name) for row in rows for name in (row.get("features") or {})})
    if not feature_names:
        raise ValueError("Bootstrap ranking examples contain no numeric features.")

    import numpy as np

    matrix = np.asarray([
        [float((row.get("features") or {}).get(name, 0.0) or 0.0) for name in feature_names]
        for row in rows
    ], dtype=float)
    target = np.asarray([float(row.get("target", 0.0) or 0.0) for row in rows], dtype=float)
    sample_weight = np.asarray([max(0.05, min(10.0, float(row.get("weight", 1.0) or 1.0))) for row in rows], dtype=float)
    splits = [str(row.get("split") or "train") for row in rows]
    ranking_groups = [str(row.get("ranking_group") or row.get("source_record_id") or row.get("group") or index) for index, row in enumerate(rows)]

    train_indices = np.asarray([index for index, split in enumerate(splits) if split == "train"], dtype=int)
    if len(train_indices) < 6:
        raise ValueError("Grouped ranking requires explicit train groups; regenerate RC11.1 weak labels first.")
    mean = matrix[train_indices].mean(axis=0)
    scale = matrix[train_indices].std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    z = (matrix - mean) / scale

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(ranking_groups):
        grouped[group].append(index)

    pair_x: list[Any] = []
    pair_y: list[float] = []
    pair_w: list[float] = []
    for indices in grouped.values():
        if not indices or splits[indices[0]] != "train":
            continue
        positives = [index for index in indices if target[index] >= 0.5]
        negatives = [index for index in indices if target[index] < 0.5]
        for positive in positives:
            for negative in negatives:
                diff = z[positive] - z[negative]
                weight = float(math.sqrt(sample_weight[positive] * sample_weight[negative]))
                pair_x.extend((diff, -diff))
                pair_y.extend((1.0, 0.0))
                pair_w.extend((weight, weight))
    if len(pair_x) < 12:
        raise ValueError("Grouped ranking needs groups with both positive and synthetic candidates; regenerate RC11.1 weak labels first.")
    pair_matrix = np.asarray(pair_x, dtype=float)
    pair_target = np.asarray(pair_y, dtype=float)
    pair_weight = np.asarray(pair_w, dtype=float)
    coef = np.zeros(len(feature_names), dtype=float)
    for _ in range(max(100, int(iterations))):
        logits = np.clip(pair_matrix @ coef, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        error = (probability - pair_target) * pair_weight
        denom = max(float(pair_weight.sum()), 1e-9)
        coef -= learning_rate * ((pair_matrix.T @ error) / denom + l2 * coef)

    scores = z @ coef

    def ranking_metrics(split: str) -> dict[str, float | int]:
        split_groups = [indices for indices in grouped.values() if indices and splits[indices[0]] == split]
        top1_total = 0.0
        top3_total = 0.0
        reciprocal_total = 0.0
        pair_correct = 0.0
        pair_count = 0
        candidate_count = 0
        eligible_groups = 0
        random_top1 = 0.0
        random_top3 = 0.0
        random_mrr = 0.0
        for indices in split_groups:
            positives = [index for index in indices if target[index] >= 0.5]
            negatives = [index for index in indices if target[index] < 0.5]
            if not positives or not negatives:
                continue
            eligible_groups += 1
            candidate_count += len(indices)
            positive = max(positives, key=lambda index: sample_weight[index])
            positive_score = float(scores[positive])
            group_scores = np.asarray([scores[index] for index in indices], dtype=float)
            higher = int(np.sum(group_scores > positive_score + 1e-12))
            tied = int(np.sum(np.abs(group_scores - positive_score) <= 1e-12))
            average_rank = 1.0 + higher + max(0, tied - 1) / 2.0
            if higher == 0:
                top1_total += 1.0 / max(1, tied)
            if average_rank <= 3.0:
                top3_total += 1.0
            reciprocal_total += 1.0 / average_rank
            for negative in negatives:
                delta = positive_score - float(scores[negative])
                pair_correct += 1.0 if delta > 1e-12 else (0.5 if abs(delta) <= 1e-12 else 0.0)
                pair_count += 1
            n = len(indices)
            random_top1 += 1.0 / n
            random_top3 += min(3, n) / n
            random_mrr += sum(1.0 / rank for rank in range(1, n + 1)) / n
        if eligible_groups == 0:
            return {
                "examples": 0,
                "groups": 0,
                "pairwise_comparisons": 0,
                "accuracy": 0.0,
                "pairwise_accuracy": 0.0,
                "top1_accuracy": 0.0,
                "top3_recall": 0.0,
                "mrr": 0.0,
                "random_top1": 0.0,
                "random_top3": 0.0,
                "random_mrr": 0.0,
            }
        return {
            "examples": int(candidate_count),
            "groups": int(eligible_groups),
            "pairwise_comparisons": int(pair_count),
            "accuracy": round(float(pair_correct / max(1, pair_count)), 6),
            "pairwise_accuracy": round(float(pair_correct / max(1, pair_count)), 6),
            "top1_accuracy": round(float(top1_total / eligible_groups), 6),
            "top3_recall": round(float(top3_total / eligible_groups), 6),
            "mrr": round(float(reciprocal_total / eligible_groups), 6),
            "random_top1": round(float(random_top1 / eligible_groups), 6),
            "random_top3": round(float(random_top3 / eligible_groups), 6),
            "random_mrr": round(float(random_mrr / eligible_groups), 6),
        }

    model_metrics = {split: ranking_metrics(split) for split in ("train", "validation", "test")}
    baseline_metrics: dict[str, dict[str, float | int]] = {}
    for split, metric in model_metrics.items():
        baseline_metrics[split] = {
            "examples": metric["examples"],
            "groups": metric["groups"],
            "pairwise_comparisons": metric["pairwise_comparisons"],
            "accuracy": 0.5,
            "pairwise_accuracy": 0.5,
            "top1_accuracy": metric["random_top1"],
            "top3_recall": metric["random_top3"],
            "mrr": metric["random_mrr"],
        }
    improvement = {
        split: round(float(model_metrics[split]["pairwise_accuracy"]) - 0.5, 6)
        for split in ("train", "validation", "test")
    }
    primary_improvement = {
        split: round(float(model_metrics[split]["top1_accuracy"]) - float(baseline_metrics[split]["top1_accuracy"]), 6)
        for split in ("train", "validation", "test")
    }

    return {
        "evaluation_kind": "grouped_pairwise_ranking",
        "primary_metric": "top1_accuracy",
        "feature_names": feature_names,
        "feature_mean": {name: float(value) for name, value in zip(feature_names, mean, strict=True)},
        "feature_scale": {name: float(value) for name, value in zip(feature_names, scale, strict=True)},
        "weights": [float(value) for value in coef],
        "bias": 0.0,
        "training_pairs": int(len(pair_target)),
        "metrics": model_metrics,
        "baseline": {"kind": "random_within_ranking_group", "metrics": baseline_metrics},
        "accuracy_improvement_vs_baseline": improvement,
        "primary_improvement_vs_baseline": primary_improvement,
    }

def _harmonic_number(size: int) -> float:
    if size <= 0:
        return 0.0
    if size <= 10000:
        return float(sum(1.0 / index for index in range(1, size + 1)))
    # Euler-Maclaurin keeps the random-candidate MRR baseline cheap even when
    # ListenBrainz contributes hundreds of thousands of destination tracks.
    n = float(size)
    return math.log(n) + 0.5772156649015329 + (1.0 / (2.0 * n)) - (1.0 / (12.0 * n * n))


def _sequence_retrieval_slice_metrics(
    connection: sqlite3.Connection,
    split: str,
    *,
    candidate_count: int,
    predicate: str = "1 = 1",
) -> dict[str, Any]:
    # `predicate` is selected only from constants in this module. It is never
    # derived from user input; split remains parameterized below.
    row = connection.execute(
        f"""
        SELECT
            COALESCE(SUM(h.count), 0) AS examples,
            COALESCE(SUM(CASE WHEN s.a IS NOT NULL THEN h.count ELSE 0 END), 0) AS known_source_examples,
            COALESCE(SUM(CASE WHEN s.a IS NULL THEN h.count ELSE 0 END), 0) AS cold_source_examples,
            COALESCE(SUM(CASE WHEN r.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS seen_edge_examples,
            COALESCE(SUM(CASE WHEN s.a IS NOT NULL AND r.rank IS NULL THEN h.count ELSE 0 END), 0) AS unseen_edge_examples,
            COALESCE(SUM(CASE WHEN g.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS target_seen_examples,
            COALESCE(SUM(CASE WHEN r.rank <= 1 THEN h.count ELSE 0 END), 0) AS model_hit_1,
            COALESCE(SUM(CASE WHEN r.rank <= 5 THEN h.count ELSE 0 END), 0) AS model_hit_5,
            COALESCE(SUM(CASE WHEN r.rank <= 10 THEN h.count ELSE 0 END), 0) AS model_hit_10,
            COALESCE(SUM(CASE WHEN r.rank IS NOT NULL THEN CAST(h.count AS REAL) / r.rank ELSE 0.0 END), 0.0) AS model_rr,
            COALESCE(SUM(CASE WHEN g.rank <= 1 THEN h.count ELSE 0 END), 0) AS global_hit_1,
            COALESCE(SUM(CASE WHEN g.rank <= 5 THEN h.count ELSE 0 END), 0) AS global_hit_5,
            COALESCE(SUM(CASE WHEN g.rank <= 10 THEN h.count ELSE 0 END), 0) AS global_hit_10,
            COALESCE(SUM(CASE WHEN g.rank IS NOT NULL THEN CAST(h.count AS REAL) / g.rank ELSE 0.0 END), 0.0) AS global_rr
        FROM heldout_edges h
        LEFT JOIN source_stats s ON s.a = h.a
        LEFT JOIN ranked_edges r ON r.a = h.a AND r.b = h.b
        LEFT JOIN target_ranks g ON g.b = h.b
        WHERE h.split = ? AND ({predicate})
        """,
        (split,),
    ).fetchone()
    assert row is not None
    (
        examples,
        known_source_examples,
        cold_source_examples,
        seen_edge_examples,
        unseen_edge_examples,
        target_seen_examples,
        model_hit_1,
        model_hit_5,
        model_hit_10,
        model_rr,
        global_hit_1,
        global_hit_5,
        global_hit_10,
        global_rr,
    ) = row
    examples = int(examples)
    denominator = max(1, examples)
    target_coverage = float(target_seen_examples) / denominator
    random_hit_1 = target_coverage * (min(1, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_5 = target_coverage * (min(5, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_10 = target_coverage * (min(10, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_mrr = target_coverage * (_harmonic_number(candidate_count) / candidate_count) if candidate_count else 0.0

    def rate(value: int | float) -> float:
        return round(float(value) / denominator, 8)

    return {
        "examples": examples,
        "known_source_examples": int(known_source_examples),
        "cold_source_examples": int(cold_source_examples),
        "seen_edge_examples": int(seen_edge_examples),
        "unseen_edge_examples": int(unseen_edge_examples),
        "target_seen_examples": int(target_seen_examples),
        "cold_target_examples": int(max(0, examples - int(target_seen_examples))),
        "source_coverage": rate(known_source_examples),
        "edge_coverage": rate(seen_edge_examples),
        "target_coverage": round(target_coverage, 8),
        "model": {
            "hit_at_1": rate(model_hit_1),
            "hit_at_5": rate(model_hit_5),
            "hit_at_10": rate(model_hit_10),
            "mrr": rate(model_rr),
        },
        "global_popularity_baseline": {
            "hit_at_1": rate(global_hit_1),
            "hit_at_5": rate(global_hit_5),
            "hit_at_10": rate(global_hit_10),
            "mrr": rate(global_rr),
        },
        "random_candidate_baseline": {
            "hit_at_1": round(random_hit_1, 8),
            "hit_at_5": round(random_hit_5, 8),
            "hit_at_10": round(random_hit_10, 8),
            "mrr": round(random_mrr, 8),
        },
    }


def _sequence_retrieval_metrics(connection: sqlite3.Connection, split: str, *, candidate_count: int) -> dict[str, Any]:
    return _sequence_retrieval_slice_metrics(connection, split, candidate_count=candidate_count)


def _sequence_generalization_metrics(connection: sqlite3.Connection, split: str, *, candidate_count: int) -> dict[str, Any]:
    predicates = {
        # Exact A->B relationship existed in training. This measures memory /
        # repeatability of the frequency prior rather than novel-pair transfer.
        "seen_edge": "r.rank IS NOT NULL",
        # Source A is known but A->B was never observed in training.
        "unseen_edge_known_source": "s.a IS NOT NULL AND r.rank IS NULL",
        # A was never observed as a train source at all.
        "cold_source": "s.a IS NULL",
        # B is outside the train destination universe and cannot be retrieved.
        "cold_target": "g.rank IS NULL",
        # Both endpoints are in the train universe, regardless of pair memory.
        "known_source_seen_target": "s.a IS NOT NULL AND g.rank IS NOT NULL",
        # Strongest pair-generalization slice: A and B are individually known,
        # but the exact A->B edge is novel in held-out data.
        "novel_pair_candidate": "s.a IS NOT NULL AND g.rank IS NOT NULL AND r.rank IS NULL",
    }
    return {
        name: _sequence_retrieval_slice_metrics(
            connection,
            split,
            candidate_count=candidate_count,
            predicate=predicate,
        )
        for name, predicate in predicates.items()
    }


def _sequence_artist_key(value: Any) -> str:
    text = " ".join(str(value or "").casefold().split())
    return _stable_id(text, prefix="artist") if text else ""


def _sequence_backoff_slice_metrics(
    connection: sqlite3.Connection,
    split: str,
    *,
    candidate_count: int,
    predicate: str = "1 = 1",
) -> dict[str, Any]:
    # The generalizer is deliberately simple and auditable: source-artist ->
    # destination-track counts are learned from train users only. Destination
    # tracks never observed after that artist back off to global popularity.
    # The materialized rank table therefore defines a full deterministic rank
    # over the train destination universe without using held-out labels.
    row = connection.execute(
        f"""
        SELECT
            COALESCE(SUM(h.count), 0) AS examples,
            COALESCE(SUM(CASE WHEN s.a IS NOT NULL THEN h.count ELSE 0 END), 0) AS known_source_examples,
            COALESCE(SUM(CASE WHEN s.a IS NULL THEN h.count ELSE 0 END), 0) AS cold_source_examples,
            COALESCE(SUM(CASE WHEN r.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS seen_edge_examples,
            COALESCE(SUM(CASE WHEN s.a IS NOT NULL AND r.rank IS NULL THEN h.count ELSE 0 END), 0) AS unseen_edge_examples,
            COALESCE(SUM(CASE WHEN g.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS target_seen_examples,
            COALESCE(SUM(CASE WHEN ast.sa IS NOT NULL THEN h.count ELSE 0 END), 0) AS artist_source_examples,
            COALESCE(SUM(CASE WHEN gr.rank <= 1 THEN h.count ELSE 0 END), 0) AS model_hit_1,
            COALESCE(SUM(CASE WHEN gr.rank <= 5 THEN h.count ELSE 0 END), 0) AS model_hit_5,
            COALESCE(SUM(CASE WHEN gr.rank <= 10 THEN h.count ELSE 0 END), 0) AS model_hit_10,
            COALESCE(SUM(CASE WHEN gr.rank IS NOT NULL THEN CAST(h.count AS REAL) / gr.rank ELSE 0.0 END), 0.0) AS model_rr,
            COALESCE(SUM(CASE WHEN g.rank <= 1 THEN h.count ELSE 0 END), 0) AS global_hit_1,
            COALESCE(SUM(CASE WHEN g.rank <= 5 THEN h.count ELSE 0 END), 0) AS global_hit_5,
            COALESCE(SUM(CASE WHEN g.rank <= 10 THEN h.count ELSE 0 END), 0) AS global_hit_10,
            COALESCE(SUM(CASE WHEN g.rank IS NOT NULL THEN CAST(h.count AS REAL) / g.rank ELSE 0.0 END), 0.0) AS global_rr
        FROM heldout_edges h
        LEFT JOIN source_stats s ON s.a = h.a
        LEFT JOIN ranked_edges r ON r.a = h.a AND r.b = h.b
        LEFT JOIN target_ranks g ON g.b = h.b
        LEFT JOIN artist_target_stats ast ON ast.sa = h.sa
        LEFT JOIN heldout_artist_backoff_ranks gr
          ON gr.split = h.split AND gr.a = h.a AND gr.b = h.b AND gr.sa = h.sa
        WHERE h.split = ? AND ({predicate})
        """,
        (split,),
    ).fetchone()
    assert row is not None
    (
        examples,
        known_source_examples,
        cold_source_examples,
        seen_edge_examples,
        unseen_edge_examples,
        target_seen_examples,
        artist_source_examples,
        model_hit_1,
        model_hit_5,
        model_hit_10,
        model_rr,
        global_hit_1,
        global_hit_5,
        global_hit_10,
        global_rr,
    ) = row
    examples = int(examples)
    denominator = max(1, examples)
    target_coverage = float(target_seen_examples) / denominator
    random_hit_1 = target_coverage * (min(1, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_5 = target_coverage * (min(5, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_10 = target_coverage * (min(10, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_mrr = target_coverage * (_harmonic_number(candidate_count) / candidate_count) if candidate_count else 0.0

    def rate(value: int | float) -> float:
        return round(float(value) / denominator, 8)

    return {
        "examples": examples,
        "known_source_examples": int(known_source_examples),
        "cold_source_examples": int(cold_source_examples),
        "seen_edge_examples": int(seen_edge_examples),
        "unseen_edge_examples": int(unseen_edge_examples),
        "target_seen_examples": int(target_seen_examples),
        "cold_target_examples": int(max(0, examples - int(target_seen_examples))),
        "source_coverage": rate(known_source_examples),
        "edge_coverage": rate(seen_edge_examples),
        "target_coverage": round(target_coverage, 8),
        "artist_source_coverage": rate(artist_source_examples),
        "model": {
            "hit_at_1": rate(model_hit_1),
            "hit_at_5": rate(model_hit_5),
            "hit_at_10": rate(model_hit_10),
            "mrr": rate(model_rr),
        },
        "global_popularity_baseline": {
            "hit_at_1": rate(global_hit_1),
            "hit_at_5": rate(global_hit_5),
            "hit_at_10": rate(global_hit_10),
            "mrr": rate(global_rr),
        },
        "random_candidate_baseline": {
            "hit_at_1": round(random_hit_1, 8),
            "hit_at_5": round(random_hit_5, 8),
            "hit_at_10": round(random_hit_10, 8),
            "mrr": round(random_mrr, 8),
        },
    }


def _sequence_backoff_metrics(connection: sqlite3.Connection, split: str, *, candidate_count: int) -> dict[str, Any]:
    return _sequence_backoff_slice_metrics(connection, split, candidate_count=candidate_count)


def _sequence_backoff_generalization_metrics(
    connection: sqlite3.Connection,
    split: str,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    predicates = {
        "seen_edge": "r.rank IS NOT NULL",
        "unseen_edge_known_source": "s.a IS NOT NULL AND r.rank IS NULL",
        "cold_source": "s.a IS NULL",
        "cold_target": "g.rank IS NULL",
        "known_source_seen_target": "s.a IS NOT NULL AND g.rank IS NOT NULL",
        "novel_pair_candidate": "s.a IS NOT NULL AND g.rank IS NOT NULL AND r.rank IS NULL",
    }
    return {
        name: _sequence_backoff_slice_metrics(
            connection,
            split,
            candidate_count=candidate_count,
            predicate=predicate,
        )
        for name, predicate in predicates.items()
    }



def _sequence_cold_target_artist_slice_metrics(
    connection: sqlite3.Connection,
    split: str,
    *,
    candidate_count: int,
    predicate: str = "1 = 1",
) -> dict[str, Any]:
    """Evaluate target-artist retrieval for track IDs unseen during training.

    RC11.6C does not pretend that an unseen recording can receive a memorized
    train-track rank. Instead it asks whether the target artist is known and,
    when it is, whether train-only source-artist -> target-artist behavior can
    rank that artist above a global target-artist popularity baseline.
    """
    row = connection.execute(
        f"""
        SELECT
            COALESCE(SUM(h.count), 0) AS examples,
            COALESCE(SUM(CASE WHEN g.rank IS NULL THEN h.count ELSE 0 END), 0) AS cold_track_examples,
            COALESCE(SUM(CASE WHEN ta.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS known_target_artist_examples,
            COALESCE(SUM(CASE WHEN g.rank IS NULL AND ta.rank IS NOT NULL THEN h.count ELSE 0 END), 0) AS cold_track_known_artist_examples,
            COALESCE(SUM(CASE WHEN g.rank IS NULL AND ta.rank IS NULL THEN h.count ELSE 0 END), 0) AS cold_track_unknown_artist_examples,
            COALESCE(SUM(CASE WHEN sas.sa IS NOT NULL THEN h.count ELSE 0 END), 0) AS known_source_artist_examples,
            COALESCE(SUM(CASE WHEN hr.rank <= 1 THEN h.count ELSE 0 END), 0) AS model_hit_1,
            COALESCE(SUM(CASE WHEN hr.rank <= 5 THEN h.count ELSE 0 END), 0) AS model_hit_5,
            COALESCE(SUM(CASE WHEN hr.rank <= 10 THEN h.count ELSE 0 END), 0) AS model_hit_10,
            COALESCE(SUM(CASE WHEN hr.rank IS NOT NULL THEN CAST(h.count AS REAL) / hr.rank ELSE 0.0 END), 0.0) AS model_rr,
            COALESCE(SUM(CASE WHEN ta.rank <= 1 THEN h.count ELSE 0 END), 0) AS global_hit_1,
            COALESCE(SUM(CASE WHEN ta.rank <= 5 THEN h.count ELSE 0 END), 0) AS global_hit_5,
            COALESCE(SUM(CASE WHEN ta.rank <= 10 THEN h.count ELSE 0 END), 0) AS global_hit_10,
            COALESCE(SUM(CASE WHEN ta.rank IS NOT NULL THEN CAST(h.count AS REAL) / ta.rank ELSE 0.0 END), 0.0) AS global_rr
        FROM heldout_edges h
        LEFT JOIN target_ranks g ON g.b = h.b
        LEFT JOIN target_artist_ranks ta ON ta.ta = h.ta
        LEFT JOIN source_artist_target_artist_stats sas ON sas.sa = h.sa
        LEFT JOIN heldout_artist_target_backoff_ranks hr
          ON hr.split = h.split AND hr.a = h.a AND hr.b = h.b AND hr.sa = h.sa AND hr.ta = h.ta
        WHERE h.split = ? AND ({predicate})
        """,
        (split,),
    ).fetchone()
    assert row is not None
    (
        examples,
        cold_track_examples,
        known_target_artist_examples,
        cold_track_known_artist_examples,
        cold_track_unknown_artist_examples,
        known_source_artist_examples,
        model_hit_1,
        model_hit_5,
        model_hit_10,
        model_rr,
        global_hit_1,
        global_hit_5,
        global_hit_10,
        global_rr,
    ) = row
    examples = int(examples)
    denominator = max(1, examples)
    target_artist_coverage = float(known_target_artist_examples) / denominator
    random_hit_1 = target_artist_coverage * (min(1, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_5 = target_artist_coverage * (min(5, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_hit_10 = target_artist_coverage * (min(10, candidate_count) / max(1, candidate_count)) if candidate_count else 0.0
    random_mrr = target_artist_coverage * (_harmonic_number(candidate_count) / candidate_count) if candidate_count else 0.0

    def rate(value: int | float) -> float:
        return round(float(value) / denominator, 8)

    return {
        "examples": examples,
        "cold_track_examples": int(cold_track_examples),
        "known_target_artist_examples": int(known_target_artist_examples),
        "cold_track_known_artist_examples": int(cold_track_known_artist_examples),
        "cold_track_unknown_artist_examples": int(cold_track_unknown_artist_examples),
        "known_source_artist_examples": int(known_source_artist_examples),
        "target_artist_coverage": round(target_artist_coverage, 8),
        "source_artist_coverage": rate(known_source_artist_examples),
        "model": {
            "hit_at_1": rate(model_hit_1),
            "hit_at_5": rate(model_hit_5),
            "hit_at_10": rate(model_hit_10),
            "mrr": rate(model_rr),
        },
        "global_target_artist_baseline": {
            "hit_at_1": rate(global_hit_1),
            "hit_at_5": rate(global_hit_5),
            "hit_at_10": rate(global_hit_10),
            "mrr": rate(global_rr),
        },
        "random_target_artist_baseline": {
            "hit_at_1": round(random_hit_1, 8),
            "hit_at_5": round(random_hit_5, 8),
            "hit_at_10": round(random_hit_10, 8),
            "mrr": round(random_mrr, 8),
        },
    }


def _sequence_cold_target_artist_metrics(
    connection: sqlite3.Connection,
    split: str,
    *,
    candidate_count: int,
) -> dict[str, Any]:
    predicates = {
        "all_heldout": "1 = 1",
        "cold_target": "g.rank IS NULL",
        "cold_target_known_artist": "g.rank IS NULL AND ta.rank IS NOT NULL",
        "cold_target_unknown_artist": "g.rank IS NULL AND ta.rank IS NULL",
        "seen_target_track": "g.rank IS NOT NULL",
    }
    return {
        name: _sequence_cold_target_artist_slice_metrics(
            connection,
            split,
            candidate_count=candidate_count,
            predicate=predicate,
        )
        for name, predicate in predicates.items()
    }

def _export_sequence_runtime_index(connection: sqlite3.Connection, destination: Path) -> dict[str, Any]:
    """Persist train-only sequence counts needed by the live hybrid ranker.

    The benchmark database also contains held-out evaluation tables, which must
    never leak into runtime scoring.  This export copies only train-derived
    adjacency, source-artist->track and source-artist->target-artist counts.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    connection.execute("ATTACH DATABASE ? AS runtime_export", (str(temporary),))
    try:
        connection.execute(
            "CREATE TABLE runtime_export.metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE runtime_export.adjacency ("
            "source_track TEXT NOT NULL, target_track TEXT NOT NULL, count INTEGER NOT NULL, "
            "outgoing_count INTEGER NOT NULL, PRIMARY KEY(source_track,target_track)) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO runtime_export.adjacency(source_track,target_track,count,outgoing_count) "
            "SELECT e.a,e.b,e.count,s.outgoing_count FROM edge_counts e JOIN source_stats s ON s.a=e.a"
        )
        connection.execute(
            "CREATE TABLE runtime_export.artist_track ("
            "source_artist_key TEXT NOT NULL, target_track TEXT NOT NULL, count INTEGER NOT NULL, "
            "outgoing_count INTEGER NOT NULL, PRIMARY KEY(source_artist_key,target_track)) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO runtime_export.artist_track(source_artist_key,target_track,count,outgoing_count) "
            "SELECT e.sa,e.b,e.count,s.outgoing_count FROM artist_target_counts e "
            "JOIN artist_target_stats s ON s.sa=e.sa"
        )
        connection.execute(
            "CREATE TABLE runtime_export.artist_pair ("
            "source_artist_key TEXT NOT NULL, target_artist_key TEXT NOT NULL, count INTEGER NOT NULL, "
            "outgoing_count INTEGER NOT NULL, PRIMARY KEY(source_artist_key,target_artist_key)) WITHOUT ROWID"
        )
        connection.execute(
            "INSERT INTO runtime_export.artist_pair(source_artist_key,target_artist_key,count,outgoing_count) "
            "SELECT e.sa,e.ta,e.count,s.outgoing_count FROM artist_pair_counts e "
            "JOIN source_artist_target_artist_stats s ON s.sa=e.sa"
        )
        counts = {
            "adjacency_rows": int(connection.execute("SELECT COUNT(*) FROM edge_counts").fetchone()[0]),
            "artist_track_rows": int(connection.execute("SELECT COUNT(*) FROM artist_target_counts").fetchone()[0]),
            "artist_pair_rows": int(connection.execute("SELECT COUNT(*) FROM artist_pair_counts").fetchone()[0]),
        }
        metadata = {
            "schema_version": "bootstrap-sequence-runtime/1.0",
            "training_split_only": "true",
            **{key: str(value) for key, value in counts.items()},
        }
        connection.executemany(
            "INSERT INTO runtime_export.metadata(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE runtime_export")
    temporary.replace(destination)
    return {
        "schema_version": "bootstrap-sequence-runtime/1.0",
        "path": destination.name,
        "training_split_only": True,
        **counts,
    }


def _aggregate_sequence_edges_streaming(records_path: Path, scratch_root: Path, *, top_limit: int = 5000) -> dict[str, Any]:
    split_examples: Counter[str] = Counter()
    total_edges = 0
    valid_train_edges = 0
    batch: Counter[tuple[str, str]] = Counter()
    artist_batch: Counter[tuple[str, str]] = Counter()
    artist_pair_batch: Counter[tuple[str, str]] = Counter()
    target_artist_batch: Counter[str] = Counter()
    heldout_batch: Counter[tuple[str, str, str, str, str]] = Counter()
    runtime_index: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory(prefix=".next-track-", dir=scratch_root) as temp_dir:
        database = Path(temp_dir) / "edge-counts.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute(
                "CREATE TABLE edge_counts (a TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (a, b)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE artist_target_counts (sa TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (sa, b)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE artist_pair_counts (sa TEXT NOT NULL, ta TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (sa, ta)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE target_artist_counts (ta TEXT PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE heldout_edges (split TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, sa TEXT NOT NULL, ta TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (split, a, b, sa, ta)) WITHOUT ROWID"
            )

            def flush_batch() -> None:
                if not batch:
                    return
                connection.executemany(
                    "INSERT INTO edge_counts(a, b, count) VALUES (?, ?, ?) "
                    "ON CONFLICT(a, b) DO UPDATE SET count = edge_counts.count + excluded.count",
                    ((a, b, count) for (a, b), count in batch.items()),
                )
                connection.commit()
                batch.clear()

            def flush_artist_batch() -> None:
                if not artist_batch:
                    return
                connection.executemany(
                    "INSERT INTO artist_target_counts(sa, b, count) VALUES (?, ?, ?) "
                    "ON CONFLICT(sa,b) DO UPDATE SET count = artist_target_counts.count + excluded.count",
                    ((sa, b, count) for (sa, b), count in artist_batch.items()),
                )
                connection.commit()
                artist_batch.clear()

            def flush_artist_pair_batch() -> None:
                if not artist_pair_batch:
                    return
                connection.executemany(
                    "INSERT INTO artist_pair_counts(sa, ta, count) VALUES (?, ?, ?) "
                    "ON CONFLICT(sa,ta) DO UPDATE SET count = artist_pair_counts.count + excluded.count",
                    ((sa, ta, count) for (sa, ta), count in artist_pair_batch.items()),
                )
                connection.commit()
                artist_pair_batch.clear()

            def flush_target_artist_batch() -> None:
                if not target_artist_batch:
                    return
                connection.executemany(
                    "INSERT INTO target_artist_counts(ta, count) VALUES (?, ?) "
                    "ON CONFLICT(ta) DO UPDATE SET count = target_artist_counts.count + excluded.count",
                    ((ta, count) for ta, count in target_artist_batch.items()),
                )
                connection.commit()
                target_artist_batch.clear()

            def flush_heldout_batch() -> None:
                if not heldout_batch:
                    return
                connection.executemany(
                    "INSERT INTO heldout_edges(split, a, b, sa, ta, count) VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(split, a, b, sa, ta) DO UPDATE SET count = heldout_edges.count + excluded.count",
                    (
                        (split, a, b, sa, ta, count)
                        for (split, a, b, sa, ta), count in heldout_batch.items()
                    ),
                )
                connection.commit()
                heldout_batch.clear()

            for row in _read_jsonl(records_path):
                if row.get("kind") != "sequence_edge":
                    continue
                total_edges += 1
                split = str(row.get("split") or "train")
                split_examples[split] += 1
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                a = str(payload.get("from_track_id") or "")
                b = str(payload.get("to_track_id") or "")
                source_artist = _sequence_artist_key(payload.get("from_artist"))
                target_artist = _sequence_artist_key(payload.get("to_artist"))
                if not a or not b or a == b:
                    continue
                # Held-out users remain held out. They are written to the same
                # temporary SQLite database only for retrieval evaluation; they
                # never contribute counts to the fitted adjacency prior.
                if split != "train":
                    heldout_batch[(split, a, b, source_artist, target_artist)] += 1
                    if len(heldout_batch) >= 25000:
                        flush_heldout_batch()
                    continue
                batch[(a, b)] += 1
                if source_artist:
                    artist_batch[(source_artist, b)] += 1
                if target_artist:
                    target_artist_batch[target_artist] += 1
                if source_artist and target_artist:
                    artist_pair_batch[(source_artist, target_artist)] += 1
                valid_train_edges += 1
                if len(batch) >= 25000:
                    flush_batch()
                if len(artist_batch) >= 25000:
                    flush_artist_batch()
                if len(artist_pair_batch) >= 25000:
                    flush_artist_pair_batch()
                if len(target_artist_batch) >= 25000:
                    flush_target_artist_batch()
            flush_batch()
            flush_artist_batch()
            flush_artist_pair_batch()
            flush_target_artist_batch()
            flush_heldout_batch()

            unique_edges = int(connection.execute("SELECT COUNT(*) FROM edge_counts").fetchone()[0])
            unique_sources = int(connection.execute("SELECT COUNT(DISTINCT a) FROM edge_counts").fetchone()[0])
            unique_targets = int(connection.execute("SELECT COUNT(DISTINCT b) FROM edge_counts").fetchone()[0])

            # Build exact train-only ranks on disk. This evaluates whether
            # conditioning on the previous track beats simply recommending the
            # globally most common next tracks. Unseen relationships and cold
            # source tracks correctly receive no adjacency-frequency credit.
            connection.execute(
                "CREATE TABLE source_stats (a TEXT PRIMARY KEY, outgoing_count INTEGER NOT NULL, candidate_count INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO source_stats(a, outgoing_count, candidate_count) "
                "SELECT a, SUM(count), COUNT(*) FROM edge_counts GROUP BY a"
            )
            connection.execute(
                "CREATE TABLE ranked_edges (a TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL, rank INTEGER NOT NULL, PRIMARY KEY (a, b)) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO ranked_edges(a, b, count, rank) "
                "SELECT a, b, count, ROW_NUMBER() OVER (PARTITION BY a ORDER BY count DESC, b ASC) FROM edge_counts"
            )
            connection.execute(
                "CREATE TABLE target_ranks (b TEXT PRIMARY KEY, count INTEGER NOT NULL, rank INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO target_ranks(b, count, rank) "
                "SELECT b, total, ROW_NUMBER() OVER (ORDER BY total DESC, b ASC) "
                "FROM (SELECT b, SUM(count) AS total FROM edge_counts GROUP BY b)"
            )
            connection.execute(
                "CREATE TABLE artist_target_stats (sa TEXT PRIMARY KEY, outgoing_count INTEGER NOT NULL, candidate_count INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO artist_target_stats(sa, outgoing_count, candidate_count) "
                "SELECT sa, SUM(count), COUNT(*) FROM artist_target_counts GROUP BY sa"
            )
            connection.execute(
                "CREATE TABLE artist_target_ranks ("
                "sa TEXT NOT NULL, b TEXT NOT NULL, count INTEGER NOT NULL, global_rank INTEGER NOT NULL, rank INTEGER NOT NULL, "
                "PRIMARY KEY (sa, b)"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO artist_target_ranks(sa, b, count, global_rank, rank) "
                "SELECT atc.sa, atc.b, atc.count, g.rank, "
                "ROW_NUMBER() OVER (PARTITION BY atc.sa ORDER BY atc.count DESC, g.rank ASC, atc.b ASC) "
                "FROM artist_target_counts atc JOIN target_ranks g ON g.b = atc.b"
            )
            connection.execute("CREATE INDEX artist_target_ranks_global ON artist_target_ranks(sa, global_rank)")
            connection.execute(
                "CREATE TABLE heldout_artist_backoff_ranks ("
                "split TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, sa TEXT NOT NULL, rank INTEGER, "
                "PRIMARY KEY (split, a, b, sa)"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO heldout_artist_backoff_ranks(split, a, b, sa, rank) "
                "SELECT h.split, h.a, h.b, h.sa, "
                "CASE "
                "WHEN g.rank IS NULL THEN NULL "
                "WHEN ar.rank IS NOT NULL THEN ar.rank "
                "WHEN ast.sa IS NOT NULL THEN ast.candidate_count + g.rank - COALESCE(("
                "SELECT COUNT(*) FROM artist_target_ranks prior "
                "WHERE prior.sa = h.sa AND prior.global_rank < g.rank"
                "), 0) "
                "ELSE g.rank END "
                "FROM heldout_edges h "
                "LEFT JOIN target_ranks g ON g.b = h.b "
                "LEFT JOIN artist_target_stats ast ON ast.sa = h.sa "
                "LEFT JOIN artist_target_ranks ar ON ar.sa = h.sa AND ar.b = h.b"
            )
            # RC11.6C: learn a second train-only backoff over artist identity.
            # This gives unseen target recordings a measurable retrieval signal
            # when their artist was observed in training, without pretending the
            # unseen track ID itself was known.
            connection.execute(
                "CREATE TABLE target_artist_ranks (ta TEXT PRIMARY KEY, count INTEGER NOT NULL, rank INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO target_artist_ranks(ta, count, rank) "
                "SELECT ta, count, ROW_NUMBER() OVER (ORDER BY count DESC, ta ASC) FROM target_artist_counts"
            )
            connection.execute(
                "CREATE TABLE source_artist_target_artist_stats (sa TEXT PRIMARY KEY, outgoing_count INTEGER NOT NULL, candidate_count INTEGER NOT NULL) WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO source_artist_target_artist_stats(sa, outgoing_count, candidate_count) "
                "SELECT sa, SUM(count), COUNT(*) FROM artist_pair_counts GROUP BY sa"
            )
            connection.execute(
                "CREATE TABLE source_artist_target_artist_ranks ("
                "sa TEXT NOT NULL, ta TEXT NOT NULL, count INTEGER NOT NULL, global_rank INTEGER NOT NULL, rank INTEGER NOT NULL, "
                "PRIMARY KEY (sa, ta)"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO source_artist_target_artist_ranks(sa, ta, count, global_rank, rank) "
                "SELECT ap.sa, ap.ta, ap.count, g.rank, "
                "ROW_NUMBER() OVER (PARTITION BY ap.sa ORDER BY ap.count DESC, g.rank ASC, ap.ta ASC) "
                "FROM artist_pair_counts ap JOIN target_artist_ranks g ON g.ta = ap.ta"
            )
            connection.execute(
                "CREATE INDEX source_artist_target_artist_global "
                "ON source_artist_target_artist_ranks(sa, global_rank)"
            )
            connection.execute(
                "CREATE TABLE heldout_artist_target_backoff_ranks ("
                "split TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL, sa TEXT NOT NULL, ta TEXT NOT NULL, rank INTEGER, "
                "PRIMARY KEY (split, a, b, sa, ta)"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "INSERT INTO heldout_artist_target_backoff_ranks(split, a, b, sa, ta, rank) "
                "SELECT h.split, h.a, h.b, h.sa, h.ta, "
                "CASE "
                "WHEN g.rank IS NULL THEN NULL "
                "WHEN ar.rank IS NOT NULL THEN ar.rank "
                "WHEN ast.sa IS NOT NULL THEN ast.candidate_count + g.rank - COALESCE(("
                "SELECT COUNT(*) FROM source_artist_target_artist_ranks prior "
                "WHERE prior.sa = h.sa AND prior.global_rank < g.rank"
                "), 0) "
                "ELSE g.rank END "
                "FROM heldout_edges h "
                "LEFT JOIN target_artist_ranks g ON g.ta = h.ta "
                "LEFT JOIN source_artist_target_artist_stats ast ON ast.sa = h.sa "
                "LEFT JOIN source_artist_target_artist_ranks ar ON ar.sa = h.sa AND ar.ta = h.ta"
            )
            connection.commit()

            evaluation_metrics = {
                split: _sequence_retrieval_metrics(connection, split, candidate_count=unique_targets)
                for split in ("validation", "test")
            }
            generalization_metrics = {
                split: _sequence_generalization_metrics(connection, split, candidate_count=unique_targets)
                for split in ("validation", "test")
            }
            backoff_metrics = {
                split: _sequence_backoff_metrics(connection, split, candidate_count=unique_targets)
                for split in ("validation", "test")
            }
            backoff_generalization_metrics = {
                split: _sequence_backoff_generalization_metrics(connection, split, candidate_count=unique_targets)
                for split in ("validation", "test")
            }
            unique_target_artists = int(connection.execute("SELECT COUNT(*) FROM target_artist_ranks").fetchone()[0])
            cold_target_artist_metrics = {
                split: _sequence_cold_target_artist_metrics(
                    connection, split, candidate_count=unique_target_artists
                )
                for split in ("validation", "test")
            }

            def score_delta(metric_row: dict[str, Any]) -> dict[str, float]:
                return {
                    metric: round(
                        float(metric_row["model"][metric])
                        - float(metric_row["global_popularity_baseline"][metric]),
                        8,
                    )
                    for metric in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr")
                }

            improvement_vs_global = {
                split: score_delta(evaluation_metrics[split])
                for split in ("validation", "test")
            }
            generalization_improvement_vs_global = {
                split: {name: score_delta(metric_row) for name, metric_row in generalization_metrics[split].items()}
                for split in ("validation", "test")
            }
            backoff_improvement_vs_global = {
                split: score_delta(backoff_metrics[split])
                for split in ("validation", "test")
            }
            backoff_generalization_improvement_vs_global = {
                split: {name: score_delta(metric_row) for name, metric_row in backoff_generalization_metrics[split].items()}
                for split in ("validation", "test")
            }

            def artist_score_delta(metric_row: dict[str, Any]) -> dict[str, float]:
                return {
                    metric: round(
                        float(metric_row["model"][metric])
                        - float(metric_row["global_target_artist_baseline"][metric]),
                        8,
                    )
                    for metric in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr")
                }

            cold_target_artist_improvement = {
                split: {
                    name: artist_score_delta(metric_row)
                    for name, metric_row in cold_target_artist_metrics[split].items()
                }
                for split in ("validation", "test")
            }

            top_rows = list(
                connection.execute(
                    "SELECT a, b, count FROM edge_counts ORDER BY count DESC, a ASC, b ASC LIMIT ?",
                    (int(top_limit),),
                )
            )
            from_ids = sorted({str(row[0]) for row in top_rows})
            from_totals: dict[str, int] = {}
            if from_ids:
                connection.execute("CREATE TEMP TABLE selected_from (a TEXT PRIMARY KEY) WITHOUT ROWID")
                connection.executemany("INSERT OR IGNORE INTO selected_from(a) VALUES (?)", ((value,) for value in from_ids))
                for a, count in connection.execute(
                    "SELECT e.a, SUM(e.count) FROM edge_counts e JOIN selected_from s ON e.a = s.a GROUP BY e.a"
                ):
                    from_totals[str(a)] = int(count)
            top_edges = [
                {
                    "from_track_id": str(a),
                    "to_track_id": str(b),
                    "count": int(count),
                    "conditional_probability": round(int(count) / max(1, from_totals.get(str(a), 0)), 8),
                }
                for a, b, count in top_rows
            ]
            artist_target_edges = int(connection.execute("SELECT COUNT(*) FROM artist_target_counts").fetchone()[0])
            unique_source_artists = int(connection.execute("SELECT COUNT(*) FROM artist_target_stats").fetchone()[0])
            top_artist_targets = [
                {
                    "source_artist_key": str(sa),
                    "to_track_id": str(b),
                    "count": int(count),
                    "artist_rank": int(rank),
                    "global_rank": int(global_rank),
                }
                for sa, b, count, rank, global_rank in connection.execute(
                    "SELECT sa, b, count, rank, global_rank FROM artist_target_ranks "
                    "ORDER BY count DESC, sa ASC, rank ASC LIMIT ?",
                    (int(top_limit),),
                )
            ]
            artist_pair_edges = int(connection.execute("SELECT COUNT(*) FROM artist_pair_counts").fetchone()[0])
            top_artist_pairs = [
                {
                    "source_artist_key": str(sa),
                    "target_artist_key": str(ta),
                    "count": int(count),
                    "artist_rank": int(rank),
                    "global_artist_rank": int(global_rank),
                }
                for sa, ta, count, rank, global_rank in connection.execute(
                    "SELECT sa, ta, count, rank, global_rank FROM source_artist_target_artist_ranks "
                    "ORDER BY count DESC, sa ASC, rank ASC LIMIT ?",
                    (int(top_limit),),
                )
            ]
            if os.environ.get("DJAI_BOOTSTRAP_EXPORT_RUNTIME_INDEX", "1").strip().lower() not in {"0", "false", "no", "off"}:
                runtime_index = _export_sequence_runtime_index(
                    connection, scratch_root / "sequence-runtime.sqlite"
                )
        finally:
            connection.close()

    return {
        "edge_examples": int(total_edges),
        "training_edge_examples": int(valid_train_edges),
        "examples_by_split": dict(split_examples),
        "unique_train_edges": int(unique_edges),
        "unique_train_sources": int(unique_sources),
        "unique_train_targets": int(unique_targets),
        "top_edges": top_edges,
        "runtime_index": runtime_index,
        "aggregation": "sqlite_exact_streaming",
        "top_edges_limit": int(top_limit),
        "evaluation": {
            "evaluation_kind": "held_out_next_track_retrieval",
            "primary_metric": "hit_at_10",
            "model_kind": "train_conditional_adjacency_frequency",
            "split_policy": "user-group-held-out; validation/test never contribute training counts",
            "candidate_universe": "unique train destination tracks",
            "candidate_count": int(unique_targets),
            "metrics": evaluation_metrics,
            "subgroups": generalization_metrics,
            "improvement_vs_global_popularity": improvement_vs_global,
            "generalization_improvement_vs_global_popularity": generalization_improvement_vs_global,
            "generalization_note": (
                "The current model is an exact train conditional-frequency prior. Unseen A->B edges have no conditional rank; "
                "novel_pair_candidate therefore measures the limitation explicitly rather than implying feature-based transfer."
            ),
            "interpretation": "behavioral next-track retrieval only; not technical mixability or professional DJ sequencing",
        },
        "generalizer": {
            "training_edge_examples": int(valid_train_edges),
            "artist_target_edges": int(artist_target_edges),
            "unique_source_artists": int(unique_source_artists),
            "top_artist_targets": top_artist_targets,
            "top_artist_targets_limit": int(top_limit),
            "aggregation": "sqlite_exact_artist_backoff",
            "evaluation": {
                "evaluation_kind": "held_out_next_track_retrieval",
                "primary_metric": "hit_at_10",
                "model_kind": "train_source_artist_to_target_track_backoff",
                "split_policy": "user-group-held-out; validation/test never contribute training counts",
                "candidate_universe": "unique train destination tracks",
                "candidate_count": int(unique_targets),
                "metrics": backoff_metrics,
                "subgroups": backoff_generalization_metrics,
                "improvement_vs_global_popularity": backoff_improvement_vs_global,
                "generalization_improvement_vs_global_popularity": backoff_generalization_improvement_vs_global,
                "generalization_note": (
                    "RC11.6 generalizes across tracks by hashing normalized source-artist identity and learning train-only "
                    "source-artist -> destination-track frequencies. Unseen artist-targets back off to global popularity; "
                    "there is no held-out label leakage and no claim of technical mixability."
                ),
                "interpretation": "behavioral source-artist sequencing prior only; not technical mixability or professional DJ sequencing",
            },
            "cold_target_intelligence": {
                "schema_version": "cold-target-artist-backoff/1.0",
                "artist_pair_edges": int(artist_pair_edges),
                "unique_target_artists": int(unique_target_artists),
                "top_source_artist_target_artists": top_artist_pairs,
                "top_source_artist_target_artists_limit": int(top_limit),
                "aggregation": "sqlite_exact_source_artist_to_target_artist_backoff",
                "evaluation": {
                    "evaluation_kind": "held_out_cold_target_artist_retrieval",
                    "primary_metric": "hit_at_10",
                    "model_kind": "train_source_artist_to_target_artist_backoff",
                    "split_policy": "user-group-held-out; validation/test never contribute training counts",
                    "candidate_universe": "unique train destination artists",
                    "candidate_count": int(unique_target_artists),
                    "subgroups": cold_target_artist_metrics,
                    "improvement_vs_global_target_artist": cold_target_artist_improvement,
                    "generalization_note": (
                        "RC11.6C gives a cold target track an artist-level behavioral prior when its destination artist "
                        "was observed in train users. Completely unseen artists remain explicitly uncovered; this stage "
                        "does not infer track-level evidence that is absent from ListenBrainz metadata."
                    ),
                    "interpretation": "behavioral artist sequencing only; not technical mixability or professional DJ sequencing",
                },
            },
        },
    }

def train_bootstrap_shadow_models(
    workspace: str | Path,
    *,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    corpus = root / "datasets" / "bootstrap-corpora" / corpus_id if corpus_id else _latest_corpus(root)
    if not corpus.exists():
        raise ValueError(f"Bootstrap corpus not found: {corpus_id}")
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = root / "config" / "bootstrap-models" / str(manifest.get("corpus_id", corpus.name))
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    tasks_dir = corpus / "tasks"
    for task in ("cue_point_prior", "transition_duration_structure"):
        task_path = tasks_dir / f"{task}.jsonl"
        if not task_path.exists():
            results.append({"model_id": task, "status": "skipped", "reason": "task_dataset_missing"})
            continue
        rows = list(_read_jsonl(task_path))
        try:
            fitted = _fit_shadow_ranking(rows)
        except ValueError as exc:
            results.append({"model_id": task, "status": "skipped", "reason": str(exc), "examples": len(rows)})
            continue
        feature_spaces = sorted({str(row.get("feature_space", "unknown")) for row in rows})
        payload = {
            "schema_version": "bootstrap-shadow/1.2",
            "model_id": task,
            "corpus_id": manifest.get("corpus_id", corpus.name),
            "trained_at_utc": _now(),
            "research_shadow": True,
            "promotable_to_live": False,
            "reason_not_promotable": "external structural feature space is intentionally separate from the runtime DJ policy feature schema",
            "feature_spaces": feature_spaces,
            **fitted,
        }
        path = output_root / f"{task}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        results.append({
            "model_id": task,
            "status": "trained",
            "path": str(path.relative_to(root)),
            "evaluation_kind": fitted["evaluation_kind"],
            "primary_metric": fitted["primary_metric"],
            "metrics": fitted["metrics"],
            "baseline": fitted["baseline"],
            "accuracy_improvement_vs_baseline": fitted["accuracy_improvement_vs_baseline"],
            "primary_improvement_vs_baseline": fitted["primary_improvement_vs_baseline"],
            "examples": len(rows),
        })

    # ListenBrainz can contribute millions of edges. Aggregate them exactly on
    # disk instead of materializing the corpus or an unbounded Counter in RAM.
    records_path = corpus / "records.jsonl"
    sequence = (
        _aggregate_sequence_edges_streaming(records_path, output_root)
        if records_path.exists()
        else {"edge_examples": 0, "training_edge_examples": 0, "examples_by_split": {}, "unique_train_edges": 0, "top_edges": []}
    )
    generalizer = sequence.get("generalizer") if isinstance(sequence.get("generalizer"), dict) else {}
    adjacency_sequence = {key: value for key, value in sequence.items() if key != "generalizer"}
    if int(sequence.get("training_edge_examples", 0)) > 0:
        payload = {
            "schema_version": "next-track-adjacency-shadow/1.3",
            "model_id": "next_track_adjacency_prior",
            "corpus_id": manifest.get("corpus_id", corpus.name),
            "trained_at_utc": _now(),
            "research_shadow": True,
            "promotable_to_live": False,
            "interpretation": "listening adjacency only; not technical mixability or professional DJ sequencing",
            "training_policy": "train_split_only; user-group validation/test edges remain held out",
            "artifact_scope": "top_edges_summary_plus_train_only_runtime_index; evaluation uses the full temporary train adjacency table",
            **adjacency_sequence,
        }
        path = output_root / "next_track_adjacency_prior.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        results.append({
            "model_id": "next_track_adjacency_prior",
            "status": "trained",
            "path": str(path.relative_to(root)),
            "examples": int(sequence["edge_examples"]),
            "training_examples": int(sequence["training_edge_examples"]),
            "examples_by_split": sequence["examples_by_split"],
            "unique_edges": int(sequence["unique_train_edges"]),
            "unique_sources": int(sequence.get("unique_train_sources", 0)),
            "unique_targets": int(sequence.get("unique_train_targets", 0)),
            "aggregation": sequence["aggregation"],
            "evaluation_kind": sequence["evaluation"]["evaluation_kind"],
            "primary_metric": sequence["evaluation"]["primary_metric"],
            "evaluation": sequence["evaluation"],
        })
    elif int(sequence.get("edge_examples", 0)) > 0:
        results.append({
            "model_id": "next_track_adjacency_prior",
            "status": "skipped",
            "reason": "no_train_sequence_edges",
            "examples": int(sequence["edge_examples"]),
            "examples_by_split": sequence.get("examples_by_split", {}),
        })
    else:
        results.append({"model_id": "next_track_adjacency_prior", "status": "skipped", "reason": "no_sequence_edges"})

    if int(generalizer.get("artist_target_edges", 0)) > 0:
        evaluation = generalizer.get("evaluation") if isinstance(generalizer.get("evaluation"), dict) else {}
        payload = {
            "schema_version": "next-track-sequence-generalizer-shadow/1.1",
            "model_id": "next_track_sequence_generalizer",
            "corpus_id": manifest.get("corpus_id", corpus.name),
            "trained_at_utc": _now(),
            "research_shadow": True,
            "promotable_to_live": False,
            "reason_not_promotable": "identity-backoff research prior requires benchmark promotion and runtime feature-schema review",
            "interpretation": "behavioral source-artist sequencing only; not technical mixability or professional DJ sequencing",
            "training_policy": "train_split_only; source artists are normalized then one-way hashed; held-out users never contribute counts",
            "artifact_scope": "top_artist_targets_summary_plus_train_only_runtime_index; evaluation uses the full temporary train source-artist table",
            **generalizer,
        }
        path = output_root / "next_track_sequence_generalizer.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        results.append({
            "model_id": "next_track_sequence_generalizer",
            "status": "trained",
            "path": str(path.relative_to(root)),
            "examples": int(sequence.get("edge_examples", 0)),
            "training_examples": int(generalizer.get("training_edge_examples", 0)),
            "examples_by_split": sequence.get("examples_by_split", {}),
            "unique_edges": int(generalizer.get("artist_target_edges", 0)),
            "unique_sources": int(generalizer.get("unique_source_artists", 0)),
            "unique_targets": int(sequence.get("unique_train_targets", 0)),
            "aggregation": str(generalizer.get("aggregation", "sqlite_exact_artist_backoff")),
            "evaluation_kind": evaluation.get("evaluation_kind", "held_out_next_track_retrieval"),
            "primary_metric": evaluation.get("primary_metric", "hit_at_10"),
            "evaluation": evaluation,
        })
    elif int(sequence.get("training_edge_examples", 0)) > 0:
        results.append({
            "model_id": "next_track_sequence_generalizer",
            "status": "skipped",
            "reason": "no_train_source_artist_metadata",
            "examples": int(sequence.get("edge_examples", 0)),
        })
    else:
        results.append({"model_id": "next_track_sequence_generalizer", "status": "skipped", "reason": "no_sequence_edges"})

    report = {
        "schema_version": "bootstrap-training-report/1.7",
        "corpus_id": manifest.get("corpus_id", corpus.name),
        "trained_at_utc": _now(),
        "shadow_only": True,
        "live_models_changed": False,
        "runtime_adapter_version": "1.1",
        "runtime_policy": "research-governed; local advisory is bounded and production defaults to shadow",
        "models": results,
    }
    (output_root / "training-report.json").write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report
