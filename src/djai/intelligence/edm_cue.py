from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

EDM_CUE_SCHEMA_VERSION = "1.0"
EDM_CUE_PAPER_URL = "https://arxiv.org/abs/2407.06823"
EDM_CUE_ISMIR_URL = "https://ismir2024program.ismir.net/poster_19.html"
EDM_CUE_MODEL_ID = "disco-eth/cue-detr"

# The ISMIR project page currently embeds four public Google Drive resources.
# We retain the IDs as provenance only. DJAI never assumes that a particular
# file is redistributable, and the fetch helper does not download track audio.
EDM_CUE_PUBLIC_DRIVE_IDS = (
    "1qAbrZKi-ItKh6F5IcxY9SVbtROgDQR2G",
    "1-MPgcWlKym7T6k4lLhTrt7VIa016_Eqq",
    "1MLy__nYjpuhpzC8A7JoXcbv2S3G-h8V0",
    "1y-Vitk0xu9FmOKt0hQsucMnMyZwiEvhP",
)


@dataclass(slots=True)
class EDMCueRecord:
    track_id: str
    artist: str | None
    title: str | None
    deezer_id: str | None
    bpm: float | None
    grid_offset_seconds: float | None
    initial_beat_number: int | None
    cue_points_seconds: list[float]
    split: str
    source: str = "EDM-CUE"
    schema_version: str = EDM_CUE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first(payload: dict[str, Any], *names: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in payload.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, "", "null", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _cue_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    elif isinstance(value, dict):
        raw = list(value.values())
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple, dict)):
                return _cue_list(parsed)
        except json.JSONDecodeError:
            pass
        separator = ";" if ";" in text else ","
        raw = [item.strip() for item in text.split(separator)]
    values: list[float] = []
    for item in raw:
        if isinstance(item, dict):
            item = _first(item, "time", "seconds", "position", "cue", "timestamp")
        number = _float_or_none(item)
        if number is not None and number >= 0.0:
            values.append(round(number, 6))
    return sorted(set(values))


def _stable_split(track_id: str) -> str:
    bucket = int(hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def normalize_edm_cue_record(payload: dict[str, Any]) -> EDMCueRecord:
    artist = _first(payload, "artist", "artist_name", "performer")
    title = _first(payload, "title", "track", "track_name", "name")
    deezer = _first(payload, "deezer_id", "deezer", "deezer_track_id")
    raw_id = _first(payload, "track_id", "id", "uuid", "rekordbox_id")
    identity = str(raw_id or deezer or f"{artist or ''}|{title or ''}").strip()
    if not identity:
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    track_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    cues = _cue_list(
        _first(
            payload,
            "cue_points_seconds",
            "cue_points",
            "cues",
            "cuepoints",
            "cue_times",
            "cue_positions",
        )
    )
    split = str(_first(payload, "split", "subset", "partition") or "").strip().lower()
    if split not in {"train", "validation", "val", "test"}:
        split = _stable_split(track_id)
    elif split == "val":
        split = "validation"
    return EDMCueRecord(
        track_id=track_id,
        artist=str(artist).strip() if artist not in (None, "") else None,
        title=str(title).strip() if title not in (None, "") else None,
        deezer_id=str(deezer).strip() if deezer not in (None, "") else None,
        bpm=_float_or_none(_first(payload, "bpm", "tempo")),
        grid_offset_seconds=_float_or_none(
            _first(payload, "grid_offset_seconds", "grid_offset", "beat_grid_offset", "offset")
        ),
        initial_beat_number=_int_or_none(
            _first(payload, "initial_beat_number", "first_beat", "beat_number")
        ),
        cue_points_seconds=cues,
        split=split,
    )


def _iter_source_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    text = path.read_text(encoding="utf-8-sig")
    if suffix in {".jsonl", ".ndjson"}:
        for line in text.splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload
        return
    payload = json.loads(text)
    if isinstance(payload, dict):
        for key in ("tracks", "data", "records", "items"):
            if isinstance(payload.get(key), list):
                for item in payload[key]:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def import_edm_cue_annotations(
    source_path: str | Path,
    output_path: str | Path = "datasets/edm-cue/annotations.jsonl",
) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [normalize_edm_cue_record(item) for item in _iter_source_records(source)]
    records = [record for record in records if record.cue_points_seconds]
    records.sort(key=lambda record: record.track_id)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return destination


def edm_cue_stats(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    counts = {"train": 0, "validation": 0, "test": 0}
    cue_count = 0
    bpm_values: list[float] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            split = str(payload.get("split", ""))
            if split in counts:
                counts[split] += 1
            cue_count += len(payload.get("cue_points_seconds", []))
            bpm = _float_or_none(payload.get("bpm"))
            if bpm is not None:
                bpm_values.append(bpm)
    total = sum(counts.values())
    return {
        "schema_version": EDM_CUE_SCHEMA_VERSION,
        "tracks": total,
        "cue_points": cue_count,
        "splits": counts,
        "bpm_min": min(bpm_values) if bpm_values else None,
        "bpm_max": max(bpm_values) if bpm_values else None,
    }


def edm_cue_source_manifest() -> dict[str, Any]:
    return {
        "dataset": "EDM-CUE",
        "paper": EDM_CUE_PAPER_URL,
        "project_page": EDM_CUE_ISMIR_URL,
        "published_model": EDM_CUE_MODEL_ID,
        "public_drive_ids": list(EDM_CUE_PUBLIC_DRIVE_IDS),
        "contains_audio": False,
        "notes": (
            "EDM-CUE contains DJ-library metadata, tempo/beat-grid information, "
            "Deezer references, and expert cue annotations. DJAI does not bundle or "
            "download copyrighted track audio."
        ),
    }
