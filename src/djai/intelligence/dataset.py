from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..cache import discover_tracks

BENCHMARK_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class BenchmarkTrack:
    track_id: str
    path: str
    split: str
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_for(track_id: str) -> str:
    bucket = int(hashlib.sha256(track_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def create_benchmark_manifest(
    folder: str | Path,
    output_path: str | Path = "benchmarks/tracks.jsonl",
    *,
    recursive: bool = True,
) -> Path:
    root = Path(folder).expanduser().resolve()
    tracks = discover_tracks(root, recursive=recursive)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for path in tracks:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
            record = BenchmarkTrack(
                track_id=digest,
                path=str(path),
                split=_split_for(digest),
                annotations={
                    "bpm": None,
                    "downbeats": [],
                    "sections": [],
                    "good_entry_zones": [],
                    "good_exit_zones": [],
                    "notes": "",
                },
            )
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return destination


def benchmark_manifest_stats(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    counts = {"train": 0, "validation": 0, "test": 0}
    total = 0
    if not source.exists():
        return {"schema_version": BENCHMARK_SCHEMA_VERSION, "total": 0, "splits": counts}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        split = str(payload.get("split", ""))
        if split in counts:
            counts[split] += 1
        total += 1
    return {"schema_version": BENCHMARK_SCHEMA_VERSION, "total": total, "splits": counts}
