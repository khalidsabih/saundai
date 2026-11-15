from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from djai.intelligence.tempo_reference_workflow import (
    HEADERS, export_annotations, prepare_csv, select_for_annotation,
)


def _make_audio(root: Path, folders: int = 3, each: int = 10):
    for i in range(folders):
        directory = root / f"folder-{i}"
        directory.mkdir(parents=True)
        for j in range(each):
            (directory / f"f{i}-track{j:02d}.wav").write_bytes(b"no audio decoded")


def test_preparation_selects_24_without_claiming_any_bpm_or_genre(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    _make_audio(root)
    output = tmp_path / "annotations.csv"
    meta = prepare_csv(root, output)
    assert meta["selected"] == 24
    assert select_for_annotation(root) == select_for_annotation(root)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert len(set(r["filename"] for r in rows)) == 24
    assert len({r["filename"].split("-")[0] for r in rows}) >= 1
    assert all(not row["reference_bpm"] and not row["reference_quality"]
               and not row["verification_method"] and not row["genre"] for row in rows)
    with pytest.raises(FileExistsError):
        prepare_csv(root, output)


def test_sample_excludes_ambiguous_duplicate_basenames(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    _make_audio(root, folders=2, each=2)
    (root / "folder-0" / "duplicate.mp3").write_bytes(b"x")
    (root / "folder-1" / "duplicate.mp3").write_bytes(b"x")
    names = select_for_annotation(root, count=10)
    assert "duplicate.mp3" not in names
    assert len(names) == 4


def test_exports_verified_and_approximate_rows_without_guessing(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow({"filename": "verified.wav", "reference_bpm": "123",
                         "reference_quality": "independently_verified",
                         "verification_method": "manual beat annotation checked twice",
                         "genre": "techno"})
        writer.writerow({"filename": "approx.wav", "reference_bpm": "78",
                         "reference_quality": "approximate"})
        writer.writerow({"filename": "pending.wav"})
    output = tmp_path / "verified.json"
    stats = export_annotations(source, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert stats["annotated_exported"] == 2
    assert stats["pending"] == 1
    assert stats["verified"] == 1 and stats["approximate"] == 1
    assert [r["filename"] for r in data] == ["verified.wav", "approx.wav"]
    assert data[0]["verification_method"] == "manual beat annotation checked twice"
    with pytest.raises(FileExistsError):
        export_annotations(source, output)


def test_export_rejects_unverified_claims_and_partial_rows(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    output = tmp_path / "references.json"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow({"filename": "a.wav", "reference_bpm": "120",
                         "reference_quality": "independently_verified"})
    with pytest.raises(ValueError, match="verification_method"):
        export_annotations(source, output)
    assert not output.exists()
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerow({"filename": "a.wav", "reference_quality": "approximate"})
    with pytest.raises(ValueError, match="partially annotated"):
        export_annotations(source, output)
    assert not output.exists()


def test_empty_or_short_library_is_honest(tmp_path: Path) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    assert select_for_annotation(root, count=24) == []
    with pytest.raises(ValueError, match="positive"):
        select_for_annotation(root, count=0)
