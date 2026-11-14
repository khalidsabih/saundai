"""Raveform offline preparation and independent-manifest safety tests."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_raveform_independent import (
    parse_annotation, prepare, track_id_from_filename,
)


def fixture(tmp_path):
    root = tmp_path / "raveform"
    beats = root / "structures" / "beats"
    beats.mkdir(parents=True)
    (root / "tracks.jsonl").write_text(
        json.dumps({"id": "abc123", "title": "Test EDM", "duration": 6.0}) + "\n",
        encoding="utf-8",
    )
    (beats / "0001.abc123.beat.csv").write_text(
        "time,downbeat,section\n"
        "0.1,1,intro\n0.6,2,intro\n1.1,3,intro\n1.6,4,intro\n"
        "2.1,1,verse\n2.6,2,verse\n",
        encoding="utf-8",
    )
    return root


def test_stages_labels_without_audio_or_manifest(tmp_path):
    root = fixture(tmp_path)
    out = tmp_path / "out"
    result = prepare(root, out)
    assert result["annotated"] == 1
    assert result["verified_audio"] == 0
    assert result["manifest"] is None
    assert (out / "labels" / "abc123.beats").read_text().splitlines() == [
        "0.100000000", "0.600000000", "1.100000000",
        "1.600000000", "2.100000000", "2.600000000",
    ]
    assert (out / "labels" / "abc123.downbeats").read_text().splitlines() == [
        "0.100000000", "2.100000000",
    ]
    assert json.loads((out / "raveform-candidates.json").read_text())["benchmark_ready"] is False
    assert not (out / "raveform-independent-manifest.json").exists()


@pytest.mark.parametrize("rows", [
    "time,downbeat\n0.1,1\n0.1,2\n",
    "time,downbeat\n0.1,0\n",
    "time,downbeat\nNaN,1\n",
    "time,downbeat\n0.1,2\n",
    "time,section\n0.1,intro\n",
])
def test_bad_annotation_rejected(tmp_path, rows):
    p = tmp_path / "0001.abc123.beat.csv"
    p.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError):
        parse_annotation(p)


def test_unmatched_metadata_id_rejected(tmp_path):
    root = fixture(tmp_path)
    (root / "tracks.jsonl").write_text('{"id":"other"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unmatched"):
        prepare(root, tmp_path / "out")


def test_manifest_only_for_attested_hash_verified_audio(tmp_path):
    root = fixture(tmp_path)
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    (audio_root / "abc123.wav").write_bytes(b"test placeholder bytes")
    sha = hashlib.sha256((audio_root / "abc123.wav").read_bytes()).hexdigest()
    entry = {
        "id": "abc123", "audio": "abc123.wav", "sha256": sha,
        "source": "Operator-owned recording; synthetic fixture",
        "recording_verified": True, "alignment_verified": True,
        "verification_notes": "Operator verified exact version and timeline",
    }
    mapping = tmp_path / "verified.json"
    mapping.write_text(json.dumps({"tracks": [entry]}), encoding="utf-8")
    out = tmp_path / "out"
    report = prepare(root, out, audio_root, mapping)
    assert report["verified_audio"] == 1
    manifest = json.loads(Path(report["manifest"]).read_text())
    assert manifest["tracks"][0]["split"] == "independent"
    assert manifest["tracks"][0]["id"] == "abc123"
    assert manifest["tracks"][0]["beats"].endswith("abc123.beats")
    assert manifest["audio_provenance"][0]["sha256"] == sha
    entry["sha256"] = "0" * 64
    mapping.write_text(json.dumps({"tracks": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare(root, tmp_path / "bad", audio_root, mapping)
    entry["sha256"] = sha
    entry["alignment_verified"] = False
    mapping.write_text(json.dumps({"tracks": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="alignment_verified"):
        prepare(root, tmp_path / "bad", audio_root, mapping)
