from __future__ import annotations

import csv
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-groove-metadata.py"
    spec = importlib.util.spec_from_file_location("m1_groove_metadata", path)
    assert spec and spec.loader
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def example_rows():
    base = dict(drummer="drummer1", session="session1", style="jazz/swing",
                bpm="120", beat_type="beat", time_signature="4-4",
                duration="13.5", midi_filename="drummer1/session1/a.mid",
                audio_filename="drummer1/session1/a.wav")
    return [dict(base, id="a", split="train"),
            dict(base, id="b", split="validation", midi_filename="drummer1/session1/b.mid",
                 audio_filename="drummer1/session1/b.wav"),
            dict(base, id="c", split="test", midi_filename="drummer1/session1/c.mid",
                 audio_filename="drummer1/session1/c.wav")]


def test_official_columns_preserve_splits_and_session_leakage():
    m = module()
    info = m.audit(example_rows())
    assert info["total"] == 3
    assert info["counts"]["split_train"] == 1
    assert info["counts"]["split_validation"] == 1
    assert info["counts"]["split_test"] == 1
    assert info["session_cross_split_count"] == 1
    assert info["entries"][0]["bpm"] == 120.0


def test_metadata_only_never_fabricates_beat_downbeat_reference(tmp_path):
    m = module()
    info = m.audit(example_rows())
    audio = tmp_path / "audio"
    (audio / "drummer1" / "session1").mkdir(parents=True)
    (audio / "drummer1" / "session1" / "a.wav").write_bytes(b"placeholder")
    manifest, stats = m.build_manifest(info, split="train", audio_root=audio)
    assert manifest["rights_confirmed"] is False
    assert manifest["split"] == "development"
    assert manifest["dataset_id"] == "groove_midi"
    assert stats["included"] == 1
    assert manifest["items"][0]["beats_seconds"] is None
    assert manifest["items"][0]["downbeats_seconds"] is None
    with pytest.raises(ValueError, match="insufficient"):
        m.build_manifest(info, split="test", audio_root=audio)


def test_zip_fixture_requires_opt_in_for_nonofficial_sha(tmp_path):
    m = module()
    path = tmp_path / "groove-fixture.zip"
    columns = list(example_rows()[0])
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(example_rows())
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("groove/info.csv", csv_buffer.getvalue())
    with pytest.raises(ValueError, match="official Groove"):
        m.read_metadata(path, verify_official=True)
    rows, digest = m.read_metadata(path, verify_official=False)
    assert len(rows) == 3
    assert len(digest) == 64


def test_missing_invalid_labels_and_paths_fail_closed():
    m = module()
    rows = example_rows()
    rows[0]["bpm"] = "-1"
    with pytest.raises(ValueError, match="invalid BPM"):
        m.audit(rows)
    rows = example_rows()
    rows[0]["midi_filename"] = "../outside.mid"
    with pytest.raises(ValueError, match="unsafe"):
        m.audit(rows)
    rows = example_rows()
    rows[0]["split"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized"):
        m.audit(rows)


def test_cli_fixture_audit_no_audio_or_approval(tmp_path):
    m = module()
    source = tmp_path / "info.csv"
    rows = example_rows()
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "groove-audit.json"
    assert m.main(["--source", str(source), "--allow-unverified-csv", "--output", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["rights_confirmed"] is False
    assert result["benchmark_ready"] is False
    assert result["beat_downbeat_ground_truth"] is False
    with pytest.raises(ValueError, match="overwrite"):
        m.main(["--source", str(source), "--allow-unverified-csv", "--output", str(out)])
