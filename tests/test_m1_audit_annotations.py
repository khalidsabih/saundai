from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-audit-annotations.py"
    spec = importlib.util.spec_from_file_location("m1_audit_annotations", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_giantsteps_format_and_missing_zero(tmp_path):
    m = load()
    root = tmp_path / "giantsteps"
    tempo = root / "annotations_v2" / "tempo"
    tempo.mkdir(parents=True)
    (tempo / "1030011.LOFI.bpm").write_text("127.0\n")
    (tempo / "1114156.LOFI.bpm").write_text("0\n")
    data = m.scan("giantsteps_tempo", root, "v2")
    assert data["annotation_file_count"] == 2
    assert data["counts"] == {"bpm_missing": 1, "bpm_present": 1}
    assert data["benchmark_ready"] is False
    assert len(data["annotations_sha256"]) == 64


def test_ballroom_downbeat_and_annotation_integrity(tmp_path):
    m = load()
    root = tmp_path / "ballroom"
    root.mkdir()
    (root / "Albums-AnaBelen_Veneo-01.beats").write_text("0.1944671200 4\n0.8499773240 1\n1.4700226750 2\n")
    data = m.scan("ballroom", root, "bar_positions_v1")
    assert data["counts"]["beat_events"] == 3
    assert data["counts"]["downbeat_events"] == 1
    assert data["counts"]["downbeat_tracks"] == 1


def test_missing_or_invalid_references_fail_closed(tmp_path):
    m = load()
    with pytest.raises(ValueError, match="no matching"):
        m.scan("ballroom", tmp_path, "bar_positions_v1")
    (tmp_path / "broken.beats").write_text("0.3 1\n0.2 2\n")
    with pytest.raises(ValueError, match="unsorted"):
        m.scan("ballroom", tmp_path, "bar_positions_v1")


def test_audit_refuses_overwrite(tmp_path):
    m = load()
    ann = tmp_path / "annotations"
    (ann / "annotations_v2" / "tempo").mkdir(parents=True)
    (ann / "annotations_v2" / "tempo" / "a.LOFI.bpm").write_text("120")
    out = tmp_path / "audit.json"
    assert m.main(["--dataset", "giantsteps_tempo", "--annotations", str(ann), "--output", str(out)]) == 0
    assert out.exists()
    with pytest.raises(ValueError, match="overwrite"):
        m.main(["--dataset", "giantsteps_tempo", "--annotations", str(ann), "--output", str(out)])
