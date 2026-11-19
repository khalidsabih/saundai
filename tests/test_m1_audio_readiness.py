from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-audio-readiness.py"
    spec = importlib.util.spec_from_file_location("m1_audio_readiness", path)
    assert spec and spec.loader
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_exact_matching_and_replica_groups(tmp_path):
    m = module()
    audio = tmp_path / "audio"
    (audio / "Quickstep").mkdir(parents=True)
    (audio / "Quickstep" / "Albums-AnaBelen_Veneo-11.wav").write_bytes(b"fake")
    (audio / "Quickstep" / "Albums-Chrisanne2-12.wav").write_bytes(b"fake")
    index = m.audio_index(root=audio)
    outcome = m.audit(["Albums-AnaBelen_Veneo-11", "Albums-Chrisanne2-12", "missing"], index, "ballroom", None)
    assert outcome["matched_unique_audio"] == 2
    assert outcome["missing_audio"] == 1
    assert outcome["known_ballroom_replica_groups"][0]["both_matched"]
    assert outcome["benchmark_ready"] is False
    assert outcome["audio_rights_confirmed"] is False


def test_ambiguous_file_stem_not_reported_as_matched(tmp_path):
    m = module()
    audio = tmp_path / "audio"
    (audio / "A").mkdir(parents=True)
    (audio / "B").mkdir(parents=True)
    (audio / "A" / "song.wav").write_bytes(b"fixture")
    (audio / "B" / "song.mp3").write_bytes(b"fixture")
    result = m.audit(["song"], m.audio_index(root=audio), "ballroom", None)
    assert result["matched_unique_audio"] == 0
    assert result["ambiguous_audio"] == 1


def test_safe_local_tar_listing_without_extraction(tmp_path):
    m = module()
    archive = tmp_path / "test.tar.gz"
    payload = tmp_path / "song.wav"
    payload.write_bytes(b"fixture")
    with tarfile.open(archive, "w:gz") as out:
        out.add(payload, arcname="BallroomData/Waltz/song.wav")
    index = m.audio_index(archive=archive)
    assert index["song"] == ["BallroomData/Waltz/song.wav"]
    assert not (tmp_path / "BallroomData").exists()


def test_tar_traversal_rejected(tmp_path):
    m = module()
    import io
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        info = tarfile.TarInfo("../escaped.wav")
        raw = b"x"
        info.size = len(raw)
        out.addfile(info, io.BytesIO(raw))
    with pytest.raises(ValueError, match="unsafe"):
        m.audio_index(archive=archive)


def test_no_audio_access_and_output_requires_explicit_source(tmp_path):
    m = module()
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "song.beats").write_text("0.5 1\n1.0 2\n")
    (tmp_path / "audio").mkdir()
    with pytest.raises(ValueError, match="choose exactly one"):
        m.audio_index()
    assert m.annotation_ids(ann, "ballroom", "v2") == ["song"]
    assert m.audit(["song"], m.audio_index(root=tmp_path / "audio"), "ballroom", None)["matched_unique_audio"] == 0
