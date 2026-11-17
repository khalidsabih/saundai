from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-prepare-ballroom-pilot.py"
    spec = importlib.util.spec_from_file_location("m1_prepare_ballroom_pilot", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    ann = tmp_path / "annotations"
    ann.mkdir()
    for genre, stem in [("Waltz", "sample-a"), ("Rumba", "sample-b"),
                        ("Rumba", "Albums-Fire-08")]:
        name = f"{genre}/{stem}.wav"
        p = audio / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"RIFF" + b"x" * 2000)
        (ann / (stem + ".beats")).write_text("0.5 1\n1.0 2\n1.5 3\n2.0 4\n")
    archive = tmp_path / "data1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for file in sorted(audio.rglob("*.wav")):
            tar.add(file, arcname=f"BallroomData/{file.relative_to(audio)}")
    return archive, ann


def test_pilot_plans_genre_diversity_and_excludes_replicas(tmp_path, monkeypatch):
    m = load()
    archive, ann = sources(tmp_path)
    # Fake archive fixture cannot have official checksum; checksum validation itself
    # is tested separately and remains mandatory in real usage.
    original = m.load_script

    def patched(path, name):
        obj = original(path, name)
        if name == "m1_audio_readiness_for_pilot":
            obj.md5 = lambda _: obj.BALLROOM_MD5
        return obj

    monkeypatch.setattr(m, "load_script", patched)
    chosen = m.plan(archive, ann, max_tracks=2)
    assert len(chosen) == 2
    assert {x[0] for x in chosen} == {"waltz", "rumba"}
    assert all(x[1] != "albums-fire-08" for x in chosen)


def test_refuse_no_ack_and_no_overwrite(tmp_path, monkeypatch):
    m = load()
    archive, ann = sources(tmp_path)
    original = m.load_script

    def patched(path, name):
        obj = original(path, name)
        if name == "m1_audio_readiness_for_pilot":
            obj.md5 = lambda _: obj.BALLROOM_MD5
        return obj

    monkeypatch.setattr(m, "load_script", patched)
    chosen = m.plan(archive, ann, max_tracks=2)
    dest = tmp_path / "pilot"
    with pytest.raises(ValueError, match="explicit"):
        m.execute(archive, ann, dest, chosen, acknowledgement=False)
    result = m.execute(archive, ann, dest, chosen, acknowledgement=True)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["rights_confirmed"] is False
    assert manifest["split"] == "development"
    assert all(x["bpm"] is None for x in manifest["items"])
    assert len(manifest["items"]) == 2
    assert all((dest / "audio" / x["audio"]).exists() for x in manifest["items"])
    with pytest.raises(ValueError, match="exists"):
        m.execute(archive, ann, dest, chosen, acknowledgement=True)


def test_invalid_archive_checksum_rejected(tmp_path):
    m = load()
    archive, ann = sources(tmp_path)
    with pytest.raises(ValueError, match="published Ballroom MD5"):
        m.plan(archive, ann, max_tracks=2)


def test_invalid_count_rejected_without_files(tmp_path):
    m = load()
    archive, ann = sources(tmp_path)
    with pytest.raises(ValueError, match="1..16"):
        m.main(["--archive", str(archive), "--annotations", str(ann),
                "--destination", str(tmp_path / "never"), "--max-tracks", "200",
                "--research-only-ack"])
    assert not (tmp_path / "never").exists()
