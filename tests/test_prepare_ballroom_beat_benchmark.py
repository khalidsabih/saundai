import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_ballroom_beat_benchmark import (
    generate_manifest, parse_ballroom_beats, split_for_hash, setup,
)


def _track(tmp_path, name, *, beats="0.0 1\n0.5 2\n1.0 3\n1.5 4\n2.0 1\n2.5 2\n3.0 3\n3.5 4\n"):
    audio = tmp_path / (name+".wav")
    audio.write_bytes(b"fake-audio-"+name.encode())
    ref = tmp_path / (name+".beats")
    ref.write_text(beats)
    return SimpleNamespace(audio_path=audio, beats_path=ref)


def test_ballroom_time_and_downbeat_normalization(tmp_path):
    t = _track(tmp_path, "one")
    beats, downs = parse_ballroom_beats(Path(t.beats_path))
    assert beats == [i*0.5 for i in range(8)]
    assert downs == [0.0, 2.0]


def test_manifest_uses_real_paths_and_annotation_pairs(tmp_path):
    tracks = {"01": _track(tmp_path, "one"), "02": _track(tmp_path, "two")}
    path = tmp_path / "data" / "manifest.json"
    result = generate_manifest(tracks, output=path)
    assert result["manifest_written"]
    assert result["included_tracks"] == 2
    data = json.loads(path.read_text())
    assert len(data["tracks"]) == 2
    for item in data["tracks"]:
        assert Path(item["audio"]).is_file()
        assert Path(item["beats"]).is_file()
        assert Path(item["downbeats"]).is_file()
        assert item["split"] in ("train", "validation", "test")
        assert len(item["audio_sha256"]) == 64


def test_exact_same_audio_bytes_share_split(tmp_path):
    first = _track(tmp_path, "first")
    second = _track(tmp_path, "second")
    Path(second.audio_path).write_bytes(Path(first.audio_path).read_bytes())
    manifest = tmp_path / "m.json"
    generate_manifest({"a": first, "b": second}, output=manifest)
    a, b = json.loads(manifest.read_text())["tracks"]
    assert a["split"] == b["split"]
    assert a["audio_sha256"] == b["audio_sha256"]


def test_missing_data_does_not_generate_empty_success_manifest(tmp_path):
    invalid = SimpleNamespace(audio_path=tmp_path/"missing.wav", beats_path=tmp_path/"missing.beats")
    out = tmp_path / "manifest.json"
    result = generate_manifest({"bad": invalid}, output=out)
    assert not result["manifest_written"]
    assert result["skipped_missing_audio"] == 1
    assert not out.exists()


def test_invalid_annotation_is_rejected(tmp_path):
    track = _track(tmp_path, "bad", beats="1.0 1\n0.5 2\n")
    result = generate_manifest({"bad": track}, output=tmp_path/"manifest.json")
    assert result["skipped_invalid_annotations"] == 1
    assert not result["manifest_written"]


def test_no_noncommercial_ack_fails_before_optional_dependency_import(tmp_path):
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        setup(data_home=tmp_path, manifest=tmp_path/"manifest.json",
              download=True, license_acknowledged=False)


def test_split_assignment_is_deterministic():
    sha = "1"*64
    assert split_for_hash(sha) == split_for_hash(sha)
