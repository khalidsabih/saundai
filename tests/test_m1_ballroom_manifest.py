from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import wave

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-manifest.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_manifest", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_wav(path, seconds=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * int(8000 * seconds))


def fixture(tmp_path):
    root = tmp_path / "B_1.0"
    for name, genre, seconds in (
        ("old", "ChaChaCha", 9),
        ("c1", "ChaChaCha", 9),
        ("w1", "Waltz", 9),
        ("short", "Waltz", 3),
    ):
        beats = root / "annotations" / "beats" / (name + ".beats")
        beats.parent.mkdir(parents=True, exist_ok=True)
        beats.write_text("0.5 1\n1.0 2\n1.5 3\n2.0 4\n2.5 1\n")
        tempo = root / "annotations" / "tempo" / (name + ".bpm")
        tempo.parent.mkdir(parents=True, exist_ok=True)
        tempo.write_text("120\n")
        write_wav(root / "audio" / genre / (name + ".wav"), seconds)
    return root


def test_importer_preserves_independent_labels_and_excludes_prior(tmp_path):
    m = load()
    root = fixture(tmp_path)
    manifest, audit = m.build(root, {"old"}, count=2)
    assert {i["id"] for i in manifest["items"]} == {"c1", "w1"}
    assert manifest["rights_confirmed"] is False
    assert {i["genre"] for i in manifest["items"]} == {"ChaChaCha", "Waltz"}
    assert audit["source_counts"]["too_short"] == 1
    assert audit["source_counts"]["excluded_prior_pilot"] == 1
    for item in manifest["items"]:
        assert item["bpm"] == 120
        assert item["beats_seconds"] == [0.5, 1, 1.5, 2, 2.5]
        assert item["downbeats_seconds"] == [0.5, 2.5]
        assert not Path(item["audio"]).is_absolute()


def test_importer_deterministic_and_never_overstating_availability(tmp_path):
    m = load()
    root = fixture(tmp_path)
    a, _ = m.build(root, {"old"}, count=2, seed=11)
    b, _ = m.build(root, {"old"}, count=2, seed=11)
    assert a == b
    with pytest.raises(ValueError, match="only 2 eligible"):
        m.build(root, {"old"}, count=3)


def test_bad_annotation_fails_closed(tmp_path):
    m = load()
    path = tmp_path / "bad.beats"
    path.write_text("0.5 1\n0.4 2\n")
    with pytest.raises(ValueError, match="nonmonotonic"):
        m.read_beats(path)
    path.write_text("0.5 1\n1.0 0\n")
    with pytest.raises(ValueError, match="invalid beat index"):
        m.read_beats(path)
