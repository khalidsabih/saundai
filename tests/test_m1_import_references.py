from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-import-references.py"
    spec = importlib.util.spec_from_file_location("m1_import_references", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_giantsteps_v2_bpm_and_missing_annotations(tmp_path):
    m = load()
    f = tmp_path / "1030011.LOFI.bpm"
    f.write_text("127.0\n", encoding="utf-8")
    assert m.giantsteps_bpm(f) == 127.0
    f.write_text("0\n", encoding="utf-8")
    assert m.giantsteps_bpm(f) is None
    f.write_text("nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        m.giantsteps_bpm(f)


def test_ballroom_beat_and_downbeat_positional_labels(tmp_path):
    m = load()
    f = tmp_path / "Albums-AnaBelen_Veneo-01.beats"
    f.write_text("0.1944671200 4\n0.8499773240 1\n1.4700226750 2\n2.0800000000 3\n", encoding="utf-8")
    beats, downbeats = m.ballroom_beats(f)
    assert beats == pytest.approx([0.19446712, 0.849977324, 1.470022675, 2.08])
    assert downbeats == pytest.approx([0.849977324])


def test_ballroom_invalid_and_missing_annotations(tmp_path):
    m = load()
    f = tmp_path / "sample.beats"
    f.write_text("2.0 1\n1.0 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsorted"):
        m.ballroom_beats(f)
    f.write_text("-1.0 0\n", encoding="utf-8")
    assert m.ballroom_beats(f) == (None, None)
    f.write_text("0.1 junk\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bar-position"):
        m.ballroom_beats(f)


def test_giantsteps_import_produces_unapproved_development_manifest(tmp_path):
    m = load()
    ann = tmp_path / "annotations"
    labeldir = ann / "annotations_v2" / "tempo"
    labeldir.mkdir(parents=True)
    (labeldir / "1030011.LOFI.bpm").write_text("127.0\n", encoding="utf-8")
    (labeldir / "1068430.LOFI.bpm").write_text("0\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "1030011.LOFI.mp3").write_bytes(b"audio-placeholder")
    output = tmp_path / "output.json"
    assert m.main([
        "--dataset", "giantsteps_tempo", "--annotations", str(ann),
        "--audio-root", str(audio), "--output", str(output)
    ]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["rights_confirmed"] is False
    assert data["split"] == "development"
    assert len(data["items"]) == 1
    assert data["items"][0]["bpm"] == 127.0
    assert data["items"][0]["beats_seconds"] is None
    assert data["items"][0]["downbeats_seconds"] is None
    with pytest.raises(ValueError, match="overwrite"):
        m.main([
            "--dataset", "giantsteps_tempo", "--annotations", str(ann),
            "--audio-root", str(audio), "--output", str(output)
        ])


def test_ballroom_import_joins_by_full_stem_and_preserves_unknown_tempo(tmp_path):
    m = load()
    ann = tmp_path / "ballroom"
    ann.mkdir()
    (ann / "Albums-AnaBelen_Veneo-01.beats").write_text("0.1 1\n0.6 2\n1.1 3\n1.6 4\n", encoding="utf-8")
    audio = tmp_path / "audio"
    (audio / "Waltz").mkdir(parents=True)
    (audio / "Waltz" / "Albums-AnaBelen_Veneo-01.wav").write_bytes(b"audio-placeholder")
    out = tmp_path / "ballroom-dev.json"
    assert m.main(["--dataset", "ballroom", "--annotations", str(ann),
                   "--audio-root", str(audio), "--output", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["items"][0]["audio"] == "Waltz/Albums-AnaBelen_Veneo-01.wav"
    assert data["items"][0]["downbeats_seconds"] == [0.1]
    assert data["items"][0]["bpm"] is None


def test_missing_audio_causes_empty_import_failure(tmp_path):
    m = load()
    ann = tmp_path / "annotations"
    (ann / "annotations_v2" / "tempo").mkdir(parents=True)
    (ann / "annotations_v2" / "tempo" / "1030011.LOFI.bpm").write_text("127.0", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    with pytest.raises(ValueError, match="no annotated audio matched"):
        m.import_sources("giantsteps_tempo", ann, audio, version="v2")


def test_ambiguous_audio_filename_fails_closed(tmp_path):
    m = load()
    audio = tmp_path / "audio"
    (audio / "genre-one").mkdir(parents=True)
    (audio / "genre-two").mkdir(parents=True)
    (audio / "genre-one" / "song.wav").write_bytes(b"a")
    (audio / "genre-two" / "song.wav").write_bytes(b"b")
    with pytest.raises(ValueError, match="ambiguous"):
        m.find_audio(audio, "song")
