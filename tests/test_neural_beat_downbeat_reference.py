from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark_neural_beat_downbeat import (
    prepare_manifest, evaluate,
)


def test_prepare_only_declares_unfilled_label_paths(tmp_path: Path):
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "song.wav").write_bytes(b"dummy")
    manifest_path = tmp_path / "manifest.json"
    prepared = prepare_manifest(audio, [{
        "filename": "song.wav",
        "reference_bpm": 120,
        "reference_quality": "approximate",
    }], manifest_path)
    assert prepared["tracks"][0]["annotation_method"] == ""
    assert prepared["tracks"][0]["beats"] == "labels/track-01.beats.txt"
    assert not (tmp_path / "labels").exists()
    with pytest.raises(FileExistsError):
        prepare_manifest(audio, [{
            "filename": "song.wav", "reference_bpm": 120,
        }], manifest_path)


def test_missing_annotations_fail_closed_before_audio_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import scripts.benchmark_neural_beat_downbeat as benchmark
    monkeypatch.setattr(benchmark, "deterministic_fallback_grid",
                        lambda path: pytest.fail("must not analyze unlabelled audio"))
    manifest = {"tracks": [{
        "id": "one", "filename": "song.wav", "audio": "song.wav",
        "beats": "labels/one.beats.txt",
        "downbeats": "labels/one.downbeats.txt",
        "split": "validation", "annotation_method": "",
    }]}
    report = evaluate(manifest, tmp_path)
    assert report["evaluated_count"] == 0
    assert report["missing_annotation_count"] == 1
    assert report["neural"]["beat"]["macro_f_measure"] is None
    assert report["ready_for_live_adoption"] is False


def test_paired_reference_scores_distinguish_beat_and_downbeat_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import scripts.benchmark_neural_beat_downbeat as benchmark
    (tmp_path / "track.wav").write_bytes(b"fixture")
    (tmp_path / "beats.txt").write_text("1.0\n1.5\n2.0\n2.5\n")
    (tmp_path / "downbeats.txt").write_text("1.0\n2.5\n")
    baseline = {"beat_times": (1., 1.5, 2., 2.5), "downbeat_times": (1.1, 2.6)}
    monkeypatch.setattr(benchmark, "deterministic_fallback_grid", lambda _: baseline)
    monkeypatch.setattr(
        "djai.intelligence.neural_ears.predict_beats",
        lambda path, **kwargs: SimpleNamespace(
            available=True, beats_seconds=(1., 1.5, 2., 2.5),
            downbeats_seconds=(1., 2.5), used=False,
        ),
    )
    manifest = {"tracks": [{
        "id": "one", "filename": "track.wav", "audio": str(tmp_path / "track.wav"),
        "beats": "beats.txt", "downbeats": "downbeats.txt",
        "split": "validation", "annotation_method": "manual external DAW beat markers",
    }]}
    report = evaluate(manifest, tmp_path)
    assert report["evaluated_count"] == 1
    assert report["deterministic"]["beat"]["macro_f_measure"] == 1.0
    assert report["neural"]["beat"]["macro_f_measure"] == 1.0
    assert report["neural"]["downbeat"]["macro_f_measure"] == 1.0
    assert report["deterministic"]["downbeat"]["macro_f_measure"] <= 1.0
    assert report["tracks"][0]["status"] == "evaluated"
    assert report["live_adoptions"] == 0


def test_duplicate_audio_basename_is_not_silently_selected(tmp_path: Path):
    root = tmp_path / "audio"
    for folder in ("a", "b"):
        (root / folder).mkdir(parents=True)
        (root / folder / "duplicate.wav").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="expected one audio basename"):
        prepare_manifest(root, [{
            "filename": "duplicate.wav", "reference_bpm": 120,
        }], tmp_path / "manifest.json")
