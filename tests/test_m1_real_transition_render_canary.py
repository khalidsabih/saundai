from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-real-transition-render-canary.py"
    spec = importlib.util.spec_from_file_location("m1_real_transition_render_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(pairs=None, *, render_permission=True, split="development"):
    return {
        "schema_version": "saund-m1-render-canary-v1",
        "split": split,
        "rights_confirmed": True,
        "render_permission_confirmed": render_permission,
        "pairs": pairs if pairs is not None else [
            {"id": "pair-1", "track_a": "a.wav", "track_b": "b.wav"}
        ],
    }


def test_requires_separately_approved_render_rights(tmp_path):
    m = load()
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    with pytest.raises(ValueError, match="rendering rights"):
        m.validate_manifest(manifest(render_permission=False), tmp_path)


def test_rejects_holdout_split(tmp_path):
    m = load()
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    with pytest.raises(ValueError, match="non-test development"):
        m.validate_manifest(manifest(split="test"), tmp_path)


def test_approved_pair_uses_only_existing_relative_sources(tmp_path):
    m = load()
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    rows = m.validate_manifest(manifest(), tmp_path)
    assert len(rows) == 1
    assert rows[0][0] == "pair-1"


def test_no_path_traversal(tmp_path):
    m = load()
    (tmp_path / "b.wav").touch()
    candidate = manifest([{"id": "pair-1", "track_a": "../secret.wav", "track_b": "b.wav"}])
    with pytest.raises(ValueError, match="missing or disallowed"):
        m.validate_manifest(candidate, tmp_path)


def test_no_duplicate_pair_ids(tmp_path):
    m = load()
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    duplicate = manifest([{"id": "pair-1", "track_a": "a.wav", "track_b": "b.wav"}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        m.validate_manifest(duplicate, tmp_path)


def test_grid_difference_detects_selected_tempo_mismatch():
    m = load()
    class Track:
        bpm = 90
        beats_seconds = [0, .5, 1, 1.5, 2]
    assert abs(m.grid_difference(Track()) - 30) < .0001


def test_audio_metrics_detect_silent_and_clipped_previews():
    import numpy as np

    m = load()
    class Preview:
        sample_rate = 100
        pre = np.ones((2, 10), dtype=np.float32) * .1
        transition = np.ones((2, 10), dtype=np.float32) * .1
        post = np.ones((2, 10), dtype=np.float32) * .1
        preview = np.concatenate([pre, transition, post], axis=-1)

    r = Preview()
    assert m.audio_metrics(r)["passed"]
    r.preview = np.ones((2, 30), dtype=np.float32) * 1.1
    assert not m.audio_metrics(r)["passed"]
    r.preview = np.zeros((2, 30), dtype=np.float32)
    assert not m.audio_metrics(r)["passed"]


def test_refuses_to_run_without_operator_confirmation(tmp_path):
    m = load()
    with pytest.raises(ValueError, match="confirm-render-rights"):
        m.main([
            "--manifest", str(tmp_path / "unread.json"),
            "--audio-root", str(tmp_path),
            "--output-dir", str(tmp_path / "output"),
        ])
