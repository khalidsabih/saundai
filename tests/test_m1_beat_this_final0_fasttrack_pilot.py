"""No-download final0 audio pilot security/benchmark preflight tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def module():
    src = Path(__file__).resolve().parents[1] / "scripts/m1-beat-this-final0-fasttrack-pilot.py"
    spec = importlib.util.spec_from_file_location("m1_final0_fasttrack", src)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixtures(tmp_path, *, split="development", rights=True):
    root = tmp_path / "audio"
    root.mkdir()
    sr = 8000
    sf.write(root / "one.wav", np.zeros(sr * 5, dtype=np.float32), sr)
    item = {
        "id": "one",
        "audio": "one.wav",
        "annotation_source": "independent local test labels",
        "beats_seconds": [0.5, 1., 1.5, 2.],
        "downbeats_seconds": [0.5, 2.],
    }
    manifest = {
        "schema_version": "saund-m1-rhythm-v1",
        "dataset_id": "ballroom",
        "split": split,
        "rights_confirmed": rights,
        "items": [item],
    }
    checkpoint = tmp_path / "final0.ckpt"
    checkpoint.write_bytes(b"fake weights")
    cli = tmp_path / "beat_this"
    cli.write_text("fake CLI")
    return root, manifest, checkpoint, cli


def test_preflight_requires_reviewed_development_or_validation(tmp_path):
    m = module()
    root, manifest, cp, cli = fixtures(tmp_path)
    items, _ = m.check_inputs(manifest, root, cp, cli, tmp_path / "new")
    assert len(items) == 1
    for split in ("test", "sealed", "train"):
        with pytest.raises(ValueError, match="development/validation"):
            m.check_inputs({**manifest, "split": split}, root, cp, cli, tmp_path / "new")


def test_rights_missing_or_false_never_runs_inference(tmp_path):
    m = module()
    root, manifest, cp, cli = fixtures(tmp_path)
    with pytest.raises(ValueError, match="rights"):
        m.check_inputs({**manifest, "rights_confirmed": False}, root, cp, cli, tmp_path / "new")


def test_outputs_and_checkpoint_must_be_explicit_and_new(tmp_path):
    m = module()
    root, manifest, cp, cli = fixtures(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        m.check_inputs(manifest, root, cp, cli, tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        m.check_inputs(manifest, root, tmp_path / "missing", cli, tmp_path / "new")


def test_audio_path_escape_refused(tmp_path):
    m = module()
    root, manifest, cp, cli = fixtures(tmp_path)
    manifest["items"][0]["audio"] = "../unauthorized.wav"
    with pytest.raises(ValueError, match="escapes root"):
        m.check_inputs(manifest, root, cp, cli, tmp_path / "new")


def test_baseline_env_disables_neural_and_network():
    m = module()
    env = m.command_env()
    assert env["DJAI_NEURAL_BEATS"] == "off"
    assert env["DJAI_MODEL_ALLOW_DOWNLOAD"] == "0"
    assert env["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] == "0"
    assert env["DJAI_EMBEDDINGS_ENABLED"] == "0"


def test_independent_score_does_not_need_inference(tmp_path):
    m = module()
    scorer = m.load_script("m1-beat-this-independent-regression.py", "unit_fasttrack_metric")
    ref = {"beats_seconds": [0.5, 1., 1.5, 2.],
           "downbeats_seconds": [0.5, 2.]}
    baseline = {"beats_seconds": [.75, 1.25, 1.75, 2.25],
                "downbeats_seconds": [.75, 2.25]}
    challenger = {"beats": [.5, 1., 1.5, 2.],
                  "downbeats": [.5, 1., 1.5, 2.]}
    result = m.independent_scores(ref, baseline, challenger, scorer)
    assert result["final0_beats"]["f1"] == 1.
    assert result["baseline_beats"]["f1"] == 0.
    assert result["final0_downbeats"]["f1"] < 1.
    assert result["beat_regression"] is False


def test_unsafe_track_id_does_not_get_file_path():
    m = module()
    with pytest.raises(ValueError, match="unsafe track ID"):
        m._safe_id("../../etc/passwd")
    assert m._safe_id("Albums-Ballroom_Classics4-06") == "Albums-Ballroom_Classics4-06"


def test_no_output_if_required_checkpoint_is_missing(tmp_path):
    m = module()
    root, manifest, cp, cli = fixtures(tmp_path)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))
    cp.unlink()
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="checkpoint"):
        m.run(manifest_file, root, cli, cp, out)
    assert not out.exists()
