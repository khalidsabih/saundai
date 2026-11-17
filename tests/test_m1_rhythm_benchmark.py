from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-rhythm-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_rhythm_benchmark", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    (root / "a.wav").write_bytes(b"fixture-only")
    manifest = {
        "schema_version": "saund-m1-rhythm-v1",
        "dataset_id": "ballroom",
        "split": "development",
        "rights_confirmed": True,
        "items": [{
            "id": "a",
            "audio": "a.wav",
            "annotation_source": "independent-fixture",
            "bpm": 120.0,
            "beats_seconds": [0.5, 1.0, 1.5, 2.0],
            "downbeats_seconds": [0.5, 2.0],
        }],
    }
    return root, manifest


def test_event_matching_one_to_one():
    m = load()
    assert m.count_matches([0.5, 0.55, 1.0], [0.52, 1.01], 0.07) == 2
    assert m.count_matches([0.5], [], 0.07) == 0
    assert m.score_events([{"match": 2, "reference": 4, "predicted": 2}])["f1"] == 0.666667


def test_real_metric_math_using_independent_fixture(tmp_path):
    m = load()
    root, reference = fixture(tmp_path)
    assert m.validate_manifest(reference, root)
    pred = {"a": {"bpm": 122.0,
                  "beats_seconds": [0.5, 1.02, 1.49, 2.05],
                  "downbeats_seconds": [0.51, 2.03]}}
    result = m.evaluate(reference, pred)
    assert result["metrics"] == {"tempo_mae_bpm": 2.0, "beat_f1": 1.0, "downbeat_f1": 1.0}
    assert result["metric_coverage"]["tempo"] == 1


def test_missing_downbeat_truth_not_scored_as_zero(tmp_path):
    m = load()
    root, ref = fixture(tmp_path)
    ref["items"][0]["downbeats_seconds"] = None
    m.validate_manifest(ref, root)
    result = m.evaluate(ref, {"a": {
        "bpm": 120.0,
        "beats_seconds": [0.5, 1.0, 1.5, 2.0],
        "downbeats_seconds": []
    }})
    assert result["metrics"]["downbeat_f1"] is None
    assert result["metric_coverage"]["downbeat"] == 0


def test_fail_closed_on_missing_rights_and_path_escape(tmp_path):
    m = load()
    root, ref = fixture(tmp_path)
    ref["rights_confirmed"] = False
    with pytest.raises(ValueError, match="rights"):
        m.validate_manifest(ref, root)
    ref["rights_confirmed"] = True
    ref["items"][0]["audio"] = "../untracked.wav"
    with pytest.raises(ValueError, match="escapes root"):
        m.validate_manifest(ref, root)


def test_validate_annotation_provenance_and_sorted_events(tmp_path):
    m = load()
    root, ref = fixture(tmp_path)
    ref["items"][0]["beats_seconds"] = [1.0, 0.5]
    with pytest.raises(ValueError, match="strictly increasing"):
        m.validate_manifest(ref, root)
    ref["items"][0]["beats_seconds"] = [0.5, 1.0]
    ref["items"][0]["annotation_source"] = ""
    with pytest.raises(ValueError, match="source"):
        m.validate_manifest(ref, root)


def test_evaluate_rejects_missing_predictions(tmp_path):
    m = load()
    root, ref = fixture(tmp_path)
    m.validate_manifest(ref, root)
    with pytest.raises(ValueError, match="missing prediction"):
        m.evaluate(ref, {})
