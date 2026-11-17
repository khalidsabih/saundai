"""Independent music regression: fail-closed split and one-to-one timing tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def load():
    p = Path(__file__).resolve().parents[1] / "scripts" / "m1-beat-this-independent-regression.py"
    spec = importlib.util.spec_from_file_location("m1_external_independent_regression", p)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_fixture(tmp_path):
    sr = 8000
    sf.write(tmp_path / "a.wav", np.zeros(sr * 5, dtype=np.float32), sr)
    (tmp_path / "reference.txt").write_text("0.5\n1.0\n1.5\n2.0\n", encoding="utf-8")
    (tmp_path / "downbeats.txt").write_text("0.5\n2.0\n", encoding="utf-8")
    (tmp_path / "beat_this.beats").write_text(
        "0.5\t1\n1.0\t2\n1.5\t3\n2.0\t1\n", encoding="utf-8"
    )
    (tmp_path / "saund.beats").write_text(
        "0.75\t1\n1.25\t2\n1.75\t3\n2.25\t1\n", encoding="utf-8"
    )
    return {
        "id": "rights-cleared-example", "split": "validation",
        "audio": "a.wav", "reference_beats": "reference.txt",
        "reference_downbeats": "downbeats.txt",
        "beat_this_predictions": "beat_this.beats",
        "saund_predictions": "saund.beats",
    }


def test_validation_comparison_scores_saved_predictions_only(tmp_path):
    module = load()
    manifest = {"tracks": [setup_fixture(tmp_path)]}
    report = module.run(manifest, tmp_path)
    assert report["selected_split"] == "validation"
    assert report["paired_baseline_track_count"] == 1
    assert report["tracks"][0]["beat_this_beats"]["f1"] == 1.0
    assert report["tracks"][0]["saund_beats"]["f1"] == 0.0
    assert report["tracks"][0]["beat_this_downbeats"]["f1"] == 1.0
    assert report["reference_used_for_inference"] is False
    assert report["production_changed"] is False


@pytest.mark.parametrize("split", ["test", "sealed", "train", None])
def test_nonvalidation_splits_are_rejected_without_access(tmp_path, split):
    module = load()
    item = setup_fixture(tmp_path)
    item["split"] = split
    with pytest.raises(ValueError, match="only explicit validation split"):
        module.run({"tracks": [item]}, tmp_path)


def test_duplicate_id_and_empty_corpus_rejected(tmp_path):
    module = load()
    item = setup_fixture(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        module.run({"tracks": [item, item]}, tmp_path)
    with pytest.raises(ValueError, match="no validation"):
        module.run({"tracks": []}, tmp_path)


def test_no_reference_as_inference_or_implicit_data(tmp_path):
    module = load()
    item = setup_fixture(tmp_path)
    (tmp_path / "reference.txt").unlink()
    with pytest.raises(ValueError, match="missing independent reference"):
        module.run({"tracks": [item]}, tmp_path)


def test_matching_tolerance_is_research_only():
    module = load()
    score = module.evaluate_events([.54, 1.04], [.5, 1.])
    assert score["matched_count"] == 2
    assert score["f1"] == 1.
    assert score["reference_events_not_matched_within_25ms"] == 2
    assert score["25ms_diagnostic_is_not_planner_gate"]
