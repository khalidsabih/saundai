import json

import pytest

from scripts.benchmark_beat_integrity_dataset import (
    benchmark, match_events, read_timestamp_labels,
)


def test_exact_event_detection_scores_one():
    m = match_events([1.0, 1.5, 2.0], [1.0, 1.5, 2.0], tolerance_ms=70)
    assert m["matched"] == 3
    assert m["f_measure"] == 1.0
    assert m["matched_over_25ms"] == 0


def test_missing_and_spurious_events_lower_precision_and_recall():
    m = match_events([1.0, 1.5, 2.0], [1.0, 1.9, 2.5], tolerance_ms=70)
    assert m["matched"] == 1
    assert m["precision"] == pytest.approx(1/3, abs=1e-5)
    assert m["recall"] == pytest.approx(1/3, abs=1e-5)


def test_optimal_one_to_one_alignment_on_overlapping_tolerance_windows():
    # Greedy nearest-first matching can starve an earlier event; global
    # matching maximizes event count before minimizing total timing error.
    m = match_events([1.0, 1.1], [1.06, 1.12], tolerance_ms=70)
    assert m["matched"] == 2


def test_screen_40ms_without_confusing_with_crossdeck_gate():
    m = match_events([1.0, 1.5], [1.04, 1.5], tolerance_ms=70)
    assert m["f_measure"] == 1.0
    assert m["matched_over_25ms"] == 1


def test_annotation_adapter_parses_optional_beat_in_bar(tmp_path):
    annotation = tmp_path / "beats.txt"
    annotation.write_text("# annotations\n0.5 1\n1.0 2\n1.5\n")
    beats, positions = read_timestamp_labels(annotation)
    assert beats == [0.5, 1.0, 1.5]
    assert positions == [1, 2, None]


def test_duplicate_or_unsorted_labels_fail_closed(tmp_path):
    annotation = tmp_path / "bad.txt"
    annotation.write_text("1.0\n0.5\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_timestamp_labels(annotation)


def test_precomputed_fixture_proves_pipeline_without_dataset_claim(tmp_path):
    (tmp_path / "reference.txt").write_text("1.0\n1.5\n2.0\n")
    (tmp_path / "downbeats.txt").write_text("1.0\n")
    (tmp_path / "analysis.json").write_text(json.dumps({
        "beats_seconds": [1.0, 1.5, 2.0],
        "downbeats_seconds": [1.0],
        "bpm": 120.0,
    }))
    manifest = {"dataset": "synthetic-fixture-not-external-evaluation", "tracks": [
        {"id": "fixture-1", "beats": "reference.txt", "downbeats": "downbeats.txt",
         "prediction": "analysis.json", "split": "test"},
    ]}
    report = benchmark(manifest, tmp_path, selected_split="test", use_precomputed_only=True)
    assert report["evaluated_count"] == 1
    assert report["beat"]["macro_f_measure"] == 1.0
    assert report["downbeat"]["macro_f_measure"] == 1.0
    assert report["tracks"][0]["prediction_source"] == "precomputed_saund_analysis"


def test_tempo_only_labels_do_not_fabricate_downbeats(tmp_path):
    (tmp_path / "beats.txt").write_text("1.0\n1.5\n2.0\n")
    (tmp_path / "analysis.json").write_text(json.dumps({
        "beats_seconds": [1.0, 1.5, 2.0], "downbeats_seconds": [1.0],
    }))
    report = benchmark(
        {"dataset": "synthetic", "tracks": [
            {"id": "a", "beats": "beats.txt", "prediction": "analysis.json", "split": "test"},
        ]}, tmp_path, use_precomputed_only=True,
    )
    assert report["downbeat"]["status"] == "no_labels"
    assert report["tracks"][0]["downbeat"] is None


def test_explicit_split_required(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        benchmark({"tracks": [{"id": "x"}]}, tmp_path)


def test_failed_audio_path_is_reported_not_a_success(tmp_path):
    (tmp_path / "beats.txt").write_text("1.0\n1.5\n2.0\n")
    report = benchmark(
        {"tracks": [
            {"id": "x", "beats": "beats.txt", "audio": "not-there.wav", "split": "test"},
        ]}, tmp_path,
    )
    assert report["evaluated_count"] == 0
    assert report["tracks"][0]["status"] == "unavailable_or_invalid_data"
