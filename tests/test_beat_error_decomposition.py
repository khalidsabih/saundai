import json

import pytest

from scripts.diagnose_beat_error_decomposition import (
    decompose_track, diagnose, diagnostic_downbeat_phase_search,
    event_rate_diagnosis, paired_offsets, reference_meter,
)


def _seq(period, n, start=1.0):
    return [start+i*period for i in range(n)]


def test_signed_offsets_and_drift_are_descriptive():
    ref = _seq(.5, 30)
    predicted = [t+.04 for t in ref]
    report = paired_offsets(ref, predicted)
    assert report["status"] == "measured"
    assert report["signed_median_ms"] == pytest.approx(40.0)
    assert report["over_25ms_count"] == 30
    assert report["late_count"] == 30
    assert report["early_count"] == 0
    assert abs(report["estimated_offset_slope_ms_per_s"]) < .001


def test_no_pairs_gives_explicit_unknown():
    report = paired_offsets(_seq(.5, 20), [t+.24 for t in _seq(.5, 20)])
    assert report["status"] == "no_close_pairs"


def test_half_and_double_time_are_not_ground_truth():
    ref = _seq(.5, 40)
    slow = event_rate_diagnosis(ref, _seq(1.0, 20))
    fast = event_rate_diagnosis(ref, _seq(.25, 80))
    assert slow["class"] == "possible_half_time"
    assert fast["class"] == "possible_double_time"
    assert slow["not_confirmed_musical_tempo_error"]


def test_reference_meter_infers_three_or_four_without_claiming_detector():
    beats3 = _seq(.5, 42, start=0.0)
    down3 = beats3[::3]
    result = reference_meter(down3, beats3)
    assert result["meter"] == 3
    assert result["not_an_autonomous_meter_prediction"]
    beats4 = _seq(.5, 40, start=0.0)
    assert reference_meter(beats4[::4], beats4)["meter"] == 4


def test_downbeat_oracle_is_explicitly_ground_truth_assisted():
    beats = _seq(.5, 40, start=0.0)
    oracle = diagnostic_downbeat_phase_search(
        beats, beats[2::4], reference_meter_value=4,
    )
    assert oracle["best_phase"] == 2
    assert oracle["best_phase_f_measure"] == 1.0
    assert oracle["oracle_used_ground_truth"]
    assert oracle["not_a_deployable_downbeat_method"]


def test_no_downbeat_labels_prevents_oracle():
    result = diagnostic_downbeat_phase_search(_seq(.5, 20), None, reference_meter_value=None)
    assert result["status"] == "missing_reference_or_metre"


def test_validation_only_and_no_test_peeking(tmp_path):
    (tmp_path/"beats.txt").write_text("\n".join(str(x) for x in _seq(.5, 32)))
    (tmp_path/"downbeats.txt").write_text("\n".join(str(x) for x in _seq(2.0, 8)))
    (tmp_path/"prediction.json").write_text(json.dumps({
        "beats_seconds": _seq(.5, 32),
        "downbeats_seconds": _seq(2.0, 8),
        "bpm": 120,
    }))
    samples = [
        {"id": "val", "split": "validation", "beats": "beats.txt",
         "downbeats": "downbeats.txt", "prediction": "prediction.json"},
        {"id": "test", "split": "test", "beats": "DOES_NOT_EXIST",
         "prediction": "DOES_NOT_EXIST"},
    ]
    output = diagnose({"dataset": "synthetic-fixture", "tracks": samples},
                      tmp_path, precomputed_only=True)
    assert output["selected_split"] == "validation"
    assert output["evaluated_count"] == 1
    assert len(output["tracks"]) == 1
    assert output["tracks"][0]["beat"]["f_measure"] == 1.0
    assert output["tracks"][0]["reference_meter"]["meter"] == 4
    assert output["tracks"][0]["downbeat_phase_oracle"]["best_phase_f_measure"] == 1.0
    assert output["production_changes"] is False


def test_missing_audio_cannot_be_success(tmp_path):
    (tmp_path/"beats.txt").write_text("\n".join(str(x) for x in _seq(.5, 20)))
    output = decompose_track({"id":"missing", "beats":"beats.txt",
                              "audio":"unavailable.wav"}, tmp_path)
    assert output["status"] == "unavailable_or_invalid_data"


def test_duplicate_track_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        diagnose({"tracks":[{"id":"a","split":"validation"},
                             {"id":"a","split":"test"}]}, tmp_path)
