import numpy as np

from scripts.validate_beat_sequence_interpretation import (
    inspect_sequence,
    onset_events,
    diagnose,
)


def test_onset_events_are_independent_of_cached_grid():
    fps = 22050 / 128
    x = np.full(2200, 0.2)
    x[0] = 0.0
    for t in [1.0, 1.5, 2.0, 2.5]:
        x[int(t * fps)] = 9
    peaks = onset_events(x, window_start=0, fps=fps, beat_period=0.5)
    assert len(peaks) == 4
    assert all(p["prominence_ratio"] >= 2 for p in peaks)


def test_long_interval_with_midpoint_onset_flags_missing_beat_candidate():
    beats = [0., .5, 1., 2., 2.5, 3.]
    independent = [{"seconds": 1.5, "prominence_ratio": 8.0}]
    result = inspect_sequence(
        beats, independent, start=0.5, end=2.5, expected_period=0.5,
    )
    assert result["anomalous_intervals"] == 1
    anomaly = result["anomalies"][0]
    assert anomaly["interpretations"][0]["hypothesis"] == "one_missing_beat_between_cached_events"
    assert anomaly["interpretations"][0]["status"] == "audio_supported_candidate"
    assert not result["index_change_authorized"]


def test_missing_middle_audio_remains_unresolved():
    result = inspect_sequence(
        [0., .5, 1., 2., 2.5], [], start=0, end=2.5, expected_period=0.5,
    )
    assert result["anomalies"][0]["interpretations"][0]["status"] == "unresolved"
    assert not result["index_change_authorized"]


def test_short_interval_is_not_deleted_automatically():
    result = inspect_sequence(
        [0., .5, 1.0, 1.18, 1.5, 2.0], [], start=0, end=2.0, expected_period=0.5,
    )
    assert result["anomalous_intervals"] >= 1
    assert any(i["hypothesis"] == "extra_or_misplaced_cached_beat"
               for a in result["anomalies"] for i in a["interpretations"])
    assert not result["index_change_authorized"]


def test_no_rejected_candidate_has_explicit_status(tmp_path):
    plan = {"plan_id": "test", "tracks": [], "transitions": [{"plan": {"metadata": {}}}]}
    row = diagnose(plan, tmp_path)["transitions"][0]
    assert row["status"] == "no_rejected_rhythmic_candidate"
