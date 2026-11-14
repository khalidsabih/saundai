import pytest

from scripts.diagnose_piecewise_beat_continuity import (
    inspect_piecewise, diagnose,
)


def _beats(n=80, period=0.5):
    return [i * period for i in range(n)]


def _events(times):
    return [{"seconds": t, "prominence_ratio": 6.0} for t in times]


def test_regular_grid_has_one_stable_run_and_no_phase_junctions():
    beats = _beats()
    report = inspect_piecewise(beats, _events(beats), start=10, end=20,
                               expected_period=0.5)
    assert report["status"] == "audited"
    assert report["stable_run_count"] == 1
    assert report["abnormal_interval_count"] == 0
    assert report["junctions"] == []
    assert not report["beat_grid_change_authorized"]


def test_three_compressed_intervals_create_measurable_run_phase_change():
    beats = _beats()
    for idx in range(29, len(beats)):
        beats[idx] -= 0.07
    for idx in range(30, len(beats)):
        beats[idx] -= 0.07
    for idx in range(31, len(beats)):
        beats[idx] -= 0.09
    report = inspect_piecewise(beats, _events(beats), start=11, end=20,
                               expected_period=0.5)
    assert report["abnormal_interval_count"] >= 2
    assert report["stable_run_count"] >= 2
    assert any(abs(j["relative_phase_shift_ms"] + 230) < 1
               for j in report["junctions"])
    assert not report["beat_grid_change_authorized"]


def test_constant_offset_is_not_classified_as_interval_anomaly():
    beats = [i * 0.5 + 0.12 for i in range(80)]
    report = inspect_piecewise(beats, [], start=10, end=20,
                               expected_period=0.5)
    assert report["abnormal_interval_count"] == 0
    assert report["stable_run_count"] == 1


def test_half_beat_subdivisions_are_reported_as_ambiguous():
    beats = _beats()
    events = _events(sorted(set(beats + [t+0.25 for t in beats])))
    report = inspect_piecewise(beats, events, start=10, end=20,
                               expected_period=0.5)
    assert report["stable_runs"][0]["half_beat_ambiguity"]
    assert not report["stable_runs"][0]["phase_choice_authorized"]


def test_invalid_grid_is_rejected():
    assert inspect_piecewise([0, 0, 0.5], [], start=0, end=4,
                             expected_period=0.5)["status"] == "invalid_input"


def test_plan_id_mismatch_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="Plan ID"):
        diagnose({"plan_id": "one"}, {"plan_id": "two"}, tmp_path)


def test_no_rejected_candidate_is_explicit(tmp_path):
    result = diagnose(
        {"plan_id": "same", "transitions": [{}]},
        {"plan_id": "same", "transitions": []}, tmp_path,
    )
    assert result["transitions"][0]["status"] == "no_rejected_rhythmic_candidate"
