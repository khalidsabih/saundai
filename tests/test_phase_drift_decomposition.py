import json
from pathlib import Path

from scripts.reconstruct_phase_drift import diagnose, reconstruct


def _beats(start: float = 0, length: int = 160, spacing: float = 0.5):
    return [round(start + i * spacing, 6) for i in range(length)]


def test_perfect_constant_grid_has_negligible_drift():
    result = reconstruct(
        _beats(), _beats(), exit_time_a=10.0, entry_time_b=14.0,
        duration=8, bpm=120.0, rate_a=1.0, rate_b=1.0,
    )
    assert result["valid_samples"] == 9
    assert result["max_abs_nominal_drift_ms"] < 0.001
    assert result["max_abs_detected_grid_drift_ms"] < 0.001
    assert all(abs(x["decomposition_error_ms"]) < 0.001 for x in result["phase_points"] if x.get("valid"))


def test_detected_grid_change_exposes_index_vs_time_discrepancy():
    a = _beats()
    b = [round(i * 0.505, 6) for i in range(160)]
    result = reconstruct(
        a, b, exit_time_a=10.0, entry_time_b=10.0,
        duration=10.0, bpm=120.0, rate_a=1.0, rate_b=1.0,
    )
    assert result["max_abs_nominal_drift_ms"] < 0.001
    assert result["max_abs_detected_grid_drift_ms"] > 50
    assert all(abs(x["decomposition_error_ms"]) < 0.001 for x in result["phase_points"] if x.get("valid"))


def test_render_rates_matching_different_source_bpms_cancel_nominal_drift():
    a = _beats(spacing=0.5)
    b = _beats(spacing=0.51)
    result = reconstruct(
        a, b, exit_time_a=10.0, entry_time_b=10.2,
        duration=9.0, bpm=120.0, rate_a=1.0, rate_b=1.02,
    )
    assert result["max_abs_nominal_drift_ms"] < 0.001
    assert result["max_abs_detected_grid_drift_ms"] < 0.001


def test_diagnose_uses_existing_full_grid_caches(tmp_path: Path):
    cache = tmp_path / "tracks"
    cache.mkdir()
    for i, name in enumerate(("a", "b")):
        (cache / (name + ".json")).write_text(json.dumps({
            "analysis": {"path": "/crate/" + name + ".wav", "bpm": 120, "beats_seconds": _beats()}
        }), encoding="utf-8")
    plan = {
        "plan_id": "abc",
        "tracks": [{"path": "/crate/a.wav"}, {"path": "/crate/b.wav"}],
        "transitions": [{
            "from_index": 0, "to_index": 1,
            "plan": {"metadata": {"decision_trace_rejected_phase": {
                "candidate": {
                    "signature": "candidate-1", "exit_time_a": 10.0,
                    "entry_time_b": 14.0, "transition_seconds": 8.0,
                    "target_bpm": 120, "render_rate_a": 1.0,
                    "render_rate_b": 1.0,
                    "phase_nudge_seconds_b": 0.0,
                },
                "phase_integrity": {
                    "max_phase_error_ms": 0.0,
                    "points": [{"phase_error_ms": 0.0} for _ in range(9)],
                },
            }}},
        }],
    }
    item = diagnose(plan, cache)["transitions"][0]
    assert item["status"] == "reconstructed"
    assert item["replay_matches_logged_1ms"] is True
    assert item["nudge_known"] is True
    assert item["reconstruction"]["max_abs_detected_grid_drift_ms"] == 0.0


def test_no_trace_is_explicit(tmp_path: Path):
    data = {"tracks": [{"path": "/crate/a.wav"}], "transitions": [{
        "from_index": 0, "to_index": 0, "plan": {"metadata": {}},
    }]}
    assert diagnose(data, tmp_path)["transitions"][0]["status"] == "missing_candidate_or_indices"
