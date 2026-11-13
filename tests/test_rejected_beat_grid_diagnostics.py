import json
from pathlib import Path

from scripts.diagnose_rejected_beat_grids import diagnose, full_beat_cache, interval_diagnostic


def _analysis(path: str, beats: list[float]) -> dict:
    return {"path": path, "bpm": 120.0, "beats_seconds": beats}


def test_interval_detection_flags_missing_beat_gap():
    beats = [i * 0.5 for i in range(60) if i != 25]
    result = interval_diagnostic(beats, start=10.0, end=17.0, expected=0.5)
    assert result["status"] == "ok"
    assert result["unusual_interval_count"] == 1
    assert result["unusual_intervals"][0]["integer_multiple"] == 2


def test_full_cache_requires_exact_path_and_unambiguous_grid(tmp_path: Path):
    p1 = tmp_path / "tracks" / "one.json"
    p1.parent.mkdir()
    beats = [i * 0.5 for i in range(80)]
    p1.write_text(json.dumps({"analysis": _analysis("/crate/a.wav", beats)}))
    result, paths = full_beat_cache(tmp_path, "/crate/a.wav")
    assert result is not None
    assert paths == [str(p1)]
    assert full_beat_cache(tmp_path, "/elsewhere/a.wav") == (None, [])

    p2 = tmp_path / "tracks" / "two.json"
    p2.write_text(json.dumps({"analysis": _analysis("/crate/a.wav", [i * 0.6 for i in range(80)])}))
    result, paths = full_beat_cache(tmp_path, "/crate/a.wav")
    assert result is None
    assert len(paths) == 2


def test_diagnose_existing_plan_with_rejected_candidate(tmp_path: Path):
    root = tmp_path / "tracks"
    root.mkdir()
    for name in ("a", "b"):
        (root / f"{name}.json").write_text(json.dumps({
            "analysis": _analysis(f"/crate/{name}.wav", [i * 0.5 for i in range(100)])
        }))
    rejected = {
        "candidate": {
            "exit_time_a": 10.0,
            "entry_time_b": 12.0,
            "transition_seconds": 8.0,
            "target_bpm": 120.0,
            "render_rate_a": 1.0,
            "render_rate_b": 1.0,
        },
        "phase_integrity": {"safe": False, "max_residual_ms": 120.0},
    }
    plan = {
        "plan_id": "abc",
        "tracks": [{"path": "/crate/a.wav"}, {"path": "/crate/b.wav"}],
        "transitions": [{
            "from_index": 0,
            "to_index": 1,
            "plan": {"metadata": {"decision_trace_rejected_phase": rejected}},
        }],
    }
    result = diagnose(plan, tmp_path)
    row = result["transitions"][0]
    assert row["status"] == "interval_diagnostics_available"
    assert row["deck_a"]["unusual_interval_count"] == 0
    assert row["deck_b"]["unusual_interval_count"] == 0
    assert row["reported_max_residual_ms"] == 120.0


def test_missing_rejected_candidate_is_explicit(tmp_path: Path):
    p = {"tracks": [{"path": "/crate/a.wav"}], "transitions": [{
        "from_index": 0, "to_index": 0, "plan": {"metadata": {}}
    }]}
    result = diagnose(p, tmp_path)
    assert result["transitions"][0]["status"] == "no_rejected_candidate_trace"
