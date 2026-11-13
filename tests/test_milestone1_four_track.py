from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.milestone1_four_track import (
    _transition_kind,
    prepare_four_track_crate,
    summarize_four_track,
)


def _edge(a: int, b: int, **plan):
    return {
        "from_index": a,
        "to_index": b,
        "plan": {
            "technique": "equal_power_blend",
            "transition_bars": 8,
            "beatmatched": True,
            "measurement_passed": True,
            "measured_score": .98,
            "fallback_level": 0,
            "metadata": {},
            **plan,
        },
    }


def test_four_track_summary_does_not_confuse_safe_with_musical_quality():
    plan = {
        "tracks": [{"title": name} for name in ("A", "B", "C", "D")],
        "transitions": [
            _edge(0, 1),
            _edge(
                1, 2,
                technique="quick_cut",
                fallback_level=1,
                metadata={"rescue_transition_kind": "structural_reset"},
            ),
            _edge(
                2, 3,
                fallback_level=4,
                fallback_reason="No evidence matrix remained",
                metadata={"deterministic_final_escape": True},
            ),
        ],
    }
    phase = {
        "summary": {"phase16_passed": False},
        "runs": [{
            "render_passed": True,
            "production_safe": False,
            "rehearsal_ready": False,
            "hard_failure_count": 0,
            "full_set_benchmark": {"story_fit": .23},
        }],
    }
    summary = summarize_four_track(phase, plan, None)
    assert summary["tracks"] == 4
    assert summary["complete_four_track_set"] is True
    assert [item["classification"] for item in summary["per_transition"]] == [
        "normal_transition", "structural_reset", "emergency_escape"
    ]
    assert summary["transition_classifications"]["normal_transition"] == 1
    assert summary["transition_classifications"]["emergency_escape"] == 1
    assert summary["musically_certified"] is False
    assert summary["render_passed"] is True
    assert summary["phase16_passed"] is False


def test_no_transition_output_is_not_a_success():
    phase = {"summary": {"phase16_passed": False}, "runs": [{"tracks": 4}]}
    summary = summarize_four_track(phase, None, None)
    assert summary["complete_four_track_set"] is False
    assert summary["normal_transition_rate"] is None
    assert summary["musically_certified"] is False


def test_four_track_crate_preserves_sources(tmp_path: Path):
    sources = [tmp_path / f"source-{i}.wav" for i in range(4)]
    for idx, path in enumerate(sources):
        path.write_bytes(bytes([idx + 10]) * 20)
    destination = tmp_path / "crate"
    prepare_four_track_crate(sources, destination)
    assert len(list(destination.iterdir())) == 4
    for source in sources:
        assert (destination / source.name).read_bytes() == source.read_bytes()


def test_four_track_crate_rejects_duplicate_paths(tmp_path: Path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"audio")
    with pytest.raises(ValueError, match="distinct"):
        prepare_four_track_crate([p, p, p, p], tmp_path / "crate")


def test_four_track_crate_rejects_duplicate_names(tmp_path: Path):
    paths = []
    for i in range(4):
        directory = tmp_path / f"folder{i}"
        directory.mkdir()
        p = directory / ("duplicate.wav" if i < 2 else f"track{i}.wav")
        p.write_bytes(b"audio")
        paths.append(p)
    with pytest.raises(ValueError, match="Duplicate audio filename"):
        prepare_four_track_crate(paths, tmp_path / "crate")
