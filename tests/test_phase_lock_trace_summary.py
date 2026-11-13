from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_phase_lock_trace import summarize


def test_summary_deduplicates_rejections_and_excludes_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "phase.jsonl"
    base = {
        "track_a": "a.wav", "track_b": "b.wav",
        "exit_time_a_seconds": 20.0, "entry_time_b_seconds": 8.0,
        "transition_seconds": 16.0, "render_rate_a": 1.0,
        "render_rate_b": 1.0, "phase_nudge_seconds_b": 0.0,
        "reason": "required correction exceeds bounded phase-lock envelope",
        "max_residual_ms": 41.0,
    }
    missing = {
        **base,
        "entry_time_b_seconds": 12.0,
        "reason": "insufficient reliable beat intervals across overlap",
        "max_residual_ms": 999.0,
    }
    path.write_text(
        "\n".join(json.dumps(record) for record in [base, base, missing])
        + "\n",
        encoding="utf-8",
    )
    result = summarize(path)
    assert result["raw_rejection_events"] == 3
    assert result["unique_candidate_configurations"] == 2
    assert result["repeated_evaluations"] == 1
    assert result["measured_residual_count"] == 1
    assert result["measured_residual_median_ms"] == 41.0
    assert result["measured_residual_p90_ms"] == 41.0


def test_trajectory_triage_is_diagnostic_only():
    from scripts.summarize_phase_lock_trace import classify_phase_motion

    points = lambda values: [{"phase_error_ms": value} for value in values]
    assert classify_phase_motion(points([0, 40, 80, 120, 160])) == "mostly_directional_drift"
    assert classify_phase_motion(points([0, 2, 5, 210, 212])) == "abrupt_phase_jump"
    assert classify_phase_motion(points([0, 10, 15, 20])) == "low_phase_excursion"
    assert classify_phase_motion(points([0, 4])) == "insufficient_trajectory"
    assert classify_phase_motion(points([0, 100, 0, 100, 0])) == "irregular_phase_motion"
