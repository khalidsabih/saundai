from __future__ import annotations

from pathlib import Path

import pytest

from djai.intelligence.neural_grid_batch_audit import (
    evaluate_batch, summarize_batch,
)


def _audited(*, existing=True, provisional=False, reasons=()):
    return {
        "neural_grid_gate_would_pass": existing,
        "counterfactual_safety_screen": {
            "would_pass_provisional_screen": provisional,
            "screen_reasons": list(reasons),
            "approved_for_live_adoption": False,
        },
        "neural_bpm": 120.0,
    }


def test_batch_summary_separates_existing_gate_and_provisional_screen():
    rows = [
        {"status": "ok", "reference_quality": "approximate",
         "audit": _audited(reasons=("same_level_bpm_disagreement_over_2_percent",))},
        {"status": "ok", "reference_quality": "approximate",
         "audit": _audited(reasons=("fallback_beat_agreement_below_85_percent",))},
        {"status": "ok", "reference_quality": "approximate",
         "audit": _audited(provisional=True)},
        {"status": "ok", "reference_quality": "approximate",
         "audit": _audited(existing=False)},
        {"status": "unavailable", "reference_quality": "approximate"},
    ]
    summary = summarize_batch(rows)
    assert summary["reference_count"] == 5
    assert summary["evaluated_count"] == 4
    assert summary["existing_gate_approvals"] == 3
    assert summary["existing_gate_rejections"] == 1
    assert summary["provisional_screen_rejections_of_existing_approvals"] == 2
    assert summary["provisional_screen_passes_of_existing_approvals"] == 1
    assert summary["rejection_reason_counts"] == {
        "fallback_beat_agreement_below_85_percent": 1,
        "same_level_bpm_disagreement_over_2_percent": 1,
    }
    assert summary["false_rejection_rate"] is None
    assert summary["ready_for_live_adoption"] is False


def test_duplicate_audio_basename_is_unavailable_not_arbitrarily_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    import djai.intelligence.neural_grid_batch_audit as batch
    for folder in ("a", "b"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "duplicate.wav").write_bytes(b"not decoded")
    monkeypatch.setattr(batch, "audit_grid_gate", lambda path: pytest.fail(
        "ambiguous audio must never be passed to inference"
    ))
    result = evaluate_batch(tmp_path, [{
        "filename": "duplicate.wav", "reference_bpm": 124,
        "reference_quality": "approximate",
    }])
    assert result["summary"]["unavailable_count"] == 1
    assert "ambiguous" in result["rows"][0]["reason"]


def test_batch_preserves_reference_quality_and_excludes_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    import djai.intelligence.neural_grid_batch_audit as batch
    (tmp_path / "one.wav").write_bytes(b"not decoded")
    monkeypatch.setattr(batch, "audit_grid_gate", lambda path: _audited(provisional=True))
    result = evaluate_batch(tmp_path, [
        {"filename": "one.wav", "reference_bpm": 121,
         "reference_quality": "approximate"},
        {"filename": "missing.wav", "reference_bpm": 125,
         "reference_quality": "approximate"},
    ])
    assert result["summary"]["evaluated_count"] == 1
    assert result["summary"]["unavailable_count"] == 1
    assert result["summary"]["approximate_reference_count"] == 2
    assert result["rows"][0]["neural_reference_absolute_error_bpm"] == 1.0
    assert result["summary"]["beat_alignment_accuracy"] is None
    assert result["summary"]["live_adoptions"] == 0
