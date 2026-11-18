from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from djai.intelligence.neural_grid_gate_audit import audit_grid_gate


def _baseline() -> dict:
    return {
        "bpm": 86.1328,
        "beat_times": (0.5, 1.2, 1.9, 2.6),
        "downbeat_times": (0.5,),
        "duration_seconds": 30.0,
        "time_signature": 4,
    }


def test_grid_gate_uses_actual_fallback_signature_without_mutating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import djai.intelligence.neural_grid_gate_audit as audit
    baseline = _baseline()
    received = {}
    monkeypatch.setattr(audit, "deterministic_fallback_grid", lambda path: baseline)

    def fake_predict(path, *, fallback_beats, fallback_downbeats):
        received["path"] = path
        received["beats"] = tuple(fallback_beats)
        received["downbeats"] = tuple(fallback_downbeats)
        return SimpleNamespace(
            available=True, bpm=88.2353, beats_seconds=(0.4, 1.0, 1.7),
            downbeats_seconds=(0.4,), confidence=.7677,
            agreement=.6, used=True, reason="neural gate passed",
        )

    monkeypatch.setattr("djai.intelligence.neural_ears.predict_beats", fake_predict)
    monkeypatch.setattr("djai.intelligence.neural_ears.tempo_only_adoption",
                        lambda result, bpm: (False, "disabled"))
    path = Path("Toc Toc.wav")
    result = audit_grid_gate(path)
    assert received == {
        "path": path,
        "beats": (0.5, 1.2, 1.9, 2.6),
        "downbeats": (0.5,),
    }
    assert baseline == _baseline()
    assert result["neural_grid_gate_would_pass"] is True
    assert result["requires_human_review"] is True
    assert result["live_grid_adoption_observed"] is False
    assert result["changes_dj_decisions"] is False
    assert result["live_analysis_executed"] is False


def test_grid_gate_rejects_false_evidence_of_live_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import djai.intelligence.neural_grid_gate_audit as audit
    monkeypatch.setattr(audit, "deterministic_fallback_grid", lambda path: _baseline())
    monkeypatch.setattr(
        "djai.intelligence.neural_ears.predict_beats",
        lambda path, **kwargs: SimpleNamespace(
            available=True, bpm=120.0, beats_seconds=(), downbeats_seconds=(),
            confidence=.25, agreement=.05, used=False, reason="grid retained",
        ),
    )
    monkeypatch.setattr("djai.intelligence.neural_ears.tempo_only_adoption",
                        lambda result, bpm: (False, "disabled"))
    result = audit_grid_gate(Path("Cola.wav"))
    assert result["neural_grid_gate_would_pass"] is False
    assert result["live_grid_adoption_observed"] is False
    assert result["requires_human_review"] is False


def test_no_neural_evidence_still_emits_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import djai.intelligence.neural_grid_gate_audit as audit
    monkeypatch.setattr(audit, "deterministic_fallback_grid", lambda path: _baseline())
    monkeypatch.setattr(
        "djai.intelligence.neural_ears.predict_beats",
        lambda path, **kwargs: SimpleNamespace(
            available=False, bpm=0.0, beats_seconds=(), downbeats_seconds=(),
            confidence=0.0, agreement=0.0, used=False,
            reason="checkpoint unavailable",
        ),
    )
    monkeypatch.setattr("djai.intelligence.neural_ears.tempo_only_adoption",
                        lambda result, bpm: (False, "disabled"))
    result = audit_grid_gate(Path("absent.wav"))
    assert result["neural_grid_gate_would_pass"] is False
    assert result["neural_available"] is False
    assert result["live_grid_adoption_observed"] is False
