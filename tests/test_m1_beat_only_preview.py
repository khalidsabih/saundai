"""M1 beat-only feature: make neural pulses useful without altering DJ authority."""
from __future__ import annotations

from copy import deepcopy

from conftest import analysis
from djai.intelligence.beat_preview import beat_preview_from_evidence
from djai.models import TrackAnalysis


def evidence(*, count=129):
    return {
        "backend": "beat_this",
        "available": True,
        "used": False,
        "reason": "neural beat grid recorded as evidence but deterministic grid retained",
        "confidence": .6599,
        "beats_seconds": [i * .5 for i in range(count)],
        # Intentionally extra/inconsistent bar labels, not authorized for use.
        "downbeats_seconds": [i * 1.5 for i in range(21)],
    }


def test_preview_exposes_rejected_neural_beats_only_when_enabled(tmp_path, monkeypatch):
    track = analysis(tmp_path / "a.wav", bpm=123)
    track.neural_evidence["beat_downbeat"] = evidence()
    baseline = track.to_dict()
    assert "beat_only_preview" not in baseline
    original_beats = deepcopy(track.beats_seconds)
    original_bars = deepcopy(track.downbeats_seconds)
    original_cues = deepcopy(track.entry_candidates)

    monkeypatch.setenv("DJAI_BEAT_ONLY_PREVIEW", "1")
    output = track.to_dict()
    preview = output["beat_only_preview"]
    assert preview["available"] is True
    assert len(preview["beats_seconds"]) == 129
    assert preview["bpm"] == 120.0
    assert preview["scope"] == "preview_only"
    assert preview["used_for_planning"] is False
    assert preview["downbeats_authoritative"] is False
    assert output["beats_seconds"] == original_beats
    assert output["downbeats_seconds"] == original_bars
    assert track.beats_seconds == original_beats
    assert track.downbeats_seconds == original_bars
    assert track.entry_candidates == original_cues


def test_preview_compact_response_limits_beat_list(tmp_path, monkeypatch):
    track = analysis(tmp_path / "a.wav", bpm=123)
    track.neural_evidence["beat_downbeat"] = evidence()
    monkeypatch.setenv("DJAI_BEAT_ONLY_PREVIEW", "yes")
    output = track.to_dict(include_beats=False)
    assert len(output["beat_only_preview"]["beats_seconds"]) == 16
    assert output["beat_only_preview"]["beat_count"] == 129
    assert len(output["beats_seconds"]) == 16
    assert len(output["downbeats_seconds"]) == 16


def test_off_switch_disables_presentation_without_deleting_neural_evidence(tmp_path, monkeypatch):
    track = analysis(tmp_path / "a.wav", bpm=123)
    track.neural_evidence["beat_downbeat"] = evidence()
    monkeypatch.setenv("DJAI_BEAT_ONLY_PREVIEW", "1")
    assert "beat_only_preview" in track.to_dict()
    monkeypatch.setenv("DJAI_BEAT_ONLY_PREVIEW", "0")
    output = track.to_dict()
    assert "beat_only_preview" not in output
    assert output["neural_evidence"]["beat_downbeat"]["available"] is True


def test_invalid_candidate_returns_abstention(tmp_path, monkeypatch):
    track = analysis(tmp_path / "a.wav", bpm=123)
    monkeypatch.setenv("DJAI_BEAT_ONLY_PREVIEW", "1")
    for beats in ([0, .5, 1, 1.5, 2, 2.5, 3, 3], [float("nan")] * 10,
                  [i * 0.5 for i in range(900)]):
        e = evidence()
        e["beats_seconds"] = beats
        track.neural_evidence["beat_downbeat"] = e
        p = track.to_dict()["beat_only_preview"]
        assert not p["available"]
        assert not p["used_for_planning"]
        assert p["beats_seconds"] == []


def test_unavailable_inference_never_exposes_preview_beats():
    e = evidence()
    e["available"] = False
    output = beat_preview_from_evidence(e, duration_seconds=120)
    assert not output["available"]
    assert not output["beats_seconds"]


def test_instability_abstains_instead_of_inventing_downbeats():
    beats = [i * .5 for i in range(50)]
    beats[30:] = [x + 4 for x in beats[30:]]
    output = beat_preview_from_evidence(
        {**evidence(), "beats_seconds": beats}, duration_seconds=60
    )
    assert not output["available"]
    assert output["downbeats_authoritative"] is False


def test_single_short_interval_abstains_even_with_regular_majority():
    beats = [i * .5 for i in range(50)]
    # One short interval should not be hidden by a 5th percentile aggregate.
    beats[30:] = [x - .35 for x in beats[30:]]
    result = beat_preview_from_evidence(
        {**evidence(), "beats_seconds": beats}, duration_seconds=60
    )
    assert not result["available"]
    assert "unstable neural beat interval" in result["reason"]
    assert result["used_for_planning"] is False
