from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import analysis
from djai.analysis import _estimate_downbeats, _estimate_time_signature
from djai.fallback import build_last_resort_plan


def _accent_pattern(meter: int, beats: int = 48) -> tuple[np.ndarray, np.ndarray]:
    beat_frames = np.arange(beats, dtype=int)
    onset = np.full(beats + 4, 0.10, dtype=np.float64)
    if meter == 3:
        onset[beat_frames[::3]] = 1.0
    elif meter == 4:
        onset[beat_frames[::4]] = 1.0
    elif meter == 6:
        onset[beat_frames[::6]] = 1.0
        onset[beat_frames[3::6]] = 0.55
    else:
        raise ValueError(meter)
    return beat_frames, onset


@pytest.mark.parametrize("meter", [3, 4, 6])
def test_meter_is_resolved_before_bar_grouping(meter: int):
    beat_frames, onset = _accent_pattern(meter)
    detected, confidence = _estimate_time_signature(
        beat_frames,
        onset,
        beat_grid_confidence=0.94,
    )
    assert detected == meter
    assert confidence > 0.35

    downbeats, downbeat_confidence = _estimate_downbeats(
        beat_frames,
        onset,
        beats_per_bar=detected,
    )
    assert downbeats.tolist() == beat_frames[::meter].tolist()
    assert downbeat_confidence > 0.0


def test_uncertain_meter_is_explicitly_unknown():
    beat_frames = np.arange(48, dtype=int)
    onset = np.full(52, 0.10, dtype=np.float64)
    meter, confidence = _estimate_time_signature(
        beat_frames,
        onset,
        beat_grid_confidence=0.12,
    )
    assert meter == 0
    assert confidence < 0.10


def _track(tmp_path: Path, name: str, bpm: float):
    path = tmp_path / f"{name}.wav"
    path.write_bytes(name.encode("utf-8"))
    return analysis(path, bpm=bpm, camelot="8A", energy=0.58)


def test_wide_bpm_rescue_uses_structural_reset_not_long_unbeatmatched_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    track_a = _track(tmp_path, "slow", 92.0)
    track_b = _track(tmp_path, "fast", 148.0)

    plan = build_last_resort_plan(track_a, track_b)

    assert plan.beatmatched is False
    assert plan.technique == "echo_out_drop_in"
    assert plan.metadata["rescue_transition_kind"] == "structural_reset"
    assert plan.metadata["continuous_mix_required"] is False
    assert plan.metadata["rescue_overlap_safe"] is False
    assert plan.transition_seconds <= 1.8 + 1e-6
    assert plan.render_rate_a == pytest.approx(1.0)
    assert plan.render_rate_b == pytest.approx(1.0)


def test_variable_tempo_rescue_uses_structural_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    track_a = _track(tmp_path, "drifting", 124.0)
    track_b = _track(tmp_path, "steady", 126.0)
    track_a.variable_tempo = True

    plan = build_last_resort_plan(track_a, track_b)

    assert plan.metadata["rescue_transition_kind"] == "structural_reset"
    assert plan.metadata["continuous_mix_required"] is False
    assert any("variable-tempo" in reason for reason in plan.metadata["rescue_overlap_risks"])


def test_compatible_stable_pair_keeps_compact_filtered_rescue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    track_a = _track(tmp_path, "compatible-a", 124.0)
    track_b = _track(tmp_path, "compatible-b", 126.0)

    plan = build_last_resort_plan(track_a, track_b)

    assert plan.technique == "filter_build"
    assert plan.metadata["rescue_transition_kind"] == "filtered_overlap"
    assert plan.metadata["continuous_mix_required"] is True
    assert plan.metadata["rescue_overlap_safe"] is True


def test_explicit_hard_cut_policy_still_allows_quick_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DJAI_ALLOW_HARD_CUTS", "1")
    track_a = _track(tmp_path, "cut-a", 92.0)
    track_b = _track(tmp_path, "cut-b", 148.0)

    plan = build_last_resort_plan(track_a, track_b)

    assert plan.technique == "quick_cut"
    assert plan.metadata["rescue_transition_kind"] == "hard_cut"
