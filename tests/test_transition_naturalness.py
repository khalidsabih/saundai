from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import analysis, candidate
from djai.fallback import build_last_resort_plan
from djai.judge.professional import judge_transition
from djai.naturalness import assess_transition_naturalness
from djai.renderer import render_transition_audio


def _pair(tmp_path: Path, *, bpm_a: float, bpm_b: float):
    a_path = tmp_path / f"a-{bpm_a}.wav"
    b_path = tmp_path / f"b-{bpm_b}.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    return (
        analysis(a_path, bpm=bpm_a, camelot="8A", energy=0.56),
        analysis(b_path, bpm=bpm_b, camelot="9A", energy=0.62),
    )


def test_difficult_rescue_uses_structural_reset_when_overlap_is_unsafe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DJAI_AUTO_RESCUE_BARS", raising=False)
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    track_a, track_b = _pair(tmp_path, bpm_a=92.0, bpm_b=148.0)
    plan = build_last_resort_plan(track_a, track_b, reason="natural rescue test")

    assert plan.technique == "echo_out_drop_in"
    assert plan.transition_bars == 1
    assert plan.metadata["transition_naturalness_profile"] == "structural_reset"
    assert plan.metadata["rescue_transition_kind"] == "structural_reset"
    assert plan.metadata["continuous_mix_required"] is False
    assert plan.metadata["rescue_overlap_safe"] is False
    assert plan.performance_program["annotations"]["transition_naturalness_profile"] == "structural_reset"


def test_compatible_rescue_keeps_balanced_phrase_handoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DJAI_AUTO_RESCUE_BARS", raising=False)
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    track_a, track_b = _pair(tmp_path, bpm_a=124.0, bpm_b=126.0)
    plan = build_last_resort_plan(track_a, track_b, reason="natural rescue test")

    assert plan.transition_bars == 4
    assert plan.metadata["transition_naturalness_profile"] == "balanced"
    assert plan.source_duration_b / max(plan.render_rate_b, 1e-9) >= plan.transition_seconds - 1e-6


def test_naturalness_rewards_phrase_structure_and_clean_sources(tmp_path: Path) -> None:
    track_a, track_b = _pair(tmp_path, bpm_a=124.0, bpm_b=126.0)
    plan = build_last_resort_plan(track_a, track_b, reason="naturalness score test")
    good = copy.deepcopy(plan)
    bad = copy.deepcopy(plan)

    good.compatibility.phrase = 0.92
    good.compatibility.boundary = 0.90
    good.compatibility.sections = 0.91
    good.compatibility.section_fit = 0.90
    good.compatibility.vocal_overlap = 0.90
    good.compatibility.bass_conflict = 0.88
    good.compatibility.harmonic = 0.90
    good.cue_confidence = 0.90

    bad.compatibility.phrase = 0.22
    bad.compatibility.boundary = 0.20
    bad.compatibility.sections = 0.28
    bad.compatibility.section_fit = 0.25
    bad.compatibility.vocal_overlap = 0.20
    bad.compatibility.bass_conflict = 0.24
    bad.compatibility.harmonic = 0.30
    bad.cue_confidence = 0.28
    bad.transition_bars = 16

    good_score = assess_transition_naturalness(good).score
    bad_score = assess_transition_naturalness(bad).score
    assert good_score > bad_score + 0.25


def test_professional_judge_exposes_naturalness_as_bounded_quality_signal(tmp_path: Path) -> None:
    track_a, track_b = _pair(tmp_path, bpm_a=124.0, bpm_b=126.0)
    plan = build_last_resort_plan(track_a, track_b, reason="judge naturalness test")
    judgement = judge_transition(plan, track_a=track_a, track_b=track_b, minimum_score=0.0)

    assert 0.0 <= judgement.naturalness <= 1.0
    assert plan.metadata["professional_judge_naturalness"] == judgement.naturalness
    assert "transition_naturalness" in plan.metadata
    assert any("Transition naturalness" in reason for reason in judgement.reasons)


def test_distant_rescue_renders_structural_reset_without_hard_cut(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DJAI_AUTO_RESCUE_BARS", raising=False)
    monkeypatch.delenv("DJAI_ALLOW_HARD_CUTS", raising=False)
    sample_rate = 22_050
    seconds = 30.0
    times = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    a_mono = (0.22 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    b_mono = (0.22 * np.sin(2.0 * np.pi * 660.0 * times)).astype(np.float32)
    a_path = tmp_path / "tone-a.wav"
    b_path = tmp_path / "tone-b.wav"
    sf.write(a_path, np.column_stack([a_mono, a_mono]), sample_rate)
    sf.write(b_path, np.column_stack([b_mono, b_mono]), sample_rate)

    track_a = analysis(a_path, bpm=92.0, camelot="8A", energy=0.52)
    track_b = analysis(b_path, bpm=148.0, camelot="10A", energy=0.68)
    track_a.duration_seconds = seconds
    track_b.duration_seconds = seconds
    track_a.exit_candidates = [
        candidate("exit", 12.0, 12, "breakdown", bpm=92.0, local_key=track_a.key, energy=0.46),
        candidate("exit", 18.0, 18, "outro", bpm=92.0, local_key=track_a.key, energy=0.42),
    ]
    track_b.entry_candidates = [
        candidate("entry", 4.0, 4, "groove", bpm=148.0, local_key=track_b.key, energy=0.62),
        candidate("entry", 10.0, 10, "drop", bpm=148.0, local_key=track_b.key, energy=0.78),
    ]

    plan = build_last_resort_plan(track_a, track_b, reason="rendered structural rescue")
    rendered = render_transition_audio(
        plan, sample_rate=sample_rate, pre_roll_seconds=0.5, post_roll_seconds=0.5
    )

    assert rendered.transition.shape[-1] > 0
    assert np.isfinite(rendered.transition).all()
    assert plan.technique == "echo_out_drop_in"
    assert plan.metadata["transition_naturalness_profile"] == "structural_reset"
    assert plan.metadata["continuous_mix_required"] is False
