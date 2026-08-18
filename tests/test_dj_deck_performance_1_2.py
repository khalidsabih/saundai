from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import analysis
from djai.decision import validate_execution_plan
from djai.planner import plan_transition
from djai.renderer import _render_tempo_release, _time_stretch_rate_curve
from djai.set_planner import make_set_plan
from djai.set_renderer import prepare_render_sequence


def _tone(seconds: float, sample_rate: int = 16_000, frequency: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    mono = 0.25 * np.sin(2.0 * np.pi * frequency * t)
    return np.stack([mono, mono]).astype(np.float32)


def _dominant_frequency(audio: np.ndarray, sample_rate: int) -> float:
    mono = np.asarray(audio[0], dtype=np.float64)
    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window))
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


def test_variable_deck_tempo_ride_preserves_pitch():
    sample_rate = 16_000
    audio = _tone(5.0, sample_rate=sample_rate, frequency=440.0)
    rendered = _time_stretch_rate_curve(
        audio,
        start_rate=0.97,
        end_rate=1.0,
        target_length=4 * sample_rate,
    )
    assert rendered.shape == (2, 4 * sample_rate)
    assert np.isfinite(rendered).all()
    assert _dominant_frequency(rendered, sample_rate) == pytest.approx(440.0, abs=6.0)


def test_transition_plan_matches_then_returns_incoming_deck_to_native_bpm(tmp_path: Path):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    track_a = analysis(a_path, bpm=124.0, camelot="8A", energy=0.55)
    track_b = analysis(b_path, bpm=128.0, camelot="8A", energy=0.60)

    plan = plan_transition(track_a, track_b, force=True, tempo_policy="safe", style="safe")
    validate_execution_plan(plan)

    assert plan.beatmatched is True
    assert plan.render_rate_b == pytest.approx(124.0 / 128.0, rel=1e-4)
    assert plan.tempo_strategy == "deck_handoff"
    assert plan.tempo_release_seconds_b > 0.0
    assert plan.tempo_release_source_duration_b > 0.0
    assert plan.post_transition_rate_b == pytest.approx(1.0)
    assert plan.post_transition_bpm_b == pytest.approx(128.0)
    assert plan.post_transition_source_end_b > plan.source_end_b
    assert plan.performance_program["annotations"]["deck_performance"]["tempo_owner_after"] == "deck_b"


def test_tempo_release_audio_reaches_native_rate_without_changing_output_length(tmp_path: Path):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    track_a = analysis(a_path, bpm=124.0, camelot="8A", energy=0.55)
    track_b = analysis(b_path, bpm=128.0, camelot="8A", energy=0.60)
    track_b.duration_seconds = 60.0
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 2.0
    plan = plan_transition(track_a, track_b, force=True, tempo_policy="safe", style="safe")

    audio_b = _tone(60.0, sample_rate=16_000, frequency=330.0)
    release, consumed = _render_tempo_release(audio_b, 16_000, plan, gain=1.0)
    assert release.shape[-1] == int(round(plan.tempo_release_seconds_b * 16_000))
    assert consumed == pytest.approx(plan.tempo_release_source_duration_b, rel=2e-3)
    assert np.isfinite(release).all()
    assert _dominant_frequency(release, 16_000) == pytest.approx(330.0, abs=7.0)


def test_full_set_uses_native_post_handoff_state_for_next_transition(tmp_path: Path):
    tracks = []
    for index, bpm in enumerate((124.0, 128.0, 126.0)):
        path = tmp_path / f"track-{index}.wav"
        path.write_bytes(str(index).encode())
        tracks.append(analysis(path, bpm=bpm, camelot="8A", energy=0.50 + index * 0.05))

    plan = make_set_plan(
        tracks,
        source="test",
        tempo_policy="safe",
        preferred_style="safe",
        journey="balanced",
        minimum_body_seconds=8.0,
        minimum_body_fraction=0.05,
    )
    states, edges = prepare_render_sequence(plan)
    assert len(edges) == 2
    first = edges[0].plan
    second = edges[1].plan
    assert first.post_transition_rate_b == pytest.approx(1.0)
    assert second.render_rate_a == pytest.approx(first.post_transition_rate_b)
    assert second.effective_bpm_a == pytest.approx(first.post_transition_bpm_b, abs=0.02)
    assert states[1].rate == pytest.approx(1.0)
    assert states[1].effective_bpm == pytest.approx(plan.tracks[edges[0].edge.to_index].bpm, abs=0.02)


def test_real_pair_render_executes_tempo_release_segment(tmp_path: Path):
    import soundfile as sf
    from djai.renderer import render_full_mix

    sample_rate = 16_000
    a_path = tmp_path / "deck-a.wav"
    b_path = tmp_path / "deck-b.wav"
    sf.write(a_path, _tone(40.0, sample_rate=sample_rate).T, sample_rate)
    sf.write(b_path, _tone(40.0, sample_rate=sample_rate, frequency=330.0).T, sample_rate)
    track_a = analysis(a_path, bpm=124.0, camelot="8A", energy=0.55)
    track_b = analysis(b_path, bpm=128.0, camelot="8A", energy=0.60)
    track_a.duration_seconds = 40.0
    track_b.duration_seconds = 40.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 12.0
        candidate.available_bars = 8
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 2.0
        candidate.available_bars = 8

    plan = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=4,
        exit_time_a=12.0,
        entry_time_b=2.0,
    )
    assert plan.tempo_release_seconds_b > 0.0
    output = render_full_mix(plan, tmp_path / "mix.wav", sample_rate=sample_rate)
    info = sf.info(output)
    expected = (
        plan.exit_time_a / plan.render_rate_a
        + plan.transition_seconds
        + plan.tempo_release_seconds_b
        + max(0.0, track_b.duration_seconds - plan.post_transition_source_end_b) / plan.post_transition_rate_b
    )
    assert info.duration == pytest.approx(expected, abs=0.15)
    assert info.channels == 2


def test_live_manifest_exposes_deck_handoff_and_tempo_release(tmp_path: Path):
    from djai.live import live_manifest

    tracks = []
    for index, bpm in enumerate((124.0, 128.0)):
        path = tmp_path / f"live-{index}.wav"
        path.write_bytes(str(index).encode())
        item = analysis(path, bpm=bpm, camelot="8A", energy=0.55 + index * 0.05)
        tracks.append(item)

    from djai.set_models import SetPlan, TransitionEdge

    transition = plan_transition(tracks[0], tracks[1], force=True, tempo_policy="safe", style="safe")
    plan = SetPlan(
        plan_id="live-manifest",
        tracks=tracks,
        order=[0, 1],
        transitions=[
            TransitionEdge(
                from_index=0,
                to_index=1,
                selected_style=transition.style,
                objective_score=transition.score,
                quality="good",
                plan=transition,
                alternatives=[],
            )
        ],
        baseline_order=[0, 1],
        baseline_score=transition.score,
        optimized_score=transition.score,
        estimated_duration_seconds=180.0,
        source="test-live-manifest",
        automatic_delivery=True,
    )
    manifest = live_manifest(plan)
    transitions = [item for item in manifest["timeline"] if item["type"] == "transition"]
    releases = [item for item in manifest["timeline"] if item["type"] == "tempo_release"]

    assert len(transitions) == 1
    assert transitions[0]["target_bpm"] == pytest.approx(plan.transitions[0].plan.target_bpm)
    assert "phase_nudge_seconds_b" in transitions[0]
    assert len(releases) == 1
    assert releases[0]["pitch_preserving"] is True
    assert releases[0]["tempo_owner"] == "deck_b"
    assert releases[0]["start_bpm"] == pytest.approx(plan.transitions[0].plan.source_bpm_b * plan.transitions[0].plan.tempo_release_start_rate_b)
    assert releases[0]["end_bpm"] == pytest.approx(plan.transitions[0].plan.post_transition_bpm_b)


def test_live_stream_executes_same_tempo_release_as_finished_render(tmp_path: Path):
    import soundfile as sf
    from djai.live import iter_live_chunks
    from djai.set_models import SetPlan, TransitionEdge

    sample_rate = 8_000
    a_path = tmp_path / "live-deck-a.wav"
    b_path = tmp_path / "live-deck-b.wav"
    sf.write(a_path, _tone(32.0, sample_rate=sample_rate).T, sample_rate)
    sf.write(b_path, _tone(32.0, sample_rate=sample_rate, frequency=330.0).T, sample_rate)
    track_a = analysis(a_path, bpm=124.0, camelot="8A", energy=0.55)
    track_b = analysis(b_path, bpm=128.0, camelot="8A", energy=0.60)
    track_a.duration_seconds = 32.0
    track_b.duration_seconds = 32.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 10.0
        candidate.available_bars = 8
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 2.0
        candidate.available_bars = 8

    transition = plan_transition(
        track_a,
        track_b,
        force=True,
        tempo_policy="safe",
        style="safe",
        requested_bars=4,
        exit_time_a=10.0,
        entry_time_b=2.0,
    )
    assert transition.tempo_release_seconds_b > 0.0
    set_plan = SetPlan(
        plan_id="live-stream-parity",
        tracks=[track_a, track_b],
        order=[0, 1],
        transitions=[TransitionEdge(0, 1, transition.style, transition.score, "good", transition)],
        baseline_order=[0, 1],
        baseline_score=transition.score,
        optimized_score=transition.score,
        estimated_duration_seconds=64.0,
        source="test-live-stream",
        automatic_delivery=True,
    )

    frames = sum(chunk.shape[0] for chunk in iter_live_chunks(set_plan, sample_rate=sample_rate, block_frames=1024))
    expected = (
        transition.exit_time_a / transition.render_rate_a
        + transition.transition_seconds
        + transition.tempo_release_seconds_b
        + max(0.0, track_b.duration_seconds - transition.post_transition_source_end_b) / transition.post_transition_rate_b
    )
    assert frames / sample_rate == pytest.approx(expected, abs=0.18)
