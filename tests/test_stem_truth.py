from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import analysis
from djai.stems import enrich_analysis_with_stems


def test_cached_stems_replace_candidate_source_heuristics(tmp_path: Path) -> None:
    track_path = tmp_path / "track.wav"
    track_path.write_bytes(b"track")
    track = analysis(track_path, bpm=120.0, energy=0.6)
    sr = 22050
    duration = 12.0
    track.duration_seconds = duration
    for index, candidate in enumerate(track.entry_candidates):
        candidate.time_seconds = 1.0 + index * 2.0
        candidate.available_bars = 2
    for index, candidate in enumerate(track.exit_candidates):
        candidate.time_seconds = 6.0 + index * 2.0
        candidate.available_bars = 2
    frames = int(sr * duration)
    t = np.arange(frames, dtype=np.float32) / sr
    stems = {
        "vocals": 0.20 * np.sin(2 * np.pi * 440 * t),
        "bass": 0.32 * np.sin(2 * np.pi * 70 * t),
        "drums": 0.05 * np.sign(np.sin(2 * np.pi * 2 * t)),
        "other": 0.04 * np.sin(2 * np.pi * 880 * t),
    }
    mapping = {}
    for name, values in stems.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, values, sr)
        mapping[name] = str(path)

    enrich_analysis_with_stems(track, mapping, sample_rate=sr, window_bars=4)
    assert track.stem_analysis["available"] == 1.0
    assert 0.0 < track.vocal_probability < 1.0
    assert 0.0 < track.bass_activity < 1.0
    for candidate in [*track.entry_candidates, *track.exit_candidates]:
        assert candidate.stem_vocal_activity > 0.0
        assert candidate.stem_bass_activity > 0.0
        assert candidate.vocal_probability == candidate.stem_vocal_activity
        assert candidate.bass_activity == candidate.stem_bass_activity


def test_stem_aware_renderer_executes_compiled_automation(tmp_path: Path) -> None:
    from conftest import analysis
    from djai.decision import execution_signature
    from djai.planner import plan_transition
    from djai.renderer import _render_stem_aware_transition_window

    sr = 22_050
    duration = 16.0
    frames = int(sr * duration)
    t = np.arange(frames, dtype=np.float32) / sr

    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    sf.write(a_path, 0.2 * np.sin(2 * np.pi * 110 * t), sr)
    sf.write(b_path, 0.2 * np.sin(2 * np.pi * 146.8 * t), sr)
    track_a = analysis(a_path, bpm=120.0, energy=0.58)
    track_b = analysis(b_path, bpm=120.0, camelot="9A", energy=0.62)
    track_a.duration_seconds = duration
    track_b.duration_seconds = duration
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 6.0
        candidate.available_bars = 4
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 1.0
        candidate.available_bars = 4

    plan = plan_transition(track_a, track_b, style="safe", force=True)
    plan.technique = "eq_bass_swap"
    plan.transition_seconds = 4.0
    plan.transition_bars = 2
    plan.exit_time_a = 6.0
    plan.entry_time_b = 1.0
    plan.render_rate_a = 1.0
    plan.render_rate_b = 1.0
    plan.source_duration_a = 4.0
    plan.source_duration_b = 4.0
    plan.compiled_execution = True

    for prefix, root in (("a", 110.0), ("b", 146.8)):
        mapping: dict[str, str] = {}
        values = {
            "vocals": 0.03 * np.sin(2 * np.pi * (root * 4) * t),
            "drums": 0.04 * np.sign(np.sin(2 * np.pi * 2 * t)),
            "bass": 0.18 * np.sin(2 * np.pi * root * 0.5 * t),
            "other": 0.06 * np.sin(2 * np.pi * root * 2 * t),
        }
        for name, audio in values.items():
            path = tmp_path / f"{prefix}-{name}.wav"
            sf.write(path, audio, sr)
            mapping[name] = str(path)
        if prefix == "a":
            plan.stem_paths_a = mapping
        else:
            plan.stem_paths_b = mapping

    plan.execution_signature = execution_signature(plan)
    output = _render_stem_aware_transition_window(
        plan,
        sample_rate=sr,
        gain_a=1.0,
        gain_b=1.0,
        target_length=4 * sr,
    )
    assert output.shape == (2, 4 * sr)
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(output))) > 0.01
    first_bass = float(np.sqrt(np.mean(np.square(output[:, : sr]))))
    last_bass = float(np.sqrt(np.mean(np.square(output[:, -sr:]))))
    assert first_bass > 0.0
    assert last_bass > 0.0
