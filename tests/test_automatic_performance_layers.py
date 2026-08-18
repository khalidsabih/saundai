from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import analysis
from djai.layers import enrich_analysis_with_performance_layers, resolve_performance_layers
from djai.planner import rank_transition_candidates
from djai.stems import ensure_cached_performance_material, stem_cache_directory


def _seed_stem_cache(track: Path, *, sr: int = 22_050, duration: float = 12.0) -> None:
    frames = int(sr * duration)
    t = np.arange(frames, dtype=np.float32) / sr
    stems = {
        "vocals": 0.04 * np.sin(2 * np.pi * 440.0 * t),
        "drums": 0.035 * np.sign(np.sin(2 * np.pi * 2.0 * t)),
        "bass": 0.10 * np.sin(2 * np.pi * 62.0 * t),
        "other": 0.05 * np.sin(2 * np.pi * 220.0 * t),
    }
    master = np.sum(np.stack(list(stems.values()), axis=0), axis=0)
    sf.write(track, master, sr, subtype="FLOAT")
    root = stem_cache_directory(track)
    root.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for name, values in stems.items():
        path = root / f"{name}.wav"
        sf.write(path, values, sr, subtype="FLOAT")
        mapping[name] = str(path)
    material = ensure_cached_performance_material(track, mapping)
    assert Path(material["instrumental"]).exists()
    assert material["quality"]["quality_confidence"] >= 0.70


def test_one_full_master_resolves_to_ai_performance_layers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    full = tmp_path / "One Master.wav"
    _seed_stem_cache(full)

    bundle = resolve_performance_layers(full)
    assert bundle is not None
    assert bundle.complete
    assert bundle.source_quality == "ai_separated"
    assert bundle.quality_confidence >= 0.70
    assert bundle.instrumental is not None and bundle.instrumental.supplied is False
    assert bundle.vocals is not None and bundle.vocals.supplied is False


def test_ai_layers_enter_real_transition_candidate_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "A.wav"
    b = tmp_path / "B.wav"
    _seed_stem_cache(a)
    _seed_stem_cache(b)
    track_a = analysis(a, bpm=124.0, camelot="8A", energy=0.58, exit_section="breakdown")
    track_b = analysis(b, bpm=125.0, camelot="8A", energy=0.62, entry_section="groove")
    track_a.duration_seconds = 12.0
    track_b.duration_seconds = 12.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 5.0
        candidate.available_bars = 3
        candidate.vocal_probability = 0.70
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 1.0
        candidate.available_bars = 3
        candidate.vocal_probability = 0.60

    enrich_analysis_with_performance_layers(track_a)
    enrich_analysis_with_performance_layers(track_b)
    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative"),
        force=True,
        requested_bars=2,
        limit_per_style=4,
    )
    layer_plans = [plan for plan in plans if plan.technique.startswith("layer_")]
    assert layer_plans
    assert any(plan.metadata.get("layer_source_quality") == "ai_separated" for plan in layer_plans)
    assert all(float(plan.metadata.get("layer_quality_confidence", 0.0)) > 0.0 for plan in layer_plans)


def test_automatic_layer_preparation_falls_back_to_stereo_on_separator_failure(tmp_path: Path, monkeypatch) -> None:
    from djai import stems as stems_module

    full = tmp_path / "Fallback Master.wav"
    full.write_bytes(b"not-used-by-mocked-separator")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "wait")
    monkeypatch.setattr(stems_module, "demucs_available", lambda: True)
    monkeypatch.setattr(stems_module, "separate_stems", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("separator unavailable")))
    monkeypatch.setattr(stems_module, "find_cached_stems", lambda *args, **kwargs: {})
    messages: list[str] = []

    result = stems_module.prepare_stems_for_tracks(
        [full],
        progress=lambda stage, current, total, message: messages.append(message),
    )

    assert result[str(full.resolve())] == {}
    assert any("continuing stereo" in message for message in messages)


def test_background_layer_preparation_reads_cache_without_running_demucs(tmp_path: Path, monkeypatch) -> None:
    from djai import stems as stems_module

    full = tmp_path / "Background Master.wav"
    full.write_bytes(b"cached-only")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "background")
    monkeypatch.setattr(stems_module, "demucs_available", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(
        stems_module,
        "separate_stems",
        lambda *args, **kwargs: calls.append("separate") or {},
    )
    monkeypatch.setattr(stems_module, "find_cached_stems", lambda path: {"vocals": "cached.wav"})

    result = stems_module.prepare_stems_for_tracks([full])

    assert calls == []
    assert result[str(full.resolve())] == {"vocals": "cached.wav"}


def test_four_stem_ownership_choreography_enters_candidate_pool_and_renders(tmp_path: Path, monkeypatch) -> None:
    from djai.audio import load_audio
    from djai.decision import validate_execution_plan
    from djai.renderer import _render_transition_window

    monkeypatch.chdir(tmp_path)
    a = tmp_path / "A Stem Brain.wav"
    b = tmp_path / "B Stem Brain.wav"
    _seed_stem_cache(a, duration=18.0)
    _seed_stem_cache(b, duration=18.0)
    track_a = analysis(a, bpm=124.0, camelot="8A", energy=0.58, exit_section="breakdown")
    track_b = analysis(b, bpm=125.0, camelot="8A", energy=0.64, entry_section="groove")
    track_a.duration_seconds = 18.0
    track_b.duration_seconds = 18.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 5.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.78
        candidate.bass_activity = 0.76
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 1.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.72
        candidate.bass_activity = 0.74

    enrich_analysis_with_performance_layers(track_a)
    enrich_analysis_with_performance_layers(track_b)
    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative"),
        force=True,
        requested_bars=2,
        stems_available=True,
        limit_per_style=4,
    )
    stem_plans = [plan for plan in plans if plan.technique == "stem_choreography"]
    assert stem_plans
    # When four real stems are available and both bass/vocal collision risks are
    # high, stem ownership should compete at the top of the candidate pool, not
    # sit behind decorative stereo/layer variants.
    assert any(plan.technique == "stem_choreography" for plan in plans[:2])
    profiles = {plan.metadata["stem_choreography"]["profile"] for plan in stem_plans}
    assert "deck_vocal_exchange" in profiles
    assert profiles & {"outgoing_vocal_bridge", "bass_ownership_swap", "instrumental_reset", "progressive_handoff", "deck_vocal_exchange"}

    selected = stem_plans[0]
    validate_execution_plan(selected)
    assert selected.performance_program["annotations"]["renderer_contract"] == "execute_stem_curves_exactly"
    assert selected.performance_program["annotations"]["single_bass_owner"] is True
    assert selected.performance_program["stem_curves_a"]["bass"] != selected.performance_program["stem_curves_b"]["bass"]
    assert selected.performance_program["stem_curves_a"]["vocals"] != selected.performance_program["stem_curves_b"]["vocals"]

    audio_a, _ = load_audio(a, sample_rate=22_050, mono=False)
    audio_b, _ = load_audio(b, sample_rate=22_050, mono=False)
    mixed, consumed_a, consumed_b, _ = _render_transition_window(
        audio_a,
        audio_b,
        selected,
        sample_rate=22_050,
    )
    assert mixed.shape == (2, int(round(selected.transition_seconds * 22_050)))
    assert np.isfinite(mixed).all()
    assert float(np.max(np.abs(mixed))) > 0.01
    assert consumed_a > 0.0 and consumed_b > 0.0
