from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from djai import candidate_renderer, fallback
from djai.candidate_renderer import MeasuredSelection, QualityGateConfig, evaluate_transition_candidates
from djai.decision import validate_execution_plan
from djai.demo import generate_demo_track
from djai.environment import config_path
from djai.execution import TransitionRenderContext
from djai.intelligence.model_registry import ModelManifest, ModelRegistry
from djai.models import TransitionMeasurement
from djai.pipeline import run_set_pipeline
from djai.planner import plan_transition_alternatives
from djai.renderer import RenderedTransitionAudio
from djai.set_renderer import _atomic_soundfile
from djai.version import MEASUREMENT_VERSION


def _tone(sample_rate: int = 8_000, seconds: float = 1.0) -> np.ndarray:
    frames = max(1, int(round(sample_rate * seconds)))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    signal = 0.1 * np.sin(2.0 * np.pi * 220.0 * time)
    return np.stack([signal, signal], axis=0).astype(np.float32)


def _passing_measurement(sample_rate: int = 8_000) -> TransitionMeasurement:
    return TransitionMeasurement(
        version=MEASUREMENT_VERSION,
        sample_rate=sample_rate,
        duration_seconds=1.0,
        loudness_continuity=0.95,
        spectral_continuity=0.95,
        bass_cleanliness=0.95,
        phase_safety=0.95,
        artifact_safety=0.95,
        silence_safety=0.95,
        overall=0.95,
        hard_failures=[],
    )


def test_signed_plan_rejects_master_content_mutation(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    validate_execution_plan(plan)

    source = Path(track_a.path)
    stat = source.stat()
    source.write_bytes(b"z")  # same size as the original fixture payload
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="content changed after planning"):
        validate_execution_plan(plan)


def test_exact_render_context_reaches_rehearsal_and_is_recorded(monkeypatch, track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    audio = _tone()
    seen: list[TransitionRenderContext | None] = []

    monkeypatch.setattr(candidate_renderer, "load_audio", lambda *args, **kwargs: (audio, 8_000))

    def fake_render(*args, render_context=None, **kwargs):
        seen.append(render_context)
        return RenderedTransitionAudio(
            pre=audio,
            transition=audio,
            post=audio,
            preview=np.concatenate([audio, audio, audio], axis=-1),
            gain_b=0.75,
            sample_rate=8_000,
        )

    monkeypatch.setattr(candidate_renderer, "render_transition_audio_from_loaded", fake_render)
    monkeypatch.setattr(candidate_renderer, "measure_transition_audio", lambda **kwargs: _passing_measurement())
    context = TransitionRenderContext(sample_rate=8_000, gain_a=0.9, gain_b=0.75)

    selection = evaluate_transition_candidates(
        [plan],
        config=QualityGateConfig(
            enabled=True,
            top_k=1,
            sample_rate=8_000,
            context_seconds=1.0,
            minimum_measured_score=0.0,
            log_path=None,
        ),
        render_context=context,
    )

    assert seen == [context.normalized()]
    assert selection.selected.metadata["render_context"] == {
        "sample_rate": 8_000,
        "gain_a": 0.9,
        "gain_b": 0.75,
        "target_loudness_dbfs": -13.0,
        "context_seconds": 1.0,
        "exact_set_context": True,
    }


def test_last_resort_fallback_requires_render_validation_when_requested(monkeypatch, track_pair) -> None:
    track_a, track_b = track_pair
    calls: dict[str, object] = {}

    def fake_evaluate(plans, *, config, render_context=None, **kwargs):
        calls["count"] = len(plans)
        calls["sample_rate"] = config.sample_rate
        calls["render_context"] = render_context
        selected = copy.deepcopy(plans[0])
        selected.measurement = _passing_measurement(config.sample_rate)
        selected.measured_score = 0.95
        selected.measurement_passed = True
        selected.quality_checked = True
        return MeasuredSelection(selected=selected, evaluated=[selected])

    monkeypatch.setattr(fallback, "evaluate_transition_candidates", fake_evaluate)
    context = TransitionRenderContext(sample_rate=8_000, gain_a=0.8, gain_b=0.7)
    plan = fallback.build_last_resort_plan(
        track_a,
        track_b,
        render_context=context,
        verify_audio=True,
    )

    assert calls["count"] == 1
    assert calls["sample_rate"] == 8_000
    assert calls["render_context"] == context
    assert plan.metadata["fallback_render_verified"] is True
    assert plan.metadata["fallback_validation_pending"] is False


def test_explicit_zero_model_weight_has_zero_authority() -> None:
    manifest = ModelManifest(
        id="zero-model",
        task="test",
        provider="test",
        model="test",
        version="1",
        authority="advisory",
        weight=0.0,
        minimum_confidence=0.0,
    )
    registry = ModelRegistry({manifest.id: manifest})

    assert manifest.effective_weight == 0.0
    assert registry.authority(manifest.id, confidence=1.0) == 0.0


def test_workspace_config_resolution_is_not_process_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("DJAI_CONFIG_DIR", raising=False)

    assert config_path("scoring-model.json") == (tmp_path / "config" / "scoring-model.json").resolve()
    assert config_path("config/model-registry.json") == (tmp_path / "config" / "model-registry.json").resolve()


def test_atomic_wav_is_not_published_after_render_failure(tmp_path: Path) -> None:
    output = tmp_path / "set.wav"
    audio = _tone()

    with pytest.raises(RuntimeError, match="verification failed"):
        with _atomic_soundfile(output, sample_rate=8_000) as handle:
            handle.write(audio.T)
            raise RuntimeError("verification failed")

    assert not output.exists()
    assert list(tmp_path.glob(".set.wav.*.tmp")) == []


def test_measured_set_renders_with_exact_default_context(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "out"
    audio_dir.mkdir()
    generate_demo_track(
        audio_dir / "a.wav",
        bpm=124,
        root_hz=55,
        minor_third_hz=65.406,
        fifth_hz=82.407,
        variant="a",
        bars=8,
    )
    generate_demo_track(
        audio_dir / "b.wav",
        bpm=124,
        root_hz=41.203,
        minor_third_hz=48.999,
        fifth_hz=61.735,
        variant="b",
        bars=8,
    )

    result = run_set_pipeline(
        audio_dir,
        output_dir=output_dir,
        use_cache=False,
        minimum_body_seconds=5.0,
        minimum_body_fraction=0.10,
        render_previews=False,
        render_audio=True,
        analysis_workers=1,
        graph_workers=1,
        measurement_top_k=1,
        measurement_log=str(tmp_path / "ratings.jsonl"),
    )

    assert result.audio_path is not None
    assert result.audio_path.exists()
    context = result.plan.transitions[0].plan.metadata["render_context"]
    assert context["sample_rate"] == 44_100
    assert context["exact_set_context"] is True
    report = json.loads(result.render_report_path.read_text(encoding="utf-8"))
    verification = report["render"]["final_transition_verification"]
    assert verification["enabled"] is True
    assert verification["passed"] is True
    assert verification["verified_transitions"] == 1
