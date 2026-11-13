from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from djai import candidate_renderer
from djai.candidate_renderer import MeasuredTransitionError, QualityGateConfig, evaluate_transition_candidates
from djai.decision import execution_signature
from djai.measurement import measure_transition_audio
from djai.models import TransitionMeasurement
from djai.planner import plan_transition_alternatives
from djai.renderer import RenderedTransitionAudio
from djai.serialization import transition_plan_from_dict
from djai.version import MEASUREMENT_VERSION


def _stereo_tone(sample_rate: int, seconds: float, frequency: float = 220.0, amplitude: float = 0.2) -> np.ndarray:
    frames = max(1, int(round(sample_rate * seconds)))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    signal = amplitude * np.sin(2.0 * np.pi * frequency * time)
    return np.stack([signal, signal], axis=0).astype(np.float32)


def _measurement(overall: float, *, failure: str | None = None) -> TransitionMeasurement:
    return TransitionMeasurement(
        version=MEASUREMENT_VERSION,
        sample_rate=8_000,
        duration_seconds=2.0,
        loudness_continuity=overall,
        spectral_continuity=overall,
        bass_cleanliness=overall,
        phase_safety=overall,
        artifact_safety=overall,
        silence_safety=overall,
        overall=overall,
        hard_failures=[failure] if failure else [],
    )


def test_measurement_distinguishes_clean_audio_from_silence(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    sample_rate = 8_000
    clean = _stereo_tone(sample_rate, 2.0)
    bad = np.zeros_like(clean)

    clean_result = measure_transition_audio(
        pre=clean,
        transition=clean,
        post=clean,
        sample_rate=sample_rate,
        plan=plan,
    )
    bad_result = measure_transition_audio(
        pre=clean,
        transition=bad,
        post=clean,
        sample_rate=sample_rate,
        plan=plan,
    )

    assert clean_result.overall >= 0.85
    assert clean_result.hard_failures == []
    assert bad_result.overall < clean_result.overall
    assert bad_result.silence_fraction >= 0.95
    assert any("silence" in reason.lower() for reason in bad_result.hard_failures)


def test_measurement_reports_individual_loudness_boundaries(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    sample_rate = 8_000
    pre = _stereo_tone(sample_rate, 2.0, amplitude=0.2)
    transition = _stereo_tone(sample_rate, 2.0, amplitude=0.05)
    post = _stereo_tone(sample_rate, 2.0, amplitude=0.05)

    result = measure_transition_audio(
        pre=pre,
        transition=transition,
        post=post,
        sample_rate=sample_rate,
        plan=plan,
    )

    assert result.pre_boundary_lufs is not None
    assert result.transition_head_lufs is not None
    assert result.transition_tail_lufs is not None
    assert result.post_boundary_lufs is not None
    assert result.entry_loudness_jump_lu is not None
    assert result.exit_loudness_jump_lu is not None
    assert result.entry_loudness_jump_lu > result.exit_loudness_jump_lu
    assert result.loudness_jump_lu == max(
        result.entry_loudness_jump_lu,
        result.exit_loudness_jump_lu,
    )


def test_quality_gate_can_rerank_analytical_candidates(
    tmp_path: Path,
    monkeypatch,
    track_pair,
) -> None:
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")
    assert len(plans) >= 2

    dummy = _stereo_tone(8_000, 2.0)
    load_calls: list[str] = []

    def fake_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return dummy, 8_000

    monkeypatch.setattr(candidate_renderer, "load_audio", fake_load)
    monkeypatch.setattr(
        candidate_renderer,
        "render_transition_audio_from_loaded",
        lambda *args, **kwargs: RenderedTransitionAudio(
            pre=dummy,
            transition=dummy,
            post=dummy,
            preview=np.concatenate([dummy, dummy, dummy], axis=-1),
            gain_b=1.0,
            sample_rate=8_000,
        ),
    )

    preferred_signature = plans[1].execution_signature

    def fake_measure(*, plan, **kwargs):
        return _measurement(0.96 if plan.execution_signature == preferred_signature else 0.30)

    monkeypatch.setattr(candidate_renderer, "measure_transition_audio", fake_measure)
    log_path = tmp_path / "measurements.jsonl"
    selection = evaluate_transition_candidates(
        plans,
        config=QualityGateConfig(
            enabled=True,
            top_k=min(3, len(plans)),
            sample_rate=8_000,
            analytical_weight=0.10,
            measured_weight=0.90,
            minimum_measured_score=0.42,
            log_path=str(log_path),
        ),
    )

    assert selection.selected.execution_signature == preferred_signature
    assert len(load_calls) == 2
    assert selection.selected.quality_checked is True
    assert selection.selected.measurement_passed is True
    assert selection.selected.measured_score == 0.96
    assert selection.selected.analytical_score > 0.0

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == min(3, len(plans))
    assert sum(bool(record["accepted"]) for record in records) == 1
    assert all(record["measurement"] is not None for record in records)


def test_measurement_round_trip_is_typed(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    plan.analytical_score = plan.score
    plan.measured_score = 0.88
    plan.measurement_passed = True
    plan.measurement_rank = 1
    plan.quality_checked = True
    plan.measurement = _measurement(0.88)

    restored = transition_plan_from_dict(plan.to_dict())

    assert restored.quality_checked is True
    assert restored.measurement_passed is True
    assert restored.measurement is not None
    assert isinstance(restored.measurement, TransitionMeasurement)
    assert restored.measurement.overall == 0.88


def test_measurement_log_failure_does_not_reject_mix(monkeypatch, track_pair) -> None:
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")
    dummy = _stereo_tone(8_000, 2.0)
    monkeypatch.setattr(candidate_renderer, "load_audio", lambda *args, **kwargs: (dummy, 8_000))
    monkeypatch.setattr(
        candidate_renderer,
        "render_transition_audio_from_loaded",
        lambda *args, **kwargs: RenderedTransitionAudio(
            pre=dummy,
            transition=dummy,
            post=dummy,
            preview=np.concatenate([dummy, dummy, dummy], axis=-1),
            gain_b=1.0,
            sample_rate=8_000,
        ),
    )
    monkeypatch.setattr(candidate_renderer, "measure_transition_audio", lambda **kwargs: _measurement(0.90))
    monkeypatch.setattr(
        candidate_renderer,
        "append_measurement_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    selection = evaluate_transition_candidates(
        plans,
        config=QualityGateConfig(
            enabled=True,
            top_k=2,
            sample_rate=8_000,
            log_path="/unwritable/measurements.jsonl",
        ),
    )

    assert selection.selected.measurement_passed is True
    assert selection.log_path is None
    assert selection.log_error is not None
    assert "read-only filesystem" in selection.log_error


def test_quality_gate_continues_after_one_candidate_render_fails(
    tmp_path: Path,
    monkeypatch,
    track_pair,
) -> None:
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")
    assert len(plans) >= 2

    first_signature = plans[0].execution_signature
    preferred_signature = plans[1].execution_signature
    dummy = _stereo_tone(8_000, 2.0)
    monkeypatch.setattr(candidate_renderer, "load_audio", lambda *args, **kwargs: (dummy, 8_000))

    def fake_render(*args, **kwargs):
        plan = args[2]
        if plan.execution_signature == first_signature:
            raise RuntimeError("synthetic renderer failure")
        return RenderedTransitionAudio(
            pre=dummy,
            transition=dummy,
            post=dummy,
            preview=np.concatenate([dummy, dummy, dummy], axis=-1),
            gain_b=1.0,
            sample_rate=8_000,
        )

    monkeypatch.setattr(candidate_renderer, "render_transition_audio_from_loaded", fake_render)
    monkeypatch.setattr(
        candidate_renderer,
        "measure_transition_audio",
        lambda *, plan, **kwargs: _measurement(
            0.92 if plan.execution_signature == preferred_signature else 0.70
        ),
    )

    log_path = tmp_path / "measurements.jsonl"
    selection = evaluate_transition_candidates(
        plans[:2],
        config=QualityGateConfig(
            enabled=True,
            top_k=2,
            sample_rate=8_000,
            analytical_weight=0.20,
            measured_weight=0.80,
            minimum_measured_score=0.42,
            log_path=str(log_path),
        ),
    )

    assert selection.selected.execution_signature == preferred_signature
    assert selection.rejected_count == 1
    failed = next(plan for plan in selection.evaluated if plan.execution_signature == first_signature)
    assert failed.measurement is not None
    assert any("synthetic renderer failure" in reason for reason in failed.measurement.hard_failures)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert sum(bool(record["accepted"]) for record in records) == 1
    assert any(record["measurement"]["hard_failures"] for record in records)



def test_candidate_render_failure_is_isolated(monkeypatch, track_pair) -> None:
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")
    dummy = _stereo_tone(8_000, 2.0)
    monkeypatch.setattr(candidate_renderer, "load_audio", lambda *args, **kwargs: (dummy, 8_000))
    # top_k=2 evaluates the two highest analytical candidates; choose one of
    # those explicitly rather than depending on style-return order.
    failed_signature = max(plans, key=lambda plan: (plan.score, plan.cue_confidence, plan.execution_signature)).execution_signature

    def fake_render(*args, **kwargs):
        plan = args[2]
        if plan.execution_signature == failed_signature:
            raise RuntimeError("synthetic render failure")
        return RenderedTransitionAudio(
            pre=dummy,
            transition=dummy,
            post=dummy,
            preview=np.concatenate([dummy, dummy, dummy], axis=-1),
            gain_b=1.0,
            sample_rate=8_000,
        )

    monkeypatch.setattr(candidate_renderer, "render_transition_audio_from_loaded", fake_render)
    monkeypatch.setattr(candidate_renderer, "measure_transition_audio", lambda **kwargs: _measurement(0.90))

    selection = evaluate_transition_candidates(
        plans,
        config=QualityGateConfig(enabled=True, top_k=2, sample_rate=8_000),
    )

    assert selection.selected.execution_signature != failed_signature
    failed = next(plan for plan in selection.evaluated if plan.execution_signature == failed_signature)
    assert failed.measurement_passed is False
    assert failed.measurement is not None
    assert any("synthetic render failure" in reason for reason in failed.measurement.hard_failures)


def test_measurement_metadata_does_not_change_execution_signature(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    original = plan.execution_signature
    plan.analytical_score = plan.score
    plan.measured_score = 0.91
    plan.measurement = _measurement(0.91)
    plan.measurement_passed = True
    plan.measurement_rank = 1
    plan.quality_checked = True

    assert execution_signature(plan) == original


def test_measurement_rejects_severe_stereo_phase_inversion(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[0]
    sample_rate = 8_000
    mono = _stereo_tone(sample_rate, 2.0)[0]
    inverted = np.stack([mono, -mono], axis=0)

    result = measure_transition_audio(
        pre=inverted,
        transition=inverted,
        post=inverted,
        sample_rate=sample_rate,
        plan=plan,
    )

    assert result.phase_correlation <= -0.99
    assert result.phase_safety == 0.0
    assert any("phase correlation" in reason.lower() for reason in result.hard_failures)


def test_quality_gate_logs_all_failures_before_raising(
    tmp_path: Path,
    monkeypatch,
    track_pair,
) -> None:
    track_a, track_b = track_pair
    plans = plan_transition_alternatives(track_a, track_b, force=True, tempo_policy="safe")[:2]
    dummy = _stereo_tone(8_000, 2.0)
    monkeypatch.setattr(candidate_renderer, "load_audio", lambda *args, **kwargs: (dummy, 8_000))
    monkeypatch.setattr(
        candidate_renderer,
        "render_transition_audio_from_loaded",
        lambda *args, **kwargs: RenderedTransitionAudio(
            pre=dummy,
            transition=dummy,
            post=dummy,
            preview=np.concatenate([dummy, dummy, dummy], axis=-1),
            gain_b=1.0,
            sample_rate=8_000,
        ),
    )
    monkeypatch.setattr(
        candidate_renderer,
        "measure_transition_audio",
        lambda **kwargs: _measurement(0.20, failure="synthetic hard failure"),
    )
    log_path = tmp_path / "failed-measurements.jsonl"

    with pytest.raises(MeasuredTransitionError) as caught:
        evaluate_transition_candidates(
            plans,
            config=QualityGateConfig(
                enabled=True,
                top_k=2,
                sample_rate=8_000,
                minimum_measured_score=0.42,
                log_path=str(log_path),
            ),
        )

    assert caught.value.best is not None
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(record["accepted"] is False for record in records)
    assert all(record["measurement"]["hard_failures"] for record in records)
