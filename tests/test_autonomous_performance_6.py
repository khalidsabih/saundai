from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np

from conftest import analysis
from djai.brain.director import direct_set_intent
from djai.brain.dj_memory import build_dj_memory
from djai.brain.proposal import transition_proposal_from_plan
from djai.brain.set_state import build_set_state
from djai.brain.unified_transition import UnifiedTransitionConfig, _candidate_pool
from djai.evaluation.release_gate import evaluate_release_gate
from djai.execution import TransitionRenderContext
from djai.judge.critics import evaluate_critic_ensemble
from djai.measurement import DEFAULT_MEASUREMENT_POLICY
from djai.models import BarFeature, TransitionMeasurement
from djai.performance.engine import execute_transition_program
from djai.performance.program import PERFORMANCE_PROGRAM_VERSION, transition_program_from_dict
from djai.perception.world_model import WORLD_MODEL_VERSION, build_track_world_model
from djai.planner import plan_transition
from djai.planning_context import build_pair_evidence_matrix
from djai.rehearsal_cache import (
    load_cached_rehearsal,
    rehearsal_cache_key,
    store_cached_rehearsal,
)
from djai.renderer import _render_transition_window
from djai.version import DJAI_VERSION, PLAN_SCHEMA_VERSION


def test_6_versions_and_executable_program(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    program = transition_program_from_dict(plan.performance_program)
    assert DJAI_VERSION == "2.0.0-rc.1"
    assert PLAN_SCHEMA_VERSION == "8.0.0"
    assert program.version == PERFORMANCE_PROGRAM_VERSION == "4.0"
    assert program.fingerprint == plan.performance_program_fingerprint
    assert program.source_mode in {"master", "stems", "layers"}

    frames = 4096
    t = np.arange(frames, dtype=np.float32) / 8_000.0
    left = (0.12 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    right = (0.12 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    result = execute_transition_program(
        program,
        {"deck_a": np.stack([left, left]), "deck_b": np.stack([right, right])},
        sample_rate=8_000,
        bpm=plan.target_bpm,
    )
    assert result.audio.shape == (2, frames)
    assert np.isfinite(result.audio).all()
    assert result.buses_used == ("deck_a", "deck_b")


def test_production_renderer_does_not_dispatch_on_technique() -> None:
    source = inspect.getsource(_render_transition_window)
    assert "if plan.technique" not in source and "elif plan.technique" not in source
    assert "execute_transition_program" in source


def test_transition_proposal_is_separate_from_executable_ir(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="creative")
    proposal = transition_proposal_from_plan(plan, set_intent="lift")
    assert proposal.technique == plan.technique
    assert proposal.intent == plan.intent
    assert proposal.duration_bars == plan.transition_bars
    assert proposal.set_intent == "lift"
    assert proposal.source_mode == plan.performance_program["source_mode"]
    assert "curves" not in proposal.to_dict()


def test_world_model_builds_phrase_roles_and_graph(tmp_path: Path) -> None:
    track = analysis(tmp_path / "world.wav", bpm=126.0, energy=0.62)
    bars: list[BarFeature] = []
    for index in range(24):
        if index < 8:
            section, energy = "groove", 0.48 + index * 0.01
        elif index < 16:
            section, energy = "build", 0.52 + (index - 8) * 0.035
        else:
            section, energy = "drop", 0.88
        bars.append(
            BarFeature(
                bar_index=index,
                start_seconds=index * 2.0,
                end_seconds=(index + 1) * 2.0,
                section_label=section,
                energy=energy,
                rhythm=0.72 if section == "drop" else 0.60,
                brightness=0.55,
                percussive_activity=0.70,
                vocal_probability=0.18,
                bass_activity=0.64,
                harmonic_tension=0.68 if section == "build" else 0.42,
                boundary_strength=0.82 if index in {7, 15} else 0.30,
                phrase_strength=0.84 if index in {7, 15} else 0.45,
                beat_grid_confidence=0.92,
                local_bpm=126.0,
            )
        )
    track.bar_features = bars
    world = build_track_world_model(track, phrase_bars=8)
    assert world.version == WORLD_MODEL_VERSION == "6.0"
    assert [phrase.role for phrase in world.phrases] == ["intro", "pre_drop", "drop"]
    assert world.phrase_graph[1].relation == "resolves_to_drop"
    assert world.structural_summary["role_sequence"] == ["intro", "pre_drop", "drop"]


def test_dj_memory_and_set_director_bound_repetition_and_risk(tmp_path: Path) -> None:
    tracks = [analysis(tmp_path / f"{index}.wav", bpm=124 + index, energy=0.55 + index * 0.03) for index in range(4)]
    for track in tracks:
        Path(track.path).write_bytes(track.title.encode())
    transitions = [plan_transition(tracks[index], tracks[index + 1], force=True, style="safe") for index in range(3)]
    memory = build_dj_memory(tracks, transitions)
    penalty = memory.transition_repetition_penalty(
        transitions[-1].technique,
        transitions[-1].transition_bars,
        transitions[-1].intent,
    )
    state = build_set_state(tracks, transitions)
    intent = direct_set_intent("balanced", 0.5, state)
    assert penalty > 0.0
    assert 0.0 <= intent.risk_budget <= 0.25 + 0.75 * memory.surprise_budget + 1e-9
    assert intent.name in {"build", "lift", "peak", "release", "bridge", "surprise"}


def test_multi_critic_returns_targeted_vocal_remedies(track_pair) -> None:
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    measurement = TransitionMeasurement(
        version="test",
        sample_rate=44_100,
        duration_seconds=10.0,
        overall=0.74,
        rhythm_coherence=0.80,
        transient_clarity=0.80,
        groove_continuity=0.80,
        harmonic_continuity=0.76,
        energy_intent_fit=0.72,
        vocal_cleanliness=0.08,
        stem_bass_cleanliness=0.75,
        stem_evidence=True,
        hard_failures=["Stem analysis found severe vocal-on-vocal overlap in a long blend."],
    )
    ensemble = evaluate_critic_ensemble(plan, measurement=measurement, track_b=track_b)
    assert "vocals" in ensemble.failure_domains
    assert "delay_incoming_vocal" in ensemble.remedies
    assert "shorten_overlap" in ensemble.remedies


def test_critic_guided_retry_prioritizes_shorter_durations(track_pair, monkeypatch) -> None:
    track_a, track_b = track_pair
    evidence = build_pair_evidence_matrix(track_a, track_b)
    route_plan = plan_transition(track_a, track_b, force=True, style="safe")
    calls: list[int] = []

    monkeypatch.setattr("djai.brain.unified_transition.duration_candidates", lambda *args, **kwargs: (16, 8, 4))
    monkeypatch.setattr(
        "djai.brain.unified_transition.rank_duration_values",
        None,
        raising=False,
    )

    def fake_rank(*args, requested_bars: int, **kwargs):
        calls.append(requested_bars)
        plan = copy.deepcopy(route_plan)
        plan.transition_bars = requested_bars
        plan.execution_signature = f"candidate-{requested_bars}"
        return [plan]

    monkeypatch.setattr("djai.brain.unified_transition.rank_transition_candidates", fake_rank)
    from djai.brain.transition_router import route_transition

    _candidate_pool(
        track_a,
        track_b,
        route=route_transition(track_a, track_b),
        evidence_matrix=evidence,
        config=UnifiedTransitionConfig(),
        tempo_policy="safe",
        preferred_style="safe",
        effective_bpm_a=None,
        render_rate_a=1.0,
        stems_available=False,
        retry_round=2,
        finalist_budget=3,
        retry_remedies=("shorten_overlap",),
        previous_failed_bars=16,
    )
    assert calls[:2] == [8, 4]


def test_rehearsal_cache_is_content_and_context_addressed(track_pair, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    track_a, track_b = track_pair
    plan = plan_transition(track_a, track_b, force=True, style="safe")
    context_a = TransitionRenderContext(sample_rate=44_100, gain_a=1.0, gain_b=0.9)
    context_b = TransitionRenderContext(sample_rate=44_100, gain_a=1.0, gain_b=0.8)
    key_a = rehearsal_cache_key(
        plan,
        sample_rate=44_100,
        context_seconds=2.0,
        policy=DEFAULT_MEASUREMENT_POLICY,
        render_context=context_a,
        capture_render_embedding=False,
    )
    key_b = rehearsal_cache_key(
        plan,
        sample_rate=44_100,
        context_seconds=2.0,
        policy=DEFAULT_MEASUREMENT_POLICY,
        render_context=context_b,
        capture_render_embedding=False,
    )
    assert key_a != key_b
    measurement = TransitionMeasurement(
        version="test",
        sample_rate=44_100,
        duration_seconds=8.0,
        overall=0.88,
    )
    assert store_cached_rehearsal(
        key_a,
        execution_signature=plan.execution_signature,
        measurement=measurement,
        metadata={"rendered_transition_embedding_captured": False},
    ) is not None
    loaded = load_cached_rehearsal(key_a)
    assert loaded is not None
    cached_measurement, metadata = loaded
    assert cached_measurement.overall == 0.88
    assert metadata["rendered_transition_embedding_captured"] is False


def test_release_gate_blocks_integrity_regressions_and_distinguishes_rc() -> None:
    blocked = evaluate_release_gate({"mutated_plan_accept_count": 1})
    assert blocked.status == "blocked" and not blocked.ready

    rc = evaluate_release_gate(
        {
            "transition_quality": 0.82,
            "phrase_alignment": 0.88,
            "transition_diversity": 0.78,
            "set_storytelling": 0.79,
        }
    )
    assert rc.status == "release_candidate"
    assert any("installation" in warning for warning in rc.warnings)

    ready = evaluate_release_gate(
        {
            "transition_quality": 0.90,
            "phrase_alignment": 0.91,
            "transition_diversity": 0.82,
            "set_storytelling": 0.84,
            "installation_success_rate": 0.99,
        }
    )
    assert ready.status == "ready" and ready.ready
