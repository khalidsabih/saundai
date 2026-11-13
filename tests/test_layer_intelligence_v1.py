from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import analysis
from djai.brain.dj_move import build_dj_state, move_brain_status, score_transition_plan_as_move
from djai.brain.move_generator import CompleteMoveCandidate
from djai.brain.move_intent import build_dj_move_intent
from djai.brain.set_state import SetState
from djai.layers import enrich_analysis_with_performance_layers, register_performance_layers
from djai.planner import rank_transition_candidates
from djai.rehearsal.headphones import AIHeadphonesConfig, listen_to_complete_moves


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_layer_bundle(
    root: Path,
    name: str,
    *,
    sr: int = 22_050,
    duration: float = 20.0,
    vocal_frequency: float = 330.0,
) -> tuple[Path, Path, Path]:
    frames = int(sr * duration)
    t = np.arange(frames, dtype=np.float32) / sr
    kick = (0.045 * np.sign(np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)
    bass = (0.11 * np.sin(2 * np.pi * 74.0 * t)).astype(np.float32)
    pad = (0.055 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    instrumental = np.clip(kick + bass + pad, -0.75, 0.75).astype(np.float32)
    phrase_gate = (np.sin(2 * np.pi * 0.25 * t) > -0.10).astype(np.float32)
    vocals = (0.075 * np.sin(2 * np.pi * vocal_frequency * t) * phrase_gate).astype(np.float32)
    full = np.clip(instrumental + vocals, -0.92, 0.92).astype(np.float32)

    full_path = root / f"{name}.wav"
    instrumental_path = root / f"{name} Instrumental.wav"
    vocal_path = root / f"{name} Acapella.wav"
    sf.write(full_path, full, sr, subtype="FLOAT")
    sf.write(instrumental_path, instrumental, sr, subtype="FLOAT")
    sf.write(vocal_path, vocals, sr, subtype="FLOAT")
    return full_path, instrumental_path, vocal_path


def _prepared_pair(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a_full, a_inst, a_vocal = _write_layer_bundle(tmp_path, "Outgoing")
    b_full, b_inst, b_vocal = _write_layer_bundle(tmp_path, "Incoming", vocal_frequency=440.0)
    register_performance_layers([a_full, a_inst, a_vocal, b_full, b_inst, b_vocal])

    track_a = analysis(a_full, bpm=120.0, camelot="8A", energy=0.58, exit_section="breakdown")
    track_b = analysis(b_full, bpm=120.0, camelot="8A", energy=0.66, entry_section="groove")
    track_a.duration_seconds = 20.0
    track_b.duration_seconds = 20.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 7.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.72
        candidate.bass_activity = 0.55
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 1.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.72
        candidate.bass_activity = 0.48

    enrich_analysis_with_performance_layers(track_a)
    enrich_analysis_with_performance_layers(track_b)
    return track_a, track_b


def _incoming_vocal_plan(track_a, track_b):
    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative"),
        force=True,
        requested_bars=2,
        limit_per_style=4,
    )
    selected = next(plan for plan in plans if plan.technique == "layer_incoming_vocal_handoff")
    return plans, selected


def test_incoming_vocal_over_outgoing_instrumental_is_a_real_djmove(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b = _prepared_pair(tmp_path, monkeypatch)
    plans, selected = _incoming_vocal_plan(track_a, track_b)

    # The existing full-master path remains present, so layer intelligence is
    # opportunistic rather than a new hard dependency.
    assert any(not plan.technique.startswith("layer_") for plan in plans)
    assert selected.metadata["layer_mix"] == "incoming_vocal_over_outgoing_instrumental"
    assert selected.metadata["layer_fallback"] == "full_master_transition"
    assert selected.metadata["ai_headphones_required_for_authority"] is True
    assert selected.metadata["layer_safety"]["feasible"] is True
    assert selected.metadata["layer_safety"]["rendered_headroom_authoritative"] is True

    choreography = selected.metadata["layer_choreography"]
    assert choreography["strategy"] == "incoming_vocal_over_outgoing_instrumental"
    assert [phase["name"] for phase in choreography["phases"]] == [
        "clear_outgoing_vocal",
        "introduce_incoming_vocal",
        "instrumental_handoff",
        "full_incoming_ownership",
    ]
    assert choreography["end"]["vocal_owner"] == "B"
    assert choreography["end"]["instrumental_owner"] == "B"

    program = selected.performance_program
    assert program["source_mode"] == "layers"
    assert program["annotations"]["vocal_ownership_transfer"] == "a_to_b_early"
    assert program["annotations"]["instrumental_ownership_transfer"] == "a_to_b_late"
    assert program["annotations"]["fallback"] == "full_master_transition"
    assert "instrumental" in program["layer_curves_a"]
    assert {"vocals", "instrumental", "master"}.issubset(program["layer_curves_b"])

    state = build_dj_state(track_a, source_cursor=4.0, effective_bpm=track_a.bpm, tempo_owner="A")
    assert state.layer_state.vocal_owner == "A"
    assert state.layer_state.instrumental_owner == "A"
    assert state.layer_state.current_track_vocal_available is True
    assert state.layer_state.current_track_instrumental_available is True

    move = score_transition_plan_as_move(track_a, track_b, selected, state=state)
    payload = move.to_dict()
    assert "LAYER_HANDOFF" in payload["structural_operations"]
    assert "VOCAL_OVER_INSTRUMENTAL" in payload["structural_operations"]
    assert payload["stem_strategy"] == "performance_layers"
    assert payload["layer_choreography"]["strategy"] == "incoming_vocal_over_outgoing_instrumental"
    assert payload["projected_state"]["layer_state"]["vocal_owner"] == "B"
    assert payload["projected_state"]["layer_state"]["instrumental_owner"] == "B"

    intent = build_dj_move_intent(
        current_index=0,
        next_index=1,
        track_a=track_a,
        track_b=track_b,
        winner_move=move,
        winner_plan=selected,
        candidate_moves=(move,),
        candidate_plans=(selected,),
    )
    assert "incoming_vocal_over_outgoing_instrumental" in intent.allowed_layer_strategies
    assert "layer_move_must_keep_master_fallback" in intent.hard_constraints
    assert intent.fallback_policy == "same_target_compiled_safe_fallback"


def test_incoming_vocal_choreography_reaches_real_ai_headphones_render(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b = _prepared_pair(tmp_path, monkeypatch)
    _, selected = _incoming_vocal_plan(track_a, track_b)
    state = build_dj_state(track_a, source_cursor=4.0, effective_bpm=track_a.bpm, tempo_owner="A")
    move = score_transition_plan_as_move(track_a, track_b, selected, state=state)
    candidate = CompleteMoveCandidate(
        current_index=0,
        next_index=1,
        move=move,
        plan=selected,
        selection_score=max(0.75, move.score),
        state=state,
    )

    decision = listen_to_complete_moves(
        preferred=candidate,
        candidates=[candidate],
        analyses=[track_a, track_b],
        set_state=SetState(current_bpm=track_a.bpm),
        journey="balanced",
        progress=0.4,
        config=AIHeadphonesConfig(
            enabled=True,
            serious_pool=1,
            minimum_renders=1,
            maximum_renders=1,
            sample_rate=22_050,
            context_seconds=1.0,
            minimum_measured_score=0.0,
            minimum_judge_score=0.0,
            measurement_log=str(tmp_path / "headphones.jsonl"),
        ),
    )

    assert len(decision.evaluations) == 1
    evaluation = decision.evaluations[0]
    assert evaluation.render_error is None
    assert evaluation.plan.technique == "layer_incoming_vocal_handoff"
    assert evaluation.plan.measurement is not None
    assert evaluation.plan.measurement.duration_seconds > 0.0
    assert evaluation.plan.measurement.peak > 0.0
    assert "ai_headphones" in evaluation.plan.metadata
    assert evaluation.plan.metadata["ai_headphones"]["hard_audio_safety_authoritative"] is True


def test_layer_intelligence_v1_can_be_disabled_without_removing_existing_fallbacks(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b = _prepared_pair(tmp_path, monkeypatch)
    monkeypatch.setenv("DJAI_LAYER_INTELLIGENCE", "0")

    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative"),
        force=True,
        requested_bars=2,
        limit_per_style=4,
    )

    assert not any(plan.technique == "layer_incoming_vocal_handoff" for plan in plans)
    assert any(plan.technique != "layer_incoming_vocal_handoff" for plan in plans)


def test_move_brain_reports_layer_ownership_capability(monkeypatch) -> None:
    monkeypatch.delenv("DJAI_LAYER_INTELLIGENCE", raising=False)
    status = move_brain_status()["layer_intelligence"]
    assert status["version"] == "1.0"
    assert status["enabled"] is True
    assert status["ownership_in_dj_state"] is True
    assert status["incoming_vocal_over_outgoing_instrumental"] is True
    assert status["full_master_fallback"] is True
    assert status["planning_triggers_separation"] is False


def test_layer_brain_patch_does_not_change_rc1_frontend_transport() -> None:
    api = (ROOT / "frontend" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert 'credentials: "include"' in api
    assert '"/api/tracks/upload"' in api
    assert '"/api/tracks/upload-intents"' in api
    assert "EventSource" in api
    next_config = (ROOT / "frontend" / "next.config.ts").read_text(encoding="utf-8")
    assert 'output: "standalone"' in next_config
    assert 'source: "/api/:path*"' in next_config
    assert 'destination: `${internalApi}/api/:path*`' in next_config
    assert 'X-Content-Type-Options' in next_config
    assert _sha256(ROOT / "compose.yaml") == (
        "762df2e22a1c21cf095ed040b92bba11357ca7c3d3c9d05a9ad785cf166bd92a"
    )
