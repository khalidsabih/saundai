from __future__ import annotations

import copy

from djai.judge.musical import _musical_hard_failures
from djai.judge.professional import judge_transition
from djai.models import TransitionMeasurement
from djai.performance.mix_contract import evaluate_professional_mix_contract
from djai.planner import plan_transition


def _long_blend(track_pair):
    a, b = track_pair
    plan = plan_transition(
        a,
        b,
        force=True,
        style="safe",
        tempo_policy="safe",
        requested_bars=8,
    )
    plan.transition_bars = 8
    plan.beatmatched = True
    plan.transition_mode = "beatmatched_blend"
    return plan


def test_long_beatmatched_blend_hard_fails_severe_rhythm_and_flam(track_pair):
    plan = _long_blend(track_pair)
    failures = _musical_hard_failures(
        plan,
        rhythm=0.12,
        transient=0.10,
        groove=0.70,
        harmony=0.70,
        payoff=0.50,
    )
    assert any("rhythmic incoherence" in item for item in failures)
    assert any("doubled transients" in item for item in failures)


def test_reset_technique_is_exempt_from_blend_continuity_hard_gates(track_pair):
    plan = _long_blend(track_pair)
    plan.technique = "echo_out_drop_in"
    plan.transition_mode = "emergency_cut"
    failures = _musical_hard_failures(
        plan,
        rhythm=0.05,
        transient=0.05,
        groove=0.05,
        harmony=0.05,
        payoff=0.05,
    )
    assert failures == []


def test_short_blend_gets_warning_space_not_long_blend_hard_gate(track_pair):
    plan = _long_blend(track_pair)
    plan.transition_bars = 2
    failures = _musical_hard_failures(
        plan,
        rhythm=0.19,
        transient=0.15,
        groove=0.10,
        harmony=0.10,
        payoff=0.50,
    )
    assert failures == []


def test_weak_committed_drop_payoff_is_a_hard_failure(track_pair):
    plan = _long_blend(track_pair)
    plan.intent = "breakdown_to_drop"
    failures = _musical_hard_failures(
        plan,
        rhythm=0.80,
        transient=0.80,
        groove=0.80,
        harmony=0.80,
        payoff=0.05,
    )
    assert any("insufficient rendered payoff" in item for item in failures)


def test_long_master_overlap_hard_fails_severe_vocal_bass_harmony_and_phrase_conflict(track_pair):
    plan = _long_blend(track_pair)
    plan.compatibility.vocal_overlap = 0.10
    plan.compatibility.bass_conflict = 0.10
    plan.compatibility.harmonic = 0.08
    plan.compatibility.local_harmonic = 0.10
    plan.compatibility.phrase = 0.10
    plan.compatibility.boundary = 0.12

    profile = dict(plan.metadata.get("professional_mix", {}) or {})
    profile.update(
        {
            "continuous_mix_required": True,
            "structural_reset": False,
            "reset_justified": False,
        }
    )
    plan.metadata = {**plan.metadata, "professional_mix": profile}

    program = copy.deepcopy(plan.performance_program)
    program["source_mode"] = "master"
    curves = dict(program.get("curves", {}) or {})
    curves.pop("deck_a_low_gain", None)
    curves.pop("deck_b_low_gain", None)
    program["curves"] = curves
    plan.performance_program = program

    contract = evaluate_professional_mix_contract(plan)

    assert contract.passed is False
    assert any("lead-vocal collision" in item for item in contract.hard_failures)
    assert any("simultaneous bass ownership" in item for item in contract.hard_failures)
    assert any("harmonic incompatibility" in item for item in contract.hard_failures)
    assert any("phrase or structural boundary" in item for item in contract.hard_failures)


def test_professional_judge_cannot_override_musical_hard_failure(track_pair, tmp_path):
    a, b = track_pair
    plan = _long_blend(track_pair)
    plan.measurement = TransitionMeasurement(
        version="test",
        sample_rate=22050,
        duration_seconds=12.0,
        loudness_continuity=0.96,
        spectral_continuity=0.95,
        bass_cleanliness=0.95,
        phase_safety=0.98,
        artifact_safety=0.98,
        silence_safety=0.98,
        overall=0.96,
        rhythm_coherence=0.10,
        transient_clarity=0.10,
        groove_continuity=0.90,
        harmonic_continuity=0.90,
        energy_intent_fit=0.90,
        drop_payoff=0.90,
        musical_overall=0.82,
        hard_failures=[
            "Rendered overlap contains persistent doubled transients/flam consistent with kick or snare competition."
        ],
    )

    result = judge_transition(
        plan,
        track_a=a,
        track_b=b,
        rejection_path=tmp_path / "none.jsonl",
        minimum_score=0.20,
    )

    assert result.feasible is False
    assert result.accepted is False
    assert any("doubled transients" in item for item in result.hard_failures)
