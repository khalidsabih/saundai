from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from djai.intelligence.model_registry import load_model_registry
from djai.intelligence.preferences import append_preference, plan_features
from djai.models import CompatibilityBreakdown, TransitionMeasurement, TransitionPlan
from djai.training.lab_dataset import create_training_snapshot
from djai.training.taste_evidence import capture_rehearsal_evidence
from djai.training.taste_learning import train_taste_learning_heads


def _plan(track_a: str, track_b: str, signature: str, *, score: float = 0.8, overall: float = 0.8) -> TransitionPlan:
    measurement = TransitionMeasurement(
        version="3.0.0",
        sample_rate=44_100,
        duration_seconds=2.0,
        loudness_continuity=overall,
        spectral_continuity=overall,
        bass_cleanliness=overall,
        phase_safety=overall,
        artifact_safety=overall,
        silence_safety=overall,
        overall=overall,
        rhythm_coherence=overall,
        transient_clarity=overall,
        groove_continuity=overall,
        harmonic_continuity=overall,
        energy_intent_fit=overall,
        drop_payoff=overall,
        musical_overall=overall,
    )
    return TransitionPlan(
        track_a=track_a,
        track_b=track_b,
        exit_time_a=30.0,
        entry_time_b=8.0,
        transition_bars=4,
        transition_seconds=2.0,
        target_bpm=120.0,
        time_stretch_rate_b=1.0,
        time_stretch_applied=False,
        technique="equal_power_blend",
        score=score,
        analytical_score=score,
        measured_score=overall,
        measurement=measurement,
        measurement_passed=True,
        quality_checked=True,
        cue_confidence=score,
        intent_score=score,
        technique_score=score,
        exit_section="outro",
        entry_section="intro",
        execution_signature=signature,
        compatibility=CompatibilityBreakdown(
            tempo=score,
            harmonic=score,
            energy=score,
            texture=score,
            phrase=score,
            sections=score,
            overall=score,
            tempo_difference_percent=1.0,
            energy_delta=0.02,
            beat_grid=score,
            vocal_overlap=score,
            bass_conflict=score,
        ),
        metadata={
            "naturalness_selection_score": score,
            "professional_judge_score": score,
            "transition_naturalness_score": score,
            "rendered_transition_embedding": [score, 1.0 - score, 0.25, -0.1],
        },
    )


def _preview(path: Path, *, sample_rate: int = 44_100, pre: float = 1.0, transition: float = 2.0, post: float = 1.0) -> None:
    duration = pre + transition + post
    t = np.arange(int(round(sample_rate * duration)), dtype=np.float32) / sample_rate
    signal = 0.18 * np.sin(2.0 * np.pi * 220.0 * t) + 0.06 * np.sin(2.0 * np.pi * 880.0 * t)
    stereo = np.column_stack([signal, signal * 0.97]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, stereo, sample_rate, subtype="PCM_24")


def _session(workspace: Path, session_id: str, plan_a: TransitionPlan, plan_b: TransitionPlan) -> None:
    root = workspace / "output" / "listening-lab" / session_id
    root.mkdir(parents=True, exist_ok=True)
    audio_a = root / "candidate-a.wav"
    audio_b = root / "candidate-b.wav"
    _preview(audio_a)
    _preview(audio_b)
    manifest = {
        "session_id": session_id,
        "track_a": plan_a.track_a,
        "track_b": plan_a.track_b,
        "preview": {"pre_roll_seconds": 1.0, "post_roll_seconds": 1.0},
        "candidates": [
            {
                "label": "A",
                "audio_path": str(audio_a),
                "pre_roll_seconds": 1.0,
                "post_roll_seconds": 1.0,
                "plan": plan_features(plan_a),
                "plan_full": plan_a.to_dict(),
            },
            {
                "label": "B",
                "audio_path": str(audio_b),
                "pre_roll_seconds": 1.0,
                "post_roll_seconds": 1.0,
                "plan": plan_features(plan_b),
                "plan_full": plan_b.to_dict(),
            },
        ],
        "active_learning": {"blind_position_randomization": True},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_runtime_evidence_is_audio_free_and_path_pseudonymized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    secret_a = str(tmp_path / "audio" / "secret-a.wav")
    secret_b = str(tmp_path / "audio" / "secret-b.wav")
    first = _plan(secret_a, secret_b, "candidate-a", score=0.82)
    second = _plan(secret_a, secret_b, "candidate-b", score=0.64)

    path = capture_rehearsal_evidence(
        [first, second], selected_signature=first.execution_signature, context={"mode": "pair", "listener": "do-not-store"}
    )

    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert secret_a not in text
    assert secret_b not in text
    assert "do-not-store" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert len(rows) == 2
    assert sum(bool(row["selected"]) for row in rows) == 1
    assert rows[0]["candidate"]["track_a_id"].startswith("track-ref-")
    assert "track_a" not in rows[0]["candidate"]


def test_snapshot_adds_actual_audio_synthetic_rendered_judge_examples(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_TASTE_SYNTHETIC_FAILURES", "1")
    track_a = str(tmp_path / "audio" / "artist-a - track-a.wav")
    track_b = str(tmp_path / "audio" / "artist-b - track-b.wav")
    good = _plan(track_a, track_b, "session-good", score=0.86)
    alt = _plan(track_a, track_b, "session-alt", score=0.70)
    _session(tmp_path, "lab-one", good, alt)
    append_preference(
        candidate_a=plan_features(good),
        candidate_b=plan_features(alt),
        choice="a",
        listener="training-expert",
        context={"session_id": "lab-one"},
        path=tmp_path / "ratings" / "transition-preferences.jsonl",
    )

    snapshot = create_training_snapshot(tmp_path, name="synthetic")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (snapshot / "rendered_judge.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["counts"]["rendered_judge_human_examples"] == 2
    assert manifest["counts"]["rendered_judge_synthetic_examples"] >= 14
    assert any(row.get("label_source") == "synthetic_technical_failure" for row in rows)
    assert any(row.get("synthetic_failure", {}).get("kind") == "timing_shift" for row in rows)
    human_groups = {row["group"] for row in rows if row.get("label_source") != "synthetic_technical_failure"}
    synthetic_groups = {row["group"] for row in rows if row.get("label_source") == "synthetic_technical_failure"}
    assert human_groups == synthetic_groups
    assert manifest["rendered_judge_training"]["synthetic_failures"]["raw_audio_persisted"] is False


def test_taste_learning_trains_both_heads_but_keeps_zero_authority(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_TASTE_SYNTHETIC_FAILURES", "1")
    monkeypatch.setenv("SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED", "0")

    pairs = []
    for pair_index in range(2):
        track_a = str(tmp_path / "audio" / f"artist-{pair_index}-a - track.wav")
        track_b = str(tmp_path / "audio" / f"artist-{pair_index}-b - track.wav")
        good = _plan(track_a, track_b, f"session-{pair_index}-good", score=0.88)
        alt = _plan(track_a, track_b, f"session-{pair_index}-alt", score=0.62)
        _session(tmp_path, f"lab-{pair_index}", good, alt)
        pairs.append((track_a, track_b))

    preference_path = tmp_path / "ratings" / "transition-preferences.jsonl"
    for index in range(24):
        track_a, track_b = pairs[index % 2]
        good = _plan(track_a, track_b, f"vote-{index}-good", score=0.90 - 0.001 * index)
        weak = _plan(track_a, track_b, f"vote-{index}-weak", score=0.58 + 0.001 * index)
        if index % 2 == 0:
            candidate_a, candidate_b, choice = good, weak, "a"
        else:
            candidate_a, candidate_b, choice = weak, good, "b"
        append_preference(
            candidate_a=plan_features(candidate_a),
            candidate_b=plan_features(candidate_b),
            choice=choice,
            listener="training-expert",
            context={"session_id": f"vote-session-{index}"},
            path=preference_path,
        )

    snapshot = create_training_snapshot(tmp_path, name="taste-two-heads")
    result = train_taste_learning_heads(tmp_path, snapshot_id=snapshot.name)

    assert result["global_taste_ranker"]["status"] == "trained"
    assert result["global_taste_ranker"]["shadow_only"] is True
    assert result["rendered_dj_judge"]["status"] == "trained"
    assert result["rendered_dj_judge"]["shadow_only"] is True
    assert result["rendered_dj_judge"]["runtime_shadow_checkpoint"]["installed"] is True
    assert (tmp_path / "config" / "neural" / "rendered-judge.json").exists()

    registry = load_model_registry(tmp_path / "config" / "model-registry.json")
    rendered_manifest = registry.get("rendered_dj_judge")
    assert rendered_manifest is not None
    assert rendered_manifest.authority == "shadow"
    assert registry.authority("rendered_dj_judge", 1.0) == 0.0
    assert result["policy"]["hard_audio_safety_authoritative"] is True
    assert result["policy"]["training_changes_transition_acceptance"] is False
