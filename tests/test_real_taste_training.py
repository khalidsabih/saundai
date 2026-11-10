from __future__ import annotations

import json
from pathlib import Path

from djai.intelligence.neural_policy import rendered_judge_payload_features
from djai.intelligence.preferences import comparison_id
from djai.training.real_taste import (
    build_teacher_pairwise_preferences,
    real_taste_status,
    split_human_preferences,
    train_real_taste_from_snapshot,
)


def _candidate(index: int, side: str, preferred: bool) -> dict:
    base = 0.78 if preferred else 0.38
    state_id = f"state-{index:03d}"
    return {
        "execution_signature": f"sig-{index:03d}-{side}",
        "track_a": f"track-a-{index:03d}",
        "track_b": f"track-b-{index:03d}",
        "pair_id": f"pair-{index:03d}",
        "style": "safe",
        "intent": "groove_handoff" if preferred else "dramatic_switch",
        "technique": "eq_bass_swap" if preferred else "quick_cut",
        "transition_bars": 8 if preferred else 2,
        "transition_seconds": 15.0 if preferred else 4.0,
        "exit_section": "outro",
        "entry_section": "intro",
        "cue_confidence": base,
        "analytical_score": base,
        "measured_score": base,
        "intent_score": base,
        "technique_score": base,
        "tempo_difference_percent": 1.5 if preferred else 9.0,
        "energy_delta": 0.05 if preferred else 0.35,
        "embedding_similarity": base,
        "set_context_score": base,
        "rejection_penalty": 0.0 if preferred else 0.4,
        "compatibility": {
            "tempo": base,
            "harmonic": base,
            "energy": base,
            "texture": base,
            "beat_grid": base,
            "phrase": base,
            "sections": base,
            "genre": base,
            "vocal_overlap": base,
            "bass_conflict": base,
            "boundary": base,
            "section_fit": base,
        },
        "measurement": {
            "overall": base,
            "musical_overall": base,
            "loudness_continuity": base,
            "spectral_continuity": base,
            "bass_cleanliness": base,
            "phase_safety": base,
            "artifact_safety": base,
            "silence_safety": base,
            "rhythm_coherence": base,
            "transient_clarity": base,
            "groove_continuity": base,
            "harmonic_continuity": base,
            "energy_intent_fit": base,
            "drop_payoff": base,
            "vocal_cleanliness": base,
        },
        "dj_state": {
            "source_bar": index % 64,
            "current_energy": 0.55,
            "current_tension": 0.45,
            "current_vocal_probability": 0.2,
            "current_bass_activity": 0.4,
            "set_energy": 0.58,
            "set_tension": 0.44,
            "set_groove": 0.7,
            "set_vocal_density": 0.25,
            "effective_bpm": 124.0,
            "pending_commitment_count": 0,
        },
        "dj_move": {
            "action": "transition",
            "transition_bars": 8 if preferred else 2,
            "wait_bars": 0,
            "matched_bpm": 124.0,
            "incoming_native_bpm": 125.0,
            "post_transition_bpm": 125.0,
            "tempo_release_seconds": 4.0,
            "phase_nudge_seconds": 0.005,
            "future_optionality": base,
            "score": base,
            "technique": "eq_bass_swap" if preferred else "quick_cut",
            "intent": "groove_handoff" if preferred else "dramatic_switch",
            "projected_state": {
                "current_energy": 0.6 if preferred else 0.9,
                "current_tension": 0.5,
                "current_vocal_probability": 0.2,
                "current_bass_activity": 0.42,
            },
            "protected_payoff_violation": False,
            "payoff_satisfied": preferred,
            "structural_operations": [],
        },
        "state_context_id": state_id,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path
    dataset_id = "dataset-20261110-real-taste-test"
    snapshot = root / "datasets/training-lab" / dataset_id
    snapshot.mkdir(parents=True)
    (root / "datasets/training-lab/LATEST").write_text(dataset_id + "\n", encoding="utf-8")
    (snapshot / "manifest.json").write_text(
        json.dumps({"dataset_id": dataset_id, "counts": {"decisive_preferences": 40}}) + "\n",
        encoding="utf-8",
    )

    human_rows = []
    for index in range(40):
        preferred_a = index % 2 == 0
        a = _candidate(index, "a", preferred_a)
        b = _candidate(index, "b", not preferred_a)
        choice = "a" if preferred_a else "b"
        human_rows.append(
            {
                "comparison_id": comparison_id(a, b),
                "listener": f"human-professional_dj-{index % 5}",
                "choice": choice,
                "weight": 1.15,
                "candidate_a": a,
                "candidate_b": b,
                "context": {
                    "label_source": "human_blind_preference",
                    "human_preference": True,
                    "teacher_scenario_id": f"human-scenario-{index:03d}",
                    "evaluator_cohort": "professional_dj" if index % 2 == 0 else "experienced_dj",
                    "reasons": ["phrasing", "bass_handoff"],
                },
            }
        )
    _write_jsonl(snapshot / "preferences.jsonl", human_rows)

    synthetic_rows = []
    for index in range(12):
        candidate = _candidate(100 + index, "s", index % 2 == 0)
        synthetic_rows.append(
            {
                "id": f"synthetic-{index}",
                "group": f"synthetic-group-{index // 2}",
                "target": 1.0 if index % 2 == 0 else 0.0,
                "weight": 1.0,
                "rights_cleared": True,
                "label_source": "synthetic_failure" if index % 2 else "synthetic_clean",
                "features": rendered_judge_payload_features(candidate),
            }
        )
    _write_jsonl(snapshot / "rendered_judge.jsonl", synthetic_rows)

    scenarios = []
    consensus = []
    for index in range(30):
        preferred_a = index % 2 == 0
        a = _candidate(200 + index, "a", preferred_a)
        b = _candidate(200 + index, "b", not preferred_a)
        scenario_id = f"teacher-scenario-{index:03d}"
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "task": "transition_preference",
                "prompt_version": "1.0",
                "context": {"journey": "balanced"},
                "candidates": [a, b],
                "requested_labels": [],
            }
        )
        consensus.append(
            {
                "scenario_id": scenario_id,
                "task": "transition_preference",
                "decisive": True,
                "consensus_choice": "a" if preferred_a else "b",
                "agreement": 1.0,
                "calibrated_mean_confidence": 0.84,
                "mean_confidence": 0.9,
                "reason_codes": ["phrasing", "bass"],
            }
        )
    teacher_root = root / "datasets/generated/teacher"
    _write_jsonl(teacher_root / "scenarios.jsonl", scenarios)
    _write_jsonl(teacher_root / "consensus.jsonl", consensus)
    return root, dataset_id


def test_human_split_keeps_comparison_groups_out_of_both_sides(tmp_path: Path):
    root, dataset_id = _workspace(tmp_path)
    snapshot = root / "datasets/training-lab" / dataset_id
    result = split_human_preferences(snapshot / "preferences.jsonl", snapshot / "real-taste-split")

    fit = [json.loads(line) for line in Path(result["fit_path"]).read_text().splitlines()]
    heldout = [json.loads(line) for line in Path(result["heldout_path"]).read_text().splitlines()]
    fit_ids = {row["comparison_id"] for row in fit}
    heldout_ids = {row["comparison_id"] for row in heldout}

    assert fit_ids
    assert heldout_ids
    assert fit_ids.isdisjoint(heldout_ids)
    assert result["fit_examples"] >= 24


def test_teacher_consensus_is_converted_to_low_weight_research_pretraining(tmp_path: Path):
    root, _ = _workspace(tmp_path)
    teacher_root = root / "datasets/generated/teacher"
    result = build_teacher_pairwise_preferences(
        teacher_root / "scenarios.jsonl",
        teacher_root / "consensus.jsonl",
        teacher_root / "pretrain.jsonl",
    )

    rows = [json.loads(line) for line in Path(result["path"]).read_text().splitlines()]
    assert len(rows) == 30
    assert all(row["context"]["human_preference"] is False for row in rows)
    assert all(row["context"]["production_authority"] is False for row in rows)
    assert max(row["weight"] for row in rows) <= 0.45


def test_real_taste_training_stages_teacher_human_and_heldout_evaluation(tmp_path: Path):
    root, dataset_id = _workspace(tmp_path)

    result = train_real_taste_from_snapshot(root, snapshot_id=dataset_id, ridge=0.12)

    assert result["teacher_pretraining"]["status"] == "trained"
    assert result["human_fine_tuning"]["status"] == "trained"
    assert result["human_fine_tuning"]["fine_tuned_from_teacher"] is True
    assert result["heldout_human_evaluation"]["status"] == "evaluated"
    assert result["heldout_human_evaluation"]["examples"] > 0
    assert result["rendered_dj_judge"]["synthetic_pretraining"]["status"] == "trained"
    assert result["rendered_dj_judge"]["human_fine_tuning"]["status"] == "trained"
    assert result["rendered_dj_judge"]["heldout_human_evaluation"]["status"] == "evaluated"
    assert result["state_move_taste"]["status"] == "trained"
    assert result["state_move_taste"]["heldout_human_evaluation"]["status"] == "evaluated"
    assert result["policy"]["all_new_models_shadow_only"] is True
    assert result["policy"]["model_registry_authority_changed"] is False

    human_model = root / result["human_fine_tuning"]["model_path"]
    payload = json.loads(human_model.read_text(encoding="utf-8"))
    assert payload["shadow_only"] is True
    assert payload["authority"] == 0.0
    assert payload["promoted_to_live"] is False
    assert payload["fine_tuned_from_model_id"]


def test_real_taste_status_reports_latest_staged_training(tmp_path: Path):
    root, dataset_id = _workspace(tmp_path)
    train_real_taste_from_snapshot(root, snapshot_id=dataset_id)

    status = real_taste_status(root)

    assert status["available"] is True
    assert status["dataset_id"] == dataset_id
    assert status["human_fine_tuning"]["status"] == "trained"
    assert status["policy"]["heldout_human_labels_are_never_used_for_fit"] is True
