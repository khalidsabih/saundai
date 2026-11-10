from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..intelligence.preferences import comparison_id, load_preferences
from ..intelligence.transition_ranker import evaluate_transition_ranker, train_transition_ranker
from .state_move_taste import evaluate_state_move_taste_model, train_state_move_taste_shadow
from .lab_dataset import _rendered_judge_training_rows, _resolve_snapshot
from .neural_heads import evaluate_neural_head, train_neural_head
from .teacher_dataset import load_scenarios

REAL_TASTE_SCHEMA_VERSION = "1.0"
TEACHER_WEIGHT_MAX = 0.45
HELDOUT_BUCKETS = {8, 9}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 10


def _human_group(row: Mapping[str, Any]) -> str:
    context = row.get("context") if isinstance(row.get("context"), Mapping) else {}
    scenario = str(context.get("teacher_scenario_id") or "")
    if scenario:
        return f"scenario:{scenario}"
    cid = str(row.get("comparison_id") or "")
    if cid:
        return f"comparison:{cid}"
    a = row.get("candidate_a") if isinstance(row.get("candidate_a"), Mapping) else {}
    b = row.get("candidate_b") if isinstance(row.get("candidate_b"), Mapping) else {}
    return f"comparison:{comparison_id(dict(a), dict(b))}"


def split_human_preferences(
    preferences_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Freeze a leakage-resistant human fit/held-out split by comparison group."""
    rows = load_preferences(preferences_path)
    decisive = [row for row in rows if str(row.get("choice") or "") in {"a", "b"}]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in decisive:
        grouped.setdefault(_human_group(row), []).append(row)

    groups = sorted(grouped)
    heldout_groups = {group for group in groups if _bucket(group) in HELDOUT_BUCKETS}
    if len(groups) >= 3 and not heldout_groups:
        heldout_groups = {groups[-1]}
    if len(groups) >= 2 and len(heldout_groups) == len(groups):
        heldout_groups = {groups[-1]}

    fit: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for group in groups:
        target = heldout if group in heldout_groups else fit
        target.extend(grouped[group])

    root = Path(output_root).expanduser().resolve()
    fit_path = _write_jsonl(root / "human-fit.jsonl", fit)
    heldout_path = _write_jsonl(root / "human-heldout.jsonl", heldout)
    return {
        "fit_path": str(fit_path),
        "heldout_path": str(heldout_path),
        "fit_examples": len(fit),
        "heldout_examples": len(heldout),
        "fit_groups": len(groups) - len(heldout_groups),
        "heldout_groups": len(heldout_groups),
        "heldout_group_ids": sorted(heldout_groups),
        "split_key": "teacher_scenario_id_else_comparison_id",
    }


def build_teacher_pairwise_preferences(
    scenarios_path: str | Path,
    consensus_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert decisive teacher consensus into low-authority pairwise pretraining rows."""
    scenarios = {
        str(row.get("scenario_id") or ""): row
        for row in load_scenarios(scenarios_path)
        if str(row.get("scenario_id") or "")
    }
    consensus_rows = _read_jsonl(Path(consensus_path).expanduser().resolve())
    output: list[dict[str, Any]] = []
    skipped = Counter()
    for row in consensus_rows:
        if not bool(row.get("decisive")):
            skipped["not_decisive"] += 1
            continue
        choice = str(row.get("consensus_choice") or "")
        if choice not in {"a", "b"}:
            skipped["non_pairwise_choice"] += 1
            continue
        scenario_id = str(row.get("scenario_id") or "")
        scenario = scenarios.get(scenario_id)
        if not isinstance(scenario, Mapping):
            skipped["missing_scenario"] += 1
            continue
        candidates = scenario.get("candidates") if isinstance(scenario.get("candidates"), list) else []
        if len(candidates) != 2 or not all(isinstance(item, Mapping) for item in candidates):
            skipped["invalid_candidate_pair"] += 1
            continue
        agreement = max(0.0, min(1.0, float(row.get("agreement", 0.0) or 0.0)))
        confidence = max(
            0.0,
            min(1.0, float(row.get("calibrated_mean_confidence", row.get("mean_confidence", 0.0)) or 0.0)),
        )
        weight = max(0.05, min(TEACHER_WEIGHT_MAX, 0.15 + 0.30 * agreement * confidence))
        candidate_a = dict(candidates[0])
        candidate_b = dict(candidates[1])
        output.append(
            {
                "schema_version": REAL_TASTE_SCHEMA_VERSION,
                "comparison_id": comparison_id(candidate_a, candidate_b),
                "listener": "llm-teacher-consensus",
                "choice": choice,
                "weight": round(weight, 6),
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "context": {
                    "label_source": "llm_teacher_consensus_research",
                    "human_preference": False,
                    "production_authority": False,
                    "teacher_scenario_id": scenario_id,
                    "agreement": agreement,
                    "calibrated_confidence": confidence,
                    "reason_codes": list(row.get("reason_codes") or []),
                },
            }
        )
    destination = _write_jsonl(Path(output_path).expanduser().resolve(), output)
    return {
        "path": str(destination),
        "examples": len(output),
        "skipped": dict(skipped),
        "maximum_teacher_weight": TEACHER_WEIGHT_MAX,
        "human_preference": False,
        "production_authority": False,
    }


def _reason_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        context = row.get("context") if isinstance(row.get("context"), Mapping) else {}
        reasons = context.get("reasons") if isinstance(context.get("reasons"), list) else []
        for reason in reasons:
            counts[str(reason)] += 1
    return dict(counts)


def train_real_taste_models(
    workspace: str | Path,
    *,
    preferences_path: str | Path,
    scenarios_path: str | Path,
    consensus_path: str | Path,
    dataset_id: str,
    ridge: float = 0.18,
) -> dict[str, Any]:
    """Teacher pretrain -> human fine-tune -> frozen human benchmark.

    All resulting checkpoints remain shadow-only. This function does not modify
    model-registry authority or live ranking.
    """
    root = Path(workspace).expanduser().resolve()
    stage_root = root / "datasets" / "training-lab" / dataset_id / "real-taste"
    stage_root.mkdir(parents=True, exist_ok=True)

    split = split_human_preferences(preferences_path, stage_root)
    teacher = build_teacher_pairwise_preferences(
        scenarios_path,
        consensus_path,
        stage_root / "teacher-pretrain.jsonl",
    )

    model_root = root / "config" / "training-lab-models"
    model_root.mkdir(parents=True, exist_ok=True)
    teacher_model = model_root / f"global-taste-teacher-{dataset_id}.json"
    human_model = model_root / f"global-taste-human-{dataset_id}.json"

    teacher_stage: dict[str, Any]
    try:
        trained_teacher = train_transition_ranker(
            teacher["path"],
            teacher_model,
            ridge=ridge,
        )
        teacher_stage = {
            "status": "trained",
            "model_path": str(trained_teacher.relative_to(root)),
            "examples": teacher["examples"],
        }
    except ValueError as exc:
        teacher_stage = {
            "status": "blocked",
            "reason": str(exc),
            "examples": teacher["examples"],
        }

    initial_model = teacher_model if teacher_stage["status"] == "trained" else None
    human_stage: dict[str, Any]
    try:
        trained_human = train_transition_ranker(
            split["fit_path"],
            human_model,
            ridge=ridge,
            initial_model_path=initial_model,
        )
        human_payload = json.loads(trained_human.read_text(encoding="utf-8"))
        would_be_authority = float(human_payload.get("authority", 0.0) or 0.0)
        acceptability = human_payload.get("acceptability")
        would_be_acceptability_authority = (
            float(acceptability.get("authority", 0.0) or 0.0)
            if isinstance(acceptability, dict)
            else 0.0
        )
        if isinstance(acceptability, dict):
            acceptability["would_be_authority"] = round(would_be_acceptability_authority, 6)
            acceptability["authority"] = 0.0
        human_payload.update(
            {
                "model_id": f"global-taste-real-{dataset_id}",
                "learning_head": "global_taste_ranker",
                "training_dataset_id": dataset_id,
                "training_hierarchy": [
                    "llm_teacher_consensus_pretrain",
                    "human_blind_preference_finetune",
                    "heldout_human_evaluation",
                ],
                "teacher_pretrain_model": (
                    str(teacher_model.relative_to(root))
                    if teacher_stage["status"] == "trained"
                    else None
                ),
                "human_fit_examples": split["fit_examples"],
                "human_heldout_examples": split["heldout_examples"],
                "shadow_only": True,
                "would_be_authority": round(would_be_authority, 6),
                "authority": 0.0,
                "promoted_to_live": False,
                "safety_contract": "Taste model reranks safe candidates only. Hard technical and musical gates remain authoritative.",
            }
        )
        trained_human.write_text(json.dumps(human_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (model_root / "GLOBAL_TASTE_LATEST").write_text(trained_human.name + "\n", encoding="utf-8")
        human_stage = {
            "status": "trained",
            "model_path": str(trained_human.relative_to(root)),
            "fit_examples": split["fit_examples"],
            "fine_tuned_from_teacher": teacher_stage["status"] == "trained",
        }
    except ValueError as exc:
        human_stage = {
            "status": "blocked",
            "reason": str(exc),
            "fit_examples": split["fit_examples"],
        }

    heldout_evaluation: dict[str, Any] = {
        "status": "blocked",
        "reason": "human fine-tune model unavailable",
    }
    if human_stage["status"] == "trained":
        if split["heldout_examples"] <= 0:
            heldout_evaluation = {
                "status": "blocked",
                "reason": "No held-out human comparisons are available.",
            }
        else:
            heldout_evaluation = {
                "status": "evaluated",
                **evaluate_transition_ranker(split["heldout_path"], human_model),
            }

    fit_rows = load_preferences(split["fit_path"])
    heldout_rows = load_preferences(split["heldout_path"])

    snapshot_root = root / "datasets" / "training-lab" / dataset_id
    rendered_source = snapshot_root / "rendered_judge.jsonl"
    rendered_rows = _read_jsonl(rendered_source)
    synthetic_rendered_rows = [
        row for row in rendered_rows
        if str(row.get("label_source") or "").startswith("synthetic")
    ]
    fit_human_rendered = _rendered_judge_training_rows(
        fit_rows,
        rights_cleared=False,
    )
    heldout_human_rendered = _rendered_judge_training_rows(
        heldout_rows,
        rights_cleared=False,
    )
    synthetic_path = _write_jsonl(stage_root / "rendered-synthetic-pretrain.jsonl", synthetic_rendered_rows)
    fit_rendered_path = _write_jsonl(stage_root / "rendered-human-fit.jsonl", fit_human_rendered)
    heldout_rendered_path = _write_jsonl(stage_root / "rendered-human-heldout.jsonl", heldout_human_rendered)

    rendered_teacher_model = model_root / f"rendered-dj-judge-synthetic-{dataset_id}.json"
    rendered_human_model = model_root / f"rendered-dj-judge-human-{dataset_id}.json"
    rendered_pretrain: dict[str, Any]
    try:
        trained = train_neural_head(
            "rendered_judge",
            synthetic_path,
            rendered_teacher_model,
            require_rights_cleared=False,
            iterations=900,
            learning_rate=0.03,
            l2=0.003,
        )
        rendered_pretrain = {
            "status": "trained",
            "model_path": str(trained.relative_to(root)),
            "examples": len(synthetic_rendered_rows),
        }
    except ValueError as exc:
        rendered_pretrain = {
            "status": "blocked",
            "reason": str(exc),
            "examples": len(synthetic_rendered_rows),
        }

    rendered_finetune: dict[str, Any]
    try:
        trained = train_neural_head(
            "rendered_judge",
            fit_rendered_path,
            rendered_human_model,
            require_rights_cleared=False,
            iterations=700,
            learning_rate=0.02,
            l2=0.003,
            initial_model_path=rendered_teacher_model if rendered_pretrain["status"] == "trained" else None,
        )
        payload = json.loads(trained.read_text(encoding="utf-8"))
        payload.update(
            {
                "model_id": f"rendered-dj-judge-real-{dataset_id}",
                "learning_head": "rendered_dj_judge",
                "training_dataset_id": dataset_id,
                "training_hierarchy": [
                    "synthetic_audio_pretrain",
                    "human_blind_preference_finetune",
                    "heldout_human_evaluation",
                ],
                "shadow_only": True,
                "authority": 0.0,
                "promoted_to_live": False,
                "safety_contract": "Rendered DJ Judge is preference evidence only; hard audio failures remain authoritative.",
            }
        )
        trained.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (model_root / "RENDERED_JUDGE_LATEST").write_text(trained.name + "\n", encoding="utf-8")
        rendered_finetune = {
            "status": "trained",
            "model_path": str(trained.relative_to(root)),
            "fit_examples": len(fit_human_rendered),
            "fine_tuned_from_synthetic": rendered_pretrain["status"] == "trained",
        }
    except ValueError as exc:
        rendered_finetune = {
            "status": "blocked",
            "reason": str(exc),
            "fit_examples": len(fit_human_rendered),
        }

    rendered_heldout: dict[str, Any] = {
        "status": "blocked",
        "reason": "rendered human fine-tune model unavailable",
    }
    if rendered_finetune["status"] == "trained":
        if heldout_human_rendered:
            rendered_heldout = {
                "status": "evaluated",
                **evaluate_neural_head(
                    "rendered_judge",
                    heldout_rendered_path,
                    rendered_human_model,
                    require_rights_cleared=False,
                ),
            }
        else:
            rendered_heldout = {
                "status": "blocked",
                "reason": "No held-out human rendered-judge rows are available.",
            }

    state_move_stage: dict[str, Any]
    try:
        trained_state_move = train_state_move_taste_shadow(
            root,
            preferences_path=split["fit_path"],
            dataset_id=f"{dataset_id}-human-fit",
            ridge=ridge,
        )
        state_eval = {
            "status": "blocked",
            "reason": "No held-out human state-move comparisons are available.",
        }
        if split["heldout_examples"] > 0:
            state_eval = {
                "status": "evaluated",
                **evaluate_state_move_taste_model(
                    split["heldout_path"],
                    root / trained_state_move["model_path"],
                ),
            }
        state_move_stage = {
            "status": "trained",
            **trained_state_move,
            "heldout_reserved": split["heldout_examples"],
            "heldout_human_evaluation": state_eval,
        }
    except ValueError as exc:
        state_move_stage = {
            "status": "blocked",
            "reason": str(exc),
            "fit_examples": split["fit_examples"],
            "heldout_reserved": split["heldout_examples"],
        }

    result = {
        "schema_version": REAL_TASTE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "teacher_pretraining": teacher_stage,
        "human_fine_tuning": human_stage,
        "heldout_human_evaluation": heldout_evaluation,
        "rendered_dj_judge": {
            "synthetic_pretraining": rendered_pretrain,
            "human_fine_tuning": rendered_finetune,
            "heldout_human_evaluation": rendered_heldout,
        },
        "state_move_taste": state_move_stage,
        "data": {
            "teacher": teacher,
            "human_split": split,
            "human_fit_reasons": _reason_counts(fit_rows),
            "human_heldout_reasons": _reason_counts(heldout_rows),
        },
        "policy": {
            "teacher_labels_are_weak_supervision": True,
            "human_labels_are_fine_tuning_truth": True,
            "heldout_human_labels_are_never_used_for_fit": True,
            "all_new_models_shadow_only": True,
            "hard_safety_authoritative": True,
            "model_registry_authority_changed": False,
        },
    }
    report_path = stage_root / "training-report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_path"] = str(report_path.relative_to(root))
    return result



def train_real_taste_from_snapshot(
    workspace: str | Path,
    *,
    snapshot_id: str | None = None,
    ridge: float = 0.18,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    snapshot = _resolve_snapshot(root, snapshot_id)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    dataset_id = str(manifest.get("dataset_id") or snapshot.name)
    scenarios = root / "datasets/generated/teacher/scenarios.jsonl"
    consensus = root / "datasets/generated/teacher/consensus.jsonl"
    return train_real_taste_models(
        root,
        preferences_path=snapshot / "preferences.jsonl",
        scenarios_path=scenarios,
        consensus_path=consensus,
        dataset_id=dataset_id,
        ridge=ridge,
    )


def real_taste_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    dataset_root = root / "datasets/training-lab"
    latest = dataset_root / "LATEST"
    if not latest.exists():
        return {
            "schema_version": REAL_TASTE_SCHEMA_VERSION,
            "available": False,
            "reason": "no training snapshot",
        }
    dataset_id = latest.read_text(encoding="utf-8").strip()
    report = dataset_root / dataset_id / "real-taste/training-report.json"
    if not report.exists():
        return {
            "schema_version": REAL_TASTE_SCHEMA_VERSION,
            "available": False,
            "dataset_id": dataset_id,
            "reason": "real taste model training has not run for latest snapshot",
        }
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": REAL_TASTE_SCHEMA_VERSION,
            "available": False,
            "dataset_id": dataset_id,
            "reason": "training report unreadable",
        }
    return {
        "schema_version": REAL_TASTE_SCHEMA_VERSION,
        "available": True,
        "dataset_id": dataset_id,
        "report_path": str(report.relative_to(root)),
        "teacher_pretraining": payload.get("teacher_pretraining", {}),
        "human_fine_tuning": payload.get("human_fine_tuning", {}),
        "heldout_human_evaluation": payload.get("heldout_human_evaluation", {}),
        "rendered_dj_judge": payload.get("rendered_dj_judge", {}),
        "state_move_taste": payload.get("state_move_taste", {}),
        "policy": payload.get("policy", {}),
    }
