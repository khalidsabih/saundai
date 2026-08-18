from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..intelligence.model_registry import load_model_registry
from ..intelligence.neural_policy import clear_neural_policy_cache
from ..intelligence.transition_ranker import predict_plan
from ..models import TransitionPlan
from .lab_dataset import _resolve_snapshot, train_shadow_ranker
from .neural_heads import train_neural_head
from .taste_evidence import runtime_evidence_status
from .state_move_taste import (
    MINIMUM_STATE_MOVE_COMPARISONS,
    state_move_taste_status,
    train_state_move_taste_shadow,
)

TASTE_LEARNING_SCHEMA_VERSION = "1.8.1"
MINIMUM_RENDERED_JUDGE_EXAMPLES = 12
MINIMUM_RENDERED_JUDGE_GROUPS = 2


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonl_stats(path: Path) -> dict[str, Any]:
    examples = positives = negatives = embeddings = rights_cleared = 0
    groups: set[str] = set()
    labels: dict[str, int] = {}
    weighted_positive = weighted_total = 0.0
    if not path.exists():
        return {
            "examples": 0,
            "groups": 0,
            "positive": 0,
            "negative": 0,
            "embedding_examples": 0,
            "rights_cleared": 0,
            "label_sources": {},
            "baseline_accuracy": None,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("features"), dict):
                continue
            examples += 1
            target = float(np.clip(float(row.get("target", 0.0) or 0.0), 0.0, 1.0))
            weight = max(0.05, min(10.0, float(row.get("weight", 1.0) or 1.0)))
            is_positive = target >= 0.5
            positives += int(is_positive)
            negatives += int(not is_positive)
            weighted_positive += weight * float(is_positive)
            weighted_total += weight
            if row.get("group"):
                groups.add(str(row["group"]))
            embeddings += int(bool(row.get("render_embedding_present")))
            rights_cleared += int(bool(row.get("rights_cleared")))
            source = str(row.get("label_source") or "unknown")
            labels[source] = labels.get(source, 0) + 1
    baseline = None
    if weighted_total > 0.0:
        share = weighted_positive / weighted_total
        baseline = max(share, 1.0 - share)
    return {
        "examples": examples,
        "groups": len(groups),
        "positive": positives,
        "negative": negatives,
        "embedding_examples": embeddings,
        "rights_cleared": rights_cleared,
        "label_sources": labels,
        "baseline_accuracy": baseline,
    }


def _model_root(workspace: Path) -> Path:
    path = workspace / "config" / "training-lab-models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_model(workspace: Path, pointer: str, glob_pattern: str) -> Path | None:
    root = _model_root(workspace)
    pointer_path = root / pointer
    if pointer_path.exists():
        candidate = root / pointer_path.read_text(encoding="utf-8").strip()
        if candidate.exists():
            return candidate
    candidates = sorted(root.glob(glob_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def shadow_global_taste_prediction(
    plan: TransitionPlan,
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = (
        Path(workspace).expanduser().resolve()
        if workspace is not None
        else Path(os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()
    )
    model_path = _latest_model(root, "GLOBAL_TASTE_LATEST", "global-taste-ranker-*.json")
    if model_path is None:
        return {"available": False, "authority": 0.0, "reason": "no trained global taste shadow"}
    model = _read_json(model_path)
    if not model:
        return {"available": False, "authority": 0.0, "reason": "global taste shadow unreadable"}
    prediction = predict_plan(plan, model=model)
    return {
        "available": True,
        "model_id": model.get("model_id", model_path.stem),
        "model_path": str(model_path),
        "score": round(float(prediction.learned_score), 6),
        "acceptability": round(float(prediction.acceptability_score), 6),
        "confidence": round(float(prediction.confidence), 6),
        "authority": 0.0,
        "would_be_authority": round(float(prediction.authority), 6),
        "shadow_only": True,
        "training_dataset_id": model.get("training_dataset_id"),
    }


def _install_rendered_shadow_checkpoint(workspace: Path, source: Path) -> dict[str, Any]:
    registry_path = workspace / "config" / "model-registry.json"
    registry = load_model_registry(registry_path)
    manifest = registry.get("rendered_dj_judge")
    if manifest is None:
        return {"installed": False, "reason": "rendered_dj_judge is absent from model registry"}
    if manifest.authority not in {"shadow", "disabled"} or manifest.production_approved:
        return {
            "installed": False,
            "reason": f"registry authority is {manifest.authority}; refusing to replace a non-shadow checkpoint",
        }
    if manifest.checkpoint_sha256:
        return {"installed": False, "reason": "registry checkpoint is hash-pinned; shadow training cannot replace it"}
    checkpoint_value = manifest.checkpoint or "config/neural/rendered-judge.json"
    destination = Path(checkpoint_value).expanduser()
    if not destination.is_absolute():
        destination = workspace / destination
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    clear_neural_policy_cache()
    return {
        "installed": True,
        "path": str(destination),
        "registry_authority": manifest.authority,
        "effective_authority": 0.0,
    }


def train_taste_learning_heads(
    workspace: str | Path,
    *,
    snapshot_id: str | None = None,
    ridge: float = 0.18,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    snapshot = _resolve_snapshot(root, snapshot_id)
    manifest = _read_json(snapshot / "manifest.json")
    dataset_id = str(manifest.get("dataset_id") or snapshot.name)
    model_root = _model_root(root)

    global_result: dict[str, Any]
    try:
        global_result = {"status": "trained", **train_shadow_ranker(root, snapshot_id=dataset_id, ridge=ridge)}
    except ValueError as exc:
        global_result = {
            "status": "blocked",
            "learning_head": "global_taste_ranker",
            "shadow_only": True,
            "reason": str(exc),
        }

    state_move_result: dict[str, Any]
    try:
        state_move_result = {
            "status": "trained",
            **train_state_move_taste_shadow(
                root,
                preferences_path=snapshot / "preferences.jsonl",
                dataset_id=dataset_id,
                ridge=ridge,
            ),
        }
    except ValueError as exc:
        state_move_result = {
            "status": "blocked",
            "learning_head": "state_move_taste",
            "shadow_only": True,
            "authority": 0.0,
            "reason": str(exc),
        }

    judge_dataset = snapshot / "rendered_judge.jsonl"
    judge_stats = _jsonl_stats(judge_dataset)
    rendered_result: dict[str, Any]
    if judge_stats["examples"] < MINIMUM_RENDERED_JUDGE_EXAMPLES:
        rendered_result = {
            "status": "blocked",
            "learning_head": "rendered_dj_judge",
            "shadow_only": True,
            "reason": f"Need at least {MINIMUM_RENDERED_JUDGE_EXAMPLES} rendered judge examples; snapshot has {judge_stats['examples']}.",
            "dataset": judge_stats,
        }
    elif judge_stats["groups"] < MINIMUM_RENDERED_JUDGE_GROUPS:
        rendered_result = {
            "status": "blocked",
            "learning_head": "rendered_dj_judge",
            "shadow_only": True,
            "reason": f"Need at least {MINIMUM_RENDERED_JUDGE_GROUPS} independent track-pair groups for held-out validation; snapshot has {judge_stats['groups']}.",
            "dataset": judge_stats,
        }
    else:
        model_name = f"rendered-dj-judge-{dataset_id.removeprefix('dataset-')}.json"
        destination = model_root / model_name
        trained = train_neural_head(
            "rendered_judge",
            judge_dataset,
            destination,
            require_rights_cleared=False,
            iterations=1000,
            learning_rate=0.03,
            l2=0.003,
        )
        payload = _read_json(trained)
        metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {}
        payload.update(
            {
                "learning_head": "rendered_dj_judge",
                "training_dataset_id": dataset_id,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "shadow_only": True,
                "promoted_to_live": False,
                "production_training_ready": bool(manifest.get("production_training_ready", False)),
                "dataset_stats": judge_stats,
                "safety_contract": "Rendered DJ Judge is preference evidence only. Hard audio safety remains authoritative and non-overridable.",
            }
        )
        if judge_stats.get("baseline_accuracy") is not None:
            metrics["baseline_accuracy"] = round(float(judge_stats["baseline_accuracy"]), 6)
        payload["metrics"] = metrics
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        (model_root / "RENDERED_JUDGE_LATEST").write_text(destination.name + "\n", encoding="utf-8")
        install = _install_rendered_shadow_checkpoint(root, destination)
        rendered_result = {
            "status": "trained",
            "learning_head": "rendered_dj_judge",
            "model_id": payload.get("model_id", "rendered_dj_judge"),
            "model_path": str(destination.relative_to(root)),
            "dataset_id": dataset_id,
            "shadow_only": True,
            "validation_accuracy": metrics.get("validation_accuracy"),
            "validation_mae": metrics.get("validation_mae"),
            "baseline_accuracy": metrics.get("baseline_accuracy"),
            "validation_gain": (
                round(float(metrics.get("validation_accuracy", 0.0)) - float(metrics.get("baseline_accuracy", 0.0)), 6)
                if metrics.get("validation_accuracy") is not None and metrics.get("baseline_accuracy") is not None
                else None
            ),
            "dataset": judge_stats,
            "runtime_shadow_checkpoint": install,
            "production_training_ready": bool(manifest.get("production_training_ready", False)),
        }

    result = {
        "schema_version": TASTE_LEARNING_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "snapshot_path": str(snapshot.relative_to(root)),
        "global_taste_ranker": global_result,
        "state_move_taste": state_move_result,
        "rendered_dj_judge": rendered_result,
        "runtime_evidence": runtime_evidence_status(root),
        "policy": {
            "new_heads_shadow_only": True,
            "hard_audio_safety_authoritative": True,
            "training_changes_transition_acceptance": False,
            "research_data_can_promote_to_production": False,
        },
    }
    # Keep the 1.1.0 training endpoint contract readable to older local clients.
    # These aliases describe only the Global Taste Ranker; the two explicit
    # head payloads above are canonical in 1.1.1.
    if global_result.get("status") == "trained":
        for key in (
            "model_id",
            "model_path",
            "comparison_count",
            "cv_accuracy",
            "baseline_accuracy",
            "win_margin",
            "shadow_only",
        ):
            if key in global_result:
                result[key] = global_result[key]
    return result


def taste_learning_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    global_path = _latest_model(root, "GLOBAL_TASTE_LATEST", "global-taste-ranker-*.json")
    rendered_path = _latest_model(root, "RENDERED_JUDGE_LATEST", "rendered-dj-judge-*.json")
    global_payload = _read_json(global_path) if global_path else {}
    rendered_payload = _read_json(rendered_path) if rendered_path else {}
    rendered_metrics = rendered_payload.get("metrics") if isinstance(rendered_payload.get("metrics"), dict) else {}
    latest_snapshot: dict[str, Any] = {}
    latest_pointer = root / "datasets" / "training-lab" / "LATEST"
    if latest_pointer.exists():
        snapshot = root / "datasets" / "training-lab" / latest_pointer.read_text(encoding="utf-8").strip()
        latest_snapshot = _read_json(snapshot / "manifest.json")
    counts = latest_snapshot.get("counts") if isinstance(latest_snapshot.get("counts"), dict) else {}
    return {
        "schema_version": TASTE_LEARNING_SCHEMA_VERSION,
        "runtime_evidence": runtime_evidence_status(root),
        "latest_snapshot_id": latest_snapshot.get("dataset_id"),
        "global_taste_ranker": {
            "available": bool(global_payload),
            "shadow_only": True,
            "model_id": global_payload.get("model_id"),
            "comparison_count": int(global_payload.get("comparison_count", 0) or 0),
            "cv_accuracy": global_payload.get("cv_accuracy"),
            "baseline_accuracy": global_payload.get("baseline_accuracy"),
            "human_decisive_examples": int(counts.get("decisive_preferences", 0) or 0),
        },
        "rendered_dj_judge": {
            "available": bool(rendered_payload),
            "shadow_only": True,
            "model_id": rendered_payload.get("model_id"),
            "validation_accuracy": rendered_metrics.get("validation_accuracy"),
            "baseline_accuracy": rendered_metrics.get("baseline_accuracy"),
            "human_examples": int(counts.get("rendered_judge_human_examples", 0) or 0),
            "synthetic_examples": int(counts.get("rendered_judge_synthetic_examples", 0) or 0),
            "embedding_examples": int(counts.get("rendered_judge_embedding_examples", 0) or 0),
        },
        "state_move_taste": state_move_taste_status(root),
        "policy": {
            "hard_audio_safety_authoritative": True,
            "shadow_predictions_change_live_ranking": False,
        },
    }
