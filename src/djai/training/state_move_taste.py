from __future__ import annotations

"""SAUND.AI 1.8.1 state-conditioned DJ taste learning.

The learned head consumes the same decision vocabulary as the Complete DJMove
planner: a sanitized current DJState plus one complete DJMove.  It is trained in
shadow mode first and therefore never overrides deterministic feasibility,
rendered-audio safety, or AI Headphones acceptance.
"""

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..brain.dj_move import DJMove, DJState

STATE_MOVE_TASTE_VERSION = "1.8.1"
STATE_MOVE_TASTE_SCHEMA_VERSION = "1.0"
MINIMUM_STATE_MOVE_COMPARISONS = 12
MINIMUM_STATE_GROUPS = 2
MAXIMUM_STATE_MOVE_AUTHORITY = 0.16

STATE_MOVE_FEATURES: tuple[str, ...] = (
    "state_source_bar",
    "state_current_energy",
    "state_current_tension",
    "state_current_vocal",
    "state_current_bass",
    "state_set_energy",
    "state_set_tension",
    "state_set_groove",
    "state_set_vocal_density",
    "pending_commitments",
    "wait_bars",
    "transition_bars",
    "tempo_shift",
    "tempo_release",
    "phase_nudge",
    "future_optionality",
    "move_score",
    "projected_energy_delta",
    "projected_tension_delta",
    "projected_vocal_delta",
    "projected_bass_delta",
    "protected_payoff_violation",
    "payoff_satisfied",
    "wait_action",
    "stem_operation",
    "structural_operation_count",
    "drop_intent",
    "vocal_exchange",
)

REASON_TAGS: tuple[str, ...] = (
    "wrong_next_track",
    "left_too_early",
    "left_too_late",
    "transition_too_long",
    "tempo_felt_forced",
    "vocal_clash",
    "energy_wrong",
    "great_payoff",
)


def _clamp(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _safe_round(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def state_taste_payload(state: DJState | Mapping[str, Any]) -> dict[str, Any]:
    raw = state.to_dict() if isinstance(state, DJState) else dict(state)
    pending = raw.get("pending_commitments")
    pending_count = len(pending) if isinstance(pending, (list, tuple)) else int(raw.get("pending_commitment_count", 0) or 0)
    return {
        "version": str(raw.get("version") or "unknown"),
        "source_bar": int(raw.get("source_bar", 0) or 0),
        "phrase_index": int(raw.get("phrase_index", 0) or 0),
        "phrase_role": str(raw.get("phrase_role") or "unknown"),
        "effective_bpm": _safe_round(raw.get("effective_bpm")),
        "native_bpm": _safe_round(raw.get("native_bpm")),
        "render_rate": _safe_round(raw.get("render_rate", 1.0)),
        "tempo_owner": str(raw.get("tempo_owner") or "unknown"),
        "current_energy": _safe_round(raw.get("current_energy")),
        "current_tension": _safe_round(raw.get("current_tension")),
        "current_vocal_probability": _safe_round(raw.get("current_vocal_probability")),
        "current_bass_activity": _safe_round(raw.get("current_bass_activity")),
        "set_energy": _safe_round(raw.get("set_energy")),
        "set_tension": _safe_round(raw.get("set_tension")),
        "set_groove": _safe_round(raw.get("set_groove")),
        "set_vocal_density": _safe_round(raw.get("set_vocal_density")),
        "pending_commitment_count": int(max(0, pending_count)),
    }


def move_taste_payload(move: DJMove | Mapping[str, Any]) -> dict[str, Any]:
    raw = move.to_dict() if isinstance(move, DJMove) else dict(move)
    projected = raw.get("projected_state") if isinstance(raw.get("projected_state"), Mapping) else {}
    operations = raw.get("structural_operations") if isinstance(raw.get("structural_operations"), (list, tuple)) else []
    deck_ops = raw.get("deck_operations") if isinstance(raw.get("deck_operations"), (list, tuple)) else []
    return {
        "version": str(raw.get("version") or "unknown"),
        "action": str(raw.get("action") or "transition"),
        "exit_bar": int(raw.get("exit_bar", 0) or 0),
        "entry_bar": int(raw.get("entry_bar", 0) or 0),
        "exit_phrase": int(raw.get("exit_phrase", 0) or 0),
        "entry_phrase": int(raw.get("entry_phrase", 0) or 0),
        "transition_bars": int(raw.get("transition_bars", 0) or 0),
        "wait_bars": int(raw.get("wait_bars", 0) or 0),
        "matched_bpm": _safe_round(raw.get("matched_bpm")),
        "incoming_native_bpm": _safe_round(raw.get("incoming_native_bpm")),
        "post_transition_bpm": _safe_round(raw.get("post_transition_bpm")),
        "tempo_release_seconds": _safe_round(raw.get("tempo_release_seconds")),
        "phase_nudge_seconds": _safe_round(raw.get("phase_nudge_seconds")),
        "technique": str(raw.get("technique") or "unknown"),
        "intent": str(raw.get("intent") or "unknown"),
        "tempo_strategy": str(raw.get("tempo_strategy") or "unknown"),
        "phase_strategy": str(raw.get("phase_strategy") or "unknown"),
        "stem_strategy": str(raw.get("stem_strategy") or "master"),
        "structural_operations": [str(item) for item in operations],
        "deck_operation_count": len(deck_ops),
        "protected_payoff_violation": bool(raw.get("protected_payoff_violation")),
        "payoff_satisfied": bool(raw.get("payoff_satisfied")),
        "landing_phrase": int(raw.get("landing_phrase", 0) or 0),
        "future_optionality": _safe_round(raw.get("future_optionality")),
        "score": _safe_round(raw.get("score")),
        "projected_state": state_taste_payload(projected) if projected else {},
    }


def state_move_features(
    state: DJState | Mapping[str, Any],
    move: DJMove | Mapping[str, Any],
) -> dict[str, float]:
    s = state_taste_payload(state)
    m = move_taste_payload(move)
    p = m.get("projected_state") if isinstance(m.get("projected_state"), Mapping) else {}
    effective_bpm = max(1.0, float(s.get("effective_bpm", 0.0) or 0.0))
    matched_bpm = float(m.get("matched_bpm", effective_bpm) or effective_bpm)
    technique = str(m.get("technique") or "").lower()
    intent = str(m.get("intent") or "").lower()
    stem_strategy = str(m.get("stem_strategy") or "").lower()
    operations = [str(item).upper() for item in m.get("structural_operations", []) if isinstance(item, str)]
    return {
        "state_source_bar": _clamp(float(s.get("source_bar", 0)) / 128.0),
        "state_current_energy": _clamp(s.get("current_energy", 0.0)),
        "state_current_tension": _clamp(s.get("current_tension", 0.0)),
        "state_current_vocal": _clamp(s.get("current_vocal_probability", 0.0)),
        "state_current_bass": _clamp(s.get("current_bass_activity", 0.0)),
        "state_set_energy": _clamp(s.get("set_energy", 0.0)),
        "state_set_tension": _clamp(s.get("set_tension", 0.0)),
        "state_set_groove": _clamp(s.get("set_groove", 0.0)),
        "state_set_vocal_density": _clamp(s.get("set_vocal_density", 0.0)),
        "pending_commitments": _clamp(float(s.get("pending_commitment_count", 0)) / 4.0),
        "wait_bars": _clamp(float(m.get("wait_bars", 0)) / 32.0),
        "transition_bars": _clamp(float(m.get("transition_bars", 0)) / 32.0),
        "tempo_shift": _clamp(abs(matched_bpm - effective_bpm) / effective_bpm / 0.08),
        "tempo_release": _clamp(float(m.get("tempo_release_seconds", 0.0)) / 24.0),
        "phase_nudge": _clamp(abs(float(m.get("phase_nudge_seconds", 0.0))) / 0.08),
        "future_optionality": _clamp(m.get("future_optionality", 0.0)),
        "move_score": _clamp(m.get("score", 0.0)),
        "projected_energy_delta": _clamp(0.5 + 0.5 * (float(p.get("current_energy", s.get("current_energy", 0.0))) - float(s.get("current_energy", 0.0)))),
        "projected_tension_delta": _clamp(0.5 + 0.5 * (float(p.get("current_tension", s.get("current_tension", 0.0))) - float(s.get("current_tension", 0.0)))),
        "projected_vocal_delta": _clamp(0.5 + 0.5 * (float(p.get("current_vocal_probability", s.get("current_vocal_probability", 0.0))) - float(s.get("current_vocal_probability", 0.0)))),
        "projected_bass_delta": _clamp(0.5 + 0.5 * (float(p.get("current_bass_activity", s.get("current_bass_activity", 0.0))) - float(s.get("current_bass_activity", 0.0)))),
        "protected_payoff_violation": float(bool(m.get("protected_payoff_violation"))),
        "payoff_satisfied": float(bool(m.get("payoff_satisfied"))),
        "wait_action": float(str(m.get("action") or "").startswith("wait")),
        "stem_operation": float(stem_strategy not in {"", "master", "none"} or any("STEM" in item for item in operations)),
        "structural_operation_count": _clamp(len(operations) / 5.0),
        "drop_intent": float("drop" in technique or "drop" in intent),
        "vocal_exchange": float("vocal" in technique or "vocal" in intent),
    }


def state_move_vector(state: Mapping[str, Any], move: Mapping[str, Any]) -> np.ndarray:
    features = state_move_features(state, move)
    return np.asarray([float(features[name]) for name in STATE_MOVE_FEATURES], dtype=np.float64)


def state_context_id(state: DJState | Mapping[str, Any]) -> str:
    payload = state_taste_payload(state)
    seed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "state-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:18]


def _model_root(workspace: Path) -> Path:
    root = workspace / "config" / "training-lab-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _latest_model(workspace: Path) -> Path | None:
    root = _model_root(workspace)
    pointer = root / "STATE_MOVE_TASTE_LATEST"
    if pointer.exists():
        candidate = root / pointer.read_text(encoding="utf-8").strip()
        if candidate.exists():
            return candidate
    candidates = sorted(root.glob("state-move-taste-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def shadow_state_move_prediction(
    state: DJState | Mapping[str, Any],
    move: DJMove | Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace or os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()
    model_path = _latest_model(root)
    if model_path is None:
        return {"available": False, "authority": 0.0, "reason": "no trained DJState x DJMove taste shadow"}
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "authority": 0.0, "reason": "state-move taste shadow unreadable"}
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    if weights.size != len(STATE_MOVE_FEATURES):
        return {"available": False, "authority": 0.0, "reason": "state-move taste feature schema mismatch"}
    vector = state_move_vector(state_taste_payload(state), move_taste_payload(move))
    score = _clamp(float(model.get("intercept", 0.5)) + float(np.dot(weights, vector)))
    confidence = _clamp(float(model.get("cv_accuracy", 0.5)) - 0.5) * 2.0
    return {
        "available": True,
        "version": STATE_MOVE_TASTE_VERSION,
        "model_id": model.get("model_id", model_path.stem),
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "authority": 0.0,
        "shadow_only": True,
        "state_context_id": state_context_id(state),
        "reason": "1.8.1 shadow prediction; does not change live ranking",
    }


def _rows_from_preferences(path: Path) -> list[tuple[np.ndarray, float, float, str, list[str]]]:
    rows: list[tuple[np.ndarray, float, float, str, list[str]]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("choice") not in {"a", "b"}:
                continue
            a = record.get("candidate_a") if isinstance(record.get("candidate_a"), dict) else {}
            b = record.get("candidate_b") if isinstance(record.get("candidate_b"), dict) else {}
            state_a = a.get("dj_state") if isinstance(a.get("dj_state"), dict) else None
            state_b = b.get("dj_state") if isinstance(b.get("dj_state"), dict) else None
            move_a = a.get("dj_move") if isinstance(a.get("dj_move"), dict) else None
            move_b = b.get("dj_move") if isinstance(b.get("dj_move"), dict) else None
            if state_a is None or state_b is None or move_a is None or move_b is None:
                continue
            xa = state_move_vector(state_a, move_a)
            xb = state_move_vector(state_b, move_b)
            target = 1.0 if record.get("choice") == "a" else 0.0
            weight = max(0.05, min(10.0, float(record.get("weight", 1.0) or 1.0)))
            context = record.get("context") if isinstance(record.get("context"), dict) else {}
            reasons = [str(item) for item in context.get("reasons", []) if str(item)] if isinstance(context.get("reasons"), list) else []
            group = str(a.get("state_context_id") or b.get("state_context_id") or a.get("pair_id") or "unknown")
            rows.append((xa - xb, target, weight, group, reasons))
    return rows


def _grouped_accuracy(x: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: Sequence[str], ridge: float) -> tuple[float, float]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        share = float(np.average(y, weights=weights)) if len(y) else 0.5
        return max(share, 1.0 - share), max(share, 1.0 - share)
    predictions = np.zeros(len(y), dtype=np.float64)
    seen = np.zeros(len(y), dtype=bool)
    group_array = np.asarray(groups, dtype=object)
    fold_count = min(5, len(unique))
    buckets: list[list[str]] = [[] for _ in range(fold_count)]
    for index, group in enumerate(unique):
        buckets[index % fold_count].append(group)
    for bucket in buckets:
        valid_mask = np.isin(group_array, np.asarray(bucket, dtype=object))
        train_mask = ~valid_mask
        if not np.any(valid_mask) or not np.any(train_mask):
            continue
        train_x = x[train_mask]
        train_y = y[train_mask]
        train_w = weights[train_mask]
        design = np.column_stack([np.ones(len(train_x)), train_x])
        root_w = np.sqrt(np.maximum(train_w, 1e-6))[:, None]
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coef = np.linalg.solve((design * root_w).T @ (design * root_w) + penalty, (design * root_w).T @ (train_y * root_w[:, 0]))
        predictions[valid_mask] = np.clip(coef[0] + x[valid_mask] @ coef[1:], 0.0, 1.0)
        seen[valid_mask] = True
    if not np.any(seen):
        return 0.5, 0.5
    accuracy = float(np.average((predictions[seen] >= 0.5) == (y[seen] >= 0.5), weights=weights[seen]))
    share = float(np.average(y[seen] >= 0.5, weights=weights[seen]))
    baseline = max(share, 1.0 - share)
    return accuracy, baseline


def train_state_move_taste_shadow(
    workspace: str | Path,
    *,
    preferences_path: str | Path,
    dataset_id: str,
    ridge: float = 0.18,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    path = Path(preferences_path).expanduser().resolve()
    rows = _rows_from_preferences(path)
    groups = [row[3] for row in rows]
    if len(rows) < MINIMUM_STATE_MOVE_COMPARISONS:
        raise ValueError(f"Need at least {MINIMUM_STATE_MOVE_COMPARISONS} decisive DJState x DJMove preferences; snapshot has {len(rows)}.")
    if len(set(groups)) < MINIMUM_STATE_GROUPS:
        raise ValueError(f"Need at least {MINIMUM_STATE_GROUPS} independent DJState groups; snapshot has {len(set(groups))}.")
    x = np.vstack([row[0] for row in rows])
    y = np.asarray([row[1] for row in rows], dtype=np.float64)
    sample_weights = np.asarray([row[2] for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    root_w = np.sqrt(np.maximum(sample_weights, 1e-6))[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve((design * root_w).T @ (design * root_w) + penalty, (design * root_w).T @ (y * root_w[:, 0]))
    train_pred = np.clip(coef[0] + x @ coef[1:], 0.0, 1.0)
    train_accuracy = float(np.average((train_pred >= 0.5) == (y >= 0.5), weights=sample_weights))
    cv_accuracy, baseline = _grouped_accuracy(x, y, sample_weights, groups, float(ridge))
    reason_counts: dict[str, int] = {}
    for *_, reasons in rows:
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    model_id = f"state-move-taste-{dataset_id.removeprefix('dataset-')}"
    destination = _model_root(root) / f"{model_id}.json"
    payload = {
        "schema_version": STATE_MOVE_TASTE_SCHEMA_VERSION,
        "version": STATE_MOVE_TASTE_VERSION,
        "model_id": model_id,
        "learning_head": "state_move_taste",
        "training_dataset_id": dataset_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(STATE_MOVE_FEATURES),
        "weights": [round(float(item), 10) for item in coef[1:]],
        "intercept": round(float(coef[0]), 10),
        "comparison_count": len(rows),
        "state_group_count": len(set(groups)),
        "train_accuracy": round(train_accuracy, 6),
        "cv_accuracy": round(cv_accuracy, 6),
        "baseline_accuracy": round(baseline, 6),
        "reason_tags": reason_counts,
        "shadow_only": True,
        "authority": 0.0,
        "maximum_future_authority": MAXIMUM_STATE_MOVE_AUTHORITY,
        "safety_contract": "DJState x DJMove taste is a preference residual only. Hard audio safety and AI Headphones feasibility remain authoritative.",
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    (_model_root(root) / "STATE_MOVE_TASTE_LATEST").write_text(destination.name + "\n", encoding="utf-8")
    return {
        "learning_head": "state_move_taste",
        "model_id": model_id,
        "model_path": str(destination.relative_to(root)),
        "dataset_id": dataset_id,
        "shadow_only": True,
        "comparison_count": len(rows),
        "state_group_count": len(set(groups)),
        "train_accuracy": round(train_accuracy, 6),
        "cv_accuracy": round(cv_accuracy, 6),
        "baseline_accuracy": round(baseline, 6),
        "reason_tags": reason_counts,
        "authority": 0.0,
    }


def state_move_taste_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    path = _latest_model(root)
    if path is None:
        return {
            "available": False,
            "version": STATE_MOVE_TASTE_VERSION,
            "shadow_only": True,
            "authority": 0.0,
            "minimum_comparisons": MINIMUM_STATE_MOVE_COMPARISONS,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "version": STATE_MOVE_TASTE_VERSION, "shadow_only": True, "authority": 0.0, "reason": "model unreadable"}
    return {
        "available": True,
        "version": STATE_MOVE_TASTE_VERSION,
        "model_id": payload.get("model_id", path.stem),
        "shadow_only": True,
        "authority": 0.0,
        "comparison_count": int(payload.get("comparison_count", 0) or 0),
        "state_group_count": int(payload.get("state_group_count", 0) or 0),
        "cv_accuracy": payload.get("cv_accuracy"),
        "baseline_accuracy": payload.get("baseline_accuracy"),
        "reason_tags": payload.get("reason_tags", {}),
        "minimum_comparisons": MINIMUM_STATE_MOVE_COMPARISONS,
    }
