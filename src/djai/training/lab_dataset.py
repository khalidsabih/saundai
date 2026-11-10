from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..cache import discover_tracks
from ..brain.crate_memory import crate_memory_status
from ..intelligence.benchmark import benchmark_brain
from ..intelligence.preferences import load_preferences
from ..intelligence.neural_policy import rendered_judge_payload_features
from ..intelligence.transition_ranker import (
    FEATURE_NAMES,
    MINIMUM_COMPARISONS,
    RANKER_VERSION,
    feature_vector,
    train_transition_ranker,
)
from ..version import (
    DECISION_VERSION,
    DJAI_VERSION,
    MEASUREMENT_VERSION,
    PLAN_SCHEMA_VERSION,
    SET_PLAN_VERSION,
)
from .provenance import audit_dataset_registry
from .rendered_judge_dataset import build_synthetic_rendered_judge_rows
from .taste_evidence import runtime_evidence_status
from .human_preference import human_preference_status
from .state_move_taste import MINIMUM_STATE_MOVE_COMPARISONS, STATE_MOVE_FEATURES, state_move_taste_status

LAB_DATASET_SCHEMA_VERSION = "1.8.2"
LAB_FEATURE_SCHEMA_VERSION = "1.0"
TRAINING_TARGET_COMPARISONS = 24
BENCHMARK_TARGET_COMPARISONS = 100


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _relative(workspace: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_track(workspace: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _track_id(workspace: Path, value: str, cache: dict[str, dict[str, Any]]) -> str:
    path = _resolve_track(workspace, value)
    key = str(path)
    existing = cache.get(key)
    if existing:
        return str(existing["track_id"])
    if path.exists() and path.is_file():
        digest = _sha256_file(path)
        size = int(path.stat().st_size)
        missing = False
    else:
        digest = _sha256_bytes(key.encode("utf-8"))
        size = 0
        missing = True
    track_id = f"track-{digest[:24]}"
    bucket = int(digest[:8], 16) % 100
    split = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
    cache[key] = {
        "track_id": track_id,
        "content_sha256": digest if not missing else None,
        "source_ref": _relative(workspace, path),
        "suffix": path.suffix.lower(),
        "size_bytes": size,
        "split": split,
        "missing": missing,
    }
    return track_id


def _sanitize_candidate(
    workspace: Path,
    candidate: dict[str, Any],
    track_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clean = json.loads(json.dumps(candidate, ensure_ascii=False))
    original_a = str(clean.get("track_a") or "unknown-a")
    original_b = str(clean.get("track_b") or "unknown-b")
    track_a_id = _track_id(workspace, original_a, track_cache)
    track_b_id = _track_id(workspace, original_b, track_cache)
    clean["track_a_ref"] = track_cache[str(_resolve_track(workspace, original_a))]["source_ref"]
    clean["track_b_ref"] = track_cache[str(_resolve_track(workspace, original_b))]["source_ref"]
    clean["track_a"] = track_a_id
    clean["track_b"] = track_b_id
    clean["candidate_id"] = str(clean.get("execution_signature") or _sha256_bytes(_json_dumps(clean).encode("utf-8"))[:24])
    split_a = next((item["split"] for item in track_cache.values() if item["track_id"] == track_a_id), "mixed")
    split_b = next((item["split"] for item in track_cache.values() if item["track_id"] == track_b_id), "mixed")
    clean["split"] = split_a if split_a == split_b else "mixed"
    clean["pair_id"] = "pair-" + _sha256_bytes(f"{track_a_id}|{track_b_id}".encode("utf-8"))[:18]
    return clean


def _iter_session_manifests(workspace: Path) -> list[tuple[Path, dict[str, Any]]]:
    root = workspace / "output" / "listening-lab"
    output: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return output
    for path in sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            output.append((path, payload))
    return output


def _latest_training_models(workspace: Path, limit: int = 8) -> list[dict[str, Any]]:
    root = workspace / "config" / "training-lab-models"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "model_path": _relative(workspace, path),
                "model_id": payload.get("model_id", path.stem),
                "kind": payload.get("learning_head", payload.get("task", "global_taste_ranker")),
                "dataset_id": payload.get("training_dataset_id"),
                "comparison_count": int(payload.get("comparison_count", payload.get("metrics", {}).get("examples", 0)) or 0),
                "cv_accuracy": payload.get("cv_accuracy", payload.get("metrics", {}).get("validation_accuracy")),
                "baseline_accuracy": payload.get("baseline_accuracy", payload.get("metrics", {}).get("baseline_accuracy")),
                "authority": payload.get("authority"),
                "shadow_only": bool(payload.get("shadow_only", True)),
                "trained_at_utc": payload.get("trained_at_utc"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def list_training_snapshots(workspace: str | Path, limit: int = 12) -> list[dict[str, Any]]:
    root = Path(workspace).expanduser().resolve() / "datasets" / "training-lab"
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "dataset_id": payload.get("dataset_id", manifest_path.parent.name),
                "created_at_utc": payload.get("created_at_utc"),
                "comparison_count": int(payload.get("counts", {}).get("decisive_preferences", 0) or 0),
                "candidate_count": int(payload.get("counts", {}).get("candidates", 0) or 0),
                "track_count": int(payload.get("counts", {}).get("tracks", 0) or 0),
                "production_training_ready": bool(payload.get("production_training_ready", False)),
                "path": _relative(Path(workspace).expanduser().resolve(), manifest_path.parent),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def training_lab_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    preferences_path = root / "ratings" / "transition-preferences.jsonl"
    rows = load_preferences(preferences_path)
    choices = Counter(str(row.get("choice", "")) for row in rows)
    decisive = [row for row in rows if row.get("choice") in {"a", "b"}]
    state_move_decisive = [
        row
        for row in decisive
        if isinstance(row.get("candidate_a"), dict)
        and isinstance(row.get("candidate_b"), dict)
        and isinstance(row["candidate_a"].get("dj_state"), dict)
        and isinstance(row["candidate_a"].get("dj_move"), dict)
        and isinstance(row["candidate_b"].get("dj_state"), dict)
        and isinstance(row["candidate_b"].get("dj_move"), dict)
    ]
    state_groups = {
        str(row["candidate_a"].get("state_context_id") or row["candidate_a"].get("pair_id") or "unknown")
        for row in state_move_decisive
    }
    pair_keys = {
        f"{row.get('candidate_a', {}).get('track_a')}|{row.get('candidate_a', {}).get('track_b')}"
        for row in decisive
        if isinstance(row.get("candidate_a"), dict)
    }
    listeners = Counter(str(row.get("listener") or "local") for row in rows)
    reasons: Counter[str] = Counter()
    for row in rows:
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        for reason in context.get("reasons", []) if isinstance(context.get("reasons"), list) else []:
            reasons[str(reason)] += 1

    sessions = _iter_session_manifests(root)
    candidate_count = sum(len(payload.get("candidates", [])) for _, payload in sessions)
    track_count = len(discover_tracks(root / "audio", recursive=True)) if (root / "audio").exists() else 0
    ranker = benchmark_brain(preferences_path, root / "config" / "transition-ranker.json").to_dict()
    registry = root / "datasets" / "registry.json"
    if registry.exists():
        try:
            rights_audit = audit_dataset_registry(registry, lane="production")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            rights_audit = {"ready": False, "blockers": [{"reason": "registry_invalid"}]}
    else:
        rights_audit = {"ready": False, "blockers": [{"reason": "registry_missing"}]}

    label_balance = {"a": int(choices.get("a", 0)), "b": int(choices.get("b", 0))}
    train_blockers: list[str] = []
    if len(decisive) < MINIMUM_COMPARISONS:
        train_blockers.append(f"need_{MINIMUM_COMPARISONS - len(decisive)}_more_decisive_preferences")
    if len(pair_keys) < 2:
        train_blockers.append("need_at_least_two_track_pairs")
    if label_balance["a"] == 0 or label_balance["b"] == 0:
        train_blockers.append("need_both_blind_positions_selected")

    source_rights_confirmed = os.environ.get("SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED", "0").strip() == "1"
    return {
        "schema_version": LAB_DATASET_SCHEMA_VERSION,
        "track_count": track_count,
        "listening_sessions": len(sessions),
        "rendered_candidates": candidate_count,
        "preferences": {
            "total": len(rows),
            "decisive": len(decisive),
            "tie": int(choices.get("tie", 0)),
            "neither": int(choices.get("neither", 0)),
            "choice_a": label_balance["a"],
            "choice_b": label_balance["b"],
            "unique_track_pairs": len(pair_keys),
            "listeners": dict(listeners.most_common(8)),
            "top_reasons": [{"reason": reason, "count": count} for reason, count in reasons.most_common(8)],
        },
        "targets": {
            "train_minimum": TRAINING_TARGET_COMPARISONS,
            "benchmark_minimum": BENCHMARK_TARGET_COMPARISONS,
        },
        "train_ready": not train_blockers,
        "train_blockers": train_blockers,
        "live_ranker": ranker,
        "runtime_evidence": runtime_evidence_status(root),
        "human_preference_program": human_preference_status(root),
        "crate_memory": crate_memory_status(workspace=root),
        "learning_heads": {
            "global_taste_ranker": {
                "human_decisive_examples": len(decisive),
                "minimum_to_train": MINIMUM_COMPARISONS,
                "ready": len(decisive) >= MINIMUM_COMPARISONS and len(pair_keys) >= 2,
                "authority": "shadow",
            },
            "state_move_taste": {
                "human_decisive_examples": len(state_move_decisive),
                "state_groups": len(state_groups),
                "minimum_to_train": MINIMUM_STATE_MOVE_COMPARISONS,
                "ready": len(state_move_decisive) >= MINIMUM_STATE_MOVE_COMPARISONS and len(state_groups) >= 2,
                "authority": "shadow",
                "feature_count": len(STATE_MOVE_FEATURES),
                "status": state_move_taste_status(root),
            },
            "rendered_dj_judge": {
                "human_labeled_candidates": len(rows) * 2,
                "synthetic_failures_enabled": os.environ.get("DJAI_TASTE_SYNTHETIC_FAILURES", "1").strip().lower() not in {"0", "false", "no", "off"},
                "authority": "shadow",
            },
        },
        "snapshots": list_training_snapshots(root),
        "shadow_models": _latest_training_models(root),
        "rights": {
            "source_rights_confirmed": source_rights_confirmed,
            "registry_ready": bool(rights_audit.get("ready")),
            "production_training_ready": bool(source_rights_confirmed and rights_audit.get("ready")),
            "blockers": rights_audit.get("blockers", []),
        },
        "policy": {
            "raw_audio_bundled_in_snapshots": False,
            "shadow_training_changes_live_model": False,
            "hard_audio_safety_can_be_overridden_by_ranker": False,
        },
    }


def list_training_sessions(workspace: str | Path, limit: int = 30) -> list[dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    preference_rows = load_preferences(root / "ratings" / "transition-preferences.jsonl")
    votes_by_session: Counter[str] = Counter()
    for row in preference_rows:
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        session_id = str(context.get("session_id") or "")
        if session_id:
            votes_by_session[session_id] += 1
    output: list[dict[str, Any]] = []
    for path, payload in _iter_session_manifests(root)[: max(1, min(200, int(limit)))]:
        session_id = str(payload.get("session_id") or path.parent.name)
        candidates = payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []
        output.append(
            {
                "session_id": session_id,
                "track_a": _relative(root, _resolve_track(root, str(payload.get("track_a") or ""))),
                "track_b": _relative(root, _resolve_track(root, str(payload.get("track_b") or ""))),
                "candidate_count": len(candidates),
                "comparison_count": len(candidates) // 2,
                "votes_recorded": int(votes_by_session.get(session_id, 0)),
                "created_at_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return output


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_digest(
    preferences: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rendered_judge_rows: list[dict[str, Any]] | None = None,
) -> str:
    signatures = sorted(str(row.get("candidate_id") or row.get("execution_signature") or "") for row in candidates)
    rendered_ids = sorted(
        f"{row.get('id','')}|{row.get('label_source','')}|{row.get('target','')}|{row.get('weight','')}"
        for row in (rendered_judge_rows or [])
    )
    seed = {
        "preferences": preferences,
        "candidate_signatures": signatures,
        "rendered_judge_rows": rendered_ids,
        "engine_version": DJAI_VERSION,
        "ranker_version": RANKER_VERSION,
        "feature_schema": FEATURE_NAMES,
    }
    return _sha256_bytes(_json_dumps(seed).encode("utf-8"))


def _rendered_judge_training_rows(
    preferences: list[dict[str, Any]],
    *,
    rights_cleared: bool,
) -> list[dict[str, Any]]:
    """Convert blind Training Lab votes into grouped rendered-judge rows.

    A/B winners are stronger positives than the relative loser is a negative.
    Ties pull both candidates toward neutral preference. `neither` is the
    strongest rejection signal. Grouping by track pair keeps both candidates
    and repeated sessions for that pair on the same train/validation side.
    """
    rows: list[dict[str, Any]] = []
    for record in preferences:
        choice = str(record.get("choice") or "").lower()
        candidate_a = record.get("candidate_a")
        candidate_b = record.get("candidate_b")
        if not isinstance(candidate_a, dict) or not isinstance(candidate_b, dict):
            continue
        comparison = str(record.get("comparison_id") or _sha256_bytes(_json_dumps(record).encode("utf-8"))[:18])
        pair_group = str(candidate_a.get("pair_id") or candidate_b.get("pair_id") or "unknown-pair")
        context = record.get("context") if isinstance(record.get("context"), dict) else {}
        session_id = str(context.get("session_id") or comparison)
        base_weight = max(0.05, min(10.0, float(record.get("weight", 1.0) or 1.0)))

        examples: list[tuple[str, dict[str, Any], float, float, str]] = []
        if choice == "a":
            examples = [
                ("a", candidate_a, 1.0, 1.0, "pairwise_winner"),
                ("b", candidate_b, 0.0, 0.55, "pairwise_relative_loser"),
            ]
        elif choice == "b":
            examples = [
                ("b", candidate_b, 1.0, 1.0, "pairwise_winner"),
                ("a", candidate_a, 0.0, 0.55, "pairwise_relative_loser"),
            ]
        elif choice == "tie":
            examples = [
                ("a", candidate_a, 0.5, 0.45, "pairwise_tie"),
                ("b", candidate_b, 0.5, 0.45, "pairwise_tie"),
            ]
        elif choice == "neither":
            examples = [
                ("a", candidate_a, 0.0, 1.35, "absolute_rejection"),
                ("b", candidate_b, 0.0, 1.35, "absolute_rejection"),
            ]
        else:
            continue

        for side, candidate, target, multiplier, label_source in examples:
            features = rendered_judge_payload_features(candidate)
            rows.append({
                "id": f"{comparison}:{side}",
                "group": pair_group,
                "context_id": session_id,
                "target": target,
                "weight": round(base_weight * multiplier, 6),
                "rights_cleared": bool(rights_cleared),
                "label_source": label_source,
                "choice": choice,
                "candidate_side": side,
                "render_embedding_present": bool(features.get("render_embedding_present", 0.0) > 0.5),
                "features": features,
            })
    return rows


def create_training_snapshot(
    workspace: str | Path,
    *,
    name: str | None = None,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    source_preferences = load_preferences(root / "ratings" / "transition-preferences.jsonl")
    if not source_preferences:
        raise ValueError("No listening preferences are available yet. Label at least one A/B comparison first.")

    track_cache: dict[str, dict[str, Any]] = {}
    candidate_map: dict[str, dict[str, Any]] = {}
    sessions_output: list[dict[str, Any]] = []
    for manifest_path, manifest in _iter_session_manifests(root):
        session_id = str(manifest.get("session_id") or manifest_path.parent.name)
        session_candidate_ids: list[str] = []
        for item in manifest.get("candidates", []) if isinstance(manifest.get("candidates"), list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("plan"), dict):
                continue
            candidate = _sanitize_candidate(root, item["plan"], track_cache)
            candidate_id = str(candidate["candidate_id"])
            candidate_map[candidate_id] = candidate
            session_candidate_ids.append(candidate_id)
        track_a = str(manifest.get("track_a") or "")
        track_b = str(manifest.get("track_b") or "")
        sessions_output.append(
            {
                "session_id": session_id,
                "track_a": _track_id(root, track_a, track_cache) if track_a else None,
                "track_b": _track_id(root, track_b, track_cache) if track_b else None,
                "candidate_ids": session_candidate_ids,
                "comparison_pairs": [session_candidate_ids[index : index + 2] for index in range(0, len(session_candidate_ids), 2) if len(session_candidate_ids[index : index + 2]) == 2],
                "active_learning": manifest.get("active_learning", {}),
            }
        )

    sanitized_preferences: list[dict[str, Any]] = []
    for row in source_preferences:
        clean = json.loads(json.dumps(row, ensure_ascii=False))
        if isinstance(clean.get("candidate_a"), dict):
            clean["candidate_a"] = _sanitize_candidate(root, clean["candidate_a"], track_cache)
            candidate_map[str(clean["candidate_a"]["candidate_id"])] = clean["candidate_a"]
        if isinstance(clean.get("candidate_b"), dict):
            clean["candidate_b"] = _sanitize_candidate(root, clean["candidate_b"], track_cache)
            candidate_map[str(clean["candidate_b"]["candidate_id"])] = clean["candidate_b"]
        sanitized_preferences.append(clean)

    candidates = sorted(candidate_map.values(), key=lambda item: str(item.get("candidate_id", "")))
    tracks = sorted(track_cache.values(), key=lambda item: str(item["track_id"]))
    source_rights_confirmed = os.environ.get("SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED", "0").strip() == "1"
    human_rendered_judge_rows = _rendered_judge_training_rows(
        sanitized_preferences, rights_cleared=source_rights_confirmed
    )
    synthetic_rendered_judge_rows, synthetic_rendered_report = build_synthetic_rendered_judge_rows(root)
    rendered_judge_rows = [*human_rendered_judge_rows, *synthetic_rendered_judge_rows]
    digest = _dataset_digest(sanitized_preferences, candidates, rendered_judge_rows)
    timestamp = _now().strftime("%Y%m%d-%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "first-train").strip()).strip("-")[:32] or "first-train"
    dataset_id = f"dataset-{timestamp}-{safe_name}-{digest[:10]}"
    dataset_root = root / "datasets" / "training-lab"
    destination = dataset_root / dataset_id
    temporary = dataset_root / f".{dataset_id}.tmp"
    if destination.exists():
        return destination
    if temporary.exists():
        for child in temporary.iterdir():
            if child.is_file():
                child.unlink()
        temporary.rmdir()
    temporary.mkdir(parents=True, exist_ok=False)

    _write_jsonl(temporary / "tracks.jsonl", tracks)
    _write_jsonl(temporary / "candidates.jsonl", candidates)
    _write_jsonl(temporary / "preferences.jsonl", sanitized_preferences)
    _write_jsonl(temporary / "sessions.jsonl", sessions_output)
    _write_jsonl(temporary / "rendered_judge.jsonl", rendered_judge_rows)

    decisive = [row for row in sanitized_preferences if row.get("choice") in {"a", "b"}]
    matrix_rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    groups: list[str] = []
    splits: list[str] = []
    for row in decisive:
        a = row.get("candidate_a")
        b = row.get("candidate_b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        matrix_rows.append(feature_vector(a) - feature_vector(b))
        targets.append(1.0 if row.get("choice") == "a" else 0.0)
        weights.append(float(row.get("weight", 1.0) or 1.0))
        groups.append(str(a.get("pair_id") or "unknown"))
        split_a = str(a.get("split") or "mixed")
        split_b = str(b.get("split") or "mixed")
        splits.append(split_a if split_a == split_b else "mixed")

    matrix = np.vstack(matrix_rows) if matrix_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    np.savez_compressed(
        temporary / "pairwise_matrix.npz",
        x=matrix,
        y=np.asarray(targets, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        groups=np.asarray(groups, dtype=str),
        splits=np.asarray(splits, dtype=str),
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
    )
    feature_schema = {
        "schema_version": LAB_FEATURE_SCHEMA_VERSION,
        "ranker_version": RANKER_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_schema_sha256": _sha256_bytes("\n".join(FEATURE_NAMES).encode("utf-8")),
        "pairwise_representation": "feature_vector(candidate_a) - feature_vector(candidate_b)",
        "target": "1 when A is preferred, 0 when B is preferred",
    }
    (temporary / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_counts = Counter(splits)
    registry = root / "datasets" / "registry.json"
    rights_audit = audit_dataset_registry(registry, lane="production") if registry.exists() else {"ready": False, "blockers": [{"reason": "registry_missing"}]}

    manifest = {
        "dataset_id": dataset_id,
        "schema_version": LAB_DATASET_SCHEMA_VERSION,
        "created_at_utc": _now().isoformat(),
        "immutable": True,
        "purpose": "SAUND.AI pairwise transition + DJState x DJMove preference training",
        "engine": {
            "djai_version": DJAI_VERSION,
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "set_plan_version": SET_PLAN_VERSION,
            "decision_version": DECISION_VERSION,
            "measurement_version": MEASUREMENT_VERSION,
            "ranker_version": RANKER_VERSION,
        },
        "counts": {
            "tracks": len(tracks),
            "sessions": len(sessions_output),
            "candidates": len(candidates),
            "preferences": len(sanitized_preferences),
            "decisive_preferences": len(matrix_rows),
            "state_move_decisive_preferences": sum(
                row.get("choice") in {"a", "b"}
                and isinstance(row.get("candidate_a"), dict)
                and isinstance(row.get("candidate_b"), dict)
                and isinstance(row["candidate_a"].get("dj_state"), dict)
                and isinstance(row["candidate_a"].get("dj_move"), dict)
                and isinstance(row["candidate_b"].get("dj_state"), dict)
                and isinstance(row["candidate_b"].get("dj_move"), dict)
                for row in sanitized_preferences
            ),
            "ties": sum(row.get("choice") == "tie" for row in sanitized_preferences),
            "neither": sum(row.get("choice") == "neither" for row in sanitized_preferences),
            "unique_pair_groups": len(set(groups)),
            "rendered_judge_examples": len(rendered_judge_rows),
            "rendered_judge_human_examples": len(human_rendered_judge_rows),
            "rendered_judge_synthetic_examples": len(synthetic_rendered_judge_rows),
            "rendered_judge_embedding_examples": sum(
                bool(row.get("render_embedding_present")) for row in rendered_judge_rows
            ),
        },
        "splits": dict(split_counts),
        "feature_schema_sha256": feature_schema["feature_schema_sha256"],
        "raw_audio_bundled": False,
        "audio_policy": "Audio stays in the local workspace. Snapshot stores content IDs, local source references, rendered/candidate features, and preference labels only.",
        "rendered_judge_training": {
            "dataset": "rendered_judge.jsonl",
            "grouping": "track_pair",
            "a_b": "winner positive; relative loser lower-weight negative",
            "tie": "both candidates target neutral 0.5 with reduced weight",
            "neither": "both candidates are strong negative acceptability evidence",
            "synthetic_failures": synthetic_rendered_report,
            "hard_audio_safety_override": False,
        },
        "source_rights_confirmed": source_rights_confirmed,
        "production_registry_ready": bool(rights_audit.get("ready")),
        "production_training_ready": bool(source_rights_confirmed and rights_audit.get("ready")),
        "rights_blockers": rights_audit.get("blockers", []),
        "source_digest": digest,
        "files": {},
    }
    for filename in ("tracks.jsonl", "candidates.jsonl", "preferences.jsonl", "sessions.jsonl", "rendered_judge.jsonl", "pairwise_matrix.npz", "feature_schema.json"):
        file_path = temporary / filename
        manifest["files"][filename] = {
            "sha256": _sha256_file(file_path),
            "bytes": int(file_path.stat().st_size),
        }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset_root.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    (dataset_root / "LATEST").write_text(dataset_id + "\n", encoding="utf-8")
    return destination


def _resolve_snapshot(workspace: Path, snapshot_id: str | None) -> Path:
    root = workspace / "datasets" / "training-lab"
    if snapshot_id:
        if not re.fullmatch(r"dataset-[A-Za-z0-9_-]+", snapshot_id):
            raise ValueError("Invalid training dataset id.")
        path = root / snapshot_id
    else:
        latest = root / "LATEST"
        if not latest.exists():
            raise ValueError("No training dataset snapshot exists yet.")
        path = root / latest.read_text(encoding="utf-8").strip()
    if not path.exists() or not (path / "manifest.json").exists():
        raise ValueError("Training dataset snapshot was not found.")
    return path


def train_shadow_ranker(
    workspace: str | Path,
    *,
    snapshot_id: str | None = None,
    ridge: float = 0.18,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    snapshot = _resolve_snapshot(root, snapshot_id)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    comparison_count = int(manifest.get("counts", {}).get("decisive_preferences", 0) or 0)
    if comparison_count < MINIMUM_COMPARISONS:
        raise ValueError(f"At least {MINIMUM_COMPARISONS} decisive A/B preferences are required; snapshot has {comparison_count}.")

    dataset_id = str(manifest.get("dataset_id") or snapshot.name)
    model_id = f"global-taste-ranker-{dataset_id.removeprefix('dataset-')}"
    model_root = root / "config" / "training-lab-models"
    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / f"{model_id}.json"
    trained = train_transition_ranker(snapshot / "preferences.jsonl", destination, ridge=ridge)
    payload = json.loads(trained.read_text(encoding="utf-8"))
    payload.update(
        {
            "model_id": model_id,
            "learning_head": "global_taste_ranker",
            "training_dataset_id": dataset_id,
            "training_dataset_manifest_sha256": _sha256_file(snapshot / "manifest.json"),
            "trained_at_utc": _now().isoformat(),
            "shadow_only": True,
            "promoted_to_live": False,
            "live_model_path": "config/transition-ranker.json",
            "safety_contract": "Learned ranker may rerank technically valid candidates only; hard rendered-audio failures remain non-overridable.",
        }
    )
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    (model_root / "GLOBAL_TASTE_LATEST").write_text(destination.name + "\n", encoding="utf-8")
    (model_root / "LATEST").write_text(destination.name + "\n", encoding="utf-8")
    benchmark = benchmark_brain(snapshot / "preferences.jsonl", destination).to_dict()
    return {
        "learning_head": "global_taste_ranker",
        "model_id": model_id,
        "model_path": _relative(root, destination),
        "dataset_id": dataset_id,
        "shadow_only": True,
        "comparison_count": int(payload.get("comparison_count", 0) or 0),
        "cv_accuracy": payload.get("cv_accuracy"),
        "baseline_accuracy": payload.get("baseline_accuracy"),
        "win_margin": (float(payload.get("cv_accuracy", 0.0)) - float(payload.get("baseline_accuracy", 0.0))) if payload.get("cv_accuracy") is not None and payload.get("baseline_accuracy") is not None else None,
        "authority_if_promoted": payload.get("authority"),
        "benchmark": benchmark,
        "production_training_ready": bool(manifest.get("production_training_ready", False)),
    }
