from __future__ import annotations

import hashlib
import json
import secrets
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from ..intelligence.preferences import append_preference, load_preferences, plan_features
from ..renderer import render_transition
from ..serialization import transition_plan_from_dict
from .teacher_dataset import load_scenarios

HUMAN_PREFERENCE_SCHEMA_VERSION = "1.0"
EvaluatorCohort = Literal["general_listener", "experienced_dj", "professional_dj"]

DEFAULT_REVIEW_QUEUE = "datasets/generated/teacher/human-review-queue.jsonl"
DEFAULT_SCENARIOS = "datasets/generated/teacher/scenarios.jsonl"
DEFAULT_REVIEW_CACHE = "output/human-review-cache"
DEFAULT_REVIEW_SESSIONS = "output/human-preference-lab"
DEFAULT_PREFERENCES = "ratings/transition-preferences.jsonl"

_COHORT_WEIGHT = {
    "general_listener": 1.0,
    "experienced_dj": 1.10,
    "professional_dj": 1.20,
}


def evaluator_id(value: str, cohort: EvaluatorCohort) -> str:
    normalized = value.strip() or "local"
    digest = hashlib.sha256(f"{cohort}|{normalized}".encode("utf-8")).hexdigest()[:20]
    return f"human-{cohort}-{digest}"


def _jsonl(path: Path) -> list[dict[str, Any]]:
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


def _reviewed_scenarios(preferences: list[dict[str, Any]], reviewer: str) -> set[str]:
    output: set[str] = set()
    for row in preferences:
        if str(row.get("listener") or "") != reviewer:
            continue
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        if str(context.get("preference_program") or "") != "active_learning_v1":
            continue
        scenario_id = str(context.get("teacher_scenario_id") or "")
        if scenario_id:
            output.add(scenario_id)
    return output


def _scenario_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("scenario_id")): row
        for row in load_scenarios(path)
        if str(row.get("scenario_id") or "")
    }


def _cache_path(workspace: Path, source_event_id: str) -> Path:
    return workspace / DEFAULT_REVIEW_CACHE / f"{source_event_id}.json"


def _reviewable(
    workspace: Path,
    queue_row: Mapping[str, Any],
    scenario: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(scenario, Mapping):
        return False
    context = scenario.get("context") if isinstance(scenario.get("context"), Mapping) else {}
    source_event_id = str(context.get("source_event_id") or "")
    candidates = scenario.get("candidates") if isinstance(scenario.get("candidates"), list) else []
    if not source_event_id or len(candidates) != 2:
        return False
    return _cache_path(workspace, source_event_id).exists()


def human_preference_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    queue = _jsonl(root / DEFAULT_REVIEW_QUEUE)
    scenarios = _scenario_map(root / DEFAULT_SCENARIOS)
    preferences = load_preferences(root / DEFAULT_PREFERENCES)

    reviewed_scenarios: set[str] = set()
    cohorts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    active_votes = 0
    confidence_values: list[float] = []
    for row in preferences:
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        if str(context.get("preference_program") or "") != "active_learning_v1":
            continue
        active_votes += 1
        scenario_id = str(context.get("teacher_scenario_id") or "")
        if scenario_id:
            reviewed_scenarios.add(scenario_id)
        cohorts[str(context.get("evaluator_cohort") or "unknown")] += 1
        for reason in context.get("reasons", []) if isinstance(context.get("reasons"), list) else []:
            reasons[str(reason)] += 1
        if context.get("human_confidence") is not None:
            confidence_values.append(float(context["human_confidence"]))

    reviewable = sum(
        1
        for row in queue
        if _reviewable(root, row, scenarios.get(str(row.get("scenario_id") or "")))
    )
    outstanding = sum(
        1
        for row in queue
        if str(row.get("scenario_id") or "") not in reviewed_scenarios
        and _reviewable(root, row, scenarios.get(str(row.get("scenario_id") or "")))
    )
    return {
        "schema_version": HUMAN_PREFERENCE_SCHEMA_VERSION,
        "queue_total": len(queue),
        "queue_reviewable": reviewable,
        "queue_outstanding": outstanding,
        "reviewed_scenarios": len(reviewed_scenarios),
        "active_learning_votes": active_votes,
        "cohorts": dict(cohorts),
        "top_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(10)
        ],
        "mean_human_confidence": (
            round(sum(confidence_values) / len(confidence_values), 6)
            if confidence_values else None
        ),
        "policy": {
            "blind_ab_required": True,
            "exact_queued_transition_replay": True,
            "teacher_scores_hidden_from_reviewers": True,
            "local_review_cache_exported_to_training_data": False,
            "human_labels_are_ground_truth_anchor": True,
            "human_labels_bypass_hard_safety": False,
        },
    }


def create_human_review_session(
    workspace: str | Path,
    *,
    evaluator: str,
    cohort: EvaluatorCohort,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    reviewer = evaluator_id(evaluator, cohort)
    preferences = load_preferences(root / DEFAULT_PREFERENCES)
    already_reviewed = _reviewed_scenarios(preferences, reviewer)
    queue = _jsonl(root / DEFAULT_REVIEW_QUEUE)
    scenarios = _scenario_map(root / DEFAULT_SCENARIOS)

    candidates = [
        row
        for row in queue
        if (
            (scenario_id is None or str(row.get("scenario_id") or "") == scenario_id)
            and str(row.get("scenario_id") or "") not in already_reviewed
            and _reviewable(root, row, scenarios.get(str(row.get("scenario_id") or "")))
        )
    ]
    candidates.sort(
        key=lambda row: (
            -float(row.get("priority", 0.0) or 0.0),
            str(row.get("scenario_id") or ""),
        )
    )
    if not candidates:
        raise ValueError("No reviewable active-learning comparison is available for this evaluator.")

    selected = candidates[0]
    selected_scenario_id = str(selected["scenario_id"])
    scenario = scenarios[selected_scenario_id]
    context = scenario.get("context") if isinstance(scenario.get("context"), Mapping) else {}
    source_event_id = str(context.get("source_event_id") or "")
    cache = json.loads(_cache_path(root, source_event_id).read_text(encoding="utf-8"))
    cached_plans = {
        str(item.get("execution_signature") or ""): item.get("plan")
        for item in cache.get("candidates", [])
        if isinstance(item, Mapping) and isinstance(item.get("plan"), Mapping)
    }

    scenario_candidates = scenario.get("candidates") if isinstance(scenario.get("candidates"), list) else []
    requested_signatures = [
        str(item.get("execution_signature") or "")
        for item in scenario_candidates
        if isinstance(item, Mapping)
    ]
    if len(requested_signatures) != 2 or any(signature not in cached_plans for signature in requested_signatures):
        raise ValueError("Queued teacher scenario no longer has exact executable replay material.")

    plans = [
        transition_plan_from_dict(dict(cached_plans[signature]))
        for signature in requested_signatures
    ]
    secrets.SystemRandom().shuffle(plans)

    session_id = f"human-review-{secrets.token_hex(6)}"
    session_root = root / DEFAULT_REVIEW_SESSIONS / session_id
    session_root.mkdir(parents=True, exist_ok=False)
    labels = ("A", "B")
    items: list[dict[str, Any]] = []
    for label, plan in zip(labels, plans, strict=True):
        audio_path = render_transition(
            plan,
            session_root / f"candidate-{label.lower()}.wav",
            pre_roll_seconds=6.0,
            post_roll_seconds=6.0,
        )
        items.append(
            {
                "label": label,
                "audio_path": str(audio_path),
                "execution_signature": plan.execution_signature,
                "plan": plan_features(plan),
            }
        )

    manifest = {
        "schema_version": HUMAN_PREFERENCE_SCHEMA_VERSION,
        "session_id": session_id,
        "teacher_scenario_id": selected_scenario_id,
        "source_event_id": source_event_id,
        "queue_priority": float(selected.get("priority", 0.0) or 0.0),
        "queue_reasons": list(selected.get("reasons") or []),
        "evaluator_id": reviewer,
        "evaluator_cohort": cohort,
        "blind": True,
        "candidate_order_randomized": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": items,
    }
    (session_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def record_human_review_vote(
    workspace: str | Path,
    *,
    session_id: str,
    choice: str,
    reasons: list[str] | tuple[str, ...],
    confidence: float,
    experience_years: float | None = None,
) -> Path:
    root = Path(workspace).expanduser().resolve()
    manifest_path = root / DEFAULT_REVIEW_SESSIONS / session_id / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Human review session does not exist.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates") if isinstance(manifest.get("candidates"), list) else []
    if len(candidates) != 2:
        raise ValueError("Human review session does not contain exactly two candidates.")

    reviewer = str(manifest.get("evaluator_id") or "")
    scenario_id = str(manifest.get("teacher_scenario_id") or "")
    if not reviewer or not scenario_id:
        raise ValueError("Human review manifest is missing evaluator/scenario identity.")

    existing = load_preferences(root / DEFAULT_PREFERENCES)
    if scenario_id in _reviewed_scenarios(existing, reviewer):
        raise ValueError("This evaluator has already reviewed the active-learning scenario.")

    cohort = str(manifest.get("evaluator_cohort") or "general_listener")
    cohort_weight = float(_COHORT_WEIGHT.get(cohort, 1.0))
    human_confidence = max(0.0, min(1.0, float(confidence)))
    confidence_weight = 0.90 + 0.20 * human_confidence
    weight = min(1.35, cohort_weight * confidence_weight)

    context = {
        "preference_program": "active_learning_v1",
        "label_source": "human_blind_preference",
        "human_preference": True,
        "teacher_scenario_id": scenario_id,
        "source_event_id": str(manifest.get("source_event_id") or ""),
        "queue_priority": float(manifest.get("queue_priority", 0.0) or 0.0),
        "queue_reasons": list(manifest.get("queue_reasons") or []),
        "session_id": session_id,
        "candidate_labels": ["A", "B"],
        "evaluator_cohort": cohort,
        "human_confidence": human_confidence,
        "experience_years": (
            max(0.0, min(80.0, float(experience_years)))
            if experience_years is not None
            else None
        ),
        "reasons": list(dict.fromkeys(str(reason).strip().lower() for reason in reasons if str(reason).strip())),
        "blind": True,
    }
    return append_preference(
        candidate_a=dict(candidates[0]["plan"]),
        candidate_b=dict(candidates[1]["plan"]),
        choice=choice,
        listener=reviewer,
        context=context,
        weight=weight,
        path=root / DEFAULT_PREFERENCES,
    )
