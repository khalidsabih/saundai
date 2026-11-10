from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .teacher_consensus import load_judgments
from .teacher_dataset import load_scenarios

TEACHER_GOVERNANCE_VERSION = "2.0"

DEFAULT_SCENARIOS = "datasets/generated/teacher/scenarios.jsonl"
DEFAULT_JUDGMENTS = "datasets/generated/teacher/judgments.jsonl"
DEFAULT_CONSENSUS = "datasets/generated/teacher/consensus.jsonl"
DEFAULT_REVIEW_QUEUE = "datasets/generated/teacher/human-review-queue.jsonl"
DEFAULT_HARD_NEGATIVES = "datasets/generated/teacher/hard-negatives.jsonl"


def teacher_data_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    scenarios_path = root / DEFAULT_SCENARIOS
    judgments_path = root / DEFAULT_JUDGMENTS
    consensus_path = root / DEFAULT_CONSENSUS
    review_path = root / DEFAULT_REVIEW_QUEUE
    hard_negative_path = root / DEFAULT_HARD_NEGATIVES

    scenarios = load_scenarios(scenarios_path)
    judgments = load_judgments(judgments_path)

    tasks = Counter(str(item.get("task") or "unknown") for item in scenarios)
    providers = Counter(str(item.get("teacher_provider") or "unknown") for item in judgments)
    models = Counter(
        f"{item.get('teacher_provider') or 'unknown'}:{item.get('teacher_model') or 'unknown'}"
        for item in judgments
    )

    consensus_total = decisive = uncertain = duplicate_removed = diversity_blocked = 0
    if consensus_path.exists():
        with consensus_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                consensus_total += 1
                is_decisive = bool(row.get("decisive"))
                decisive += int(is_decisive)
                uncertain += int(not is_decisive)
                duplicate_removed += int(row.get("duplicate_count", 0) or 0)
                flags = set(str(item) for item in (row.get("quality_flags") or []))
                diversity_blocked += int(
                    bool(
                        flags.intersection(
                            {
                                "insufficient_provider_diversity",
                                "insufficient_model_diversity",
                                "correlated_teacher_family_risk",
                            }
                        )
                    )
                )

    def _jsonl_count(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
        return count

    return {
        "version": TEACHER_GOVERNANCE_VERSION,
        "lane": "research",
        "label_source": "llm_teacher_research",
        "production_authority": False,
        "human_preference": False,
        "raw_audio_required": False,
        "provider_neutral": True,
        "paths": {
            "scenarios": str(scenarios_path),
            "judgments": str(judgments_path),
            "consensus": str(consensus_path),
            "human_review_queue": str(review_path),
            "hard_negatives": str(hard_negative_path),
        },
        "scenario_count": len(scenarios),
        "judgment_count": len(judgments),
        "consensus_count": consensus_total,
        "decisive_consensus_count": decisive,
        "uncertain_consensus_count": uncertain,
        "duplicate_judgments_removed": duplicate_removed,
        "diversity_flagged_consensus_count": diversity_blocked,
        "human_review_queue_count": _jsonl_count(review_path),
        "hard_negative_count": _jsonl_count(hard_negative_path),
        "tasks": dict(tasks),
        "providers": dict(providers),
        "models": dict(models),
        "policy": {
            "synthetic_labels_are_weak_supervision": True,
            "human_validation_required_for_production_authority": True,
            "teacher_scores_do_not_override_hard_failures": True,
            "teacher_prompts_hide_saund_aggregate_scores": True,
            "raw_audio_not_stored_in_teacher_scenarios": True,
            "local_paths_not_stored_in_teacher_scenarios": True,
            "decisive_consensus_requires_teacher_diversity": True,
            "teacher_confidence_is_calibrated_or_shrunk": True,
            "uncertain_or_correlated_cases_route_to_humans": True,
            "hard_negative_candidates_remain_research_only": True,
        },
    }
