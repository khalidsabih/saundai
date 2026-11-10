from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .teacher_consensus import load_judgments
from .teacher_dataset import load_scenarios

TEACHER_GOVERNANCE_VERSION = "1.0"

DEFAULT_SCENARIOS = "datasets/generated/teacher/scenarios.jsonl"
DEFAULT_JUDGMENTS = "datasets/generated/teacher/judgments.jsonl"
DEFAULT_CONSENSUS = "datasets/generated/teacher/consensus.jsonl"


def teacher_data_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    scenarios_path = root / DEFAULT_SCENARIOS
    judgments_path = root / DEFAULT_JUDGMENTS
    consensus_path = root / DEFAULT_CONSENSUS

    scenarios = load_scenarios(scenarios_path)
    judgments = load_judgments(judgments_path)

    tasks = Counter(str(item.get("task") or "unknown") for item in scenarios)
    providers = Counter(str(item.get("teacher_provider") or "unknown") for item in judgments)
    models = Counter(
        f"{item.get('teacher_provider') or 'unknown'}:{item.get('teacher_model') or 'unknown'}"
        for item in judgments
    )

    consensus_total = decisive = 0
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
                decisive += int(bool(row.get("decisive")))

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
        },
        "scenario_count": len(scenarios),
        "judgment_count": len(judgments),
        "consensus_count": consensus_total,
        "decisive_consensus_count": decisive,
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
        },
    }
