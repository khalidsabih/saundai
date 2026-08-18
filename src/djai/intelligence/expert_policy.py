from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import TransitionPlan

MODEL_VERSION = "1.0"
DEFAULT_PATH = "config/expert-policy.json"


@dataclass(frozen=True, slots=True)
class ExpertPrior:
    score: float = 0.5
    authority: float = 0.0
    context_count: int = 0
    global_count: int = 0


def _context_key(plan: dict[str, Any], journey: str = "unspecified") -> str:
    exit_section = str(plan.get("exit_section") or "unknown")
    entry_section = str(plan.get("entry_section") or "unknown")
    tempo_diff = abs(float(plan.get("tempo_difference_percent", plan.get("tempo_difference", 0.0)) or 0.0))
    tempo_bin = "near" if tempo_diff < 2.0 else "moderate" if tempo_diff < 6.0 else "wide"
    return f"{exit_section}->{entry_section}|{journey}|{tempo_bin}"


def _winning_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    choice = str(record.get("choice", ""))
    if choice == "a":
        return record.get("candidate_a") if isinstance(record.get("candidate_a"), dict) else None
    if choice == "b":
        return record.get("candidate_b") if isinstance(record.get("candidate_b"), dict) else None
    return None


def train_expert_policy(
    source: str | Path = "ratings/expert-decisions.jsonl",
    output: str | Path = DEFAULT_PATH,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    contexts: dict[str, Counter[str]] = defaultdict(Counter)
    global_actions: Counter[str] = Counter()
    examples = 0
    if source_path.exists():
        for line in source_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            schema = str(record.get("schema", ""))
            if schema.startswith("expert-set"):
                journey = str(record.get("journey", "unspecified"))
                for transition in record.get("transitions", []):
                    if not isinstance(transition, dict):
                        continue
                    technique = str(transition.get("technique", ""))
                    intent = str(transition.get("intent", ""))
                    if not technique or not intent:
                        continue
                    action = f"{intent}|{technique}"
                    contexts[_context_key(transition, journey)][action] += 1
                    global_actions[action] += 1
                    examples += 1
            elif schema.startswith("expert-transition"):
                winner = _winning_candidate(record)
                if winner is None:
                    continue
                technique = str(winner.get("technique", ""))
                intent = str(winner.get("intent", ""))
                if not technique or not intent:
                    continue
                journey = str((record.get("context") or {}).get("journey", "unspecified")) if isinstance(record.get("context"), dict) else "unspecified"
                action = f"{intent}|{technique}"
                contexts[_context_key(winner, journey)][action] += 1
                global_actions[action] += 1
                examples += 1
    payload = {
        "version": MODEL_VERSION,
        "example_count": examples,
        "contexts": {key: dict(value) for key, value in sorted(contexts.items())},
        "global_actions": dict(global_actions),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def expert_action_prior(
    plan: TransitionPlan,
    *,
    journey: str = "unspecified",
    model_path: str | Path = DEFAULT_PATH,
) -> ExpertPrior:
    source = Path(model_path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ExpertPrior()
    plan_payload = {
        "exit_section": plan.exit_section,
        "entry_section": plan.entry_section,
        "tempo_difference_percent": plan.compatibility.tempo_difference_percent,
    }
    key = _context_key(plan_payload, journey)
    contexts = payload.get("contexts", {}) if isinstance(payload, dict) else {}
    counts = Counter(contexts.get(key, {})) if isinstance(contexts, dict) else Counter()
    global_counts = Counter(payload.get("global_actions", {})) if isinstance(payload, dict) else Counter()
    action = f"{plan.intent}|{plan.technique}"
    context_total = sum(counts.values())
    global_total = sum(global_counts.values())
    # Smoothed probability. Context gets authority only once it has repeated
    # expert evidence; the global prior is deliberately weak.
    global_p = (global_counts[action] + 1.0) / (global_total + max(1, len(global_counts)) + 1.0)
    if context_total:
        local_p = (counts[action] + 1.0) / (context_total + max(1, len(counts)) + 1.0)
        raw = 0.78 * local_p + 0.22 * global_p
    else:
        raw = global_p if global_total else 0.5
    # Convert distribution probability to a softer 0..1 preference signal.
    score = 1.0 / (1.0 + math.exp(-5.0 * (raw - 0.18))) if global_total else 0.5
    evidence = context_total + 0.2 * global_total
    authority = min(0.20, 0.20 * evidence / 800.0)
    return ExpertPrior(
        score=round(max(0.0, min(1.0, score)), 4),
        authority=round(max(0.0, min(0.20, authority)), 4),
        context_count=context_total,
        global_count=global_total,
    )
