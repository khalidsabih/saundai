from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..intelligence.preferences import plan_features

TEACHER_SCHEMA_VERSION = "1.0"
TEACHER_PROMPT_VERSION = "1.0"

TeacherTask = Literal[
    "transition_preference",
    "next_track_ranking",
    "transition_family",
    "technique_ranking",
    "duration_preference",
    "hard_negative",
    "set_story",
]

_MACHINE_LABEL_SOURCE = "llm_teacher_research"
_FORBIDDEN_CANDIDATE_KEYS = {
    "analytical_score",
    "measured_score",
    "measurement_passed",
    "professional_judge_score",
    "selection_utility",
    "state_move_taste_shadow",
    "rejection_penalty",
    "rendered_transition_embedding",
    "track_a",
    "track_b",
}


@dataclass(frozen=True, slots=True)
class TeacherScenario:
    scenario_id: str
    task: TeacherTask
    prompt_version: str
    context: dict[str, Any]
    candidates: tuple[dict[str, Any], ...]
    requested_labels: tuple[str, ...]
    lane: str = "research"
    raw_audio_included: bool = False
    local_paths_included: bool = False
    schema_version: str = TEACHER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [dict(item) for item in self.candidates]
        payload["requested_labels"] = list(self.requested_labels)
        return payload


@dataclass(frozen=True, slots=True)
class TeacherJudgment:
    scenario_id: str
    task: TeacherTask
    teacher_provider: str
    teacher_model: str
    teacher_model_version: str
    prompt_version: str
    ranking: tuple[str, ...] = ()
    choice: str | None = None
    confidence: float = 0.0
    reason_codes: tuple[str, ...] = ()
    rationale: str = ""
    temperature: float | None = None
    seed: str | None = None
    label_source: str = _MACHINE_LABEL_SOURCE
    human_preference: bool = False
    production_authority: bool = False
    schema_version: str = TEACHER_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ranking"] = list(self.ranking)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def _stable_id(*parts: object, prefix: str) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return round(float(value), 6)
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(item)
            for key, item in value.items()
            if str(key) not in _FORBIDDEN_CANDIDATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return str(value)


def teacher_candidate(plan) -> dict[str, Any]:
    """Return an interpretable, score-blind plan payload for LLM teachers."""
    raw = plan_features(plan)
    candidate = {
        key: value
        for key, value in raw.items()
        if key not in _FORBIDDEN_CANDIDATE_KEYS
    }
    measurement = candidate.get("measurement")
    if isinstance(measurement, Mapping):
        # Keep concrete audio observations, but not the aggregate score or
        # pass/fail verdict which would leak SAUND's existing judgment.
        candidate["measurement"] = {
            key: value
            for key, value in measurement.items()
            if key not in {
                "overall",
                "musical_overall",
                "hard_failures",
                "warnings",
            }
        }
    return _sanitize_value(candidate)


def transition_preference_scenario(
    plan_a,
    plan_b,
    *,
    context: Mapping[str, Any] | None = None,
) -> TeacherScenario:
    candidate_a = teacher_candidate(plan_a)
    candidate_b = teacher_candidate(plan_b)
    if candidate_a.get("execution_signature") == candidate_b.get("execution_signature"):
        raise ValueError("Teacher comparison candidates must be different transitions.")
    scenario_context = _sanitize_value(dict(context or {}))
    scenario_id = _stable_id(
        "transition_preference",
        candidate_a.get("execution_signature", ""),
        candidate_b.get("execution_signature", ""),
        json.dumps(scenario_context, sort_keys=True),
        prefix="teacher-scenario",
    )
    return TeacherScenario(
        scenario_id=scenario_id,
        task="transition_preference",
        prompt_version=TEACHER_PROMPT_VERSION,
        context=scenario_context,
        candidates=(candidate_a, candidate_b),
        requested_labels=(
            "choice:a|b|tie|neither",
            "confidence:0..1",
            "reason_codes",
            "short_rationale",
        ),
    )


def write_scenarios(path: str | Path, scenarios: Iterable[TeacherScenario]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("scenario_id"):
                rows.append(payload)
    return rows


def teacher_prompt_payload(scenario: TeacherScenario | Mapping[str, Any]) -> dict[str, Any]:
    raw = scenario.to_dict() if isinstance(scenario, TeacherScenario) else dict(scenario)
    return {
        "role": (
            "You are an experienced professional DJ evaluating structured musical choices. "
            "Judge musical plausibility and style fit, not SAUND's existing scores. "
            "Do not assume metadata proves how the audio sounds."
        ),
        "task": raw.get("task"),
        "context": _sanitize_value(raw.get("context", {})),
        "candidates": _sanitize_value(raw.get("candidates", [])),
        "requested_output": {
            "choice_or_ranking": "Use only candidate IDs present in the scenario.",
            "confidence": "0.0 to 1.0",
            "reason_codes": [
                "beats",
                "phrasing",
                "vocals",
                "bass",
                "energy",
                "harmony",
                "style",
                "too_long",
                "too_abrupt",
                "creativity",
                "uncertain",
            ],
            "rationale": "Brief musical reasoning, no chain-of-thought requirement.",
        },
        "prompt_version": str(raw.get("prompt_version") or TEACHER_PROMPT_VERSION),
    }


def validate_teacher_judgment(
    scenario: TeacherScenario | Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    model_version: str = "",
    temperature: float | None = None,
    seed: str | None = None,
) -> TeacherJudgment:
    raw = scenario.to_dict() if isinstance(scenario, TeacherScenario) else dict(scenario)
    scenario_id = str(raw.get("scenario_id") or "")
    if not scenario_id:
        raise ValueError("Teacher scenario is missing scenario_id.")
    task = str(raw.get("task") or "")
    candidate_ids = {
        str(item.get("execution_signature") or item.get("candidate_id") or "")
        for item in raw.get("candidates", [])
        if isinstance(item, Mapping)
    }
    candidate_ids.discard("")

    choice = payload.get("choice")
    if choice is not None:
        choice = str(choice).strip().lower()
        if task == "transition_preference" and choice not in {"a", "b", "tie", "neither"}:
            raise ValueError("Transition preference teacher choice must be a, b, tie, or neither.")

    ranking = tuple(str(item) for item in payload.get("ranking", []) if str(item))
    if ranking and candidate_ids and any(item not in candidate_ids for item in ranking):
        raise ValueError("Teacher ranking references a candidate outside the scenario.")

    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
    reason_codes = tuple(
        dict.fromkeys(
            str(item).strip().lower()
            for item in payload.get("reason_codes", [])
            if str(item).strip()
        )
    )
    rationale = str(payload.get("rationale") or "")[:1200]

    return TeacherJudgment(
        scenario_id=scenario_id,
        task=task,  # type: ignore[arg-type]
        teacher_provider=str(provider).strip() or "unknown",
        teacher_model=str(model).strip() or "unknown",
        teacher_model_version=str(model_version).strip(),
        prompt_version=str(raw.get("prompt_version") or TEACHER_PROMPT_VERSION),
        ranking=ranking,
        choice=choice,
        confidence=confidence,
        reason_codes=reason_codes,
        rationale=rationale,
        temperature=temperature,
        seed=seed,
    )


def append_teacher_judgment(path: str | Path, judgment: TeacherJudgment) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(judgment.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return destination
