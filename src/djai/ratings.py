from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RATING_FIELDS = (
    "beat_alignment",
    "phrase_timing",
    "bass_cleanliness",
    "energy_flow",
    "cue_quality",
    "overall",
)


def _validate_score(name: str, value: int) -> int:
    score = int(value)
    if score < 1 or score > 5:
        raise ValueError(f"{name} must be an integer from 1 to 5")
    return score


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Report is not valid JSON: {report_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Report JSON must contain an object")
    data["_report_path"] = str(report_path)
    return data


def _select_plan(
    report: dict[str, Any],
    alternative_id: str | None,
    transition_index: int | None = None,
) -> dict[str, Any]:
    alternatives = report.get("alternatives")
    if isinstance(alternatives, list):
        if not alternative_id:
            raise ValueError(
                "This is an alternatives report; provide --alternative safe, creative, or emergency"
            )
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id") or item.get("alternative_id")
            plan = item.get("plan", item)
            if item_id == alternative_id and isinstance(plan, dict):
                return plan
        raise ValueError(f"Alternative '{alternative_id}' was not found in the report")

    previews = report.get("previews")
    if isinstance(previews, list):
        if transition_index is None:
            raise ValueError("This is a set-preview report; provide a transition index")
        if transition_index < 0 or transition_index >= len(previews):
            raise ValueError("Transition index is outside the preview report")
        item = previews[transition_index]
        if isinstance(item, dict) and isinstance(item.get("plan"), dict):
            return item["plan"]

    transitions = report.get("transitions")
    if isinstance(transitions, list):
        if transition_index is None:
            raise ValueError("This is a set report; provide a transition index")
        if transition_index < 0 or transition_index >= len(transitions):
            raise ValueError("Transition index is outside the set report")
        item = transitions[transition_index]
        if isinstance(item, dict):
            plan = item.get("plan", item)
            if isinstance(plan, dict):
                return plan

    transition = report.get("transition")
    if isinstance(transition, dict):
        return transition
    raise ValueError("The report contains neither alternatives nor a transition plan")


def append_listening_rating(
    report_path: str | Path,
    *,
    alternative_id: str | None,
    transition_index: int | None = None,
    beat_alignment: int,
    phrase_timing: int,
    bass_cleanliness: int,
    energy_flow: int,
    cue_quality: int,
    overall: int,
    notes: str = "",
    listener: str = "local",
    ratings_path: str | Path = "ratings/listening-tests.jsonl",
) -> Path:
    report = _load_report(report_path)
    plan = _select_plan(report, alternative_id, transition_index)
    scores = {
        "beat_alignment": _validate_score("beat_alignment", beat_alignment),
        "phrase_timing": _validate_score("phrase_timing", phrase_timing),
        "bass_cleanliness": _validate_score("bass_cleanliness", bass_cleanliness),
        "energy_flow": _validate_score("energy_flow", energy_flow),
        "cue_quality": _validate_score("cue_quality", cue_quality),
        "overall": _validate_score("overall", overall),
    }

    output = Path(ratings_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "rated_at_utc": datetime.now(timezone.utc).isoformat(),
        "listener": listener,
        "source_report": report["_report_path"],
        "alternative_id": plan.get("alternative_id") or alternative_id or "safe",
        "transition_index": transition_index,
        "style": plan.get("style", alternative_id or "safe"),
        "track_a": plan.get("track_a"),
        "track_b": plan.get("track_b"),
        "exit_time_a": plan.get("exit_time_a"),
        "entry_time_b": plan.get("entry_time_b"),
        "transition_bars": plan.get("transition_bars"),
        "technique": plan.get("technique"),
        "tempo_policy": plan.get("tempo_policy"),
        "scores": scores,
        "notes": notes.strip(),
        "plan": plan,
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return output
