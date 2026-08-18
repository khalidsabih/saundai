from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .personalization import update_profile_from_event
from .privacy import sanitize_feedback_metadata
from .version import FEEDBACK_SCHEMA_VERSION


ALLOWED_EVENTS = {
    "set_like",
    "set_dislike",
    "transition_like",
    "transition_dislike",
    "alternative_selected",
    "replanned_after_preview",
    "render_completed",
    "downloaded",
    "skip",
    "replay",
    "playback_completed",
}


def append_feedback_event(
    *,
    event: str,
    listener: str = "local",
    plan_id: str | None = None,
    transition_index: int | None = None,
    value: float | None = None,
    journey: str | None = None,
    plan: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    feedback_path: str | Path = "ratings/feedback-events.jsonl",
    profiles_path: str | Path = "ratings/listener-profiles.json",
    personalization_enabled: bool = True,
    training_consent: bool = False,
) -> tuple[Path, dict[str, Any]]:
    clean_event = event.strip().lower()
    if clean_event not in ALLOWED_EVENTS:
        raise ValueError(f"Unsupported feedback event: {event}")
    record: dict[str, Any] = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "event": clean_event,
        "listener": listener.strip() or "local",
        "plan_id": plan_id,
        "transition_index": transition_index,
        "value": value,
        "journey": journey,
        "plan": dict(plan or {}),
        "metadata": sanitize_feedback_metadata(metadata),
        "consent": {
            "personalization": bool(personalization_enabled),
            "training": bool(training_consent),
        },
        "training_eligible": bool(training_consent),
    }
    destination = Path(feedback_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if not personalization_enabled:
        return destination, {}
    profile = update_profile_from_event(record, path=profiles_path)
    return destination, profile.to_dict()
