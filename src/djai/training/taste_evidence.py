from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..intelligence.preferences import plan_features
from ..models import TransitionPlan

TASTE_EVIDENCE_SCHEMA_VERSION = "1.8.1"
DEFAULT_EVIDENCE_PATH = "ratings/runtime-candidate-evidence.jsonl"


def _enabled() -> bool:
    return os.environ.get("DJAI_TASTE_EVIDENCE_CAPTURE", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _workspace() -> Path:
    return Path(os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()


def _track_id(value: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]
    return f"track-ref-{digest}"


def _safe_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    allowed = {
        "mode",
        "automatic_delivery",
        "professional_brain",
        "fallback_level",
        "adaptive_rehearsal_stage",
        "unified_transition",
        "unified_transition_emergency",
        "primary_transition_family",
        "transition_family",
        "strategy_attempt",
        "family_finalist_budget",
        "total_edge_finalist_budget",
        "resolution_action",
    }
    output: dict[str, Any] = {}
    for key, value in dict(context or {}).items():
        if key not in allowed:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
    return output


def _candidate_payload(plan: TransitionPlan) -> dict[str, Any]:
    payload = plan_features(plan)
    track_a = str(payload.pop("track_a", ""))
    track_b = str(payload.pop("track_b", ""))
    payload["track_a_id"] = _track_id(track_a)
    payload["track_b_id"] = _track_id(track_b)
    payload["pair_id"] = "pair-" + hashlib.sha256(
        f"{payload['track_a_id']}|{payload['track_b_id']}".encode("utf-8")
    ).hexdigest()[:18]
    return payload


def capture_rehearsal_evidence(
    plans: Sequence[TransitionPlan],
    *,
    selected_signature: str | None,
    context: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path | None:
    """Append structured, audio-free evidence for every real rehearsal candidate.

    This captures the candidate population SAUND actually considered, including
    rejected/hard-failed renders. It intentionally stores pseudonymous track
    references instead of local filenames and never copies raw audio.
    """

    if not _enabled() or not plans:
        return None
    destination = Path(path).expanduser() if path is not None else _workspace() / DEFAULT_EVIDENCE_PATH
    if not destination.is_absolute():
        destination = _workspace() / destination
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    event_seed = f"{time.time_ns()}|{selected_signature or ''}|{len(plans)}"
    event_id = "rehearsal-" + hashlib.sha256(event_seed.encode("utf-8")).hexdigest()[:20]
    safe_context = _safe_context(context)

    with destination.open("a", encoding="utf-8") as handle:
        for rank, plan in enumerate(plans, start=1):
            candidate = _candidate_payload(plan)
            measurement = candidate.get("measurement") if isinstance(candidate.get("measurement"), dict) else {}
            row = {
                "schema_version": TASTE_EVIDENCE_SCHEMA_VERSION,
                "event_id": event_id,
                "candidate_rank": int(getattr(plan, "measurement_rank", 0) or rank),
                "selected": bool(selected_signature and plan.execution_signature == selected_signature),
                "gate_passed": bool(plan.measurement_passed),
                "hard_failure_count": len(measurement.get("hard_failures", []) or []),
                "candidate": candidate,
                "context": safe_context,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def runtime_evidence_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    path = root / DEFAULT_EVIDENCE_PATH
    total = selected = passed = hard_failed = 0
    events: set[str] = set()
    modes: Counter[str] = Counter()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                total += 1
                selected += int(bool(row.get("selected")))
                passed += int(bool(row.get("gate_passed")))
                hard_failed += int(int(row.get("hard_failure_count", 0) or 0) > 0)
                if row.get("event_id"):
                    events.add(str(row["event_id"]))
                context = row.get("context") if isinstance(row.get("context"), dict) else {}
                modes[str(context.get("mode") or "unknown")] += 1
    return {
        "schema_version": TASTE_EVIDENCE_SCHEMA_VERSION,
        "path": DEFAULT_EVIDENCE_PATH,
        "events": len(events),
        "candidates": total,
        "selected": selected,
        "gate_passed": passed,
        "hard_failed": hard_failed,
        "modes": dict(modes),
        "raw_audio_stored": False,
        "track_paths_stored": False,
    }
