from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .environment import current_environment
from .execution import TransitionRenderContext
from .measurement import MeasurementPolicy
from .models import TransitionMeasurement, TransitionPlan
from .version import MEASUREMENT_VERSION

REHEARSAL_CACHE_VERSION = "6.0"


@dataclass(frozen=True, slots=True)
class RehearsalCacheEntry:
    version: str
    key: str
    execution_signature: str
    measurement: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rehearsal_cache_enabled() -> bool:
    return os.environ.get("DJAI_REHEARSAL_CACHE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _render_context_payload(
    render_context: TransitionRenderContext | None,
    *,
    sample_rate: int,
    context_seconds: float,
) -> dict[str, Any]:
    if render_context is None:
        return {
            "sample_rate": int(sample_rate),
            "context_seconds": round(float(context_seconds), 6),
            "exact_set_context": False,
        }
    value = render_context.normalized()
    return {
        "sample_rate": int(value.sample_rate),
        "gain_a": round(float(value.gain_a), 10),
        "gain_b": None if value.gain_b is None else round(float(value.gain_b), 10),
        "target_loudness_dbfs": round(float(value.target_loudness_dbfs), 6),
        "context_seconds": round(float(context_seconds), 6),
        "exact_set_context": True,
    }


def rehearsal_cache_key(
    plan: TransitionPlan,
    *,
    sample_rate: int,
    context_seconds: float,
    policy: MeasurementPolicy,
    render_context: TransitionRenderContext | None,
    capture_render_embedding: bool,
) -> str:
    payload = {
        "cache_version": REHEARSAL_CACHE_VERSION,
        "measurement_version": MEASUREMENT_VERSION,
        "execution_signature": plan.execution_signature,
        "performance_program_fingerprint": plan.performance_program_fingerprint,
        "render_context": _render_context_payload(
            render_context,
            sample_rate=sample_rate,
            context_seconds=context_seconds,
        ),
        "measurement_policy": asdict(policy),
        "capture_render_embedding": bool(capture_render_embedding),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _path_for(key: str) -> Path:
    return current_environment().cache_dir / "rehearsals" / f"{key}.json"


def load_cached_rehearsal(key: str) -> tuple[TransitionMeasurement, dict[str, Any]] | None:
    if not rehearsal_cache_enabled():
        return None
    path = _path_for(key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != REHEARSAL_CACHE_VERSION or payload.get("key") != key:
            return None
        measurement = TransitionMeasurement(**dict(payload["measurement"]))
        metadata = dict(payload.get("metadata", {}))
        return measurement, metadata
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def store_cached_rehearsal(
    key: str,
    *,
    execution_signature: str,
    measurement: TransitionMeasurement,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    if not rehearsal_cache_enabled():
        return None
    # Never persist infrastructure/decoder exceptions. Musical hard failures
    # are deterministic evidence and are safe to cache; transient I/O failures
    # are not.
    if any(str(value).startswith("Candidate rendering or measurement failed:") for value in measurement.hard_failures):
        return None
    destination = _path_for(key)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        entry = RehearsalCacheEntry(
            version=REHEARSAL_CACHE_VERSION,
            key=key,
            execution_signature=execution_signature,
            measurement=measurement.to_dict(),
            metadata=dict(metadata or {}),
        )
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination
    except OSError:
        return None
