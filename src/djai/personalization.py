from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import TransitionPlan
from .version import PERSONALIZATION_VERSION


@dataclass(slots=True)
class ListenerProfile:
    listener: str
    sample_count: int = 0
    confidence: float = 0.0
    style_bias: dict[str, float] = field(default_factory=dict)
    technique_bias: dict[str, float] = field(default_factory=dict)
    journey_bias: dict[str, float] = field(default_factory=dict)
    energy_delta_target: float = 0.04
    version: str = PERSONALIZATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_bias(value: float) -> float:
    return max(-0.18, min(0.18, float(value)))


def _profiles_path(path: str | Path | None = None) -> Path:
    return Path(path or "ratings/listener-profiles.json").expanduser().resolve()


def load_profiles(path: str | Path | None = None) -> dict[str, ListenerProfile]:
    source = _profiles_path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = payload.get("profiles", payload) if isinstance(payload, dict) else {}
    if not isinstance(records, dict):
        return {}
    result: dict[str, ListenerProfile] = {}
    for listener, item in records.items():
        if not isinstance(item, dict):
            continue
        try:
            result[str(listener)] = ListenerProfile(
                listener=str(listener),
                sample_count=int(item.get("sample_count", 0)),
                confidence=float(item.get("confidence", 0.0)),
                style_bias={str(k): float(v) for k, v in dict(item.get("style_bias", {})).items()},
                technique_bias={str(k): float(v) for k, v in dict(item.get("technique_bias", {})).items()},
                journey_bias={str(k): float(v) for k, v in dict(item.get("journey_bias", {})).items()},
                energy_delta_target=float(item.get("energy_delta_target", 0.04)),
                version=str(item.get("version", PERSONALIZATION_VERSION)),
            )
        except (TypeError, ValueError):
            continue
    return result


def save_profiles(
    profiles: dict[str, ListenerProfile],
    path: str | Path | None = None,
) -> Path:
    destination = _profiles_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "version": PERSONALIZATION_VERSION,
                "profiles": {name: profile.to_dict() for name, profile in sorted(profiles.items())},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def get_listener_profile(
    listener: str | None,
    path: str | Path | None = None,
) -> ListenerProfile | None:
    clean = (listener or "").strip()
    if not clean or clean in {"anonymous", "none"}:
        return None
    return load_profiles(path).get(clean)


def update_profile_from_event(
    event: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> ListenerProfile:
    listener = str(event.get("listener") or "local").strip() or "local"
    profiles = load_profiles(path)
    profile = profiles.get(listener, ListenerProfile(listener=listener))

    event_type = str(event.get("event") or "")
    value = float(event.get("value", 0.0) or 0.0)
    if event_type.endswith("like") or event_type in {"set_like", "transition_like", "replay"}:
        signal = 1.0 if value == 0.0 else max(-1.0, min(1.0, value))
    elif event_type.endswith("dislike") or event_type in {"set_dislike", "transition_dislike", "skip"}:
        signal = -1.0 if value == 0.0 else max(-1.0, min(1.0, value))
    else:
        signal = max(-1.0, min(1.0, value))

    plan = event.get("plan") if isinstance(event.get("plan"), dict) else {}
    style = str(plan.get("style") or event.get("style") or "")
    technique = str(plan.get("technique") or event.get("technique") or "")
    journey = str(event.get("journey") or "")
    learning_rate = 0.045 / math.sqrt(1.0 + profile.sample_count / 12.0)

    if style:
        profile.style_bias[style] = _clamp_bias(profile.style_bias.get(style, 0.0) + learning_rate * signal)
    if technique:
        profile.technique_bias[technique] = _clamp_bias(
            profile.technique_bias.get(technique, 0.0) + learning_rate * signal
        )
    if journey:
        profile.journey_bias[journey] = _clamp_bias(
            profile.journey_bias.get(journey, 0.0) + learning_rate * signal
        )
    energy_delta = event.get("energy_delta")
    if energy_delta is not None:
        try:
            observed = max(-0.5, min(0.5, float(energy_delta)))
            profile.energy_delta_target = (
                0.92 * profile.energy_delta_target + 0.08 * observed * signal
            )
        except (TypeError, ValueError):
            pass

    profile.sample_count += 1
    profile.confidence = min(1.0, profile.sample_count / 80.0)
    profiles[listener] = profile
    save_profiles(profiles, path)
    return profile


def plan_personalization_adjustment(
    plan: TransitionPlan,
    profile: ListenerProfile | None,
) -> float:
    if profile is None or profile.confidence <= 0.0:
        return 0.0
    raw = (
        profile.style_bias.get(plan.style, 0.0)
        + profile.technique_bias.get(plan.technique, 0.0)
    )
    delta = float(plan.compatibility.energy_delta)
    energy_fit = math.exp(-(((delta - profile.energy_delta_target) / 0.24) ** 2))
    raw += 0.05 * (energy_fit - 0.5)
    return float(max(-0.12, min(0.12, raw * profile.confidence)))
