from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


CROWD_CONTROL_VERSION = "1.0.0"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return _clamp(value, low, high)


@dataclass(slots=True)
class CrowdState:
    engagement: float = 0.5
    momentum: float = 0.5
    like_rate: float = 0.0
    skip_rate: float = 0.0
    replay_rate: float = 0.0
    sample_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CrowdControlConfig:
    """Hard bounds for live crowd influence.

    Crowd feedback is advisory control input, never a safety override. The
    limits below cap how much it may move the Set Director target, complete
    DJMove ranking, or transition-policy ranking in one Executive cycle.
    """

    enabled: bool = True
    minimum_confidence: float = 0.20
    max_energy_shift: float = 0.08
    max_novelty_shift: float = 0.06
    max_risk_shift: float = 0.08
    max_track_adjustment: float = 0.04
    max_transition_adjustment: float = 0.05

    @classmethod
    def from_environment(cls) -> "CrowdControlConfig":
        return cls(
            enabled=_env_bool("DJAI_LIVE_CROWD_CONTROL", True),
            minimum_confidence=_env_float(
                "DJAI_LIVE_CROWD_MIN_CONFIDENCE", 0.20, low=0.0, high=1.0
            ),
            max_energy_shift=_env_float(
                "DJAI_LIVE_CROWD_MAX_ENERGY_SHIFT", 0.08, low=0.0, high=0.18
            ),
            max_novelty_shift=_env_float(
                "DJAI_LIVE_CROWD_MAX_NOVELTY_SHIFT", 0.06, low=0.0, high=0.15
            ),
            max_risk_shift=_env_float(
                "DJAI_LIVE_CROWD_MAX_RISK_SHIFT", 0.08, low=0.0, high=0.18
            ),
            max_track_adjustment=_env_float(
                "DJAI_LIVE_CROWD_MAX_TRACK_ADJUSTMENT", 0.04, low=0.0, high=0.10
            ),
            max_transition_adjustment=_env_float(
                "DJAI_LIVE_TRANSITION_POLICY_MAX_ADJUSTMENT", 0.05, low=0.0, high=0.10
            ),
        )


@dataclass(frozen=True, slots=True)
class CrowdDirective:
    """Executive-owned, confidence-gated interpretation of CrowdState."""

    active: bool = False
    mode: str = "neutral"
    confidence: float = 0.0
    strength: float = 0.0
    energy_shift: float = 0.0
    novelty_shift: float = 0.0
    risk_shift: float = 0.0
    vocal_simplicity: float = 0.0
    prefer_short_transition: float = 0.0
    prefer_creative_transition: float = 0.0
    max_track_adjustment: float = 0.0
    max_transition_adjustment: float = 0.0
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["version"] = CROWD_CONTROL_VERSION
        payload["authority"] = "executive_bounded_advisory"
        payload["hard_safety_authoritative"] = False
        return payload


def crowd_state_from_feedback(
    path: str | Path = "ratings/feedback-events.jsonl",
    *,
    listener: str | None = None,
    limit: int = 80,
) -> CrowdState:
    """Turn privacy-preserving implicit feedback into a bounded live signal.

    This is intentionally not a camera/microphone crowd detector. It consumes
    explicit or product interaction events such as likes, skips, replays and
    playback completion and can later be combined with venue sensors when the
    product has a clear consent and rights model.
    """
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return CrowdState()
    records: list[dict[str, object]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if listener and str(item.get("listener", "")) != listener:
            continue
        records.append(item)
    records = records[-max(1, limit):]
    if not records:
        return CrowdState()
    likes = sum(str(item.get("event")) in {"set_like", "transition_like"} for item in records)
    skips = sum(str(item.get("event")) in {"skip", "set_dislike", "transition_dislike"} for item in records)
    replays = sum(str(item.get("event")) == "replay" for item in records)
    completions = sum(str(item.get("event")) == "playback_completed" for item in records)
    n = len(records)
    like_rate = likes / n
    skip_rate = skips / n
    replay_rate = replays / n
    engagement = max(0.0, min(1.0, 0.50 + 0.55 * like_rate + 0.50 * replay_rate + 0.20 * completions / n - 0.75 * skip_rate))
    recent = records[-min(12, n):]
    recent_positive = sum(str(item.get("event")) in {"set_like", "transition_like", "replay", "playback_completed"} for item in recent)
    recent_negative = sum(str(item.get("event")) in {"skip", "set_dislike", "transition_dislike"} for item in recent)
    momentum = max(0.0, min(1.0, 0.5 + 0.08 * (recent_positive - recent_negative)))
    return CrowdState(
        engagement=round(engagement, 4),
        momentum=round(momentum, 4),
        like_rate=round(like_rate, 4),
        skip_rate=round(skip_rate, 4),
        replay_rate=round(replay_rate, 4),
        sample_count=n,
        confidence=round(min(1.0, n / 50.0), 4),
    )


def crowd_state_from_mapping(payload: Mapping[str, Any] | None) -> CrowdState:
    values = dict(payload or {})

    def number(name: str, default: float) -> float:
        value = values.get(name, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return CrowdState(
        engagement=_clamp(number("engagement", 0.5)),
        momentum=_clamp(number("momentum", 0.5)),
        like_rate=_clamp(number("like_rate", 0.0)),
        skip_rate=_clamp(number("skip_rate", 0.0)),
        replay_rate=_clamp(number("replay_rate", 0.0)),
        sample_count=max(0, int(number("sample_count", 0.0))),
        confidence=_clamp(number("confidence", 0.0)),
    )


def crowd_directive_from_state(
    state: CrowdState,
    *,
    config: CrowdControlConfig | None = None,
) -> CrowdDirective:
    """Convert observations into a small, auditable Executive directive.

    The mapping is intentionally conservative. Low-confidence observations are
    neutral. Strong negative feedback prefers simpler, shorter, lower-risk
    future moves while still allowing a modest energy lift. Positive momentum
    permits a little more novelty/creative transition preference. All changes
    are capped by configuration and remain downstream of hard safety gates.
    """

    cfg = config or CrowdControlConfig.from_environment()
    confidence = _clamp(state.confidence)
    if not cfg.enabled or confidence < cfg.minimum_confidence:
        reason = (
            "crowd control disabled"
            if not cfg.enabled
            else f"confidence {confidence:.0%} below {cfg.minimum_confidence:.0%} activation floor"
        )
        return CrowdDirective(confidence=confidence, reasons=(reason,))

    negative = max(
        _clamp((0.46 - state.momentum) / 0.46),
        _clamp((0.42 - state.engagement) / 0.42),
        _clamp(state.skip_rate / 0.22),
    )
    positive = max(
        _clamp((state.momentum - 0.58) / 0.42),
        _clamp((state.engagement - 0.60) / 0.40),
        _clamp((state.like_rate + state.replay_rate) / 0.24),
    )
    signal = max(negative, positive)
    strength = _clamp(confidence * signal)
    if strength < 0.04:
        return CrowdDirective(
            confidence=confidence,
            strength=strength,
            reasons=("feedback is confident but close to neutral; no musical steering applied",),
        )

    reasons: list[str] = []
    if state.skip_rate >= 0.12 or state.engagement < 0.38:
        mode = "recover"
        energy_direction = 0.55
        novelty_direction = -1.0
        risk_direction = -1.0
        vocal_simplicity = 1.0
        short_bias = 1.0
        creative_bias = -0.85
        reasons.append("negative/skip pressure asks for a cleaner low-risk recovery")
    elif state.momentum < 0.44:
        mode = "lift"
        energy_direction = 1.0
        novelty_direction = 0.25
        risk_direction = -0.20
        vocal_simplicity = 0.35
        short_bias = 0.45
        creative_bias = -0.10
        reasons.append("momentum is soft, so future selection receives a bounded energy lift")
    elif positive > negative and positive >= 0.20:
        mode = "reward"
        energy_direction = 0.35
        novelty_direction = 1.0
        risk_direction = 0.75
        vocal_simplicity = 0.0
        short_bias = -0.20
        creative_bias = 1.0
        reasons.append("strong positive momentum permits slightly more novelty and creative phrasing")
    else:
        mode = "hold"
        energy_direction = 0.0
        novelty_direction = 0.0
        risk_direction = -0.10
        vocal_simplicity = 0.10
        short_bias = 0.10
        creative_bias = 0.0
        reasons.append("crowd state is stable; preserve the current musical trajectory")

    return CrowdDirective(
        active=True,
        mode=mode,
        confidence=round(confidence, 6),
        strength=round(strength, 6),
        energy_shift=round(cfg.max_energy_shift * energy_direction * strength, 6),
        novelty_shift=round(cfg.max_novelty_shift * novelty_direction * strength, 6),
        risk_shift=round(cfg.max_risk_shift * risk_direction * strength, 6),
        vocal_simplicity=round(_clamp(vocal_simplicity * strength), 6),
        prefer_short_transition=round(max(-1.0, min(1.0, short_bias * strength)), 6),
        prefer_creative_transition=round(max(-1.0, min(1.0, creative_bias * strength)), 6),
        max_track_adjustment=round(cfg.max_track_adjustment, 6),
        max_transition_adjustment=round(cfg.max_transition_adjustment, 6),
        reasons=tuple(reasons),
    )


def crowd_directive_from_mapping(
    payload: Mapping[str, Any] | None,
    *,
    config: CrowdControlConfig | None = None,
) -> CrowdDirective:
    return crowd_directive_from_state(crowd_state_from_mapping(payload), config=config)


def crowd_alignment_adjustment(
    *,
    current_energy: float,
    candidate_energy: float,
    candidate_vocal_density: float,
    directive: CrowdDirective | None,
) -> float:
    """Small residual used by retrieval/move ranking after hard feasibility."""

    if directive is None or not directive.active or directive.strength <= 0.0:
        return 0.0
    energy_sign = 1.0 if directive.energy_shift > 1e-9 else -1.0 if directive.energy_shift < -1e-9 else 0.0
    energy_delta = max(-1.0, min(1.0, (float(candidate_energy) - float(current_energy)) / 0.24))
    energy_alignment = energy_sign * energy_delta
    vocal_alignment = (0.50 - _clamp(candidate_vocal_density)) * 2.0 * directive.vocal_simplicity
    raw = max(-1.0, min(1.0, 0.78 * energy_alignment + 0.22 * vocal_alignment))
    return round(directive.max_track_adjustment * directive.strength * raw, 6)


def crowd_directive_from_payload(payload: Mapping[str, Any] | None) -> CrowdDirective:
    """Rehydrate an Executive-issued directive while re-enforcing local caps."""
    values = dict(payload or {})
    cfg = CrowdControlConfig.from_environment()
    reasons = values.get("reasons", ())
    if not isinstance(reasons, (list, tuple)):
        reasons = (str(reasons),) if reasons else ()

    def signed(name: str, limit: float) -> float:
        try:
            value = float(values.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return max(-limit, min(limit, value))

    return CrowdDirective(
        active=bool(values.get("active", False)) and cfg.enabled,
        mode=str(values.get("mode", "neutral")),
        confidence=_clamp(float(values.get("confidence", 0.0) or 0.0)),
        strength=_clamp(float(values.get("strength", 0.0) or 0.0)),
        energy_shift=signed("energy_shift", cfg.max_energy_shift),
        novelty_shift=signed("novelty_shift", cfg.max_novelty_shift),
        risk_shift=signed("risk_shift", cfg.max_risk_shift),
        vocal_simplicity=_clamp(float(values.get("vocal_simplicity", 0.0) or 0.0)),
        prefer_short_transition=max(-1.0, min(1.0, float(values.get("prefer_short_transition", 0.0) or 0.0))),
        prefer_creative_transition=max(-1.0, min(1.0, float(values.get("prefer_creative_transition", 0.0) or 0.0))),
        max_track_adjustment=_clamp(
            float(values.get("max_track_adjustment", 0.0) or 0.0),
            0.0,
            cfg.max_track_adjustment,
        ),
        max_transition_adjustment=_clamp(
            float(values.get("max_transition_adjustment", 0.0) or 0.0),
            0.0,
            cfg.max_transition_adjustment,
        ),
        reasons=tuple(str(item) for item in reasons),
    )
