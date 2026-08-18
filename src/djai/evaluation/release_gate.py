from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

RELEASE_GATE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    version: str
    ready: bool
    status: str
    hard_blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    scorecard: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hard_blockers"] = list(self.hard_blockers)
        payload["warnings"] = list(self.warnings)
        return payload


def _value(metrics: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default


def evaluate_release_gate(metrics: Mapping[str, Any]) -> ReleaseGateResult:
    """Evaluate non-negotiable SAUND.AI 1.0 launch invariants and quality guidance.

    The function does not invent benchmark evidence: callers must supply real
    run metrics from CI/listening evaluation. Missing quality metrics become
    warnings, while safety/integrity counters default to zero so unit-level
    release checks can be composed from independent evidence sources.
    """

    blockers: list[str] = []
    warnings: list[str] = []
    hard_zero = {
        "validation_bypass_count": "automatic transitions bypassed rendered validation",
        "partial_output_count": "partial renders were published",
        "mutated_plan_accept_count": "mutated/content-mismatched execution plans were accepted",
        "nan_inf_output_count": "NaN/Inf audio reached output",
        "unverified_fallback_count": "fallback transitions bypassed rendered verification",
        "model_governance_blocker_count": "production model-governance blockers remain",
        "privacy_violation_count": "privacy boundary violations were observed",
        "consent_training_violation_count": "training consumed feedback without training consent",
        "cross_account_profile_access_count": "cross-account personalization access was observed",
    }
    for key, label in hard_zero.items():
        value = _value(metrics, key, 0.0)
        if value > 0.0:
            blockers.append(f"{label}: {value:g}")

    clipping = _value(metrics, "uncontrolled_clipping_rate", 0.0)
    if clipping > 0.001:
        blockers.append(f"uncontrolled clipping rate {clipping:.3%} exceeds 0.100%")
    crash_rate = _value(metrics, "crash_rate", 0.0)
    if crash_rate > 0.01:
        blockers.append(f"crash rate {crash_rate:.2%} exceeds 1.00%")

    guidance = {
        "transition_quality": (0.80, "transition quality"),
        "phrase_alignment": (0.84, "phrase alignment"),
        "transition_diversity": (0.72, "transition diversity"),
        "set_storytelling": (0.76, "set storytelling"),
        "installation_success_rate": (0.95, "installation success rate"),
    }
    scorecard: dict[str, float] = {}
    for key, (threshold, label) in guidance.items():
        if key not in metrics:
            warnings.append(f"No {label} benchmark was supplied.")
            continue
        value = max(0.0, min(1.0, _value(metrics, key)))
        scorecard[key] = round(value, 4)
        if value < threshold:
            warnings.append(f"{label} {value:.1%} is below the 1.0 target of {threshold:.0%}.")

    if blockers:
        status = "blocked"
    elif warnings:
        status = "release_candidate"
    else:
        status = "ready"
    return ReleaseGateResult(
        version=RELEASE_GATE_VERSION,
        ready=not blockers and not warnings,
        status=status,
        hard_blockers=tuple(blockers),
        warnings=tuple(warnings),
        scorecard=scorecard,
    )
