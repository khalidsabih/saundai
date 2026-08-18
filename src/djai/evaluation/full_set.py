from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from ..set_models import SetPlan

FULL_SET_BENCHMARK_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class FullSetBenchmark:
    version: str
    overall: float
    transition_quality: float
    worst_transition: float
    story_fit: float
    transition_diversity: float
    quality_gate_pass_rate: float
    fallback_rate: float
    emergency_rate: float
    hard_failure_count: int
    production_safe: bool
    transition_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _story_fit(plan: SetPlan) -> float:
    arc = dict(plan.metadata.get("journey_arc", {}))
    for key in ("curve_fit_score", "curve_fit", "score"):
        if key in arc:
            return _clamp(float(arc[key]))
    story = dict(plan.metadata.get("professional_story", {}))
    for key in ("story_score", "score", "fit"):
        if key in story:
            return _clamp(float(story[key]))
    # Absence of story telemetry is neutral rather than silently excellent.
    return 0.5


def evaluate_full_set(plan: SetPlan) -> FullSetBenchmark:
    transitions = list(plan.transitions)
    if not transitions:
        story = _story_fit(plan)
        return FullSetBenchmark(
            version=FULL_SET_BENCHMARK_VERSION,
            overall=story,
            transition_quality=1.0,
            worst_transition=1.0,
            story_fit=story,
            transition_diversity=1.0,
            quality_gate_pass_rate=1.0,
            fallback_rate=0.0,
            emergency_rate=0.0,
            hard_failure_count=0,
            production_safe=True,
            transition_count=0,
        )

    scores: list[float] = []
    passed = 0
    hard_failures = 0
    fallbacks = 0
    emergencies = 0
    techniques: set[str] = set()
    for edge in transitions:
        transition = edge.plan
        measured = float(transition.measured_score or 0.0)
        analytical = float(transition.score or edge.objective_score or 0.0)
        score = measured if transition.quality_checked and measured > 0.0 else analytical
        scores.append(_clamp(score))
        if transition.measurement_passed or (
            not transition.quality_checked and edge.quality == "good"
        ):
            passed += 1
        measurement = transition.measurement
        if measurement is not None:
            hard_failures += len(measurement.hard_failures)
        fallbacks += int(transition.fallback_level > 0)
        emergencies += int(transition.transition_mode == "emergency_cut")
        techniques.add(str(transition.technique or transition.transition_mode))

    n = len(transitions)
    transition_quality = mean(scores)
    worst = min(scores)
    story = _story_fit(plan)
    diversity = _clamp(len(techniques) / max(1.0, min(float(n), 4.0)))
    gate_pass_rate = passed / n
    fallback_rate = fallbacks / n
    emergency_rate = emergencies / n

    overall = _clamp(
        0.42 * transition_quality
        + 0.16 * worst
        + 0.22 * story
        + 0.08 * diversity
        + 0.12 * gate_pass_rate
        - 0.08 * fallback_rate
        - 0.12 * emergency_rate
    )
    safe = hard_failures == 0 and gate_pass_rate >= 1.0 and emergency_rate == 0.0
    if hard_failures:
        overall = 0.0

    return FullSetBenchmark(
        version=FULL_SET_BENCHMARK_VERSION,
        overall=round(overall, 6),
        transition_quality=round(transition_quality, 6),
        worst_transition=round(worst, 6),
        story_fit=round(story, 6),
        transition_diversity=round(diversity, 6),
        quality_gate_pass_rate=round(gate_pass_rate, 6),
        fallback_rate=round(fallback_rate, 6),
        emergency_rate=round(emergency_rate, 6),
        hard_failure_count=hard_failures,
        production_safe=safe,
        transition_count=n,
    )


def compare_full_sets(champion: SetPlan, challenger: SetPlan) -> dict[str, Any]:
    left = evaluate_full_set(champion)
    right = evaluate_full_set(challenger)
    delta = right.overall - left.overall
    # Never allow aggregate taste to defeat a hard safety result.
    challenger_wins = right.production_safe and (
        not left.production_safe or delta > 1e-6
    )
    return {
        "version": FULL_SET_BENCHMARK_VERSION,
        "champion": left.to_dict(),
        "challenger": right.to_dict(),
        "delta": round(delta, 6),
        "challenger_wins": challenger_wins,
    }
