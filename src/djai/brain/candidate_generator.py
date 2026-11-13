from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..models import TrackAnalysis, TransitionPlan
from ..planner import rank_transition_candidates


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    analytical_pool: int = 48
    finalists: int = 8
    diversity_weight: float = 0.34
    minimum_analytical_score: float = 0.18

    def normalized(self) -> "CandidateGenerationConfig":
        return CandidateGenerationConfig(
            analytical_pool=max(8, min(200, int(self.analytical_pool))),
            finalists=max(2, min(24, int(self.finalists))),
            diversity_weight=max(0.0, min(0.75, float(self.diversity_weight))),
            minimum_analytical_score=max(0.0, min(1.0, float(self.minimum_analytical_score))),
        )


def _bar(plan: TransitionPlan, key: str, fallback: float) -> float:
    try:
        return float(plan.metadata.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def candidate_distance(left: TransitionPlan, right: TransitionPlan) -> float:
    """Musical-program distance used to stop top-K collapsing to one idea."""
    exit_delta = min(1.0, abs(_bar(left, "exit_bar_index", left.exit_time_a) - _bar(right, "exit_bar_index", right.exit_time_a)) / 16.0)
    entry_delta = min(1.0, abs(_bar(left, "entry_bar_index", left.entry_time_b) - _bar(right, "entry_bar_index", right.entry_time_b)) / 16.0)
    length_delta = min(1.0, abs(left.transition_bars - right.transition_bars) / 16.0)
    technique_delta = 0.0 if left.technique == right.technique else 1.0
    intent_delta = 0.0 if left.intent == right.intent else 1.0
    section_delta = 0.0 if (left.exit_section, left.entry_section) == (right.exit_section, right.entry_section) else 1.0
    style_delta = 0.0 if left.style == right.style else 1.0
    return float(
        np.clip(
            0.16 * exit_delta
            + 0.14 * entry_delta
            + 0.08 * length_delta
            + 0.22 * technique_delta
            + 0.18 * intent_delta
            + 0.14 * section_delta
            + 0.08 * style_delta,
            0.0,
            1.0,
        )
    )


def diversify_transition_candidates(
    plans: Sequence[TransitionPlan],
    *,
    count: int,
    diversity_weight: float = 0.34,
    score_key: str = "selection_utility",
) -> list[TransitionPlan]:
    """Maximal-marginal-relevance selection over executable DJ ideas.

    A validated taste score may guide the shortlist only when explicitly
    requested. The default preserves deterministic production ranking.
    """
    if not plans:
        return []
    count = max(1, min(int(count), len(plans)))
    weight = max(0.0, min(0.75, float(diversity_weight)))
    ranked = sorted(
        plans,
        key=lambda plan: (
            float(plan.metadata.get(score_key, plan.metadata.get("selection_utility", plan.score))),
            plan.score,
            plan.cue_confidence,
            plan.execution_signature,
        ),
        reverse=True,
    )
    chosen = [ranked[0]]
    remaining = ranked[1:]
    while remaining and len(chosen) < count:
        best: TransitionPlan | None = None
        best_value = -1e9
        for plan in remaining:
            analytical = float(plan.metadata.get(score_key, plan.metadata.get("selection_utility", plan.score)))
            minimum_distance = min(candidate_distance(plan, existing) for existing in chosen)
            # Encourage a new technique/intent particularly strongly early in the list.
            novelty_bonus = 0.04 if all(plan.technique != item.technique for item in chosen) else 0.0
            novelty_bonus += 0.025 if all(plan.intent != item.intent for item in chosen) else 0.0
            value = (1.0 - weight) * analytical + weight * minimum_distance + novelty_bonus
            if value > best_value:
                best = plan
                best_value = value
        assert best is not None
        best.metadata = {
            **best.metadata,
            "candidate_diversity": round(min(candidate_distance(best, item) for item in chosen), 6),
            "candidate_mmr_score": round(best_value, 6),
        }
        chosen.append(best)
        remaining.remove(best)
    if chosen:
        chosen[0].metadata = {**chosen[0].metadata, "candidate_diversity": 1.0}
    return chosen


def generate_transition_ideas(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    config: CandidateGenerationConfig | None = None,
    tempo_policy: str = "safe",
    effective_bpm_a: float | None = None,
    render_rate_a: float = 1.0,
    stems_available: bool = False,
    evidence_matrix=None,
) -> list[TransitionPlan]:
    """Generate a broad analytical pool, then return genuinely different ideas."""
    cfg = (config or CandidateGenerationConfig()).normalized()
    per_style = max(4, (cfg.analytical_pool + 2) // 3)
    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative", "emergency"),
        force=True,
        tempo_policy=tempo_policy,
        effective_bpm_a=effective_bpm_a,
        render_rate_a=render_rate_a,
        stems_available=stems_available,
        evidence_matrix=evidence_matrix,
        limit_per_style=per_style,
    )
    plans = [plan for plan in plans if plan.score >= cfg.minimum_analytical_score]
    plans = plans[: cfg.analytical_pool]
    return diversify_transition_candidates(
        plans,
        count=cfg.finalists,
        diversity_weight=cfg.diversity_weight,
    )
