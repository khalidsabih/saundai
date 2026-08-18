from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

from ..models import TrackAnalysis

ADAPTIVE_THINKING_VERSION = "1.0"
BrittlenessLevel = Literal["easy", "normal", "brittle"]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class AdaptiveThinkingConfig:
    enabled: bool = True
    easy_threshold: float = 0.30
    brittle_threshold: float = 0.62
    preflight_easy_threshold: float = 0.10
    easy_lookahead: int = 2
    easy_headphones_min: int = 1
    easy_headphones_max: int = 2
    brittle_headphones_max: int = 5
    easy_professional_pilot: int = 1
    brittle_force_full_professional: bool = True

    @classmethod
    def from_environment(cls) -> "AdaptiveThinkingConfig":
        easy = _clamp(_float_env("DJAI_THINKING_EASY_THRESHOLD", 0.30))
        brittle = _clamp(_float_env("DJAI_THINKING_BRITTLE_THRESHOLD", 0.62))
        if brittle <= easy:
            brittle = min(1.0, easy + 0.10)
        return cls(
            enabled=_bool_env("DJAI_ADAPTIVE_THINKING", True),
            easy_threshold=easy,
            brittle_threshold=brittle,
            preflight_easy_threshold=_clamp(_float_env("DJAI_THINKING_PREFLIGHT_EASY_THRESHOLD", 0.10)),
            easy_lookahead=max(1, min(2, int(_float_env("DJAI_THINKING_EASY_LOOKAHEAD", 2)))),
            easy_headphones_min=max(1, min(2, int(_float_env("DJAI_THINKING_EASY_HEADPHONES_MIN", 1)))),
            easy_headphones_max=max(1, min(3, int(_float_env("DJAI_THINKING_EASY_HEADPHONES_MAX", 2)))),
            brittle_headphones_max=max(4, min(6, int(_float_env("DJAI_THINKING_BRITTLE_HEADPHONES_MAX", 5)))),
            easy_professional_pilot=max(1, min(2, int(_float_env("DJAI_THINKING_EASY_PRO_PILOT", 1)))),
            brittle_force_full_professional=_bool_env("DJAI_THINKING_BRITTLE_FULL_PRO", True),
        )


@dataclass(frozen=True, slots=True)
class PreflightThinkingAssessment:
    version: str
    score: float
    easy_candidate_context: bool
    candidate_count: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True, slots=True)
class BrittlenessAssessment:
    version: str
    score: float
    level: BrittlenessLevel
    candidate_margin: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        payload["candidate_margin"] = round(float(self.candidate_margin), 6)
        payload["components"] = {key: round(float(value), 6) for key, value in self.components.items()}
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True, slots=True)
class ThinkingBudget:
    version: str
    enabled: bool
    brittleness: BrittlenessAssessment
    preflight: PreflightThinkingAssessment
    lookahead_initial: int
    lookahead_target: int
    lookahead_used: int
    lookahead_escalated: bool
    headphones_serious_pool: int
    headphones_minimum_renders: int
    headphones_maximum_renders: int
    professional_pilot_finalists: int
    professional_adaptive_rehearsal: bool
    hard_safety_unchanged: bool = True
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["brittleness"] = self.brittleness.to_dict()
        payload["preflight"] = self.preflight.to_dict()
        payload["rationale"] = list(self.rationale)
        return payload


def _tempo_relation_risk(bpm_a: float, bpm_b: float) -> float:
    a = max(1.0, float(bpm_a))
    candidates = (float(bpm_b), float(bpm_b) * 2.0, float(bpm_b) * 0.5)
    difference = min(abs(value - a) / a for value in candidates)
    return _clamp(difference / 0.06)


def _camelot_risk(track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    left = str(track_a.key.camelot or "")
    right = str(track_b.key.camelot or "")
    if left == right and left:
        return 0.0
    try:
        left_num, left_mode = int(left[:-1]), left[-1]
        right_num, right_mode = int(right[:-1]), right[-1]
    except (TypeError, ValueError, IndexError):
        return 0.65
    circular = min((left_num - right_num) % 12, (right_num - left_num) % 12)
    if left_mode == right_mode and circular == 1:
        return 0.28
    if left_num == right_num and left_mode != right_mode:
        return 0.34
    if left_mode == right_mode and circular == 2:
        return 0.52
    return 0.78


def assess_context_preflight(
    *,
    track_a: TrackAnalysis,
    candidate_tracks: Sequence[TrackAnalysis],
    dj_state: Any | None = None,
    config: AdaptiveThinkingConfig | None = None,
) -> PreflightThinkingAssessment:
    """Cheap O(N) screen deciding whether it is safe to *start* with shallow search.

    This intentionally avoids graph construction and transition generation. A
    normal/brittle-looking context goes directly to the configured V3 search
    depth, so adaptive thinking never pays for a redundant shallow pass first.
    """

    cfg = config or AdaptiveThinkingConfig.from_environment()
    if not candidate_tracks:
        return PreflightThinkingAssessment(
            version=ADAPTIVE_THINKING_VERSION,
            score=1.0,
            easy_candidate_context=False,
            candidate_count=0,
            reasons=("no candidate tracks",),
        )
    pair_scores: list[float] = []
    for track_b in candidate_tracks:
        tempo = _tempo_relation_risk(track_a.bpm, track_b.bpm)
        harmonic = _camelot_risk(track_a, track_b)
        vocal = _clamp(float(track_a.vocal_probability) * float(track_b.vocal_probability) * 1.5)
        confidence = _confidence_risk(track_a, track_b)
        pair_scores.append(_clamp(0.48 * tempo + 0.27 * harmonic + 0.10 * vocal + 0.15 * confidence))
    best = min(pair_scores)
    commitment_risk = 0.0
    if dj_state is not None and tuple(getattr(dj_state, "pending_commitments", ()) or ()):
        commitment_risk = 0.45
    if track_a.variable_tempo:
        commitment_risk = max(commitment_risk, 0.55)
    score = _clamp(max(best, commitment_risk))
    easy = bool(cfg.enabled and score <= cfg.preflight_easy_threshold)
    reasons = [f"best cheap pair risk {best:.2f}"]
    if commitment_risk > 0.0:
        reasons.append(f"current-state commitment risk {commitment_risk:.2f}")
    reasons.append("start shallow" if easy else "go directly to configured deep search")
    return PreflightThinkingAssessment(
        version=ADAPTIVE_THINKING_VERSION,
        score=round(score, 6),
        easy_candidate_context=easy,
        candidate_count=len(candidate_tracks),
        reasons=tuple(reasons),
    )


def _candidate_margin(candidates: Sequence[Any]) -> float:
    scores = sorted((float(getattr(item, "selection_score", 0.0)) for item in candidates), reverse=True)
    if len(scores) < 2:
        return 1.0
    return max(0.0, scores[0] - scores[1])


def _ambiguity_risk(margin: float) -> float:
    if margin <= 0.025:
        return 1.0
    if margin <= 0.055:
        return 0.78
    if margin <= 0.10:
        return 0.48
    if margin <= 0.16:
        return 0.24
    return 0.05


def _confidence_risk(track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    values = [
        track_a.bpm_confidence,
        track_b.bpm_confidence,
        track_a.beat_grid_confidence,
        track_b.beat_grid_confidence,
        track_a.downbeat_confidence,
        track_b.downbeat_confidence,
        track_a.structure_confidence,
        track_b.structure_confidence,
        track_a.key.confidence,
        track_b.key.confidence,
    ]
    reliable = sum(_clamp(value) for value in values) / max(1, len(values))
    risk = 1.0 - reliable
    if track_a.variable_tempo or track_b.variable_tempo:
        risk = max(risk, 0.60)
    return _clamp(risk)


def assess_move_brittleness(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    winner: Any,
    candidates: Sequence[Any],
    config: AdaptiveThinkingConfig | None = None,
) -> BrittlenessAssessment:
    """Estimate how fragile a provisional DJMove is before expensive rehearsal.

    The score intentionally uses only evidence already available to Complete
    DJMove. It never replaces rendered safety or Professional Brain judgement;
    it decides only how much compute should be spent before the existing hard
    gates commit an exact program.
    """

    cfg = config or AdaptiveThinkingConfig.from_environment()
    plan = winner.plan
    move = winner.move
    compatibility = plan.compatibility
    margin = _candidate_margin(candidates)

    tempo_percent = max(
        abs(float(getattr(compatibility, "tempo_difference_percent", 0.0))),
        abs(float(getattr(compatibility, "local_tempo_difference_percent", 0.0))),
    )
    tempo_risk = _clamp(tempo_percent / 8.0)
    rhythm_risk = 1.0 - _clamp(float(getattr(compatibility, "beat_grid", 0.5)))
    phrase_risk = 1.0 - _clamp(float(getattr(compatibility, "phrase", 0.5)))
    local_harmonic = float(getattr(compatibility, "local_harmonic", 0.0))
    harmonic_quality = local_harmonic if local_harmonic > 0.0 else float(getattr(compatibility, "harmonic", 0.5))
    harmonic_risk = 1.0 - _clamp(harmonic_quality)
    overlap_risk = 1.0 - 0.5 * (
        _clamp(float(getattr(compatibility, "vocal_overlap", 0.5)))
        + _clamp(float(getattr(compatibility, "bass_conflict", 0.5)))
    )
    ambiguity_risk = _ambiguity_risk(margin)

    complexity_risk = 0.0
    if getattr(move, "layer_choreography", None):
        complexity_risk = max(complexity_risk, 0.56)
    if getattr(move, "structural_operations", ()):
        complexity_risk = max(complexity_risk, 0.58)
    if str(getattr(move, "tempo_strategy", "")) not in {"native_blend", "native_cut"}:
        complexity_risk = max(complexity_risk, 0.36)
    if int(getattr(move, "wait_bars", 0)) >= 8:
        complexity_risk = max(complexity_risk, 0.24)

    payoff_risk = 1.0 if bool(getattr(move, "protected_payoff_violation", False)) else 0.0
    confidence_risk = _confidence_risk(track_a, track_b)

    components = {
        "tempo": tempo_risk,
        "rhythm": rhythm_risk,
        "phrase": phrase_risk,
        "harmonic": harmonic_risk,
        "vocal_bass_overlap": overlap_risk,
        "candidate_ambiguity": ambiguity_risk,
        "execution_complexity": complexity_risk,
        "payoff_commitment": payoff_risk,
        "perception_confidence": confidence_risk,
    }
    weights = {
        "tempo": 0.16,
        "rhythm": 0.13,
        "phrase": 0.12,
        "harmonic": 0.10,
        "vocal_bass_overlap": 0.12,
        "candidate_ambiguity": 0.17,
        "execution_complexity": 0.08,
        "payoff_commitment": 0.07,
        "perception_confidence": 0.05,
    }
    score = _clamp(sum(weights[key] * components[key] for key in weights))
    level: BrittlenessLevel
    if not cfg.enabled:
        level = "normal"
    elif score <= cfg.easy_threshold:
        level = "easy"
    elif score >= cfg.brittle_threshold:
        level = "brittle"
    else:
        level = "normal"

    reasons: list[str] = []
    ranked_components = sorted(components.items(), key=lambda item: item[1], reverse=True)
    for name, value in ranked_components[:3]:
        if value >= 0.45:
            reasons.append(f"{name} risk {value:.2f}")
    if level == "easy":
        reasons.append("robust move: shallow planning and a smaller private-listening budget are sufficient")
    elif level == "brittle":
        reasons.append("fragile move: retain deep lookahead and expand real-audio rehearsal")
    else:
        reasons.append("normal move: retain the established RC1/V3 thinking budget")

    return BrittlenessAssessment(
        version=ADAPTIVE_THINKING_VERSION,
        score=round(score, 6),
        level=level,
        candidate_margin=round(margin, 6),
        components=components,
        reasons=tuple(reasons),
    )


def merge_brittleness(
    first: BrittlenessAssessment,
    second: BrittlenessAssessment,
    *,
    config: AdaptiveThinkingConfig | None = None,
) -> BrittlenessAssessment:
    """Keep the more conservative risk when a deeper search changes the winner."""

    cfg = config or AdaptiveThinkingConfig.from_environment()
    score = max(float(first.score), float(second.score))
    components = {
        key: max(float(first.components.get(key, 0.0)), float(second.components.get(key, 0.0)))
        for key in set(first.components) | set(second.components)
    }
    if not cfg.enabled:
        level: BrittlenessLevel = "normal"
    elif score <= cfg.easy_threshold:
        level = "easy"
    elif score >= cfg.brittle_threshold:
        level = "brittle"
    else:
        level = "normal"
    return BrittlenessAssessment(
        version=ADAPTIVE_THINKING_VERSION,
        score=round(score, 6),
        level=level,
        candidate_margin=min(float(first.candidate_margin), float(second.candidate_margin)),
        components=components,
        reasons=tuple(dict.fromkeys([*first.reasons, *second.reasons])),
    )


def build_thinking_budget(
    *,
    assessment: BrittlenessAssessment,
    configured_lookahead: int,
    preflight: PreflightThinkingAssessment | None = None,
    headphones_serious_pool: int,
    headphones_minimum_renders: int,
    headphones_maximum_renders: int,
    professional_pilot_finalists: int = 2,
    config: AdaptiveThinkingConfig | None = None,
    lookahead_initial: int | None = None,
    lookahead_used: int | None = None,
    lookahead_escalated: bool = False,
) -> ThinkingBudget:
    cfg = config or AdaptiveThinkingConfig.from_environment()
    configured_depth = max(1, int(configured_lookahead))
    resolved_preflight = preflight or PreflightThinkingAssessment(
        version=ADAPTIVE_THINKING_VERSION,
        score=1.0,
        easy_candidate_context=False,
        candidate_count=0,
        reasons=("preflight unavailable",),
    )
    initial = configured_depth
    if cfg.enabled and resolved_preflight.easy_candidate_context:
        initial = min(configured_depth, max(1, cfg.easy_lookahead))
    if lookahead_initial is not None:
        initial = max(1, min(configured_depth, int(lookahead_initial)))

    if not cfg.enabled:
        target = configured_depth
        serious_pool = int(headphones_serious_pool)
        min_renders = int(headphones_minimum_renders)
        max_renders = int(headphones_maximum_renders)
        pro_pilot = int(professional_pilot_finalists)
        pro_adaptive = True
        rationale = ("adaptive thinking disabled; preserve configured V3 budgets",)
    elif assessment.level == "easy":
        target = (
            min(configured_depth, max(1, cfg.easy_lookahead))
            if resolved_preflight.easy_candidate_context
            else configured_depth
        )
        serious_pool = min(int(headphones_serious_pool), 8)
        min_renders = min(int(headphones_minimum_renders), cfg.easy_headphones_min)
        min_renders = max(1, min_renders)
        max_renders = min(int(headphones_maximum_renders), max(min_renders, cfg.easy_headphones_max))
        pro_pilot = min(int(professional_pilot_finalists), cfg.easy_professional_pilot)
        pro_pilot = max(1, pro_pilot)
        pro_adaptive = True
        rationale = (
            (
                "easy move: preflight cleared bounded 2-move search and 1-2 private rehearsals"
                if resolved_preflight.easy_candidate_context
                else "easy final move: preflight conservatively kept configured search depth; rehearsal may still use the small budget"
            ),
            "Professional Brain may early-accept after one strong finalist but expands on ambiguity/rejection",
        )
    elif assessment.level == "brittle":
        target = configured_depth
        serious_pool = max(int(headphones_serious_pool), 12)
        min_renders = max(2, int(headphones_minimum_renders))
        max_renders = max(int(headphones_maximum_renders), cfg.brittle_headphones_max)
        max_renders = min(6, max(max_renders, min_renders))
        pro_pilot = max(2, int(professional_pilot_finalists))
        pro_adaptive = not cfg.brittle_force_full_professional
        rationale = (
            "brittle move: retain configured deep search and allow expanded AI Headphones rehearsal",
            "brittle Professional Brain path disables early acceptance when configured, forcing full finalist rehearsal",
        )
    else:
        target = configured_depth
        serious_pool = int(headphones_serious_pool)
        min_renders = int(headphones_minimum_renders)
        max_renders = int(headphones_maximum_renders)
        pro_pilot = int(professional_pilot_finalists)
        pro_adaptive = True
        rationale = ("normal move: preserve established V3 search and rehearsal budgets",)

    used = int(lookahead_used if lookahead_used is not None else target)
    return ThinkingBudget(
        version=ADAPTIVE_THINKING_VERSION,
        enabled=bool(cfg.enabled),
        brittleness=assessment,
        preflight=resolved_preflight,
        lookahead_initial=initial,
        lookahead_target=max(1, min(configured_depth, target)),
        lookahead_used=max(1, min(configured_depth, used)),
        lookahead_escalated=bool(lookahead_escalated),
        headphones_serious_pool=max(1, serious_pool),
        headphones_minimum_renders=max(1, min_renders),
        headphones_maximum_renders=max(max(1, min_renders), max_renders),
        professional_pilot_finalists=max(1, pro_pilot),
        professional_adaptive_rehearsal=bool(pro_adaptive),
        hard_safety_unchanged=True,
        rationale=tuple(rationale),
    )


def adaptive_thinking_status() -> dict[str, Any]:
    cfg = AdaptiveThinkingConfig.from_environment()
    return {
        "version": ADAPTIVE_THINKING_VERSION,
        "enabled": cfg.enabled,
        "purpose": "allocate planning/rehearsal compute by musical brittleness without weakening hard safety",
        "thresholds": {
            "preflight_easy_max": cfg.preflight_easy_threshold,
            "easy_max": cfg.easy_threshold,
            "brittle_min": cfg.brittle_threshold,
        },
        "easy": {
            "lookahead_moves": cfg.easy_lookahead,
            "headphones_renders": {"minimum": cfg.easy_headphones_min, "maximum": cfg.easy_headphones_max},
            "professional_pilot_finalists": cfg.easy_professional_pilot,
        },
        "normal": "preserve configured V3 budgets",
        "brittle": {
            "lookahead": "preserve configured maximum",
            "headphones_maximum_renders": cfg.brittle_headphones_max,
            "force_full_professional_rehearsal": cfg.brittle_force_full_professional,
        },
        "preflight": "cheap global/state screen avoids redundant shallow+deep search for normal decisions",
        "hard_audio_safety_unchanged": True,
        "can_disable_with": "DJAI_ADAPTIVE_THINKING=0",
    }
