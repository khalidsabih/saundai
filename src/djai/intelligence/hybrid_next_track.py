from __future__ import annotations

"""Confidence-aware next-track ranking used by the SAUND.AI 1.0 set brain.

The hybrid ranker deliberately has a useful deterministic path when no learned
checkpoint is installed.  Learned sequence/taste heads are residual signals and
can only contribute through the model registry's authority gate.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

import numpy as np

from ..brain.director import direct_set_intent, intent_track_score
from ..brain.set_state import SetState, next_track_context_score
from ..compatibility import genre_compatibility, harmonic_compatibility, match_bpm
from ..models import TrackAnalysis, TransitionPlan

HYBRID_NEXT_TRACK_VERSION = "1.0"


def _clamp(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _cosine01(a: list[float], b: list[float]) -> tuple[float, bool]:
    if not a or not b:
        return 0.5, False
    size = min(len(a), len(b))
    if size <= 0:
        return 0.5, False
    left = np.asarray(a[:size], dtype=np.float64)
    right = np.asarray(b[:size], dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.5, False
    cosine = float(np.clip(float(np.dot(left, right)) / denominator, -1.0, 1.0))
    return _clamp(0.5 + 0.5 * cosine), True


def _target_evidence_confidence(track: TrackAnalysis) -> tuple[float, dict[str, float]]:
    """Estimate how much non-ID musical evidence exists for a candidate.

    This is the runtime cold-target bridge.  It does not claim ListenBrainz
    knowledge for an unseen recording; it quantifies how well SAUND can reason
    from audio when memorized sequence evidence is unavailable.
    """
    embedding = 1.0 if track.music_embedding else 0.0
    beat = _clamp(track.beat_grid_confidence)
    structure = _clamp(track.structure_confidence)
    key = _clamp(track.key.confidence)
    genre = _clamp(track.genre.confidence)
    perception = _clamp(track.perception_confidence)
    components = {
        "embedding": embedding,
        "beat": beat,
        "structure": structure,
        "key": key,
        "genre": genre,
        "perception": perception,
    }
    confidence = (
        0.16
        + 0.20 * embedding
        + 0.17 * beat
        + 0.17 * structure
        + 0.12 * key
        + 0.08 * genre
        + 0.10 * perception
    )
    return _clamp(confidence), components


def _plan_safety_factor(plan: TransitionPlan | None) -> tuple[float, dict[str, Any]]:
    if plan is None:
        return 1.0, {"known": False, "hard_failures": []}
    measurement = plan.measurement
    hard_failures = list(measurement.hard_failures if measurement else [])
    if hard_failures:
        return 0.0, {"known": True, "hard_failures": hard_failures}
    factor = 1.0
    if plan.transition_mode == "emergency_cut":
        factor *= 0.82
    if plan.requires_confirmation:
        factor *= 0.84
    if plan.quality == "poor":
        factor *= 0.68
    elif plan.quality == "risky":
        factor *= 0.88
    if measurement is not None:
        factor *= 0.78 + 0.22 * _clamp(measurement.overall)
    return _clamp(factor), {"known": True, "hard_failures": []}


@dataclass(frozen=True, slots=True)
class HybridNextTrackPrediction:
    score: float
    confidence: float
    deterministic_score: float
    learned_score: float
    learned_authority: float
    target_evidence_confidence: float
    components: dict[str, float] = field(default_factory=dict)
    learned: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    version: str = HYBRID_NEXT_TRACK_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = {key: round(float(value), 6) for key, value in self.components.items()}
        payload["score"] = round(self.score, 6)
        payload["confidence"] = round(self.confidence, 6)
        payload["deterministic_score"] = round(self.deterministic_score, 6)
        payload["learned_score"] = round(self.learned_score, 6)
        payload["learned_authority"] = round(self.learned_authority, 6)
        payload["target_evidence_confidence"] = round(self.target_evidence_confidence, 6)
        return payload


def score_hybrid_next_track(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    edge_score: float = 0.5,
    set_state: SetState | None = None,
    journey: str = "balanced",
    progress: float = 0.5,
    plan: TransitionPlan | None = None,
) -> HybridNextTrackPrediction:
    """Score A -> B using sequence intelligence plus cold-target audio reasoning.

    `edge_score` is the already-governed technical/transition objective supplied
    by the set graph.  Hard rendered failures remain an absolute veto here.
    """
    _, perceived_delta, _ = match_bpm(max(track_a.bpm, 1e-6), max(track_b.bpm, 1e-6))
    tempo = _clamp(math.exp(-((float(perceived_delta) / 0.12) ** 2)))
    harmonic = _clamp(harmonic_compatibility(track_a.key, track_b.key))
    genre = _clamp(genre_compatibility(track_a, track_b))
    embedding, embedding_present = _cosine01(track_a.music_embedding, track_b.music_embedding)

    if set_state is not None:
        context, _ = next_track_context_score(set_state, track_b, journey=journey, progress=progress)
        intent = direct_set_intent(journey, progress, set_state)
        director, _ = intent_track_score(intent, set_state, track_b)
    else:
        context = 0.5
        director = 0.5

    energy_delta = float(track_b.energy - track_a.energy)
    energy_continuity = _clamp(math.exp(-((energy_delta / 0.28) ** 2)))
    target_confidence, target_evidence = _target_evidence_confidence(track_b)
    safety_factor, safety = _plan_safety_factor(plan)

    # The deterministic path is intentionally strong enough to ship without a
    # learned checkpoint.  Sequence/taste models improve ranking but are never
    # required for technical correctness.
    heuristic_deterministic = _clamp(
        0.27 * _clamp(edge_score)
        + 0.16 * context
        + 0.10 * director
        + 0.11 * tempo
        + 0.10 * harmonic
        + 0.07 * genre
        + 0.08 * embedding
        + 0.06 * energy_continuity
        + 0.05 * target_confidence
    )

    learned_payload: dict[str, Any] = {}

    # SAUND.AI 1.1 bootstrap sequence knowledge is a bounded residual on the
    # existing cold-target-aware musical score.  It participates only when a
    # local track/artist identity can be resolved against the exported
    # ListenBrainz summaries; otherwise the hybrid brain remains purely
    # musical.  Production research mode is forced to shadow by the adapter.
    bootstrap_sequence_weight = 0.0
    bootstrap_sequence_score = 0.5
    try:
        from .bootstrap_runtime import sequence_pair_prediction

        bootstrap_sequence = sequence_pair_prediction(track_a, track_b)
        learned_payload["bootstrap_next_track"] = bootstrap_sequence.to_dict()
        if bootstrap_sequence.available:
            bootstrap_sequence_weight = float(bootstrap_sequence.effective_weight)
            bootstrap_sequence_score = float(bootstrap_sequence.score)
    except Exception as exc:  # pragma: no cover - defensive research-model isolation
        learned_payload["bootstrap_next_track"] = {"available": False, "reason": str(exc)}

    deterministic = _clamp(
        (1.0 - bootstrap_sequence_weight) * heuristic_deterministic
        + bootstrap_sequence_weight * bootstrap_sequence_score
    )

    learned_values: list[tuple[float, float]] = []
    try:
        from .neural_policy import next_track_prediction

        prediction = next_track_prediction(track_a, track_b, set_state=set_state)
        learned_payload["next_track_policy"] = prediction.to_dict()
        if prediction.available and prediction.authority > 0.0:
            learned_values.append((float(prediction.score), float(prediction.authority)))
    except Exception as exc:  # pragma: no cover - defensive model isolation
        learned_payload["next_track_policy"] = {"available": False, "reason": str(exc)}

    # RC11.8: the existing professional transition preference ranker is global
    # taste evidence.  It only participates when the transition has a plan and
    # its own cross-validation-derived authority is non-zero.
    if plan is not None:
        try:
            from .transition_ranker import predict_plan

            taste = predict_plan(plan)
            learned_payload["global_taste_ranker"] = {
                "available": taste.authority > 0.0,
                "score": round(float(taste.learned_score), 6),
                "acceptability": round(float(taste.acceptability_score), 6),
                "authority": round(float(taste.authority), 6),
                "confidence": round(float(taste.confidence), 6),
            }
            if taste.authority > 0.0:
                learned_values.append((float(taste.learned_score), float(taste.authority)))
        except Exception as exc:  # pragma: no cover - defensive model isolation
            learned_payload["global_taste_ranker"] = {"available": False, "reason": str(exc)}

    authority_sum = sum(weight for _, weight in learned_values)
    if authority_sum > 0.0:
        learned_score = sum(score * weight for score, weight in learned_values) / authority_sum
        # Learned authority is a residual ceiling, not a replacement for the
        # deterministic score. Cold-target audio evidence reduces how much an
        # ID-memorizing head can dominate a poorly observed target.
        learned_authority = min(0.55, authority_sum) * (0.60 + 0.40 * target_confidence)
    else:
        learned_score = 0.5
        learned_authority = 0.0

    blended = (1.0 - learned_authority) * deterministic + learned_authority * learned_score
    blended *= safety_factor
    confidence = _clamp(
        0.50 * target_confidence
        + 0.22 * _clamp(track_a.perception_confidence)
        + 0.18 * _clamp(edge_score)
        + 0.10 * min(1.0, authority_sum / 0.55 if authority_sum else 0.0)
    )
    if safety_factor <= 0.0:
        confidence = 1.0

    components = {
        "edge": _clamp(edge_score),
        "context": _clamp(context),
        "director": _clamp(director),
        "tempo": tempo,
        "harmonic": harmonic,
        "genre": genre,
        "embedding": embedding,
        "embedding_present": 1.0 if embedding_present else 0.0,
        "energy_continuity": energy_continuity,
        "safety_factor": safety_factor,
        "heuristic_deterministic": heuristic_deterministic,
        "bootstrap_sequence_score": _clamp(bootstrap_sequence_score),
        "bootstrap_sequence_weight": _clamp(bootstrap_sequence_weight),
        **{f"target_{key}": value for key, value in target_evidence.items()},
    }
    return HybridNextTrackPrediction(
        score=_clamp(blended),
        confidence=confidence,
        deterministic_score=deterministic,
        learned_score=_clamp(learned_score),
        learned_authority=_clamp(learned_authority),
        target_evidence_confidence=target_confidence,
        components=components,
        learned=learned_payload,
        safety=safety,
    )


def rank_hybrid_candidates(
    track_a: TrackAnalysis,
    candidates: Mapping[int, tuple[TrackAnalysis, float, TransitionPlan | None]],
    *,
    set_state: SetState | None = None,
    journey: str = "balanced",
    progress: float = 0.5,
) -> list[tuple[int, HybridNextTrackPrediction]]:
    ranked = [
        (
            index,
            score_hybrid_next_track(
                track_a,
                track,
                edge_score=edge_score,
                set_state=set_state,
                journey=journey,
                progress=progress,
                plan=plan,
            ),
        )
        for index, (track, edge_score, plan) in candidates.items()
    ]
    return sorted(ranked, key=lambda item: (item[1].score, item[1].confidence, -item[0]), reverse=True)
