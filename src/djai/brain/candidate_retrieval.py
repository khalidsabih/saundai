from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from ..compatibility import genre_compatibility, harmonic_compatibility, tempo_compatibility
from ..intelligence.embeddings import embedding_similarity
from ..intelligence.neural_policy import next_track_prediction
from ..models import TrackAnalysis
from .crowd_state import CrowdDirective, crowd_alignment_adjustment
from .set_state import SetState, next_track_context_score, track_state_vector

RETRIEVAL_VERSION = "1.8.0"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class CandidateRetrievalConfig:
    width: int = 6
    expansion_width: int = 4
    minimum_diverse: int = 2
    learned_weight: float = 0.68
    experimental_learned_filter: bool = False

    @classmethod
    def from_environment(cls) -> "CandidateRetrievalConfig":
        return cls(
            width=max(3, min(32, _int_env("DJAI_LAZY_CANDIDATE_WIDTH", 6))),
            expansion_width=max(2, min(32, _int_env("DJAI_LAZY_EXPANSION_WIDTH", 4))),
            minimum_diverse=max(1, min(6, _int_env("DJAI_LAZY_MIN_DIVERSE", 2))),
            learned_weight=max(0.0, min(0.85, _float_env("DJAI_NEURAL_RETRIEVAL_WEIGHT", 0.68))),
            experimental_learned_filter=_bool_env("DJAI_EXPERIMENTAL_NEURAL_RETRIEVAL", False),
        )


@dataclass(frozen=True, slots=True)
class RetrievedTrack:
    index: int
    score: float
    components: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "score": round(self.score, 6),
            "components": {key: round(float(value), 6) for key, value in self.components.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateRetrievalResult:
    current_index: int
    ranked: tuple[RetrievedTrack, ...]
    selected: tuple[int, ...]
    total_available: int
    width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": RETRIEVAL_VERSION,
            "current_index": self.current_index,
            "total_available": self.total_available,
            "width": self.width,
            "selected": list(self.selected),
            "ranked": [item.to_dict() for item in self.ranked[: self.width]],
        }


def _energy_bridge_score(current: TrackAnalysis, candidate: TrackAnalysis) -> float:
    delta = abs(float(candidate.energy) - float(current.energy))
    # Deliberately reward both continuity and useful medium-size energy moves.
    continuity = math.exp(-((delta / 0.22) ** 2))
    bridge = math.exp(-(((delta - 0.22) / 0.14) ** 2))
    return _clamp(max(continuity * 0.88, bridge))


def _embedding_score(current: TrackAnalysis, candidate: TrackAnalysis) -> float:
    if not current.music_embedding or not candidate.music_embedding:
        return 0.5
    return _clamp(embedding_similarity(current.music_embedding, candidate.music_embedding))


def _cheap_score(
    current: TrackAnalysis,
    candidate: TrackAnalysis,
    *,
    set_state: SetState,
    journey: str,
    progress: float,
    crowd_directive: CrowdDirective | None = None,
) -> tuple[float, dict[str, float]]:
    tempo, _ = tempo_compatibility(current.bpm, candidate.bpm)
    harmonic = harmonic_compatibility(current.key, candidate.key)
    energy = _energy_bridge_score(current, candidate)
    embedding = _embedding_score(current, candidate)
    genre = genre_compatibility(current, candidate)
    context, _ = next_track_context_score(set_state, candidate, journey=journey, progress=progress)

    # Cheap retrieval is intentionally broad. Exact cue compatibility belongs
    # to expensive complete-move generation after shortlisting.
    score = _clamp(
        0.22 * tempo
        + 0.16 * harmonic
        + 0.14 * energy
        + 0.14 * embedding
        + 0.10 * genre
        + 0.24 * context
    )
    vector = track_state_vector(candidate)
    crowd_adjustment = crowd_alignment_adjustment(
        current_energy=set_state.energy,
        candidate_energy=vector.energy,
        candidate_vocal_density=vector.vocal_density,
        directive=crowd_directive,
    )
    score = _clamp(score + crowd_adjustment)
    return score, {
        "tempo": tempo,
        "harmonic": harmonic,
        "energy_bridge": energy,
        "embedding": embedding,
        "genre": genre,
        "set_context": context,
        "crowd_adjustment": crowd_adjustment,
    }


def retrieve_candidate_tracks(
    *,
    current_index: int,
    candidate_indices: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    set_state: SetState,
    journey: str,
    progress: float,
    width: int | None = None,
    config: CandidateRetrievalConfig | None = None,
    crowd_directive: CrowdDirective | None = None,
) -> CandidateRetrievalResult:
    """Cheaply shortlist plausible successors without constructing DJ edges.

    The shortlist keeps strong continuity candidates but reserves a small
    amount of space for musically useful bridges/novel choices so retrieval
    does not collapse into nearest-neighbour similarity only.
    """

    cfg = config or CandidateRetrievalConfig.from_environment()
    requested_width = max(1, int(width if width is not None else cfg.width))
    available = [index for index in candidate_indices if index != current_index]
    current = analyses[current_index]
    ranked: list[RetrievedTrack] = []
    for index in available:
        deterministic_score, components = _cheap_score(
            current,
            analyses[index],
            set_state=set_state,
            journey=journey,
            progress=progress,
            crowd_directive=crowd_directive,
        )
        learned = (
            next_track_prediction(current, analyses[index], set_state=set_state)
            if cfg.experimental_learned_filter
            else None
        )
        if learned is not None and learned.available:
            learned_weight = cfg.learned_weight
            score = _clamp(
                (1.0 - learned_weight) * deterministic_score
                + learned_weight * float(learned.score)
            )
            components = {
                **components,
                "deterministic_retrieval": deterministic_score,
                "learned_retrieval": float(learned.score),
                "learned_confidence": float(learned.confidence),
                "learned_weight": learned_weight,
            }
            reason = "learned next-track filter + deterministic musical guard"
        else:
            score = deterministic_score
            components = {
                **components,
                "deterministic_retrieval": deterministic_score,
                "learned_retrieval": 0.5,
                "learned_confidence": 0.0,
                "learned_weight": 0.0,
            }
            reason = (
                "deterministic retrieval; learned next-track filter unavailable"
                if cfg.experimental_learned_filter
                else "deterministic retrieval; learned next-track filter disabled"
            )
        ranked.append(
            RetrievedTrack(
                index=index,
                score=score,
                components=components,
                reason=reason,
            )
        )
    ranked.sort(key=lambda item: (item.score, -item.index), reverse=True)

    if len(ranked) <= requested_width:
        selected = [item.index for item in ranked]
    else:
        selected = [item.index for item in ranked[:requested_width]]
        # Reserve a little diversity from outside the nearest neighbourhood:
        # useful bridge choices have moderate embedding similarity and a
        # meaningful energy movement rather than simply being random tracks.
        diverse_pool = sorted(
            ranked[requested_width:],
            key=lambda item: (
                item.components["energy_bridge"],
                1.0 - abs(item.components["embedding"] - 0.58),
                item.components["set_context"],
                item.score,
            ),
            reverse=True,
        )
        diverse_count = min(cfg.minimum_diverse, requested_width // 3, len(diverse_pool))
        if diverse_count:
            selected[-diverse_count:] = [item.index for item in diverse_pool[:diverse_count]]
            # Keep deterministic order by the original retrieval score.
            selected_set = set(selected)
            selected = [item.index for item in ranked if item.index in selected_set][:requested_width]

    return CandidateRetrievalResult(
        current_index=current_index,
        ranked=tuple(ranked),
        selected=tuple(selected),
        total_available=len(available),
        width=min(requested_width, len(available)),
    )
