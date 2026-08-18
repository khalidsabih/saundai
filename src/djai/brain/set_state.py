from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from ..intelligence.embeddings import embedding_similarity
from ..models import TrackAnalysis, TransitionPlan
from .dj_memory import build_dj_memory


@dataclass(frozen=True, slots=True)
class JourneyVector:
    energy: float
    tension: float
    groove: float
    vocal_density: float
    brightness: float
    novelty: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class SetState:
    """Rolling musical memory used by transition and set policies."""

    position: int = 0
    elapsed_seconds: float = 0.0
    energy: float = 0.5
    tension: float = 0.35
    groove: float = 0.5
    vocal_density: float = 0.35
    brightness: float = 0.5
    novelty: float = 0.5
    current_bpm: float = 0.0
    recent_track_paths: list[str] = field(default_factory=list)
    recent_embeddings: list[list[float]] = field(default_factory=list)
    recent_techniques: list[str] = field(default_factory=list)
    recent_intents: list[str] = field(default_factory=list)
    recent_energy: list[float] = field(default_factory=list)
    recent_tension: list[float] = field(default_factory=list)
    dj_memory: dict[str, Any] = field(default_factory=dict)
    tracks_since_breakdown: int = 0
    tracks_since_drop: int = 0
    tracks_since_vocal: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _mean_bar(track: TrackAnalysis, field_name: str, fallback: float) -> float:
    values = [float(getattr(bar, field_name)) for bar in track.bar_features]
    return float(np.mean(values)) if values else float(fallback)


def _track_tension(track: TrackAnalysis) -> float:
    harmonic = _mean_bar(track, "harmonic_tension", 0.35)
    if track.bar_features:
        energies = np.asarray([item.energy for item in track.bar_features], dtype=np.float64)
        slope = float(np.mean(np.diff(energies[-min(8, len(energies)) :]))) if len(energies) > 1 else 0.0
        dynamic = _clamp(0.5 + slope * 4.0)
    else:
        dynamic = track.energy
    return _clamp(0.58 * harmonic + 0.42 * dynamic)


def track_state_vector(track: TrackAnalysis) -> JourneyVector:
    rhythm = _mean_bar(track, "rhythm", track.percussive_ratio)
    vocal = _mean_bar(track, "vocal_probability", track.vocal_probability)
    brightness = _mean_bar(track, "brightness", track.brightness)
    return JourneyVector(
        energy=_clamp(track.energy),
        tension=_track_tension(track),
        groove=_clamp(0.62 * rhythm + 0.38 * track.percussive_ratio),
        vocal_density=_clamp(vocal),
        brightness=_clamp(brightness),
        novelty=0.5,
    )


def _smoothstep(x: float) -> float:
    x = _clamp(x)
    return x * x * (3.0 - 2.0 * x)


def journey_target(mode: str, progress: float) -> JourneyVector:
    """Multi-dimensional target for the musical story, not just energy."""
    p = _clamp(progress)
    if mode == "build":
        return JourneyVector(
            energy=0.34 + 0.56 * _smoothstep(p),
            tension=0.28 + 0.55 * _smoothstep(p),
            groove=0.50 + 0.32 * p,
            vocal_density=0.30 + 0.12 * math.sin(p * math.pi),
            brightness=0.40 + 0.32 * p,
            novelty=0.42 + 0.25 * p,
        )
    if mode == "warmup":
        return JourneyVector(
            energy=0.30 + 0.40 * p,
            tension=0.24 + 0.30 * p,
            groove=0.42 + 0.28 * p,
            vocal_density=0.26 + 0.14 * p,
            brightness=0.38 + 0.18 * p,
            novelty=0.40 + 0.12 * p,
        )
    if mode == "peak":
        wave = 0.5 + 0.5 * math.sin((p * 2.0 - 0.4) * math.pi)
        return JourneyVector(
            energy=0.78 + 0.14 * wave,
            tension=0.62 + 0.22 * wave,
            groove=0.76 + 0.16 * wave,
            vocal_density=0.34 + 0.12 * (1.0 - wave),
            brightness=0.58 + 0.24 * wave,
            novelty=0.60 + 0.18 * wave,
        )
    if mode == "sunset":
        wave = math.sin(p * math.pi)
        return JourneyVector(
            energy=0.48 + 0.24 * wave,
            tension=0.32 + 0.20 * wave,
            groove=0.55 + 0.18 * wave,
            vocal_density=0.38 + 0.14 * wave,
            brightness=0.52 + 0.20 * wave,
            novelty=0.48 + 0.16 * wave,
        )
    if mode == "smooth":
        return JourneyVector(
            energy=0.55 + 0.10 * math.sin(p * math.pi),
            tension=0.42 + 0.08 * math.sin(p * math.pi),
            groove=0.62,
            vocal_density=0.36,
            brightness=0.52,
            novelty=0.42,
        )
    # balanced
    wave = math.sin(p * math.pi)
    return JourneyVector(
        energy=0.48 + 0.28 * wave,
        tension=0.35 + 0.24 * wave,
        groove=0.56 + 0.16 * wave,
        vocal_density=0.34 + 0.10 * math.sin(p * 2.0 * math.pi),
        brightness=0.48 + 0.18 * wave,
        novelty=0.48 + 0.12 * p,
    )


def _embedding_novelty(track: TrackAnalysis, recent: Sequence[list[float]]) -> float:
    if not track.music_embedding or not recent:
        return 0.5
    similarities = [embedding_similarity(track.music_embedding, item) for item in recent if item]
    if not similarities:
        return 0.5
    # High similarity to anything recently played means low novelty.
    return _clamp(1.0 - max(similarities))


def build_set_state(
    tracks: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
    *,
    memory_tracks: int = 6,
) -> SetState:
    if not tracks:
        return SetState()
    recent_tracks = list(tracks[-memory_tracks:])
    vectors = [track_state_vector(track) for track in recent_tracks]
    weights = np.linspace(0.45, 1.0, len(vectors), dtype=np.float64)
    weights /= max(float(np.sum(weights)), 1e-9)
    def weighted(name: str) -> float:
        return float(sum(float(getattr(vector, name)) * float(weight) for vector, weight in zip(vectors, weights, strict=True)))
    transition_values = list((transitions or [])[-8:])
    state = SetState(
        position=len(tracks) - 1,
        elapsed_seconds=sum(track.duration_seconds for track in tracks),
        energy=_clamp(weighted("energy")),
        tension=_clamp(weighted("tension")),
        groove=_clamp(weighted("groove")),
        vocal_density=_clamp(weighted("vocal_density")),
        brightness=_clamp(weighted("brightness")),
        novelty=_clamp(float(np.mean([_embedding_novelty(track, [item.music_embedding for item in recent_tracks[:-1]]) for track in recent_tracks]))),
        current_bpm=tracks[-1].bpm,
        recent_track_paths=[track.path for track in recent_tracks],
        recent_embeddings=[list(track.music_embedding) for track in recent_tracks if track.music_embedding],
        recent_techniques=[item.technique for item in transition_values],
        recent_intents=[item.intent for item in transition_values],
        recent_energy=[track.energy for track in recent_tracks],
        recent_tension=[_track_tension(track) for track in recent_tracks],
        dj_memory=build_dj_memory(tracks, transitions).to_dict(),
    )
    sections = [bar.section_label for track in recent_tracks for bar in track.bar_features]
    state.tracks_since_breakdown = next((i for i, track in enumerate(reversed(recent_tracks)) if any(bar.section_label == "breakdown" for bar in track.bar_features)), len(recent_tracks))
    state.tracks_since_drop = next((i for i, track in enumerate(reversed(recent_tracks)) if any(bar.section_label == "drop" for bar in track.bar_features)), len(recent_tracks))
    state.tracks_since_vocal = next((i for i, track in enumerate(reversed(recent_tracks)) if track.vocal_probability >= 0.48), len(recent_tracks))
    del sections
    return state


def next_track_context_score(
    state: SetState,
    candidate: TrackAnalysis,
    *,
    journey: str,
    progress: float,
) -> tuple[float, dict[str, float]]:
    """Score how useful a track is *now*, independent of pair compatibility."""
    vector = track_state_vector(candidate)
    novelty = _embedding_novelty(candidate, state.recent_embeddings[-5:])
    vector = JourneyVector(
        energy=vector.energy,
        tension=vector.tension,
        groove=vector.groove,
        vocal_density=vector.vocal_density,
        brightness=vector.brightness,
        novelty=novelty,
    )
    target = journey_target(journey, progress)
    fields = ("energy", "tension", "groove", "vocal_density", "brightness", "novelty")
    weights = {
        "energy": 0.28,
        "tension": 0.20,
        "groove": 0.18,
        "vocal_density": 0.12,
        "brightness": 0.08,
        "novelty": 0.14,
    }
    component: dict[str, float] = {}
    total = 0.0
    for name in fields:
        fit = math.exp(-((float(getattr(vector, name)) - float(getattr(target, name))) / 0.28) ** 2)
        component[name] = _clamp(fit)
        total += weights[name] * component[name]

    # Anti-repetition memory: repeated vocals/techniques need an intentional reason.
    if state.tracks_since_vocal == 0 and vector.vocal_density > 0.65:
        total -= 0.08
        component["vocal_repetition_penalty"] = 0.08
    if novelty < 0.12:
        total -= 0.08
        component["similarity_penalty"] = 0.08
    return _clamp(total), component


def set_story_report(order: Sequence[int], analyses: Sequence[TrackAnalysis], *, journey: str) -> list[dict[str, Any]]:
    """Return the actual vs target multidimensional set journey."""
    report: list[dict[str, Any]] = []
    prefix: list[TrackAnalysis] = []
    total = max(1, len(order) - 1)
    for position, index in enumerate(order):
        track = analyses[index]
        prefix.append(track)
        state = build_set_state(prefix)
        vector = track_state_vector(track)
        novelty = _embedding_novelty(track, state.recent_embeddings[:-1][-5:])
        actual = JourneyVector(
            energy=vector.energy,
            tension=vector.tension,
            groove=vector.groove,
            vocal_density=vector.vocal_density,
            brightness=vector.brightness,
            novelty=novelty,
        )
        target = journey_target(journey, position / total)
        report.append(
            {
                "position": position,
                "track_index": index,
                "track": track.path,
                "actual": actual.to_dict(),
                "target": target.to_dict(),
                "set_state": state.to_dict(),
            }
        )
    return report
