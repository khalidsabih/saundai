from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import TrackAnalysis
from ..music.timeline import TimelinePhrase, build_musical_timeline

WORLD_MODEL_VERSION = "6.0"


@dataclass(frozen=True, slots=True)
class PhraseState:
    start_bar: int
    end_bar: int
    start_seconds: float
    end_seconds: float
    section: str
    energy: float
    rhythm: float
    brightness: float
    vocal_probability: float
    bass_activity: float
    boundary_strength: float
    phrase_strength: float
    beat_confidence: float
    embedding: tuple[float, ...] = ()
    role: str = "unknown"
    tension: float = 0.0
    energy_delta: float = 0.0
    transition_readiness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["embedding"] = list(self.embedding)
        return payload


@dataclass(frozen=True, slots=True)
class PhraseLink:
    source_phrase: int
    target_phrase: int
    relation: str
    energy_delta: float
    tension_delta: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackWorldModel:
    version: str
    path: str
    bpm: float
    key: str
    global_embedding: tuple[float, ...]
    phrases: tuple[PhraseState, ...]
    stem_activity: dict[str, float] = field(default_factory=dict)
    neural_evidence: dict[str, Any] = field(default_factory=dict)
    phrase_graph: tuple[PhraseLink, ...] = ()
    structural_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "bpm": self.bpm,
            "key": self.key,
            "global_embedding": list(self.global_embedding),
            "phrases": [item.to_dict() for item in self.phrases],
            "stem_activity": dict(self.stem_activity),
            "neural_evidence": dict(self.neural_evidence),
            "phrase_graph": [item.to_dict() for item in self.phrase_graph],
            "structural_summary": dict(self.structural_summary),
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _relation(a: PhraseState, b: PhraseState) -> str:
    if b.role in {"build", "pre_drop"} and b.energy_delta >= -0.02:
        return "escalates"
    if b.role == "drop" and a.role in {"build", "pre_drop", "breakdown"}:
        return "resolves_to_drop"
    if b.role in {"breakdown", "outro"} and b.energy < a.energy - 0.08:
        return "releases"
    if b.role == "bridge":
        return "bridges"
    return "continues"


def _phrase_state(track: TrackAnalysis, phrase: TimelinePhrase) -> PhraseState:
    embedding = tuple(float(value) for value in track.section_embeddings.get(phrase.section, []))
    return PhraseState(
        start_bar=phrase.start_bar,
        end_bar=phrase.end_bar,
        start_seconds=phrase.start_seconds,
        end_seconds=phrase.end_seconds,
        section=phrase.section,
        energy=phrase.energy,
        rhythm=phrase.rhythm,
        brightness=phrase.brightness,
        vocal_probability=phrase.vocal_probability,
        bass_activity=phrase.bass_activity,
        boundary_strength=phrase.boundary_strength,
        phrase_strength=phrase.phrase_strength,
        beat_confidence=phrase.beat_confidence,
        embedding=embedding,
        role=phrase.role,
        tension=phrase.tension,
        energy_delta=phrase.energy_delta,
        transition_readiness=phrase.transition_readiness,
    )


def build_track_world_model(track: TrackAnalysis, *, phrase_bars: int = 8) -> TrackWorldModel:
    """Project the canonical MusicalTimeline into the legacy world-model shape.

    ``phrase_bars`` remains accepted for API compatibility, but 1.6.0 no
    longer creates a second phrase grid here. Phrase boundaries and roles come
    from the one canonical timeline shared with planning and performance.
    """
    timeline = build_musical_timeline(track)
    phrases = tuple(_phrase_state(track, phrase) for phrase in timeline.phrases)

    links: list[PhraseLink] = []
    for index, (left, right) in enumerate(zip(phrases, phrases[1:])):
        confidence = _clamp(
            0.40 * right.transition_readiness
            + 0.30 * left.beat_confidence
            + 0.30 * right.beat_confidence
        )
        links.append(
            PhraseLink(
                source_phrase=index,
                target_phrase=index + 1,
                relation=_relation(left, right),
                energy_delta=round(right.energy - left.energy, 4),
                tension_delta=round(right.tension - left.tension, 4),
                confidence=round(confidence, 4),
            )
        )

    stem_state = dict(track.stem_analysis or {})
    activity = {
        "vocals": float(stem_state.get("vocals_activity", track.vocal_probability)),
        "bass": float(stem_state.get("bass_activity", track.bass_activity)),
        "drums": float(stem_state.get("drums_activity", track.percussive_ratio)),
        "other": float(stem_state.get("other_activity", max(0.0, 1.0 - track.percussive_ratio))),
    }
    role_counts: dict[str, int] = {}
    for phrase in phrases:
        role_counts[phrase.role] = role_counts.get(phrase.role, 0) + 1
    summary = {
        "phrase_bars": "adaptive",
        "legacy_requested_phrase_bars": max(1, int(phrase_bars)),
        "canonical_timeline": True,
        "timeline_version": timeline.version,
        "timeline_fingerprint": timeline.fingerprint,
        "phrase_count": len(phrases),
        "role_counts": role_counts,
        "role_sequence": [phrase.role for phrase in phrases],
        "transition_ready_phrases": [
            index for index, phrase in enumerate(phrases) if phrase.transition_readiness >= 0.62
        ],
        "energy_peak_phrase": (
            int(max(range(len(phrases)), key=lambda index: phrases[index].energy)) if phrases else None
        ),
    }
    return TrackWorldModel(
        version=WORLD_MODEL_VERSION,
        path=track.path,
        bpm=float(track.bpm),
        key=str(getattr(track.key, "label", track.key)),
        global_embedding=tuple(float(value) for value in track.music_embedding),
        phrases=phrases,
        stem_activity=activity,
        neural_evidence=dict(track.neural_evidence or {}),
        phrase_graph=tuple(links),
        structural_summary=summary,
    )
