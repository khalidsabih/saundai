from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from ..models import BarFeature, SectionLabel, TrackAnalysis


@dataclass(slots=True)
class SectionSpan:
    label: SectionLabel
    start_bar: int
    end_bar: int
    start_seconds: float
    end_seconds: float
    energy: float
    rhythm: float
    vocal_probability: float
    bass_activity: float
    harmonic_tension: float
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




@dataclass(slots=True)
class MusicalSpan:
    scale_bars: int
    start_bar: int
    end_bar: int
    start_seconds: float
    end_seconds: float
    section_label: str
    feature_vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class TrackMap:
    """Compact bar-synchronous representation consumed by the DJ brain.

    `TrackAnalysis` remains the authoritative signal-analysis result. TrackMap
    is the policy-facing view: it exposes trajectories and structural spans in
    a stable form so ranking/search code never needs to know how the signal
    features were computed.
    """

    path: str
    duration_seconds: float
    bpm: float
    key: str
    genre: str
    global_energy: float
    global_vocal_probability: float
    global_bass_activity: float
    brightness: float
    percussive_ratio: float
    perception_confidence: float
    bars: list[BarFeature] = field(default_factory=list)
    sections: list[SectionSpan] = field(default_factory=list)
    phrase_spans: list[MusicalSpan] = field(default_factory=list)
    long_spans: list[MusicalSpan] = field(default_factory=list)
    global_embedding: list[float] = field(default_factory=list)
    embedding_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def bar_count(self) -> int:
        return len(self.bars)

    def trajectory(self, field_name: str) -> list[float]:
        if field_name not in {
            "energy",
            "rhythm",
            "brightness",
            "percussive_activity",
            "vocal_probability",
            "bass_activity",
            "harmonic_tension",
        }:
            raise ValueError(f"Unsupported TrackMap trajectory: {field_name}")
        return [float(getattr(bar, field_name)) for bar in self.bars]


def _section_spans(analysis: TrackAnalysis) -> list[SectionSpan]:
    bars = analysis.bar_features
    if not bars:
        return []
    spans: list[SectionSpan] = []
    start = 0
    for index in range(1, len(bars) + 1):
        boundary = index == len(bars) or bars[index].section_label != bars[start].section_label
        if not boundary:
            continue
        group = bars[start:index]
        label = bars[start].section_label
        spans.append(
            SectionSpan(
                label=label,
                start_bar=group[0].bar_index,
                end_bar=group[-1].bar_index + 1,
                start_seconds=group[0].start_seconds,
                end_seconds=group[-1].end_seconds,
                energy=round(mean(item.energy for item in group), 4),
                rhythm=round(mean(item.rhythm for item in group), 4),
                vocal_probability=round(mean(item.vocal_probability for item in group), 4),
                bass_activity=round(mean(item.bass_activity for item in group), 4),
                harmonic_tension=round(mean(item.harmonic_tension for item in group), 4),
                embedding=list(analysis.section_embeddings.get(label, [])),
            )
        )
        start = index
    return spans


def _span_vector(group: list[BarFeature]) -> list[float]:
    if not group:
        return []
    fields = (
        "energy", "rhythm", "brightness", "percussive_activity",
        "vocal_probability", "bass_activity", "harmonic_tension",
        "boundary_strength", "phrase_strength", "beat_grid_confidence",
    )
    values: list[float] = []
    for name in fields:
        series = [float(getattr(item, name)) for item in group]
        values.extend([mean(series), min(series), max(series)])
    chroma = [item.chroma for item in group if item.chroma]
    if chroma:
        size = min(len(item) for item in chroma)
        for index in range(size):
            values.append(mean(float(item[index]) for item in chroma))
    return [round(float(value), 6) for value in values]


def _multi_scale_spans(bars: list[BarFeature], scale: int, hop: int) -> list[MusicalSpan]:
    if not bars:
        return []
    spans: list[MusicalSpan] = []
    for start in range(0, len(bars), max(1, hop)):
        group = bars[start : min(len(bars), start + scale)]
        if len(group) < max(2, scale // 2):
            continue
        labels: dict[str, int] = {}
        for item in group:
            labels[item.section_label] = labels.get(item.section_label, 0) + 1
        label = max(labels, key=labels.get) if labels else "unknown"
        spans.append(MusicalSpan(
            scale_bars=scale,
            start_bar=group[0].bar_index, end_bar=group[-1].bar_index + 1,
            start_seconds=group[0].start_seconds, end_seconds=group[-1].end_seconds,
            section_label=label, feature_vector=_span_vector(group),
        ))
    return spans


def build_track_map(analysis: TrackAnalysis) -> TrackMap:
    return TrackMap(
        path=analysis.path,
        duration_seconds=analysis.duration_seconds,
        bpm=analysis.bpm,
        key=analysis.key.camelot,
        genre=analysis.genre.label,
        global_energy=analysis.energy,
        global_vocal_probability=analysis.vocal_probability,
        global_bass_activity=analysis.bass_activity,
        brightness=analysis.brightness,
        percussive_ratio=analysis.percussive_ratio,
        perception_confidence=analysis.perception_confidence,
        bars=list(analysis.bar_features),
        sections=_section_spans(analysis),
        phrase_spans=_multi_scale_spans(list(analysis.bar_features), 8, 4),
        long_spans=_multi_scale_spans(list(analysis.bar_features), 16, 8),
        global_embedding=list(analysis.music_embedding),
        embedding_backend=analysis.embedding_backend,
    )
