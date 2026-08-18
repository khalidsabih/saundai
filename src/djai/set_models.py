from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import QualityClass, TrackAnalysis, TransitionPlan
from .music.timeline import build_musical_timeline
from .version import PLAN_SCHEMA_VERSION, SET_PLAN_VERSION


@dataclass(slots=True)
class TransitionEdge:
    from_index: int
    to_index: int
    selected_style: str
    objective_score: float
    quality: QualityClass
    plan: TransitionPlan
    alternatives: list[TransitionPlan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_index": self.from_index,
            "to_index": self.to_index,
            "selected_style": self.selected_style,
            "objective_score": self.objective_score,
            "quality": self.quality,
            "plan": self.plan.to_dict(),
            "alternatives": [plan.to_dict() for plan in self.alternatives],
            "warnings": self.warnings,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class SetPlan:
    plan_id: str
    tracks: list[TrackAnalysis]
    order: list[int]
    transitions: list[TransitionEdge]
    baseline_order: list[int]
    baseline_score: float
    optimized_score: float
    estimated_duration_seconds: float
    source: str
    algorithm: str = "multi_start_greedy_directed_local_search"
    manual: bool = False
    confirmed: bool = False
    confirmation_token: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = SET_PLAN_VERSION
    schema_version: str = PLAN_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    automatic_delivery: bool = False
    listener: str = "local"

    @property
    def has_risks(self) -> bool:
        return any(edge.quality != "good" for edge in self.transitions)

    @property
    def requires_confirmation(self) -> bool:
        return self.has_risks and not self.automatic_delivery

    @property
    def ordered_tracks(self) -> list[TrackAnalysis]:
        return [self.tracks[index] for index in self.order]

    def to_dict(self, *, include_analysis: bool = False) -> dict[str, Any]:
        tracks: list[dict[str, Any]] = []
        for index, analysis in enumerate(self.tracks):
            payload: dict[str, Any] = {
                "index": index,
                "path": analysis.path,
                "title": analysis.title,
                "duration_seconds": analysis.duration_seconds,
                "bpm": analysis.bpm,
                "key": asdict(analysis.key),
                "genre": asdict(analysis.genre),
                "energy": analysis.energy,
                "variable_tempo": analysis.variable_tempo,
                "time_signature": analysis.time_signature,
                "waveform_peaks": analysis.waveform_peaks,
                "musical_timeline": build_musical_timeline(analysis).to_summary_dict(),
            }
            if include_analysis:
                payload["analysis"] = analysis.to_dict(include_beats=False)
            tracks.append(payload)
        return {
            "version": self.version,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "source": self.source,
            "created_at_utc": self.created_at_utc,
            "algorithm": self.algorithm,
            "manual": self.manual,
            "confirmed": self.confirmed,
            "requires_confirmation": self.requires_confirmation,
            "has_risks": self.has_risks,
            "automatic_delivery": self.automatic_delivery,
            "listener": self.listener,
            "confirmation_token": self.confirmation_token,
            "baseline_order": self.baseline_order,
            "order": self.order,
            "baseline_score": round(self.baseline_score, 6),
            "optimized_score": round(self.optimized_score, 6),
            "improvement_percent": round(
                100.0 * (self.optimized_score - self.baseline_score)
                / max(abs(self.baseline_score), 1e-9),
                3,
            ),
            "estimated_duration_seconds": round(self.estimated_duration_seconds, 3),
            "tracks": tracks,
            "ordered_tracks": [tracks[index] for index in self.order],
            "transitions": [edge.to_dict() for edge in self.transitions],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
