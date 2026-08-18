from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar

from .models import (
    BarFeature,
    CompatibilityBreakdown,
    GenreEstimate,
    KeyEstimate,
    TechniqueAssessment,
    TransitionMeasurement,
    TempoPoint,
    TrackAnalysis,
    TransitionCandidate,
    TransitionPlan,
)

T = TypeVar("T")


def _filter_fields(cls: type[T], payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def _key_from_dict(value: Any) -> KeyEstimate | None:
    if value is None or isinstance(value, KeyEstimate):
        return value
    if not isinstance(value, dict):
        raise TypeError("Expected a key estimate object or null.")
    return KeyEstimate(**_filter_fields(KeyEstimate, value))


def transition_candidate_from_dict(data: dict[str, Any]) -> TransitionCandidate:
    payload = dict(data)
    payload["local_key"] = _key_from_dict(payload.get("local_key"))
    return TransitionCandidate(**_filter_fields(TransitionCandidate, payload))


def track_analysis_from_dict(data: dict[str, Any]) -> TrackAnalysis:
    payload = dict(data)
    payload.pop("beat_count", None)
    payload.pop("downbeat_count", None)
    payload["key"] = _key_from_dict(payload.get("key"))
    if payload["key"] is None:
        raise ValueError("Track analysis is missing its global key estimate.")
    genre = payload.get("genre", {})
    payload["genre"] = genre if isinstance(genre, GenreEstimate) else GenreEstimate(**_filter_fields(GenreEstimate, dict(genre)))
    payload["tempo_curve"] = [
        item if isinstance(item, TempoPoint) else TempoPoint(**_filter_fields(TempoPoint, dict(item)))
        for item in payload.get("tempo_curve", [])
    ]
    payload["entry_candidates"] = [
        item if isinstance(item, TransitionCandidate) else transition_candidate_from_dict(dict(item))
        for item in payload.get("entry_candidates", [])
    ]
    payload["exit_candidates"] = [
        item if isinstance(item, TransitionCandidate) else transition_candidate_from_dict(dict(item))
        for item in payload.get("exit_candidates", [])
    ]
    payload["bar_features"] = [
        item if isinstance(item, BarFeature) else BarFeature(**_filter_fields(BarFeature, dict(item)))
        for item in payload.get("bar_features", [])
    ]
    return TrackAnalysis(**_filter_fields(TrackAnalysis, payload))


def technique_assessment_from_dict(data: dict[str, Any]) -> TechniqueAssessment:
    return TechniqueAssessment(**_filter_fields(TechniqueAssessment, dict(data)))


def transition_measurement_from_dict(data: dict[str, Any]) -> TransitionMeasurement:
    return TransitionMeasurement(**_filter_fields(TransitionMeasurement, dict(data)))


def transition_plan_from_dict(data: dict[str, Any]) -> TransitionPlan:
    payload = dict(data)
    compatibility = payload.get("compatibility")
    if isinstance(compatibility, CompatibilityBreakdown):
        payload["compatibility"] = compatibility
    elif isinstance(compatibility, dict):
        payload["compatibility"] = CompatibilityBreakdown(
            **_filter_fields(CompatibilityBreakdown, compatibility)
        )
    else:
        raise ValueError("Transition plan is missing compatibility evidence.")
    if isinstance(payload.get("genre_pair"), list):
        payload["genre_pair"] = tuple(payload["genre_pair"])
    payload["technique_assessments"] = [
        item if isinstance(item, TechniqueAssessment) else technique_assessment_from_dict(dict(item))
        for item in payload.get("technique_assessments", [])
    ]
    measurement = payload.get("measurement")
    if isinstance(measurement, dict):
        payload["measurement"] = transition_measurement_from_dict(measurement)
    elif measurement is not None and not isinstance(measurement, TransitionMeasurement):
        raise TypeError("Transition measurement must be an object or null.")
    return TransitionPlan(**_filter_fields(TransitionPlan, payload))
