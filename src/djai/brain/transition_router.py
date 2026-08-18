from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Literal, Sequence

import numpy as np

from ..compatibility import genre_compatibility, harmonic_compatibility, match_bpm
from ..models import TrackAnalysis, TransitionPlan

TransitionFamily = Literal[
    "direct_blend",
    "layer_handoff",
    "breakdown_drop",
    "rhythmic_switch",
    "hard_reset",
    "bridge_preferred",
]


@dataclass(frozen=True, slots=True)
class TempoRelationship:
    bpm_a: float
    bpm_b: float
    direct_difference: float
    perceived_difference: float
    relation: str
    direct_stretch_safe: bool
    matched_bpm_b: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransitionRoute:
    family: TransitionFamily
    confidence: float
    direct_transition_confidence: float
    bridge_preference: float
    tempo: TempoRelationship
    genre_similarity: float
    groove_similarity: float
    harmonic_similarity: float
    layer_capability: float
    reasons: tuple[str, ...]
    fallback_families: tuple[TransitionFamily, ...] = ()
    neural_policy: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        payload["fallback_families"] = list(self.fallback_families)
        return payload


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def tempo_relationship(track_a: TrackAnalysis, track_b: TrackAnalysis) -> TempoRelationship:
    a = max(float(track_a.bpm), 1e-6)
    b = max(float(track_b.bpm), 1e-6)
    direct = abs(b - a) / a
    matched, perceived, _ = match_bpm(a, b)
    ratio = matched / b if b > 0 else 1.0
    if abs(ratio - 0.5) < 0.05:
        relation = "double_time_b"
    elif abs(ratio - 2.0) < 0.10:
        relation = "half_time_b"
    else:
        relation = "direct"
    return TempoRelationship(
        bpm_a=round(a, 4),
        bpm_b=round(b, 4),
        direct_difference=round(direct, 6),
        perceived_difference=round(perceived, 6),
        relation=relation,
        direct_stretch_safe=direct <= 0.065,
        matched_bpm_b=round(float(matched), 4),
    )


def groove_similarity(track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    """Conservative style/groove proximity independent of a genre label."""
    scales = np.asarray([0.34, 0.34, 0.38, 0.25, 2.6], dtype=np.float64)
    left = np.asarray(
        [
            track_a.percussive_ratio,
            track_a.bass_activity,
            track_a.energy,
            track_a.brightness,
            track_a.onset_density,
        ],
        dtype=np.float64,
    )
    right = np.asarray(
        [
            track_b.percussive_ratio,
            track_b.bass_activity,
            track_b.energy,
            track_b.brightness,
            track_b.onset_density,
        ],
        dtype=np.float64,
    )
    distance = float(np.sqrt(np.mean(np.square((left - right) / np.maximum(scales, 1e-6)))))
    return _clamp(math.exp(-0.72 * distance * distance))


def _layer_capability(track: TrackAnalysis) -> float:
    info = dict(track.performance_layer_analysis or {})
    layer_value = 0.0
    if float(info.get("available", 0.0)) > 0.0:
        confidence = float(info.get("quality_confidence", info.get("alignment_confidence", 0.75)))
        complete = float(info.get("complete", 0.0))
        layer_value = _clamp((0.55 + 0.45 * complete) * confidence)
    stem = dict(track.stem_analysis or {})
    stem_value = 0.0
    if float(stem.get("available", 0.0)) > 0.0:
        stem_value = _clamp(float(stem.get("quality_confidence", 0.72)))
    return max(layer_value, stem_value)


def route_transition(track_a: TrackAnalysis, track_b: TrackAnalysis) -> TransitionRoute:
    tempo = tempo_relationship(track_a, track_b)
    genre = genre_compatibility(track_a, track_b)
    groove = groove_similarity(track_a, track_b)
    harmonic = harmonic_compatibility(track_a.key, track_b.key)
    layer = max(_layer_capability(track_a), _layer_capability(track_b))
    reasons: list[str] = []

    easy_tempo = tempo.direct_difference <= 0.055
    moderate_tempo = tempo.direct_difference <= 0.105
    pulse_equivalent = tempo.relation != "direct" and tempo.perceived_difference <= 0.045
    musical_neighborhood = 0.48 * groove + 0.30 * genre + 0.22 * harmonic

    if easy_tempo and musical_neighborhood >= 0.63:
        family: TransitionFamily = "direct_blend"
        confidence = _clamp(0.48 * musical_neighborhood + 0.30 * (1.0 - tempo.direct_difference / 0.055) + 0.22 * track_a.beat_grid_confidence * track_b.beat_grid_confidence)
        direct = _clamp(0.72 + 0.28 * confidence)
        bridge = _clamp(0.10 * (1.0 - confidence))
        reasons.append("Tempo, groove and style are close enough for a normal phrase-aligned blend.")
    elif easy_tempo and layer >= 0.55:
        family = "layer_handoff"
        confidence = _clamp(0.38 * layer + 0.30 * groove + 0.18 * genre + 0.14 * harmonic)
        direct = _clamp(0.55 + 0.30 * confidence)
        bridge = _clamp(0.18 * (1.0 - confidence))
        reasons.append("Clean performance layers can reduce vocal/bass competition across a less-natural style handoff.")
    elif pulse_equivalent:
        family = "rhythmic_switch"
        confidence = _clamp(0.50 * (1.0 - tempo.perceived_difference / 0.045) + 0.30 * groove + 0.20 * genre)
        direct = _clamp(0.30 + 0.42 * confidence)
        bridge = _clamp(0.30 + 0.28 * (1.0 - confidence))
        reasons.append("The tracks have a half/double-time pulse relationship; a switch/reset is safer than a conventional long blend.")
    elif moderate_tempo and (groove >= 0.50 or genre >= 0.58):
        family = "breakdown_drop"
        confidence = _clamp(0.38 * groove + 0.28 * genre + 0.18 * harmonic + 0.16 * (1.0 - tempo.direct_difference / 0.105))
        direct = _clamp(0.34 + 0.36 * confidence)
        bridge = _clamp(0.28 + 0.30 * (1.0 - confidence))
        reasons.append("The BPM gap is too large for an invisible long blend; prefer a structural reset or breakdown-to-drop handoff.")
    elif layer >= 0.62 and tempo.direct_difference <= 0.15:
        family = "layer_handoff"
        confidence = _clamp(0.44 * layer + 0.22 * groove + 0.18 * genre + 0.16 * harmonic)
        direct = _clamp(0.28 + 0.38 * confidence)
        bridge = _clamp(0.34 + 0.28 * (1.0 - confidence))
        reasons.append("Layer isolation gives the brain a credible reset path despite the larger musical distance.")
    elif tempo.direct_difference > 0.14 or (genre < 0.42 and groove < 0.42):
        family = "bridge_preferred"
        confidence = _clamp(0.50 + 0.30 * min(1.0, tempo.direct_difference / 0.20) + 0.20 * (1.0 - musical_neighborhood))
        direct = _clamp(0.12 + 0.22 * musical_neighborhood)
        bridge = _clamp(0.68 + 0.28 * confidence)
        reasons.append("The pair is musically distant enough that the set brain should prefer an intermediate track when the crate permits it.")
    else:
        family = "hard_reset"
        confidence = _clamp(0.42 + 0.28 * (1.0 - musical_neighborhood) + 0.18 * min(1.0, tempo.direct_difference / 0.14))
        direct = _clamp(0.22 + 0.30 * musical_neighborhood)
        bridge = _clamp(0.48 + 0.30 * (1.0 - musical_neighborhood))
        reasons.append("A short artistic reset is safer than forcing the two full masters to coexist for a long overlap.")

    fallback_map: dict[str, tuple[TransitionFamily, ...]] = {
        "direct_blend": ("layer_handoff", "breakdown_drop") if layer >= 0.50 else ("breakdown_drop", "hard_reset"),
        "layer_handoff": ("direct_blend", "breakdown_drop", "hard_reset"),
        "breakdown_drop": ("layer_handoff", "hard_reset", "bridge_preferred") if layer >= 0.50 else ("hard_reset", "bridge_preferred"),
        "rhythmic_switch": ("hard_reset", "bridge_preferred"),
        "hard_reset": ("layer_handoff", "bridge_preferred") if layer >= 0.50 else ("bridge_preferred",),
        "bridge_preferred": ("hard_reset", "layer_handoff") if layer >= 0.50 else ("hard_reset",),
    }

    route = TransitionRoute(
        family=family, confidence=round(confidence,4), direct_transition_confidence=round(direct,4),
        bridge_preference=round(bridge,4), tempo=tempo, genre_similarity=round(genre,4),
        groove_similarity=round(groove,4), harmonic_similarity=round(harmonic,4),
        layer_capability=round(layer,4), reasons=tuple(reasons), fallback_families=fallback_map.get(family,()),
    )
    try:
        from ..intelligence.neural_policy import TRANSITION_FAMILIES, transition_family_prediction
        prediction=transition_family_prediction(track_a,track_b,route=route); neural=prediction.to_dict(); neural["heuristic_family"]=route.family; top=prediction.top_label
        if prediction.authority>=0.20 and top in TRANSITION_FAMILIES and top!=route.family and prediction.probability(str(top))>=prediction.probability(route.family)+0.20:
            previous=route.family; family=top  # type: ignore[assignment]
            neural.update(changed_family=True,previous_family=previous); reasons.append(f"Validated neural transition policy prefers {str(top).replace('_',' ')} over the heuristic {previous.replace('_',' ')} route.")
            route=replace(route,family=family,confidence=round(max(route.confidence,prediction.confidence),4),fallback_families=fallback_map.get(family,()),reasons=tuple(reasons),neural_policy=neural)
        else:
            neural["changed_family"]=False; route=replace(route,neural_policy=neural)
    except Exception as exc:
        route=replace(route,neural_policy={"available":False,"reason":f"neural policy unavailable: {exc}"})
    return route



def route_family_sequence(track_a: TrackAnalysis, track_b: TrackAnalysis) -> tuple[TransitionRoute, ...]:
    """Primary musical route followed by bounded, intentional fallback families."""
    primary = route_transition(track_a, track_b)
    routes = [primary]
    fallback_families=list(primary.fallback_families); neural=dict(primary.neural_policy or {})
    if float(neural.get("authority",0.0) or 0.0)>0.0:
        probabilities={str(label):float(probability) for label,probability in zip(neural.get("labels",[]),neural.get("probabilities",[]),strict=False)}
        fallback_families.sort(key=lambda family:probabilities.get(family,0.0),reverse=True)
    for index, family in enumerate(fallback_families, start=1):
        routes.append(
            replace(
                primary,
                family=family,
                confidence=round(max(0.20, primary.confidence - 0.10 * index), 4),
                reasons=(
                    *primary.reasons,
                    f"Fallback family {family.replace('_', ' ')} is tried only after the higher-priority family fails rendered-audio acceptance.",
                ),
                fallback_families=(),
            )
        )
    return tuple(routes)


_FAMILY_TECHNIQUE_FIT: dict[str, dict[str, float]] = {
    "direct_blend": {
        "stem_choreography": 1.0,
        "eq_bass_swap": 0.98,
        "equal_power_blend": 0.96,
        "filter_build": 0.84,
        "stem_vocal_overlay": 0.90,
        "layer_instrumental_handoff": 0.90,
        "layer_incoming_vocal_handoff": 0.94,
        "layer_vocal_bridge": 0.82,
        "double_drop": 0.64,
        "echo_out_drop_in": 0.52,
        "quick_cut": 0.36,
        "backspin": 0.30,
        "loop_roll": 0.56,
    },
    "layer_handoff": {
        "stem_choreography": 1.0,
        "layer_vocal_bridge": 0.96,
        "layer_incoming_vocal_handoff": 1.0,
        "layer_instrumental_handoff": 0.98,
        "stem_vocal_overlay": 0.96,
        "filter_build": 0.78,
        "eq_bass_swap": 0.72,
        "equal_power_blend": 0.62,
        "echo_out_drop_in": 0.55,
        "quick_cut": 0.46,
        "double_drop": 0.62,
        "loop_roll": 0.52,
        "backspin": 0.38,
    },
    "breakdown_drop": {
        "stem_choreography": 0.94,
        "filter_build": 1.0,
        "double_drop": 0.96,
        "echo_out_drop_in": 0.90,
        "layer_vocal_bridge": 0.82,
        "layer_incoming_vocal_handoff": 0.84,
        "layer_instrumental_handoff": 0.78,
        "stem_vocal_overlay": 0.78,
        "quick_cut": 0.70,
        "eq_bass_swap": 0.56,
        "equal_power_blend": 0.48,
        "loop_roll": 0.72,
        "backspin": 0.54,
    },
    "rhythmic_switch": {
        "stem_choreography": 0.78,
        "echo_out_drop_in": 1.0,
        "quick_cut": 0.94,
        "backspin": 0.86,
        "double_drop": 0.82,
        "layer_vocal_bridge": 0.78,
        "layer_incoming_vocal_handoff": 0.80,
        "layer_instrumental_handoff": 0.72,
        "loop_roll": 0.70,
        "filter_build": 0.52,
        "eq_bass_swap": 0.34,
        "equal_power_blend": 0.28,
        "stem_vocal_overlay": 0.62,
    },
    "hard_reset": {
        "stem_choreography": 0.74,
        "echo_out_drop_in": 1.0,
        "quick_cut": 0.98,
        "backspin": 0.90,
        "loop_roll": 0.76,
        "double_drop": 0.70,
        "layer_vocal_bridge": 0.68,
        "layer_incoming_vocal_handoff": 0.70,
        "layer_instrumental_handoff": 0.64,
        "filter_build": 0.46,
        "eq_bass_swap": 0.22,
        "equal_power_blend": 0.18,
        "stem_vocal_overlay": 0.54,
    },
    "bridge_preferred": {
        "stem_choreography": 0.58,
        "echo_out_drop_in": 0.86,
        "quick_cut": 0.84,
        "backspin": 0.72,
        "layer_vocal_bridge": 0.62,
        "layer_incoming_vocal_handoff": 0.64,
        "layer_instrumental_handoff": 0.58,
        "loop_roll": 0.56,
        "double_drop": 0.52,
        "filter_build": 0.32,
        "stem_vocal_overlay": 0.36,
        "eq_bass_swap": 0.18,
        "equal_power_blend": 0.14,
    },
}


def route_fit(route: TransitionRoute, plan: TransitionPlan) -> float:
    return _clamp(_FAMILY_TECHNIQUE_FIT.get(route.family, {}).get(plan.technique, 0.45))


def apply_transition_route(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plans: Sequence[TransitionPlan],
    *,
    route: TransitionRoute | None = None,
) -> tuple[TransitionRoute, list[TransitionPlan]]:
    route = route or route_transition(track_a, track_b)
    output: list[TransitionPlan] = []
    for plan in plans:
        fit = route_fit(route, plan)
        # Route authority is intentionally bounded. The actual rendered-audio
        # judge still decides among finalists, while the set graph receives a
        # much stronger bridge penalty through _edge_objective.
        utility = float(plan.metadata.get("selection_utility", plan.score))
        utility += 0.12 * (fit - 0.5)
        if route.family == "bridge_preferred":
            utility -= 0.08 * (1.0 - fit)
        plan.metadata = {
            **plan.metadata,
            "selection_utility": round(utility, 6),
            "transition_router": {
                **route.to_dict(),
                "route_fit": round(fit, 4),
            },
        }
        output.append(plan)
    output.sort(
        key=lambda plan: (
            float(plan.metadata.get("selection_utility", plan.score)),
            plan.score,
            plan.cue_confidence,
            plan.execution_signature,
        ),
        reverse=True,
    )
    return route, output
