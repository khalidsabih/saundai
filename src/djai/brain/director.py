from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

from ..models import TrackAnalysis, TransitionPlan
from .crowd_state import CrowdDirective
from .set_state import JourneyVector, SetState, journey_target, track_state_vector

SetIntentName = Literal[
    "open",
    "build",
    "hold",
    "lift",
    "peak",
    "release",
    "reset",
    "bridge",
    "surprise",
    "recover",
    "close",
]


@dataclass(frozen=True, slots=True)
class SetIntent:
    name: SetIntentName
    phase: str
    progress: float
    target: JourneyVector
    horizon_tracks: int
    target_energy_delta: float
    preserve_groove: bool
    vocal_tolerance: float
    risk_budget: float
    novelty_target: float
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target"] = self.target.to_dict()
        payload["rationale"] = list(self.rationale)
        return payload


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _intent_name(journey: str, progress: float, state: SetState) -> SetIntentName:
    p = _clamp(progress)
    if p <= 0.10:
        return "open"
    if p >= 0.92:
        return "close"

    # Musical recovery rules override the nominal journey when the recent set
    # has become too dense or too static.
    recent = list(state.recent_energy[-4:])
    if len(recent) >= 3:
        slope = float(np.mean(np.diff(np.asarray(recent, dtype=np.float64))))
        spread = float(np.ptp(np.asarray(recent, dtype=np.float64)))
        if state.energy > 0.82 and state.tension > 0.70 and state.tracks_since_breakdown >= 4:
            return "release"
        if spread < 0.035 and abs(slope) < 0.015 and 0.20 < p < 0.82:
            return "surprise" if float(state.dj_memory.get("surprise_budget", 0.5)) >= 0.42 else "lift"
    if state.vocal_density > 0.72 and state.tracks_since_vocal == 0:
        return "bridge"

    if journey == "build":
        return "build" if p < 0.52 else ("lift" if p < 0.78 else "peak")
    if journey == "warmup":
        return "hold" if p < 0.42 else ("build" if p < 0.76 else "lift")
    if journey == "peak":
        if p < 0.28:
            return "lift"
        if p < 0.58:
            return "peak"
        if p < 0.70:
            return "release"
        return "peak"
    if journey == "sunset":
        if p < 0.36:
            return "build"
        if p < 0.64:
            return "peak"
        return "release"
    if journey == "smooth":
        return "hold" if p < 0.70 else "release"

    # balanced: deliberate two-wave story.
    if p < 0.28:
        return "build"
    if p < 0.43:
        return "lift"
    if p < 0.55:
        return "peak"
    if p < 0.67:
        return "release"
    if p < 0.84:
        return "build"
    return "peak"


def direct_set_intent(
    journey: str,
    progress: float,
    state: SetState,
    *,
    crowd_directive: CrowdDirective | None = None,
) -> SetIntent:
    name = _intent_name(journey, progress, state)
    p = _clamp(progress)
    base = journey_target(journey, p)
    energy_adjust = {
        "open": -0.10,
        "build": 0.04,
        "hold": 0.0,
        "lift": 0.10,
        "peak": 0.16,
        "release": -0.16,
        "reset": -0.22,
        "bridge": -0.04,
        "surprise": 0.05,
        "recover": -0.10,
        "close": -0.12,
    }[name]
    target = JourneyVector(
        energy=_clamp(base.energy + energy_adjust),
        tension=_clamp(base.tension + (0.10 if name in {"lift", "peak"} else -0.10 if name in {"release", "reset", "close"} else 0.0)),
        groove=_clamp(base.groove + (0.06 if name in {"hold", "build", "bridge"} else 0.0)),
        vocal_density=_clamp(base.vocal_density - (0.18 if name in {"bridge", "reset"} else 0.0)),
        brightness=_clamp(base.brightness + (0.06 if name in {"lift", "peak"} else -0.06 if name == "close" else 0.0)),
        novelty=_clamp(base.novelty + (0.22 if name == "surprise" else -0.05 if name in {"hold", "close"} else 0.0)),
    )
    risk_budget = {
        "open": 0.18,
        "hold": 0.20,
        "close": 0.18,
        "bridge": 0.24,
        "build": 0.35,
        "release": 0.30,
        "reset": 0.34,
        "recover": 0.18,
        "lift": 0.46,
        "peak": 0.58,
        "surprise": 0.70,
    }[name]
    surprise_budget = float(state.dj_memory.get("surprise_budget", 0.5))
    risk_budget = min(risk_budget, 0.25 + 0.75 * surprise_budget)
    rationale_items = [
        f"Set director selected {name} at {p:.0%} journey progress.",
        f"Target energy {target.energy:.0%} from current {state.energy:.0%}.",
        f"Risk budget {risk_budget:.0%}; novelty target {target.novelty:.0%}.",
    ]
    if crowd_directive is not None and crowd_directive.active:
        target = JourneyVector(
            energy=_clamp(target.energy + crowd_directive.energy_shift),
            tension=target.tension,
            groove=target.groove,
            vocal_density=_clamp(target.vocal_density - 0.10 * crowd_directive.vocal_simplicity),
            brightness=target.brightness,
            novelty=_clamp(target.novelty + crowd_directive.novelty_shift),
        )
        risk_budget = _clamp(risk_budget + crowd_directive.risk_shift)
        rationale_items.append(
            "Executive crowd directive "
            f"{crowd_directive.mode} applied at {crowd_directive.strength:.0%} bounded strength."
        )
    rationale = tuple(rationale_items)
    return SetIntent(
        name=name,
        phase=f"{journey}:{name}",
        progress=p,
        target=target,
        horizon_tracks=3 if name not in {"surprise", "reset", "close"} else 2,
        target_energy_delta=round(target.energy - state.energy, 4),
        preserve_groove=(
            True
            if crowd_directive is not None and crowd_directive.active and crowd_directive.mode == "recover"
            else name not in {"reset", "surprise"}
        ),
        vocal_tolerance=_clamp(
            (0.28 if name in {"bridge", "reset"} else 0.52)
            - (0.18 * crowd_directive.vocal_simplicity if crowd_directive is not None and crowd_directive.active else 0.0)
        ),
        risk_budget=round(_clamp(risk_budget), 4),
        novelty_target=round(target.novelty, 4),
        rationale=rationale,
    )


def intent_track_score(intent: SetIntent, state: SetState, candidate: TrackAnalysis) -> tuple[float, dict[str, float]]:
    vector = track_state_vector(candidate)
    target = intent.target
    fits = {
        "energy": float(np.exp(-((vector.energy - target.energy) / 0.24) ** 2)),
        "tension": float(np.exp(-((vector.tension - target.tension) / 0.28) ** 2)),
        "groove": float(np.exp(-((vector.groove - target.groove) / 0.28) ** 2)),
        "vocal": float(np.exp(-((vector.vocal_density - target.vocal_density) / 0.30) ** 2)),
        "brightness": float(np.exp(-((vector.brightness - target.brightness) / 0.32) ** 2)),
    }
    actual_delta = vector.energy - state.energy
    desired_delta = intent.target_energy_delta
    direction = float(np.exp(-((actual_delta - desired_delta) / 0.20) ** 2))
    fits["energy_direction"] = direction
    score = (
        0.28 * fits["energy"]
        + 0.16 * fits["tension"]
        + 0.18 * fits["groove"]
        + 0.10 * fits["vocal"]
        + 0.08 * fits["brightness"]
        + 0.20 * direction
    )
    if intent.preserve_groove and vector.groove < 0.42:
        score -= 0.08
        fits["groove_floor_penalty"] = 0.08
    if vector.vocal_density > 0.72 and intent.vocal_tolerance < 0.35:
        score -= 0.09
        fits["vocal_density_penalty"] = 0.09
    return _clamp(score), {key: round(_clamp(value), 4) for key, value in fits.items()}


def set_director_report(
    order: Sequence[int],
    analyses: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
    *,
    journey: str,
) -> list[dict[str, object]]:
    from .set_state import build_set_state

    report: list[dict[str, object]] = []
    prefix_tracks: list[TrackAnalysis] = []
    prefix_transitions: list[TransitionPlan] = []
    total = max(1, len(order) - 1)
    transition_values = list(transitions or [])
    for position, index in enumerate(order):
        if position > 0 and position - 1 < len(transition_values):
            transition = transition_values[position - 1]
            prefix_transitions.append(getattr(transition, "plan", transition))
        state = build_set_state(prefix_tracks or [analyses[index]], prefix_transitions)
        progress = position / total
        intent = direct_set_intent(journey, progress, state)
        candidate_score, components = intent_track_score(intent, state, analyses[index])
        report.append({
            "position": position,
            "track_index": index,
            "track": analyses[index].path,
            "intent": intent.to_dict(),
            "intent_track_score": round(candidate_score, 4),
            "components": components,
        })
        prefix_tracks.append(analyses[index])
    return report
