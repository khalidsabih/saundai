from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..models import TransitionPlan
from .operations import ProgramDeckOperation, deck_operations_from_payload, validate_deck_operation

# 6.0 turns the performance program from descriptive metadata into the
# authoritative executable intermediate representation consumed by the DSP
# engine. Renderer code must not infer transition strategy from `technique`.
PERFORMANCE_PROGRAM_VERSION = "4.0"


@dataclass(frozen=True, slots=True)
class CurvePoint:
    position: float
    value: float

    def to_dict(self) -> dict[str, float]:
        return {
            "position": round(float(self.position), 6),
            "value": round(float(self.value), 6),
        }


@dataclass(frozen=True, slots=True)
class ProgramProcessor:
    """Deterministic pre-mix transform for one named source bus."""

    bus: str
    kind: str
    parameters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bus": self.bus,
            "kind": self.kind,
            "parameters": {
                key: round(float(value), 6)
                for key, value in sorted(self.parameters.items())
            },
        }


@dataclass(slots=True)
class TransitionProgram:
    """Executable immutable DJ transition program.

    The brain/compiler owns this representation. The DSP engine only knows
    source buses, automation curves and deterministic processors; it never
    chooses a transition technique or musical strategy.
    """

    version: str
    intent: str
    technique: str
    transition_bars: int
    transition_seconds: float
    target_bpm: float
    tempo_strategy: str
    source_mode: str = "master"
    curves: dict[str, list[CurvePoint]] = field(default_factory=dict)
    stem_curves_a: dict[str, list[CurvePoint]] = field(default_factory=dict)
    stem_curves_b: dict[str, list[CurvePoint]] = field(default_factory=dict)
    layer_curves_a: dict[str, list[CurvePoint]] = field(default_factory=dict)
    layer_curves_b: dict[str, list[CurvePoint]] = field(default_factory=dict)
    processors: list[ProgramProcessor] = field(default_factory=list)
    deck_operations: list[ProgramDeckOperation] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def curves(values: dict[str, list[CurvePoint]]) -> dict[str, list[dict[str, float]]]:
            return {
                key: [point.to_dict() for point in points]
                for key, points in sorted(values.items())
            }

        return {
            "version": self.version,
            "intent": self.intent,
            "technique": self.technique,
            "transition_bars": self.transition_bars,
            "transition_seconds": round(float(self.transition_seconds), 6),
            "target_bpm": round(float(self.target_bpm), 4),
            "tempo_strategy": self.tempo_strategy,
            "source_mode": self.source_mode,
            "curves": curves(self.curves),
            "stem_curves_a": curves(self.stem_curves_a),
            "stem_curves_b": curves(self.stem_curves_b),
            "layer_curves_a": curves(self.layer_curves_a),
            "layer_curves_b": curves(self.layer_curves_b),
            "processors": [item.to_dict() for item in self.processors],
            "deck_operations": [item.to_dict() for item in self.deck_operations],
            "annotations": self.annotations,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _curve(*pairs: tuple[float, float]) -> list[CurvePoint]:
    return [CurvePoint(float(position), float(value)) for position, value in pairs]


def _sample(function: Callable[[float], float], *, samples: int = 33) -> list[CurvePoint]:
    count = max(3, int(samples))
    return [
        CurvePoint(index / (count - 1), float(function(index / (count - 1))))
        for index in range(count)
    ]


def _smooth(value: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value >= end else 0.0
    t = max(0.0, min(1.0, (value - start) / (end - start)))
    return t * t * (3.0 - 2.0 * t)


def _smooth_in(start: float, end: float) -> list[CurvePoint]:
    return _sample(lambda x: _smooth(x, start, end))


def _smooth_out(start: float, end: float) -> list[CurvePoint]:
    return _sample(lambda x: 1.0 - _smooth(x, start, end))


def _constant(value: float) -> list[CurvePoint]:
    return _curve((0.0, value), (1.0, value))


def _equal_power_out() -> list[CurvePoint]:
    return _sample(lambda x: math.cos(x * math.pi / 2.0))


def _equal_power_in() -> list[CurvePoint]:
    return _sample(lambda x: math.sin(x * math.pi / 2.0))


def _product(*functions: Callable[[float], float]) -> list[CurvePoint]:
    return _sample(lambda x: math.prod(function(x) for function in functions))


def _complete_stems(plan: TransitionPlan) -> bool:
    required = ("vocals", "drums", "bass", "other")
    return all(plan.stem_paths_a.get(name) and plan.stem_paths_b.get(name) for name in required)


def _layer_path(bundle: dict[str, Any], role: str) -> bool:
    value = bundle.get(role)
    return isinstance(value, dict) and bool(value.get("path"))


def _source_mode(plan: TransitionPlan) -> str:
    if plan.technique in {
        "layer_vocal_bridge",
        "layer_incoming_vocal_handoff",
        "layer_instrumental_handoff",
    }:
        return "layers"
    if plan.technique == "stem_choreography":
        return "stems"
    if plan.technique in {
        "eq_bass_swap",
        "filter_build",
        "double_drop",
        "stem_vocal_overlay",
    } and _complete_stems(plan):
        return "stems"
    return "master"


def _hold_curve(points: list[CurvePoint], *, start: float, end: float, minimum: float) -> list[CurvePoint]:
    """Raise a stem automation curve to a deterministic minimum in a window."""
    if not points:
        return points
    positions = sorted({point.position for point in points} | {float(start), float(end)})

    def value_at(position: float) -> float:
        if position <= points[0].position:
            return float(points[0].value)
        for left, right in zip(points, points[1:]):
            if left.position <= position <= right.position:
                span = max(1e-9, right.position - left.position)
                t = (position - left.position) / span
                return float(left.value + t * (right.value - left.value))
        return float(points[-1].value)

    return [
        CurvePoint(position, max(value_at(position), minimum) if start <= position <= end else value_at(position))
        for position in positions
    ]


def _apply_structural_curve_operations(
    stem_a: dict[str, list[CurvePoint]],
    stem_b: dict[str, list[CurvePoint]],
    operations: list[ProgramDeckOperation],
) -> None:
    for operation in operations:
        if str(operation.kind) != "HOLD_STEM" or str(operation.phase) != "transition":
            continue
        params = operation.parameters
        target = stem_a if operation.deck == "a" else stem_b
        stem = str(params.get("stem", ""))
        if stem not in target:
            continue
        target[stem] = _hold_curve(
            target[stem],
            start=float(params.get("start_position", 0.0)),
            end=float(params.get("end_position", 1.0)),
            minimum=float(params.get("minimum_gain", 1.0)),
        )


def compile_transition_program(plan: TransitionPlan) -> TransitionProgram:
    source_mode = _source_mode(plan)
    curves: dict[str, list[CurvePoint]] = {
        "deck_a_gain": _equal_power_out(),
        "deck_b_gain": _equal_power_in(),
    }
    stem_a = {name: _equal_power_out() for name in ("drums", "bass", "vocals", "other")}
    stem_b = {name: _equal_power_in() for name in ("drums", "bass", "vocals", "other")}
    layer_a: dict[str, list[CurvePoint]] = {}
    layer_b: dict[str, list[CurvePoint]] = {}
    processors: list[ProgramProcessor] = []
    deck_operations = deck_operations_from_payload(plan.metadata.get("deck_operations", []))
    incoming_extension = any(
        str(operation.phase) == "transition"
        and str(operation.deck) == "b"
        and str(operation.kind) in {"LOOP_INCOMING", "EXTEND_INTRO"}
        for operation in deck_operations
    )
    professional_mix = dict(plan.metadata.get("professional_mix", {}) or {})
    annotations: dict[str, Any] = {
        "section_pair": f"{plan.exit_section}->{plan.entry_section}",
        "beatmatched": bool(plan.beatmatched),
        "source_duration_a": round(plan.source_duration_a, 6),
        "source_duration_b": round(plan.source_duration_b, 6),
        "renderer_contract": "execute_program_ir",
        "execution_engine": "generic_program_ir",
        "incoming_alignment": (
            "loop_extend"
            if incoming_extension
            and (plan.source_duration_b / max(plan.render_rate_b, 1e-9)) < plan.transition_seconds - 1e-5
            else "tail"
            if (plan.source_duration_b / max(plan.render_rate_b, 1e-9)) < plan.transition_seconds - 1e-5
            else "full"
        ),
        "professional_mix": professional_mix,
        "mix_family": str(professional_mix.get("family", "legacy")),
        "continuous_mix_required": bool(professional_mix.get("continuous_mix_required", False)),
        "musical_ownership": dict(professional_mix.get("ownership", {}) or {}),
        "deck_performance": dict(plan.metadata.get("deck_performance", {}) or {}),
        "source_start_b": round(plan.source_start_b, 6),
        "post_transition_source_end_b": round(plan.post_transition_source_end_b, 6),
    }

    technique = plan.technique
    annotations["structural_reset"] = bool(
        professional_mix.get("structural_reset", False)
        or technique in {"echo_out_drop_in", "quick_cut", "backspin"}
    )
    annotations["reset_justified"] = bool(professional_mix.get("reset_justified", False))
    if technique == "equal_power_blend":
        annotations["choreography"] = "baseline continuous blend; incoming establishes while outgoing progressively yields"
    if technique == "eq_bass_swap":
        curves = {
            "deck_a_gain": _constant(1.0),
            "deck_b_gain": _constant(1.0),
            "deck_a_high_gain": _equal_power_out(),
            "deck_b_high_gain": _equal_power_in(),
            "deck_a_low_gain": _smooth_out(0.34, 0.56),
            "deck_b_low_gain": _smooth_in(0.48, 0.70),
        }
        stem_a["bass"] = _smooth_out(0.32, 0.55)
        stem_b["bass"] = _smooth_in(0.54, 0.76)
        annotations["choreography"] = "single low-frequency ownership handoff around the phrase midpoint"
    elif technique == "filter_build":
        natural_profile = str(plan.metadata.get("transition_naturalness_profile", "balanced"))
        if natural_profile == "compact":
            # Difficult pairs still need a real handoff: B establishes rhythmic
            # presence before A releases ownership. A compact mix is shorter,
            # not a disguised fade-to-reset.
            curves = {
                "deck_a_gain": _smooth_out(0.46, 0.96),
                "deck_b_gain": _constant(1.0),
                "deck_b_high_gain": _smooth_in(0.02, 0.30),
                "deck_b_low_gain": _smooth_in(0.50, 0.74),
            }
            a_out = _smooth_out(0.44, 0.96)
            stem_a = {name: a_out for name in ("drums", "bass", "vocals", "other")}
            stem_b["drums"] = _smooth_in(0.02, 0.32)
            stem_b["other"] = _smooth_in(0.12, 0.48)
            stem_b["bass"] = _smooth_in(0.58, 0.84)
            stem_b["vocals"] = _smooth_in(0.70, 0.94)
            annotations["choreography"] = "compact phrase handoff: incoming rhythm first, outgoing master clears early, bass/vocal establish late"
        else:
            curves = {
                "deck_a_gain": _smooth_out(0.52, 0.98),
                "deck_b_gain": _constant(1.0),
                "deck_b_high_gain": _smooth_in(0.02, 0.34),
                "deck_b_low_gain": _smooth_in(0.54, 0.78),
            }
            a_out = _smooth_out(0.50, 0.98)
            stem_a = {name: a_out for name in ("drums", "bass", "vocals", "other")}
            stem_b["drums"] = _smooth_in(0.02, 0.48)
            stem_b["other"] = _smooth_in(0.18, 0.72)
            stem_b["bass"] = _smooth_in(0.60, 0.94)
            stem_b["vocals"] = _smooth_in(0.74, 0.98)
            annotations["choreography"] = "drums first, musical body second, bass and vocal late for build payoff"
        annotations["transition_naturalness_profile"] = natural_profile
    elif technique == "echo_out_drop_in":
        curves = {
            "deck_a_gain": _smooth_out(0.04, 0.90),
            "deck_b_gain": _smooth_in(0.80, 0.96),
        }
        processors.append(ProgramProcessor(bus="deck_a", kind="echo", parameters={}))
        annotations["drop_target"] = 0.82
    elif technique == "quick_cut":
        curves = {
            "deck_a_gain": _smooth_out(0.41, 0.59),
            "deck_b_gain": _smooth_in(0.41, 0.59),
        }
        annotations["hard_switch_fraction"] = 0.50
    elif technique == "backspin":
        curves = {
            "deck_a_gain": _smooth_out(0.10, 0.72),
            "deck_b_gain": _smooth_in(0.48, 0.76),
        }
        processors.append(ProgramProcessor(bus="deck_a", kind="reverse", parameters={}))
    elif technique == "loop_roll":
        curves = {
            "deck_a_gain": _smooth_out(0.30, 0.92),
            "deck_b_gain": _smooth_in(0.16, 0.78),
        }
        processors.append(ProgramProcessor(bus="deck_a", kind="loop_tail", parameters={"beats": 1.0}))
        annotations["loop_beats"] = 1.0
    elif technique == "double_drop":
        curves = {
            "deck_a_gain": _smooth_out(0.42, 0.60),
            "deck_b_gain": _smooth_in(0.05, 0.48),
        }
        for name in ("drums", "bass", "other"):
            stem_a[name] = _smooth_out(0.42, 0.58)
            stem_b[name] = _smooth_in(0.10, 0.48)
        stem_a["vocals"] = _smooth_out(0.20, 0.46)
        stem_b["vocals"] = _smooth_in(0.58, 0.86)
        annotations["drop_target"] = 0.50
        annotations["choreography"] = "protect the drop by suppressing vocal competition around the pivot"
    elif technique == "stem_vocal_overlay":
        curves = {
            "deck_a_gain": _constant(1.0),
            "deck_b_gain": _constant(1.0),
            "deck_a_high_gain": _equal_power_out(),
            "deck_b_high_gain": _equal_power_in(),
            "deck_a_low_gain": _smooth_out(0.30, 0.55),
            "deck_b_low_gain": _smooth_in(0.55, 0.80),
        }
        for name in ("drums", "other"):
            stem_a[name] = _equal_power_out()
            stem_b[name] = _equal_power_in()
        stem_a["bass"] = _smooth_out(0.30, 0.55)
        stem_b["bass"] = _smooth_in(0.55, 0.80)
        stem_a["vocals"] = _sample(lambda x: 0.78 * (1.0 - _smooth(x, 0.58, 0.92)))
        stem_b["vocals"] = _sample(lambda x: 0.65 * _smooth(x, 0.72, 0.96))
        annotations["choreography"] = "carry outgoing vocal above an instrumental handoff, then yield to incoming vocal"
    elif technique == "stem_choreography":
        profile = str(dict(plan.metadata.get("stem_choreography", {})).get("profile", "progressive_handoff"))
        if profile == "deck_vocal_exchange":
            # Deck-style ownership transfer: B rhythm establishes under A, A's
            # lead vocal clears before B's lead arrives, bass has one owner,
            # and A's musical bed yields progressively rather than as a master fade.
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.30, 0.96), (0.72, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.38, 1.0), (0.56, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 1.0), (0.24, 0.96), (0.52, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.32, 0.88), (0.84, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.06, 0.18), (0.40, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.54, 0.0), (0.68, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.60, 0.0), (0.76, 0.28), (0.94, 1.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.16, 0.12), (0.68, 0.78), (0.92, 1.0), (1.0, 1.0)),
            }
        elif profile == "outgoing_vocal_bridge":
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.18, 0.95), (0.48, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.34, 1.0), (0.54, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 0.92), (0.62, 0.92), (0.90, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.26, 0.82), (0.64, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.10, 0.28), (0.42, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.52, 0.0), (0.70, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.86, 0.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.24, 0.24), (0.72, 1.0), (1.0, 1.0)),
            }
        elif profile == "instrumental_reset":
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.34, 0.86), (0.62, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.36, 1.0), (0.52, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 1.0), (0.18, 0.76), (0.34, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.38, 0.70), (0.70, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.18, 0.20), (0.60, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.52, 0.0), (0.72, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.78, 0.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.28, 0.20), (0.76, 1.0), (1.0, 1.0)),
            }
        elif profile == "drum_first_build":
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.32, 0.88), (0.68, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.44, 1.0), (0.60, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 1.0), (0.46, 0.85), (0.76, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.40, 0.72), (0.76, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.05, 0.16), (0.40, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.58, 0.0), (0.76, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.80, 0.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.38, 0.12), (0.78, 1.0), (1.0, 1.0)),
            }
        elif profile == "bass_ownership_swap":
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.48, 0.82), (0.82, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.40, 1.0), (0.56, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 1.0), (0.58, 0.88), (0.86, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.44, 0.78), (0.84, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.20, 0.18), (0.66, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.56, 0.0), (0.72, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.82, 0.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.26, 0.16), (0.78, 1.0), (1.0, 1.0)),
            }
        else:  # progressive_handoff
            stem_a = {
                "drums": _curve((0.0, 1.0), (0.34, 0.90), (0.72, 0.0), (1.0, 0.0)),
                "bass": _curve((0.0, 1.0), (0.40, 1.0), (0.58, 0.0), (1.0, 0.0)),
                "vocals": _curve((0.0, 1.0), (0.56, 0.92), (0.88, 0.0), (1.0, 0.0)),
                "other": _curve((0.0, 1.0), (0.42, 0.78), (0.82, 0.0), (1.0, 0.0)),
            }
            stem_b = {
                "drums": _curve((0.0, 0.0), (0.12, 0.18), (0.58, 1.0), (1.0, 1.0)),
                "bass": _curve((0.0, 0.0), (0.58, 0.0), (0.74, 1.0), (1.0, 1.0)),
                "vocals": _curve((0.0, 0.0), (0.82, 0.0), (1.0, 1.0)),
                "other": _curve((0.0, 0.0), (0.30, 0.18), (0.80, 1.0), (1.0, 1.0)),
            }
        curves = {"deck_a_gain": _constant(1.0), "deck_b_gain": _constant(1.0)}
        annotations.update({
            "choreography": "four-stem musical-role ownership program",
            "stem_choreography_profile": profile,
            "renderer_contract": "execute_stem_curves_exactly",
            "single_bass_owner": True,
            "avoid_dual_lead_vocals": True,
            "vocal_ownership_transfer": "a_to_b",
            "incoming_rhythm_first": True,
        })
    elif technique == "layer_vocal_bridge":
        # Full-master levels and supplied alternate sources are all explicit
        # buses in the IR. Source preparation may level-match an alternate to
        # its full master, but it cannot alter choreography.
        curves = {}
        layer_a["master"] = _smooth_out(0.08, 0.34)
        layer_a["vocals"] = _sample(
            lambda x: 0.90 * _smooth(x, 0.04, 0.12) * (1.0 - _smooth(x, 0.66, 0.90))
        )
        layer_b["instrumental"] = _sample(
            lambda x: _smooth(x, 0.08, 0.38) * (1.0 - _smooth(x, 0.72, 1.0))
        )
        layer_b["master"] = _smooth_in(0.72, 1.0)
        annotations.update({
            "choreography": "outgoing supplied vocal bridges over incoming supplied instrumental before incoming full-master reveal",
            "layer_bundle_a": plan.performance_layers_a.get("fingerprint", ""),
            "layer_bundle_b": plan.performance_layers_b.get("fingerprint", ""),
            "source_priority": plan.metadata.get("layer_source_quality", "performance_layers"),
            "layer_quality_confidence": float(plan.metadata.get("layer_quality_confidence", 1.0)),
            "layer_level_ratios": {"a:vocals": 0.56, "b:instrumental": 0.96},
        })
    elif technique == "layer_incoming_vocal_handoff":
        # Layer Intelligence V1.  A's lead is cleared first by moving from the
        # full master to its instrumental.  B's vocal then becomes the lead
        # over A's instrumental before the accompaniment is transferred to B.
        # The final short phase crossfades the separated B layers back to the
        # full B master so post-transition playback has one authoritative source.
        curves = {}
        layer_a["master"] = _smooth_out(0.04, 0.22)
        layer_a["instrumental"] = _sample(
            lambda x: 0.82 * _smooth(x, 0.05, 0.20) * (1.0 - _smooth(x, 0.58, 0.88))
        )
        layer_b["vocals"] = _sample(
            lambda x: 0.52 * _smooth(x, 0.12, 0.30) * (1.0 - _smooth(x, 0.82, 0.98))
        )
        layer_b["instrumental"] = _sample(
            lambda x: 0.80 * _smooth(x, 0.46, 0.76) * (1.0 - _smooth(x, 0.84, 1.0))
        )
        layer_b["master"] = _smooth_in(0.84, 1.0)
        annotations.update({
            "choreography": "incoming vocal takes lead over outgoing instrumental, then accompaniment ownership transfers to B before full-master reveal",
            "layer_choreography": dict(plan.metadata.get("layer_choreography", {}) or {}),
            "layer_bundle_a": plan.performance_layers_a.get("fingerprint", ""),
            "layer_bundle_b": plan.performance_layers_b.get("fingerprint", ""),
            "source_priority": plan.metadata.get("layer_source_quality", "performance_layers"),
            "layer_quality_confidence": float(plan.metadata.get("layer_quality_confidence", 1.0)),
            "layer_level_ratios": {
                "a:instrumental": 0.90,
                "b:vocals": 0.72,
                "b:instrumental": 0.90,
            },
            "vocal_ownership_transfer": "a_to_b_early",
            "instrumental_ownership_transfer": "a_to_b_late",
            "avoid_dual_lead_vocals": True,
            "rendered_headroom_authoritative": True,
            "fallback": "full_master_transition",
        })
    elif technique == "layer_instrumental_handoff":
        curves = {}
        a_out = lambda x: 1.0 - _smooth(x, 0.18, 0.78)
        if _layer_path(dict(plan.performance_layers_a or {}), "instrumental"):
            reveal = lambda x: _smooth(x, 0.06, 0.24)
            layer_a["master"] = _product(a_out, lambda x: 1.0 - reveal(x))
            layer_a["instrumental"] = _product(a_out, reveal)
        else:
            layer_a["master"] = _sample(a_out)
        layer_b["instrumental"] = _sample(
            lambda x: _smooth(x, 0.08, 0.58) * (1.0 - _smooth(x, 0.72, 1.0))
        )
        layer_b["master"] = _smooth_in(0.72, 1.0)
        annotations.update({
            "choreography": "remove competing vocals during overlap, exchange instrumentals, then reveal incoming full master",
            "layer_bundle_a": plan.performance_layers_a.get("fingerprint", ""),
            "layer_bundle_b": plan.performance_layers_b.get("fingerprint", ""),
            "source_priority": plan.metadata.get("layer_source_quality", "performance_layers"),
            "layer_quality_confidence": float(plan.metadata.get("layer_quality_confidence", 1.0)),
            "layer_level_ratios": {"a:instrumental": 0.96, "b:instrumental": 0.96},
        })

    _apply_structural_curve_operations(stem_a, stem_b, deck_operations)

    program = TransitionProgram(
        version=PERFORMANCE_PROGRAM_VERSION,
        intent=plan.intent,
        technique=plan.technique,
        transition_bars=plan.transition_bars,
        transition_seconds=plan.transition_seconds,
        target_bpm=plan.target_bpm,
        tempo_strategy=plan.tempo_strategy,
        source_mode=source_mode,
        curves=curves,
        stem_curves_a=stem_a,
        stem_curves_b=stem_b,
        layer_curves_a=layer_a,
        layer_curves_b=layer_b,
        processors=processors,
        deck_operations=deck_operations,
        annotations=annotations,
    )
    validate_performance_program(program)
    return program


def validate_performance_program(program: TransitionProgram) -> None:
    if program.version != PERFORMANCE_PROGRAM_VERSION:
        raise ValueError(f"Unsupported performance program version: {program.version}")
    if program.source_mode not in {"master", "stems", "layers"}:
        raise ValueError(f"Unsupported performance-program source mode: {program.source_mode}")
    if program.transition_bars <= 0 or program.transition_seconds <= 0.0:
        raise ValueError("Performance program duration must be positive.")
    if not program.intent or not program.technique:
        raise ValueError("Performance program must bind intent and technique for auditability.")
    groups = (
        program.curves,
        program.stem_curves_a,
        program.stem_curves_b,
        program.layer_curves_a,
        program.layer_curves_b,
    )
    for group in groups:
        for name, points in group.items():
            if not points:
                raise ValueError(f"Performance curve {name!r} is empty.")
            previous = -1.0
            for point in points:
                if not 0.0 <= point.position <= 1.0:
                    raise ValueError(f"Performance curve {name!r} has an out-of-range position.")
                if point.position < previous:
                    raise ValueError(f"Performance curve {name!r} is not monotonic in time.")
                if not math.isfinite(point.value):
                    raise ValueError(f"Performance curve {name!r} contains a non-finite value.")
                previous = point.position
    for operation in program.deck_operations:
        validate_deck_operation(operation)
    for processor in program.processors:
        if not processor.bus or processor.kind not in {"echo", "reverse", "loop_tail"}:
            raise ValueError(f"Unsupported performance processor: {processor.kind!r}")
        if any(not math.isfinite(float(value)) for value in processor.parameters.values()):
            raise ValueError("Performance processor contains non-finite parameters.")


def transition_program_from_dict(payload: dict[str, Any]) -> TransitionProgram:
    def points(values: Any) -> list[CurvePoint]:
        return [
            value if isinstance(value, CurvePoint) else CurvePoint(**dict(value))
            for value in list(values or [])
        ]

    def group(name: str) -> dict[str, list[CurvePoint]]:
        return {
            str(key): points(value)
            for key, value in dict(payload.get(name, {}) or {}).items()
        }

    processors = [
        value if isinstance(value, ProgramProcessor) else ProgramProcessor(
            bus=str(dict(value).get("bus", "")),
            kind=str(dict(value).get("kind", "")),
            parameters={
                str(key): float(item)
                for key, item in dict(dict(value).get("parameters", {}) or {}).items()
            },
        )
        for value in list(payload.get("processors", []) or [])
    ]
    deck_operations = deck_operations_from_payload(payload.get("deck_operations", []))
    program = TransitionProgram(
        version=str(payload.get("version", "")),
        intent=str(payload.get("intent", "")),
        technique=str(payload.get("technique", "")),
        transition_bars=int(payload.get("transition_bars", 0)),
        transition_seconds=float(payload.get("transition_seconds", 0.0)),
        target_bpm=float(payload.get("target_bpm", 0.0)),
        tempo_strategy=str(payload.get("tempo_strategy", "")),
        source_mode=str(payload.get("source_mode", "master")),
        curves=group("curves"),
        stem_curves_a=group("stem_curves_a"),
        stem_curves_b=group("stem_curves_b"),
        layer_curves_a=group("layer_curves_a"),
        layer_curves_b=group("layer_curves_b"),
        processors=processors,
        deck_operations=deck_operations,
        annotations=dict(payload.get("annotations", {}) or {}),
    )
    validate_performance_program(program)
    return program


def attach_performance_program(plan: TransitionPlan) -> TransitionPlan:
    program = compile_transition_program(plan)
    plan.performance_program = program.to_dict()
    plan.performance_program_fingerprint = program.fingerprint
    return plan
