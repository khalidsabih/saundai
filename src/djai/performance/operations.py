from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

DECK_OPERATION_VERSION = "1.0"
DeckOperationPhase = Literal["pre_transition", "transition", "post_transition"]
DeckOperationKind = Literal[
    "WAIT",
    "LOOP_OUTGOING",
    "LOOP_INCOMING",
    "EXTEND_INTRO",
    "BEAT_JUMP",
    "HOT_CUE_JUMP",
    "HOLD_STEM",
    "SKIP_BREAKDOWN",
]

_ALLOWED_KINDS: set[str] = {
    "WAIT",
    "LOOP_OUTGOING",
    "LOOP_INCOMING",
    "EXTEND_INTRO",
    "BEAT_JUMP",
    "HOT_CUE_JUMP",
    "HOLD_STEM",
    "SKIP_BREAKDOWN",
}
_ALLOWED_PHASES: set[str] = {"pre_transition", "transition", "post_transition"}
_ALLOWED_DECKS: set[str] = {"a", "b"}


@dataclass(frozen=True, slots=True)
class ProgramDeckOperation:
    """Deterministic structural deck instruction.

    The planner may choose *which* operation is useful, but execution is bound
    to explicit source coordinates/parameters here. Renderers are forbidden
    from inferring musical intent from the operation name.
    """

    kind: DeckOperationKind | str
    phase: DeckOperationPhase | str
    deck: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    version: str = DECK_OPERATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": str(self.kind),
            "phase": str(self.phase),
            "deck": str(self.deck),
            "parameters": {
                str(key): _normalize_value(value)
                for key, value in sorted(self.parameters.items())
            },
            "reason": str(self.reason),
        }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Deck operation contains a non-finite numeric value.")
        return round(float(value), 6)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return [_normalize_value(item) for item in value]
    raise ValueError(f"Unsupported deck-operation parameter type: {type(value).__name__}")


def validate_deck_operation(operation: ProgramDeckOperation) -> None:
    if operation.version != DECK_OPERATION_VERSION:
        raise ValueError(f"Unsupported deck-operation version: {operation.version}")
    if str(operation.kind) not in _ALLOWED_KINDS:
        raise ValueError(f"Unsupported deck operation: {operation.kind!r}")
    if str(operation.phase) not in _ALLOWED_PHASES:
        raise ValueError(f"Unsupported deck-operation phase: {operation.phase!r}")
    if str(operation.deck) not in _ALLOWED_DECKS:
        raise ValueError(f"Unsupported deck-operation deck: {operation.deck!r}")
    for value in operation.parameters.values():
        _normalize_value(value)

    params = operation.parameters
    kind = str(operation.kind)
    if kind in {"LOOP_OUTGOING", "LOOP_INCOMING", "EXTEND_INTRO"}:
        start = float(params.get("source_start_seconds", -1.0))
        end = float(params.get("source_end_seconds", -1.0))
        if start < 0.0 or end <= start:
            raise ValueError(f"{kind} requires a positive source window.")
        if int(params.get("repeats", 1)) < 1:
            raise ValueError(f"{kind} requires repeats >= 1.")
    if kind in {"BEAT_JUMP", "HOT_CUE_JUMP", "SKIP_BREAKDOWN"}:
        trigger = float(params.get("trigger_source_seconds", -1.0))
        target = float(params.get("target_source_seconds", -1.0))
        if trigger < 0.0 or target < 0.0:
            raise ValueError(f"{kind} requires non-negative trigger and target positions.")
    if kind == "HOLD_STEM":
        stem = str(params.get("stem", ""))
        if stem not in {"drums", "bass", "vocals", "other"}:
            raise ValueError("HOLD_STEM requires one of drums/bass/vocals/other.")
        start = float(params.get("start_position", -1.0))
        end = float(params.get("end_position", -1.0))
        gain = float(params.get("minimum_gain", -1.0))
        if not (0.0 <= start < end <= 1.0):
            raise ValueError("HOLD_STEM requires 0 <= start_position < end_position <= 1.")
        if not (0.0 <= gain <= 1.5):
            raise ValueError("HOLD_STEM minimum_gain must be between 0 and 1.5.")


def deck_operation_from_dict(payload: Mapping[str, Any]) -> ProgramDeckOperation:
    operation = ProgramDeckOperation(
        version=str(payload.get("version", DECK_OPERATION_VERSION)),
        kind=str(payload.get("kind", "")),
        phase=str(payload.get("phase", "")),
        deck=str(payload.get("deck", "")),
        parameters=dict(payload.get("parameters", {}) or {}),
        reason=str(payload.get("reason", "")),
    )
    validate_deck_operation(operation)
    return operation


def deck_operations_from_payload(values: Any) -> list[ProgramDeckOperation]:
    operations: list[ProgramDeckOperation] = []
    for value in list(values or []):
        if isinstance(value, ProgramDeckOperation):
            operation = value
            validate_deck_operation(operation)
        elif isinstance(value, Mapping):
            operation = deck_operation_from_dict(value)
        else:
            raise ValueError("Deck-operation payload must contain mappings.")
        operations.append(operation)
    return operations


def structural_deck_runtime_status() -> dict[str, Any]:
    return {
        "schema_version": DECK_OPERATION_VERSION,
        "deterministic_ir": True,
        "supported_operations": sorted(_ALLOWED_KINDS),
        "phases": sorted(_ALLOWED_PHASES),
        "learned_code_manipulates_samples": False,
        "renderer_executes_bound_coordinates": True,
    }
