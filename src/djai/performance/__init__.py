"""Immutable DJ performance programs and deterministic execution helpers."""

from .engine import ProgramExecutionResult, execute_transition_program
from .operations import (
    DECK_OPERATION_VERSION,
    ProgramDeckOperation,
    deck_operation_from_dict,
    structural_deck_runtime_status,
    validate_deck_operation,
)
from .program import (
    PERFORMANCE_PROGRAM_VERSION,
    CurvePoint,
    ProgramProcessor,
    TransitionProgram,
    attach_performance_program,
    compile_transition_program,
    transition_program_from_dict,
    validate_performance_program,
)

__all__ = [
    "PERFORMANCE_PROGRAM_VERSION",
    "DECK_OPERATION_VERSION",
    "ProgramDeckOperation",
    "CurvePoint",
    "ProgramProcessor",
    "TransitionProgram",
    "ProgramExecutionResult",
    "attach_performance_program",
    "compile_transition_program",
    "transition_program_from_dict",
    "validate_performance_program",
    "execute_transition_program",
    "deck_operation_from_dict",
    "validate_deck_operation",
    "structural_deck_runtime_status",
]
