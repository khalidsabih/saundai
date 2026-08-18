from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..models import TrackAnalysis, TransitionPlan
from ..music.timeline import MusicalTimeline
from ..performance.operations import ProgramDeckOperation

STRUCTURAL_MOVE_VERSION = "1.7.2"


def _enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class StructuralOperationConfig:
    enabled: bool = True
    outgoing_loop: bool = False
    incoming_intro_extension: bool = False
    stem_hold: bool = False
    jump_operations: bool = False
    minimum_loop_wait_bars: int = 20

    @classmethod
    def from_environment(cls) -> "StructuralOperationConfig":
        try:
            minimum_loop_wait_bars = int(os.environ.get("DJAI_STRUCTURAL_MIN_LOOP_WAIT_BARS", "20"))
        except ValueError:
            minimum_loop_wait_bars = 20
        return cls(
            enabled=_enabled("DJAI_STRUCTURAL_DECK_OPS", "1"),
            outgoing_loop=_enabled("DJAI_STRUCTURAL_LOOP_OUTGOING", "0"),
            incoming_intro_extension=_enabled("DJAI_STRUCTURAL_EXTEND_INTRO", "1"),
            stem_hold=_enabled("DJAI_STRUCTURAL_HOLD_STEM", "0"),
            jump_operations=_enabled("DJAI_STRUCTURAL_JUMPS", "0"),
            minimum_loop_wait_bars=max(8, min(64, minimum_loop_wait_bars)),
        )


def compile_move_deck_operations(
    *,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    source_cursor_seconds: float,
    source_bar: int,
    exit_bar: int,
    entry_bar: int,
    wait_bars: int,
    timeline_a: MusicalTimeline,
    timeline_b: MusicalTimeline,
    stem_profile: str | None,
    wait_threshold_bars: int,
    config: StructuralOperationConfig | None = None,
) -> tuple[ProgramDeckOperation, ...]:
    """Compile a complete move into bounded deterministic deck operations.

    WAIT is always represented explicitly when the move deliberately delays a
    transition. More invasive structural edits are capability-gated so a new
    release can expose/test the tools without silently rewriting users' tracks.
    """

    cfg = config or StructuralOperationConfig.from_environment()
    operations: list[ProgramDeckOperation] = []
    if wait_bars >= max(1, wait_threshold_bars):
        operations.append(
            ProgramDeckOperation(
                kind="WAIT",
                phase="pre_transition",
                deck="a",
                parameters={
                    "bars": int(wait_bars),
                    "source_start_bar": int(source_bar),
                    "source_end_bar": int(exit_bar),
                    "source_start_seconds": round(float(source_cursor_seconds), 6),
                    "source_end_seconds": round(float(plan.exit_time_a), 6),
                },
                reason="preserve the stronger phrase/payoff before the handoff",
            )
        )
    if not cfg.enabled:
        return tuple(operations)

    if cfg.outgoing_loop and wait_bars >= cfg.minimum_loop_wait_bars:
        # The loop is intentionally placed well before the transition so the
        # final rehearsal context remains the same raw phrase the compiler
        # measured. It extends a stable low-vocal phrase, then releases back
        # into the original timeline before the selected exit.
        loop_end_bar = int(exit_bar) - 8
        loop_start_bar = loop_end_bar - 4
        moment = timeline_a.phrase_for_bar(loop_start_bar)
        if (
            loop_start_bar >= int(source_bar) + 4
            and loop_end_bar <= int(exit_bar) - 4
            and moment is not None
            and moment.rhythm >= 0.55
            and moment.vocal_probability <= 0.42
            and moment.importance <= 0.72
        ):
            start_seconds = timeline_a.bar_start_seconds(loop_start_bar)
            end_seconds = timeline_a.bar_start_seconds(loop_end_bar)
            if end_seconds > start_seconds + 0.25:
                operations.append(
                    ProgramDeckOperation(
                        kind="LOOP_OUTGOING",
                        phase="pre_transition",
                        deck="a",
                        parameters={
                            "source_start_bar": loop_start_bar,
                            "source_end_bar": loop_end_bar,
                            "source_start_seconds": round(start_seconds, 6),
                            "source_end_seconds": round(end_seconds, 6),
                            "repeats": 2,
                        },
                        reason="extend a stable outgoing phrase without moving the selected exit",
                    )
                )

    entry_phrase = timeline_b.phrase_for_bar(entry_bar)
    if (
        cfg.incoming_intro_extension
        and entry_phrase is not None
        and entry_phrase.role == "intro"
        and entry_phrase.vocal_probability <= 0.35
        and int(plan.transition_bars) >= 8
    ):
        loop_bars = max(1, min(4, entry_phrase.end_bar - max(entry_phrase.start_bar, entry_bar)))
        end_bar = min(entry_phrase.end_bar, entry_bar + loop_bars)
        start_seconds = timeline_b.bar_start_seconds(entry_bar)
        end_seconds = timeline_b.bar_start_seconds(end_bar)
        if end_seconds > start_seconds + 0.20:
            operations.append(
                ProgramDeckOperation(
                    kind="EXTEND_INTRO",
                    phase="transition",
                    deck="b",
                    parameters={
                        "source_start_bar": int(entry_bar),
                        "source_end_bar": int(end_bar),
                        "source_start_seconds": round(start_seconds, 6),
                        "source_end_seconds": round(end_seconds, 6),
                        "repeats": 2,
                        "loop_beats": int(max(1, loop_bars) * max(3, track_b.time_signature or 4)),
                        "hold_fraction": 0.84,
                        "source_fraction": round(
                            min(1.0, (plan.source_duration_b / max(plan.render_rate_b, 1e-9)) / max(plan.transition_seconds, 1e-9)),
                            6,
                        ),
                        "runway_strategy": "loop_prefix_then_release_to_planned_source",
                    },
                    reason="hold an instrumental intro under the handoff before releasing to the planned landing",
                )
            )

    if cfg.stem_hold and stem_profile:
        operations.append(
            ProgramDeckOperation(
                kind="HOLD_STEM",
                phase="transition",
                deck="b",
                parameters={
                    "stem": "drums",
                    "start_position": 0.12,
                    "end_position": 0.42,
                    "minimum_gain": 0.32,
                },
                reason="keep incoming rhythmic ownership stable while other stems hand off",
            )
        )

    # BEAT_JUMP, HOT_CUE_JUMP and SKIP_BREAKDOWN are fully executable in the
    # renderer but are intentionally not auto-generated in 1.7.2. They require
    # explicit cue/section intent from the planner rather than a generic rule.
    return tuple(operations)


def structural_operation_status() -> dict[str, Any]:
    cfg = StructuralOperationConfig.from_environment()
    return {
        "version": STRUCTURAL_MOVE_VERSION,
        "enabled": cfg.enabled,
        "true_wait": True,
        "automatic_outgoing_loop": cfg.outgoing_loop,
        "automatic_intro_extension": cfg.incoming_intro_extension,
        "automatic_stem_hold": cfg.stem_hold,
        "automatic_jumps": cfg.jump_operations,
        "minimum_loop_wait_bars": cfg.minimum_loop_wait_bars,
        "supported": [
            "WAIT",
            "LOOP_OUTGOING",
            "LOOP_INCOMING",
            "EXTEND_INTRO",
            "BEAT_JUMP",
            "HOT_CUE_JUMP",
            "HOLD_STEM",
            "SKIP_BREAKDOWN",
        ],
        "advanced_operations_default": "safe instrumental intro extension is enabled; invasive jumps/outgoing loops remain opt-in",
        "deterministic_compiler_authoritative": True,
    }
