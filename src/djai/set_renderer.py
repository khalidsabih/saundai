from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .audio import load_audio, write_audio
from .brain.authority import record_execution_result, record_runtime_feedback, validate_set_commit_receipt
from .decision import validate_execution_plan
from .execution import TransitionRenderContext, normalization_gain
from .models import TransitionPlan
from .performance.executor import DeterministicSetExecutor
from .performance.program import transition_program_from_dict
from .measurement import measure_transition_audio
from .renderer import (
    DEFAULT_RENDER_SAMPLE_RATE,
    _edge_fades,
    _fade_in,
    _fade_out,
    _mix_transition,
    _render_segment,
    _render_tempo_release,
    _render_transition_window,
    _smoothstep,
    _soft_limit,
)
from .set_models import SetPlan, TransitionEdge
from .version import DJAI_VERSION

ProgressCallback = Callable[[str, int, int, str], None]


@contextmanager
def _atomic_soundfile(path: Path, *, sample_rate: int):
    """Publish a completed WAV only after the entire render succeeds."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with sf.SoundFile(
            temporary,
            mode="w",
            samplerate=sample_rate,
            channels=2,
            subtype="PCM_24",
            format="WAV",
        ) as handle:
            yield handle
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class RenderTrackState:
    track_index: int
    rate: float
    effective_bpm: float
    gain: float


@dataclass(slots=True)
class RenderEdgeState:
    edge: TransitionEdge
    plan: TransitionPlan
    from_state: RenderTrackState
    to_state: RenderTrackState


def _render_pre_transition_body(
    audio: np.ndarray,
    *,
    sample_rate: int,
    source_cursor: float,
    exit_time: float,
    rate: float,
    gain: float,
    plan: TransitionPlan,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Execute signed pre-transition deck operations and return the audible body."""

    program = transition_program_from_dict(dict(plan.performance_program or {}))
    operations = [
        operation
        for operation in program.deck_operations
        if str(operation.phase) == "pre_transition" and operation.deck == "a"
    ]
    cursor = float(source_cursor)
    chunks: list[np.ndarray] = []
    events: list[dict[str, object]] = []

    def render_until(target: float) -> None:
        nonlocal cursor
        bounded = max(cursor, min(float(target), float(exit_time)))
        if bounded <= cursor + 1e-8:
            return
        chunks.append(
            _render_segment(
                audio, sample_rate, cursor, bounded, rate=rate, gain=gain
            )
        )
        cursor = bounded

    ordered = sorted(
        operations,
        key=lambda operation: float(
            operation.parameters.get(
                "trigger_source_seconds",
                operation.parameters.get("source_start_seconds", source_cursor),
            )
        ),
    )
    for operation in ordered:
        kind = str(operation.kind)
        params = operation.parameters
        if kind == "WAIT":
            events.append(operation.to_dict())
            continue
        if kind == "LOOP_OUTGOING":
            start = float(params.get("source_start_seconds", cursor))
            end = float(params.get("source_end_seconds", start))
            repeats = max(1, min(8, int(params.get("repeats", 1))))
            if start + 1e-7 < cursor or end > exit_time + 1e-7 or end <= start:
                raise RuntimeError("Compiled outgoing loop is outside the executable body window.")
            render_until(start)
            loop = _render_segment(audio, sample_rate, start, end, rate=rate, gain=gain)
            if loop.shape[-1]:
                chunks.append(np.tile(loop, (1, repeats)))
            cursor = end
            events.append(operation.to_dict())
            continue
        if kind in {"BEAT_JUMP", "HOT_CUE_JUMP", "SKIP_BREAKDOWN"}:
            trigger = float(params.get("trigger_source_seconds", cursor))
            target = float(params.get("target_source_seconds", trigger))
            if trigger + 1e-7 < cursor or trigger > exit_time + 1e-7:
                raise RuntimeError(f"Compiled {kind} trigger is outside the executable body window.")
            if target + 1e-7 < trigger or target > exit_time + 1e-7:
                raise RuntimeError(f"Compiled {kind} target must jump forward within the outgoing body.")
            render_until(trigger)
            cursor = target
            events.append(operation.to_dict())
            continue

    render_until(exit_time)
    if not chunks:
        body = np.zeros((audio.shape[0], 0), dtype=np.float32)
    else:
        body = np.concatenate(chunks, axis=-1).astype(np.float32)
    return body, events


def _track_gain(loudness_dbfs: float, target_dbfs: float = -13.0) -> float:
    # Kept as a private compatibility alias; normalization policy lives in the
    # shared execution contract so rehearsal and final rendering cannot drift.
    return normalization_gain(loudness_dbfs, target_dbfs)


def prepare_render_edge_state(
    plan: SetPlan,
    edge: TransitionEdge,
    current: RenderTrackState,
) -> RenderEdgeState:
    """Validate one immutable transition against the current deck state.

    This helper is used by continuous live execution after DJ Executive has
    replaced only the FUTURE schedule. It deliberately validates the same
    render-rate/BPM continuity as prepared-set rendering without requiring the
    original full-set commit receipt to describe the newly committed suffix.
    """

    if int(edge.from_index) != int(current.track_index):
        raise ValueError("Transition source track does not match the current render state.")
    selected = edge.plan
    validate_execution_plan(selected)
    if abs(selected.render_rate_a - current.rate) > 1e-5:
        raise ValueError("Compiled outgoing render rate is inconsistent with the current deck state.")
    if abs(selected.effective_bpm_a - current.effective_bpm) > 0.02:
        raise ValueError("Compiled outgoing BPM is inconsistent with the current deck state.")
    next_analysis = plan.tracks[edge.to_index]
    next_state = RenderTrackState(
        track_index=edge.to_index,
        rate=selected.post_transition_rate_b,
        effective_bpm=(selected.post_transition_bpm_b or next_analysis.bpm * selected.post_transition_rate_b),
        gain=_track_gain(next_analysis.loudness_dbfs),
    )
    return RenderEdgeState(edge, selected, current, next_state)


def prepare_render_sequence(plan: SetPlan) -> tuple[list[RenderTrackState], list[RenderEdgeState]]:
    """Validate and expose the immutable execution states stored in a set plan."""
    # Authoritative V6 plans fail closed if the Executive receipt no longer
    # matches the exact order and immutable transition signatures presented to
    # the deterministic executor. Legacy/observe plans remain compatible.
    validate_set_commit_receipt(plan)
    if not plan.order:
        raise ValueError("Set plan contains no tracks.")
    if len(plan.transitions) != max(0, len(plan.order) - 1):
        raise ValueError("Set plan transition count does not match its track order.")

    if not plan.transitions:
        first = plan.tracks[plan.order[0]]
        return [RenderTrackState(plan.order[0], 1.0, first.bpm, _track_gain(first.loudness_dbfs))], []

    states: list[RenderTrackState] = []
    edge_states: list[RenderEdgeState] = []
    first_plan = plan.transitions[0].plan
    validate_execution_plan(first_plan)
    first_state = RenderTrackState(
        track_index=plan.order[0],
        rate=first_plan.render_rate_a,
        effective_bpm=first_plan.effective_bpm_a,
        gain=_track_gain(plan.tracks[plan.order[0]].loudness_dbfs),
    )
    states.append(first_state)

    for position, edge in enumerate(plan.transitions):
        expected_from = plan.order[position]
        expected_to = plan.order[position + 1]
        if edge.from_index != expected_from or edge.to_index != expected_to:
            raise ValueError("Set plan transition endpoints do not match the ordered track list.")
        current = states[-1]
        edge_state = prepare_render_edge_state(plan, edge, current)
        states.append(edge_state.to_state)
        edge_states.append(edge_state)
    return states, edge_states


def _stem_transition_audio(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    edge_state: RenderEdgeState,
    sample_rate: int,
) -> tuple[np.ndarray, float, float]:
    plan = edge_state.plan
    target_length = max(1, int(round(plan.transition_seconds * sample_rate)))
    try:
        vocals_a_audio, _ = load_audio(plan.stem_paths_a["vocals"], sample_rate=sample_rate, mono=False)
        vocals_b_audio, _ = load_audio(plan.stem_paths_b["vocals"], sample_rate=sample_rate, mono=False)
        drums_b_audio, _ = load_audio(plan.stem_paths_b["drums"], sample_rate=sample_rate, mono=False)
        bass_b_audio, _ = load_audio(plan.stem_paths_b["bass"], sample_rate=sample_rate, mono=False)
        other_b_audio, _ = load_audio(plan.stem_paths_b["other"], sample_rate=sample_rate, mono=False)

        part_a = _render_segment(
            audio_a, sample_rate, plan.exit_time_a, plan.source_end_a,
            rate=plan.render_rate_a, gain=edge_state.from_state.gain,
            target_length=target_length,
        )
        vocals_a = _render_segment(
            vocals_a_audio, sample_rate, plan.exit_time_a, plan.source_end_a,
            rate=plan.render_rate_a, gain=edge_state.from_state.gain,
            target_length=target_length,
        )
        vocals_b = _render_segment(
            vocals_b_audio, sample_rate, plan.source_start_b, plan.source_end_b,
            rate=plan.render_rate_b, gain=edge_state.to_state.gain,
            target_length=target_length,
        )
        b_instrumental = np.zeros_like(part_a)
        for stem_audio in (drums_b_audio, bass_b_audio, other_b_audio):
            b_instrumental += _render_segment(
                stem_audio, sample_rate, plan.source_start_b, plan.source_end_b,
                rate=plan.render_rate_b, gain=edge_state.to_state.gain,
                target_length=target_length,
            )
        base = _mix_transition(
            part_a - 0.65 * vocals_a,
            b_instrumental,
            sample_rate=sample_rate,
            bpm=plan.target_bpm,
            technique="eq_bass_swap",
        )
        vocal_a_out = 1.0 - _smoothstep(target_length, 0.58, 0.92)
        vocal_b_in = _smoothstep(target_length, 0.72, 0.96)
        mixed = base + 0.78 * vocals_a * vocal_a_out + 0.65 * vocals_b * vocal_b_in
        return _soft_limit(mixed), plan.source_duration_a, plan.source_duration_b
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "The compiled stem transition could not be executed. "
            "Rebuild the stems or re-plan with a stereo technique."
        ) from exc


def _transition_audio(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    edge_state: RenderEdgeState,
    sample_rate: int,
) -> tuple[np.ndarray, float, float]:
    plan = edge_state.plan
    mixed, consumed_a, consumed_b, _ = _render_transition_window(
        audio_a,
        audio_b,
        plan,
        sample_rate=sample_rate,
        gain_a=edge_state.from_state.gain,
        gain_b=edge_state.to_state.gain,
    )
    return mixed, consumed_a, consumed_b


def _validate_rehearsal_render_contract(
    edges: list[RenderEdgeState],
    sample_rate: int,
) -> None:
    for edge_state in edges:
        context = dict(edge_state.plan.metadata.get("render_context", {}) or {})
        if not bool(context.get("exact_set_context", False)):
            continue
        planned_rate = int(context.get("sample_rate", sample_rate))
        if planned_rate != int(sample_rate):
            raise ValueError(
                "Final set sample rate differs from the rate used for rendered-audio rehearsal "
                f"({planned_rate} != {sample_rate}); re-plan at the target render rate."
            )
        planned_gain_a = context.get("gain_a")
        planned_gain_b = context.get("gain_b")
        if planned_gain_a is not None and abs(float(planned_gain_a) - edge_state.from_state.gain) > 1e-6:
            raise ValueError("Outgoing normalization gain differs from the rehearsed render context; re-plan.")
        if planned_gain_b is not None and abs(float(planned_gain_b) - edge_state.to_state.gain) > 1e-6:
            raise ValueError("Incoming normalization gain differs from the rehearsed render context; re-plan.")


def _verify_final_transition_audio(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    transition: np.ndarray,
    edge_state: RenderEdgeState,
    sample_rate: int,
) -> dict[str, object]:
    """Measure the exact transition samples that are about to enter the set."""

    plan = edge_state.plan
    recorded_context = dict(plan.metadata.get("render_context", {}) or {})
    context_seconds = max(0.5, min(8.0, float(recorded_context.get("context_seconds", 2.0))))
    pre_source = context_seconds * edge_state.from_state.rate
    pre = _soft_limit(
        _render_segment(
            audio_a,
            sample_rate,
            max(0.0, plan.exit_time_a - pre_source),
            plan.exit_time_a,
            rate=edge_state.from_state.rate,
            gain=edge_state.from_state.gain,
            target_length=int(round(context_seconds * sample_rate)),
        )
    )
    release_seconds = min(context_seconds, max(0.0, plan.tempo_release_seconds_b))
    release, _ = _render_tempo_release(
        audio_b,
        sample_rate,
        plan,
        gain=edge_state.to_state.gain,
        output_seconds=release_seconds,
    )
    remaining = max(0.0, context_seconds - release_seconds)
    if remaining > 0.0:
        post_source = remaining * edge_state.to_state.rate
        native = _render_segment(
            audio_b,
            sample_rate,
            plan.post_transition_source_end_b,
            plan.post_transition_source_end_b + post_source,
            rate=edge_state.to_state.rate,
            gain=edge_state.to_state.gain,
            target_length=int(round(remaining * sample_rate)),
        )
        post = _soft_limit(np.concatenate([release, native], axis=-1))
    else:
        post = _soft_limit(release)
    measurement = measure_transition_audio(
        pre=pre,
        transition=transition,
        post=post,
        sample_rate=sample_rate,
        plan=plan,
    )
    if measurement.hard_failures:
        raise RuntimeError(
            "Final-set transition failed technical verification before commit: "
            + "; ".join(measurement.hard_failures)
        )
    rehearsal = plan.measurement
    comparison: dict[str, float] = {}
    if rehearsal is not None:
        comparison = {
            "overall_delta": round(float(measurement.overall - rehearsal.overall), 6),
            "peak_delta": round(float(measurement.peak - rehearsal.peak), 6),
            "clipping_fraction_delta": round(
                float(measurement.clipping_fraction - rehearsal.clipping_fraction), 6
            ),
            "loudness_jump_delta_db": round(
                float(measurement.loudness_jump_db - rehearsal.loudness_jump_db), 6
            ),
            "low_frequency_buildup_delta_db": round(
                float(measurement.low_frequency_buildup_db - rehearsal.low_frequency_buildup_db), 6
            ),
            "phase_correlation_delta": round(
                float(measurement.phase_correlation - rehearsal.phase_correlation), 6
            ),
            "silence_fraction_delta": round(
                float(measurement.silence_fraction - rehearsal.silence_fraction), 6
            ),
        }
    return {
        "passed": True,
        "sample_rate": sample_rate,
        "context_seconds": context_seconds,
        "execution_signature": plan.execution_signature,
        "measurement": measurement.to_dict(),
        "rehearsal_comparison": comparison,
        "exact_transition_samples": True,
    }


def render_set(
    plan: SetPlan,
    output_path: str | Path,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    report_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, Path]:
    states, edges = prepare_render_sequence(plan)
    _validate_rehearsal_render_contract(edges, sample_rate)
    executor = DeterministicSetExecutor(
        plan,
        sample_rate=sample_rate,
        transitions=plan.transitions,
    )
    executor.arm()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline: list[dict[str, object]] = []
    final_verifications: list[dict[str, object]] = []
    set_time = 0.0
    source_cursor = 0.0
    maximum_peak = 0.0

    current_index = plan.order[0]
    initial_load_command = executor.begin_load_track()
    current_audio, loaded_sr = load_audio(
        plan.tracks[current_index].path,
        sample_rate=sample_rate,
        mono=False,
    )
    executor.complete_load_track(
        initial_load_command,
        loaded_sample_rate=loaded_sr,
        set_time=set_time,
    )

    with _atomic_soundfile(output, sample_rate=sample_rate) as handle:
        for position, edge_state in enumerate(edges):
            if progress:
                progress(
                    "render",
                    position,
                    len(edges) + 1,
                    f"{plan.tracks[edge_state.edge.from_index].title} -> {plan.tracks[edge_state.edge.to_index].title}",
                )
            state = edge_state.from_state
            transition_plan = edge_state.plan
            body_command = executor.begin_body(
                edge_position=position,
                source_cursor=source_cursor,
            )
            if transition_plan.exit_time_a + 1e-6 < source_cursor:
                raise RuntimeError(
                    "A compiled exit occurs before the already-consumed source position. Re-plan the set."
                )
            body, body_operations = _render_pre_transition_body(
                current_audio,
                sample_rate=sample_rate,
                source_cursor=source_cursor,
                exit_time=transition_plan.exit_time_a,
                rate=state.rate,
                gain=state.gain,
                plan=transition_plan,
            )
            body = _soft_limit(body)
            if position == 0:
                body = _fade_in(body, sample_rate)
            executor.observe_audio(body, body_command)
            if body.shape[-1]:
                maximum_peak = max(maximum_peak, float(np.max(np.abs(body))))
                handle.write(body.T)
            body_duration = body.shape[-1] / sample_rate
            timeline.append(
                {
                    "type": "body",
                    "track_index": edge_state.edge.from_index,
                    "track": plan.tracks[edge_state.edge.from_index].path,
                    "set_start": round(set_time, 6),
                    "set_end": round(set_time + body_duration, 6),
                    "source_start": round(source_cursor, 6),
                    "source_end": round(transition_plan.exit_time_a, 6),
                    "rate": round(state.rate, 6),
                    "rendered_bpm": round(state.effective_bpm, 4),
                    "deck_operations": body_operations,
                }
            )
            for operation in body_operations:
                timeline.append(
                    {
                        "type": "deck_operation",
                        "track_index": edge_state.edge.from_index,
                        "track": plan.tracks[edge_state.edge.from_index].path,
                        "set_time": round(set_time, 6),
                        "operation": operation,
                    }
                )
            set_time += body_duration
            executor.complete_body(
                body_command,
                source_cursor_after=transition_plan.exit_time_a,
                set_time=set_time,
                frames=body.shape[-1],
            )

            load_command = executor.begin_load_track(edge_position=position)
            next_audio, next_sr = load_audio(
                plan.tracks[edge_state.edge.to_index].path,
                sample_rate=sample_rate,
                mono=False,
            )
            executor.complete_load_track(
                load_command,
                loaded_sample_rate=next_sr,
                set_time=set_time,
            )
            transition_command = executor.begin_transition(edge_position=position)
            transition, consumed_a, consumed_b = _transition_audio(
                current_audio,
                next_audio,
                edge_state,
                sample_rate,
            )
            verification = _verify_final_transition_audio(
                current_audio,
                next_audio,
                transition,
                edge_state,
                sample_rate,
            )
            final_verifications.append(verification)
            executor.observe_audio(transition, transition_command)
            maximum_peak = max(maximum_peak, float(np.max(np.abs(transition))))
            handle.write(transition.T)
            transition_duration = transition.shape[-1] / sample_rate
            timeline.append(
                {
                    "type": "transition",
                    "from_index": edge_state.edge.from_index,
                    "to_index": edge_state.edge.to_index,
                    "set_start": round(set_time, 6),
                    "set_end": round(set_time + transition_duration, 6),
                    "exit_time_a": transition_plan.exit_time_a,
                    "entry_time_b": transition_plan.entry_time_b,
                    "intent": transition_plan.intent,
                    "technique": transition_plan.technique,
                    "style": transition_plan.style,
                    "quality": edge_state.edge.quality,
                    "target_bpm": transition_plan.target_bpm,
                    "rate_a": state.rate,
                    "matched_rate_b": transition_plan.render_rate_b,
                    "post_transition_rate_b": edge_state.to_state.rate,
                    "phase_nudge_seconds_b": transition_plan.phase_nudge_seconds_b,
                    "source_consumed_a": round(consumed_a, 6),
                    "source_consumed_b": round(consumed_b, 6),
                    "execution_signature": transition_plan.execution_signature,
                }
            )
            set_time += transition_duration
            executor.complete_transition(
                transition_command,
                source_cursor_after=transition_plan.source_end_b,
                set_time=set_time,
                frames=transition.shape[-1],
                verified=bool(verification.get("passed", False)),
            )

            release_command = executor.begin_tempo_release(edge_position=position)
            tempo_release, release_source = _render_tempo_release(
                next_audio,
                sample_rate,
                transition_plan,
                gain=edge_state.to_state.gain,
            )
            tempo_release = _soft_limit(tempo_release)
            executor.observe_audio(tempo_release, release_command)
            if tempo_release.shape[-1]:
                maximum_peak = max(maximum_peak, float(np.max(np.abs(tempo_release))))
                handle.write(tempo_release.T)
                release_duration = tempo_release.shape[-1] / sample_rate
                timeline.append(
                    {
                        "type": "tempo_release",
                        "track_index": edge_state.edge.to_index,
                        "set_start": round(set_time, 6),
                        "set_end": round(set_time + release_duration, 6),
                        "source_start": round(transition_plan.source_end_b, 6),
                        "source_end": round(transition_plan.source_end_b + release_source, 6),
                        "start_rate": round(transition_plan.tempo_release_start_rate_b, 6),
                        "end_rate": round(transition_plan.tempo_release_end_rate_b, 6),
                        "start_bpm": round(transition_plan.source_bpm_b * transition_plan.tempo_release_start_rate_b, 4),
                        "end_bpm": round(transition_plan.post_transition_bpm_b, 4),
                        "pitch_preserving": True,
                        "tempo_owner": "deck_b",
                    }
                )
                set_time += release_duration
            executor.complete_tempo_release(
                release_command,
                source_cursor_after=transition_plan.post_transition_source_end_b,
                set_time=set_time,
                frames=tempo_release.shape[-1],
            )
            source_cursor = transition_plan.post_transition_source_end_b
            current_audio = next_audio
            current_index = edge_state.edge.to_index
            if progress:
                progress("render", position + 1, len(edges) + 1, "transition complete")

        final_state = states[-1]
        final_analysis = plan.tracks[current_index]
        final_body_command = executor.begin_final_body(source_cursor=source_cursor)
        final_body = _render_segment(
            current_audio,
            sample_rate,
            source_cursor,
            final_analysis.duration_seconds,
            rate=final_state.rate,
            gain=final_state.gain,
        )
        final_body = _fade_out(_soft_limit(final_body), sample_rate)
        executor.observe_audio(final_body, final_body_command)
        if final_body.shape[-1]:
            maximum_peak = max(maximum_peak, float(np.max(np.abs(final_body))))
            handle.write(final_body.T)
        final_duration = final_body.shape[-1] / sample_rate
        timeline.append(
            {
                "type": "body",
                "track_index": current_index,
                "track": final_analysis.path,
                "set_start": round(set_time, 6),
                "set_end": round(set_time + final_duration, 6),
                "source_start": round(source_cursor, 6),
                "source_end": final_analysis.duration_seconds,
                "rate": round(final_state.rate, 6),
                "rendered_bpm": round(final_state.effective_bpm, 4),
            }
        )
        set_time += final_duration
        executor.complete_final_body(
            final_body_command,
            source_cursor_after=final_analysis.duration_seconds,
            set_time=set_time,
            frames=final_body.shape[-1],
        )

    executor.finish(set_time=set_time)
    executor_snapshot = executor.to_dict()
    runtime_feedback = record_runtime_feedback(
        plan,
        executor_snapshot=executor_snapshot,
        mode="offline_full_set",
    )

    if progress:
        progress("render", len(edges) + 1, len(edges) + 1, "set complete")

    execution_result = record_execution_result(
        plan,
        duration_seconds=set_time,
        peak=maximum_peak,
        verified_transitions=len(final_verifications),
    )

    report = {
        "djai_version": DJAI_VERSION,
        "render": {
            "output_wav": str(output),
            "sample_rate": sample_rate,
            "channels": 2,
            "duration_seconds": round(set_time, 6),
            "peak": round(maximum_peak, 6),
            "mode": "offline_full_set",
            "immutable_plan_execution": True,
            "final_transition_verification": {
                "enabled": True,
                "passed": len(final_verifications) == len(edges),
                "verified_transitions": len(final_verifications),
            },
        },
        "final_transition_verifications": final_verifications,
        "executive_execution_result": execution_result,
        "executive_runtime_feedback": runtime_feedback,
        "deterministic_executor": executor_snapshot,
        "set_plan": plan.to_dict(include_analysis=False),
        "rendered_tracks": [
            {
                "track_index": state.track_index,
                "path": plan.tracks[state.track_index].path,
                "native_bpm": plan.tracks[state.track_index].bpm,
                "rendered_bpm": round(state.effective_bpm, 4),
                "time_stretch_rate": round(state.rate, 6),
                "gain": round(state.gain, 6),
            }
            for state in states
        ],
        "timeline": timeline,
    }
    report_output = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else output.with_suffix(".json")
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(report_output, json.dumps(report, indent=2) + "\n")
    return output, report_output


def render_set_previews(
    plan: SetPlan,
    output_dir: str | Path,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    pre_roll_seconds: float = 8.0,
    post_roll_seconds: float = 8.0,
    progress: ProgressCallback | None = None,
) -> tuple[list[Path], Path]:
    states, edges = prepare_render_sequence(plan)
    _validate_rehearsal_render_contract(edges, sample_rate)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    index_items: list[dict[str, object]] = []

    if progress:
        progress("render-preview", 0, max(1, len(edges)), "Preparing transition previews")

    for position, edge_state in enumerate(edges, start=1):
        analysis_a = plan.tracks[edge_state.edge.from_index]
        analysis_b = plan.tracks[edge_state.edge.to_index]
        audio_a, _ = load_audio(analysis_a.path, sample_rate=sample_rate, mono=False)
        audio_b, _ = load_audio(analysis_b.path, sample_rate=sample_rate, mono=False)
        pre_source = pre_roll_seconds * edge_state.from_state.rate
        pre = _soft_limit(
            _render_segment(
                audio_a,
                sample_rate,
                max(0.0, edge_state.plan.exit_time_a - pre_source),
                edge_state.plan.exit_time_a,
                rate=edge_state.from_state.rate,
                gain=edge_state.from_state.gain,
                target_length=int(round(pre_roll_seconds * sample_rate)),
            )
        )
        transition, _, consumed_b = _transition_audio(audio_a, audio_b, edge_state, sample_rate)
        del consumed_b
        release_preview_seconds = min(post_roll_seconds, max(0.0, edge_state.plan.tempo_release_seconds_b))
        release, _ = _render_tempo_release(
            audio_b, sample_rate, edge_state.plan, gain=edge_state.to_state.gain, output_seconds=release_preview_seconds
        )
        remaining = max(0.0, post_roll_seconds - release_preview_seconds)
        if remaining > 0.0:
            post_source = remaining * edge_state.to_state.rate
            native = _render_segment(
                audio_b,
                sample_rate,
                edge_state.plan.post_transition_source_end_b,
                edge_state.plan.post_transition_source_end_b + post_source,
                rate=edge_state.to_state.rate,
                gain=edge_state.to_state.gain,
                target_length=int(round(remaining * sample_rate)),
            )
            post = _soft_limit(np.concatenate([release, native], axis=-1))
        else:
            post = _soft_limit(release)
        preview = _edge_fades(
            np.concatenate([pre, transition, post], axis=-1),
            sample_rate,
        )
        filename = (
            f"{position:02d}-{analysis_a.title[:36]}-to-{analysis_b.title[:36]}.wav".replace("/", "-")
        )
        output = write_audio(root / filename, preview, sample_rate)
        outputs.append(output)
        index_items.append(
            {
                "index": position,
                "from": analysis_a.path,
                "to": analysis_b.path,
                "output_wav": str(output),
                "execution_signature": edge_state.plan.execution_signature,
                "pre_roll_seconds": pre_roll_seconds,
                "post_roll_seconds": post_roll_seconds,
                "transition_start_seconds": pre_roll_seconds,
                "transition_end_seconds": pre_roll_seconds + edge_state.plan.transition_seconds,
                "plan": edge_state.plan.to_dict(),
            }
        )
        if progress:
            progress(
                "render-preview",
                position,
                max(1, len(edges)),
                f"{analysis_a.title} -> {analysis_b.title}",
            )

    report = root / "index.json"
    report.write_text(
        json.dumps({"plan_id": plan.plan_id, "previews": index_items}, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs, report
