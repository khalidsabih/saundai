from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np

from .audio import load_audio
from .brain.authority import record_runtime_feedback
from .brain.crowd_state import crowd_state_from_feedback
from .brain.live_orchestrator import (
    LiveExecutiveConfig,
    LiveExecutiveController,
    LiveReplanRequest,
)
from .live_telemetry import publish_live_telemetry
from .performance.executor import DeterministicSetExecutor, ExecutorSafetyHold
from .set_models import SetPlan
from .set_planner import replan_live_future_candidate
from .set_renderer import (
    DEFAULT_RENDER_SAMPLE_RATE,
    _fade_in,
    _fade_out,
    _render_segment,
    _render_tempo_release,
    _transition_audio,
    prepare_render_edge_state,
    prepare_render_sequence,
)


def _render_live_hold_loop(
    audio: np.ndarray,
    *,
    sample_rate: int,
    exit_time: float,
    rate: float,
    target_bpm: float,
    time_signature: int,
    bars: int = 4,
) -> np.ndarray:
    """Render a phrase-aligned outgoing loop while Safety holds NEXT.

    The source cursor does not advance: this is an explicit live structural
    hold, not a hidden schedule mutation. A short edge taper suppresses clicks
    when the loop jumps back to its phrase start.
    """
    beats = max(1, int(time_signature or 4)) * max(1, int(bars))
    output_seconds = beats * 60.0 / max(float(target_bpm), 1.0)
    source_seconds = output_seconds * max(float(rate), 1e-9)
    start = max(0.0, float(exit_time) - source_seconds)
    target_length = max(1, int(round(output_seconds * sample_rate)))
    loop = _render_segment(
        audio,
        sample_rate,
        start,
        float(exit_time),
        rate=max(float(rate), 1e-9),
        gain=1.0,
        target_length=target_length,
    )
    taper = min(max(16, int(round(sample_rate * 0.012))), max(1, loop.shape[-1] // 8))
    if taper > 1:
        loop = np.array(loop, copy=True, dtype=np.float32)
        loop[:, :taper] *= np.linspace(0.92, 1.0, taper, dtype=np.float32)
        loop[:, -taper:] *= np.linspace(1.0, 0.92, taper, dtype=np.float32)
    return np.asarray(loop, dtype=np.float32)


def _manifest_payload(
    plan: SetPlan,
    *,
    order: list[int],
    edges: list[Any],
    initial_rate: float,
    dynamic: bool,
    schedule_generation: int = 0,
    initial_schedule: bool = True,
) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    set_time = 0.0
    source_cursor = 0.0
    current_index = order[0]
    current_rate = float(initial_rate)
    for edge_state in edges:
        plan_edge = edge_state.plan
        body_duration = max(0.0, plan_edge.exit_time_a - source_cursor) / max(current_rate, 1e-9)
        if body_duration > 0.0:
            timeline.append(
                {
                    "type": "track",
                    "track_index": current_index,
                    "title": plan.tracks[current_index].title,
                    "set_start": round(set_time, 6),
                    "set_end": round(set_time + body_duration, 6),
                }
            )
            set_time += body_duration
        alternatives = list(edge_state.edge.alternatives)
        backup = next(
            (
                value
                for value in alternatives
                if value.execution_signature != plan_edge.execution_signature and value.style != "emergency"
            ),
            None,
        )
        emergency = next(
            (
                value
                for value in alternatives
                if value.style == "emergency" and value.execution_signature != plan_edge.execution_signature
            ),
            None,
        )
        stem_meta = dict(plan_edge.metadata.get("stem_choreography", {}) or {})
        timeline.append(
            {
                "type": "transition",
                "from_index": edge_state.edge.from_index,
                "to_index": edge_state.edge.to_index,
                "from_title": plan.tracks[edge_state.edge.from_index].title,
                "to_title": plan.tracks[edge_state.edge.to_index].title,
                "set_start": round(set_time, 6),
                "set_end": round(set_time + plan_edge.transition_seconds, 6),
                "technique": plan_edge.technique,
                "intent": plan_edge.intent,
                "fallback_level": plan_edge.fallback_level,
                "target_bpm": round(plan_edge.target_bpm, 4),
                "matched_rate_b": round(plan_edge.render_rate_b, 6),
                "post_transition_rate_b": round(plan_edge.post_transition_rate_b, 6),
                "phase_nudge_seconds_b": round(plan_edge.phase_nudge_seconds_b, 6),
                "stem_profile": stem_meta.get("profile"),
                "stem_confidence": stem_meta.get("quality_confidence"),
                "primary": {
                    "technique": plan_edge.technique,
                    "intent": plan_edge.intent,
                    "execution_signature": plan_edge.execution_signature,
                },
                "backup": (
                    None
                    if backup is None
                    else {
                        "technique": backup.technique,
                        "intent": backup.intent,
                        "execution_signature": backup.execution_signature,
                    }
                ),
                "emergency": (
                    None
                    if emergency is None
                    else {
                        "technique": emergency.technique,
                        "intent": emergency.intent,
                        "execution_signature": emergency.execution_signature,
                    }
                ),
                "backup_techniques": [value.technique for value in alternatives],
            }
        )
        set_time += plan_edge.transition_seconds
        if plan_edge.tempo_release_seconds_b > 0.0:
            timeline.append(
                {
                    "type": "tempo_release",
                    "track_index": edge_state.edge.to_index,
                    "title": plan.tracks[edge_state.edge.to_index].title,
                    "set_start": round(set_time, 6),
                    "set_end": round(set_time + plan_edge.tempo_release_seconds_b, 6),
                    "start_rate": round(plan_edge.tempo_release_start_rate_b, 6),
                    "end_rate": round(plan_edge.tempo_release_end_rate_b, 6),
                    "start_bpm": round(plan_edge.source_bpm_b * plan_edge.tempo_release_start_rate_b, 4),
                    "end_bpm": round(plan_edge.post_transition_bpm_b, 4),
                    "pitch_preserving": True,
                    "tempo_owner": "deck_b",
                }
            )
            set_time += plan_edge.tempo_release_seconds_b
        source_cursor = plan_edge.post_transition_source_end_b
        current_index = edge_state.edge.to_index
        current_rate = edge_state.to_state.rate
    final_duration = max(0.0, plan.tracks[current_index].duration_seconds - source_cursor) / max(current_rate, 1e-9)
    timeline.append(
        {
            "type": "track",
            "track_index": current_index,
            "title": plan.tracks[current_index].title,
            "set_start": round(set_time, 6),
            "set_end": round(set_time + final_duration, 6),
        }
    )
    set_time += final_duration
    return {
        "plan_id": plan.plan_id,
        "duration_seconds": round(set_time, 6),
        "timeline": timeline,
        "runtime_control": {
            "continuous_replanning": dynamic,
            "now_mutable": False,
            "next_mutable": False,
            "future_replannable": dynamic,
            "manifest_is_initial_schedule": bool(initial_schedule and dynamic),
            "telemetry_supported": True,
            "schedule_generation": int(schedule_generation),
        },
        "ordered_tracks": [
            {
                "index": index,
                "title": plan.tracks[index].title,
                "bpm": plan.tracks[index].bpm,
                "key": plan.tracks[index].key.camelot,
            }
            for index in order
        ],
    }


def live_manifest(plan: SetPlan) -> dict[str, Any]:
    """Return the initial exact playback manifest for browser now/next indicators."""
    states, edges = prepare_render_sequence(plan)
    live_cfg = LiveExecutiveConfig.from_environment()
    authoritative = bool(dict(plan.metadata.get("dj_executive", {}) or {}).get("authoritative"))
    dynamic = bool(live_cfg.enabled and authoritative and plan.automatic_delivery)
    return _manifest_payload(
        plan,
        order=list(plan.order),
        edges=list(edges),
        initial_rate=states[0].rate,
        dynamic=dynamic,
        schedule_generation=0,
        initial_schedule=True,
    )


def _runtime_manifest(
    plan: SetPlan,
    runtime: DeterministicSetExecutor,
    *,
    dynamic: bool,
) -> dict[str, Any]:
    """Build a browser schedule from the Executor's currently committed live route.

    The original set receipt is validated first. FUTURE replacements are then
    reconstructed from their individually validated immutable transition plans;
    they are authorized by ExecutiveFutureCommit rather than by mutating the
    original whole-set receipt.
    """
    states, _ = prepare_render_sequence(plan)
    current_state = states[0]
    edge_states: list[Any] = []
    order = list(runtime.active_order)
    if not order or order[0] != plan.order[0]:
        raise ValueError("Live schedule changed the immutable first track.")
    for position, edge in enumerate(runtime.transitions):
        if position + 1 >= len(order):
            raise ValueError("Live schedule transition count does not match active order.")
        if int(edge.from_index) != int(order[position]) or int(edge.to_index) != int(order[position + 1]):
            raise ValueError("Live schedule transition endpoints do not match active order.")
        edge_state = prepare_render_edge_state(plan, edge, current_state)
        edge_states.append(edge_state)
        current_state = edge_state.to_state
    return _manifest_payload(
        plan,
        order=order,
        edges=edge_states,
        initial_rate=states[0].rate,
        dynamic=dynamic,
        schedule_generation=runtime.schedule_generation,
        initial_schedule=False,
    )


def iter_live_chunks(
    plan: SetPlan,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    block_frames: int = 4096,
    telemetry_session_id: str | None = None,
) -> Iterator[np.ndarray]:
    """Stream a prepared set under the continuous Executive control loop.

    NOW and NEXT remain immutable. While NEXT executes, a background planner may
    reconsider FUTURE from the exact projected post-NEXT state. A candidate can
    only be installed at the following safe boundary before it becomes NEXT.
    Late, stale, invalid, or rejected work is discarded and the already
    prepared schedule continues unchanged.
    """

    states, _ = prepare_render_sequence(plan)
    executive_cfg = LiveExecutiveConfig.from_environment()
    authoritative = bool(dict(plan.metadata.get("dj_executive", {}) or {}).get("authoritative"))
    live_replanning = bool(executive_cfg.enabled and authoritative and plan.automatic_delivery)
    if live_replanning != executive_cfg.enabled:
        executive_cfg = LiveExecutiveConfig(
            enabled=live_replanning,
            future_depth=executive_cfg.future_depth,
            replan_every_edges=executive_cfg.replan_every_edges,
            max_replans=executive_cfg.max_replans,
            minimum_future_tracks=executive_cfg.minimum_future_tracks,
        )
    executive = LiveExecutiveController(config=executive_cfg)
    runtime = DeterministicSetExecutor(
        plan,
        sample_rate=sample_rate,
        transitions=plan.transitions,
        continuous_replanning=live_replanning,
    )
    runtime.arm()
    set_time = 0.0
    current_index = runtime.active_order[0]
    current_render_state = states[0]
    initial_load_command = runtime.begin_load_track()
    current_audio, loaded_sr = load_audio(
        plan.tracks[current_index].path,
        sample_rate=sample_rate,
        mono=False,
    )
    runtime.complete_load_track(
        initial_load_command,
        loaded_sample_rate=loaded_sr,
        set_time=set_time,
    )
    source_cursor = 0.0

    feedback_path = os.environ.get("DJAI_CROWD_FEEDBACK_PATH", "ratings/feedback-events.jsonl")
    journey = str(plan.metadata.get("journey", "balanced"))

    def emit(segment: np.ndarray) -> Iterator[np.ndarray]:
        for offset in range(0, segment.shape[-1], block_frames):
            yield np.ascontiguousarray(segment[:, offset : offset + block_frames].T)

    pending_request: LiveReplanRequest | None = None
    pending_future: Future[Any] | None = None
    last_crowd_state: dict[str, Any] = {}
    last_crowd_directive: dict[str, Any] = {}
    last_published_safety_action: str | None = None

    def publish_telemetry(*, status: str = "streaming", detail: str = "") -> None:
        if not telemetry_session_id:
            return
        manifest = _runtime_manifest(plan, runtime, dynamic=live_replanning)
        horizons = runtime.horizons.snapshot()
        safety = runtime.safety.to_dict()
        safety_decisions = list(safety.get("decisions", []))
        last_safety = safety_decisions[-1] if safety_decisions else {}
        executive_snapshot = executive.to_dict()
        executive_events = list(executive_snapshot.get("events", []))
        last_executive_event = executive_events[-1] if executive_events else None
        publish_live_telemetry(
            plan.plan_id,
            telemetry_session_id,
            {
                "detail": detail,
                "schedule_generation": runtime.schedule_generation,
                "active_order": list(runtime.active_order),
                "duration_seconds": manifest["duration_seconds"],
                "timeline": manifest["timeline"],
                "ordered_tracks": manifest["ordered_tracks"],
                "runtime_control": manifest["runtime_control"],
                "playback_state": horizons["playback_state"],
                "horizons": {
                    "now": horizons["now"],
                    "next": horizons["next"],
                    "future": horizons["future"],
                },
                "safety": {
                    "action": safety.get("last_action", "allow"),
                    "reason": last_safety.get("reason", ""),
                    "decision": last_safety or None,
                },
                "executive": {
                    "enabled": bool(executive_snapshot.get("enabled", False)),
                    "cycle": int(executive_snapshot.get("cycle", 0)),
                    "replanning": bool(pending_future is not None and not pending_future.done()),
                    "last_committed_generation": int(executive_snapshot.get("last_committed_generation", 0)),
                    "crowd_state_authority": executive_snapshot.get("crowd_state_authority", "bounded_advisory"),
                    "transition_policy_authority": executive_snapshot.get("transition_policy_authority", "bounded_advisory"),
                    "last_event": last_executive_event,
                },
                "crowd": {
                    "state": dict(last_crowd_state),
                    "directive": dict(last_crowd_directive),
                },
            },
            status=status,
        )

    def publish_safety_change(decision: dict[str, Any], *, detail: str = "") -> None:
        nonlocal last_published_safety_action
        action = str(decision.get("action", "allow"))
        if action != last_published_safety_action:
            publish_telemetry(
                detail=detail or f"Safety Governor: {action} - {decision.get('reason', '')}"
            )
            last_published_safety_action = action

    def discard_request(request: LiveReplanRequest, *, reason: str) -> None:
        executive.discard(request, reason=reason)
        publish_telemetry(detail=reason)

    publish_telemetry(detail="Live executor armed with the initial committed schedule")

    def submit_future_replan(planner_pool: ThreadPoolExecutor) -> None:
        nonlocal pending_request, pending_future, last_crowd_state, last_crowd_directive
        if not live_replanning or pending_future is not None:
            return
        position = runtime.edge_position
        if position < 0 or position >= len(runtime.transitions):
            return
        locked_next = runtime.transitions[position]
        replace_from = position + 1
        if replace_from >= len(runtime.transitions):
            return
        history_transitions = tuple(runtime.transitions[:replace_from])
        history_order = tuple(runtime.active_order[: replace_from + 1])
        remaining = tuple(runtime.active_order[replace_from + 1 :])
        crowd = crowd_state_from_feedback(feedback_path, listener=plan.listener).to_dict()
        request = executive.make_request(
            schedule_generation=runtime.schedule_generation,
            current_edge_position=position,
            locked_next=locked_next,
            history_order=history_order,
            history_transitions=history_transitions,
            remaining_indices=remaining,
            journey=journey,
            listener=plan.listener,
            crowd_state=crowd,
            reason=(
                "safety_pressure"
                if runtime.safety.last_action in {"hold", "use_backup"}
                else "periodic_receding_horizon"
            ),
        )
        if request is None:
            return
        last_crowd_state = dict(crowd)
        last_crowd_directive = dict(request.crowd_directive)
        pending_request = request
        pending_future = planner_pool.submit(
            replan_live_future_candidate,
            plan,
            request,
            history_transitions=history_transitions,
        )
        publish_telemetry(detail="Executive is reconsidering FUTURE while NOW/NEXT remain locked")

    def accept_ready_future() -> None:
        nonlocal pending_request, pending_future
        if pending_request is None or pending_future is None:
            return
        if not pending_future.done():
            discard_request(
                pending_request,
                reason="FUTURE planning missed the safe boundary; keeping the already prepared NEXT schedule",
            )
            # Running ThreadPool work cannot be killed safely. Mark the result
            # stale and let the single planning worker finish in the background.
            return
        request = pending_request
        future = pending_future
        pending_request = None
        pending_future = None
        try:
            candidate = future.result()
        except Exception as exc:
            discard_request(
                request,
                reason=f"FUTURE planner failed: {type(exc).__name__}: {exc}",
            )
            return
        if candidate is None:
            discard_request(request, reason="FUTURE planner found no safer/valid complete route")
            return
        if request.replace_from_edge != runtime.edge_position:
            discard_request(request, reason="FUTURE candidate became stale before reaching its safe boundary")
            return
        current_suffix = tuple(runtime.active_order[runtime.edge_position :])
        current_signatures = tuple(
            str(edge.plan.execution_signature or "")
            for edge in runtime.transitions[runtime.edge_position :]
        )
        candidate_signatures = tuple(
            str(edge.plan.execution_signature or "") for edge in candidate.transitions
        )
        if candidate.order_suffix == current_suffix and candidate_signatures == current_signatures:
            discard_request(request, reason="FUTURE reconsideration confirmed the existing prepared schedule")
            return
        committed = executive.commit_candidate(
            request,
            candidate,
            current_schedule_generation=runtime.schedule_generation,
        )
        if committed is None:
            return
        runtime.replace_future_schedule(committed.commit, committed.transitions)
        publish_telemetry(detail="Executive committed and installed a replacement FUTURE schedule")

    planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="djai-live-executive")
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="djai-live-prefetch") as prefetch_pool:
            submit_future_replan(planner_pool)
            while runtime.edge_position < len(runtime.transitions):
                position = runtime.edge_position
                edge = runtime.transitions[position]
                edge_state = prepare_render_edge_state(plan, edge, current_render_state)
                next_index = edge.to_index
                next_future = prefetch_pool.submit(
                    load_audio,
                    plan.tracks[next_index].path,
                    sample_rate=sample_rate,
                    mono=False,
                )
                body_command = runtime.begin_body(
                    edge_position=position,
                    source_cursor=source_cursor,
                )
                body = _render_segment(
                    current_audio,
                    sample_rate,
                    source_cursor,
                    edge_state.plan.exit_time_a,
                    rate=edge_state.from_state.rate,
                    gain=edge_state.from_state.gain,
                )
                if position == 0:
                    body = _fade_in(body, sample_rate)
                runtime.observe_audio(body, body_command)
                yield from emit(body)
                set_time += body.shape[-1] / sample_rate
                runtime.complete_body(
                    body_command,
                    source_cursor_after=edge_state.plan.exit_time_a,
                    set_time=set_time,
                    frames=body.shape[-1],
                )

                safety_decision = runtime.update_telemetry(prefetch_ready=next_future.done())
                publish_safety_change(safety_decision)

                # HOLD/USE_BACKUP are real audible barriers. Preparation may
                # continue, but A remains the musical owner until B is ready.
                # A phrase-aligned loop supplies continuous program audio instead
                # of blocking the generator and draining the playback buffer.
                hold_loops = 0
                max_hold_loops = max(1, int(os.environ.get("DJAI_LIVE_HOLD_MAX_LOOPS", "64")))
                while not next_future.done():
                    hold_loops += 1
                    if hold_loops > max_hold_loops:
                        raise RuntimeError("Safety HOLD exceeded the configured live phrase-loop budget before NEXT became ready.")
                    hold = _render_live_hold_loop(
                        current_audio,
                        sample_rate=sample_rate,
                        exit_time=edge_state.plan.exit_time_a,
                        rate=edge_state.from_state.rate,
                        target_bpm=edge_state.plan.target_bpm,
                        time_signature=plan.tracks[current_index].time_signature,
                        bars=4,
                    )
                    yield from emit(hold)
                    set_time += hold.shape[-1] / sample_rate
                    runtime.record_hold_audio(
                        set_time=set_time,
                        frames=hold.shape[-1],
                        detail="Safety HOLD kept the outgoing musical owner alive with a phrase-aligned loop while NEXT prepared.",
                    )
                    publish_telemetry(detail=f"Safety HOLD: outgoing phrase loop {hold_loops}; NEXT is still preparing")

                load_command = runtime.begin_load_track(edge_position=position)
                next_audio, next_sr = next_future.result()
                safety_decision = runtime.update_telemetry(prefetch_ready=True)
                publish_safety_change(safety_decision, detail="Safety Governor: ALLOW - NEXT material is prepared")
                runtime.complete_load_track(
                    load_command,
                    loaded_sample_rate=next_sr,
                    set_time=set_time,
                )
                try:
                    transition_command = runtime.begin_transition(edge_position=position)
                except ExecutorSafetyHold as hold_error:
                    publish_safety_change(hold_error.decision)
                    raise RuntimeError("Safety Governor still blocks the audible transition after NEXT preparation.") from hold_error
                transition, _, _ = _transition_audio(
                    current_audio,
                    next_audio,
                    edge_state,
                    sample_rate,
                )
                runtime.observe_audio(transition, transition_command)
                yield from emit(transition)
                set_time += transition.shape[-1] / sample_rate
                runtime.complete_transition(
                    transition_command,
                    source_cursor_after=edge_state.plan.source_end_b,
                    set_time=set_time,
                    frames=transition.shape[-1],
                    verified=True,
                )
                release_command = runtime.begin_tempo_release(edge_position=position)
                release, _ = _render_tempo_release(
                    next_audio,
                    sample_rate,
                    edge_state.plan,
                    gain=edge_state.to_state.gain,
                )
                runtime.observe_audio(release, release_command)
                if release.shape[-1]:
                    yield from emit(release)
                set_time += release.shape[-1] / sample_rate
                runtime.complete_tempo_release(
                    release_command,
                    source_cursor_after=edge_state.plan.post_transition_source_end_b,
                    set_time=set_time,
                    frames=release.shape[-1],
                )
                source_cursor = edge_state.plan.post_transition_source_end_b
                current_audio = next_audio
                current_index = next_index
                current_render_state = edge_state.to_state
                publish_telemetry(detail="Executor advanced to the next committed horizon")

                # The former FUTURE edge is now at the NEXT boundary. Only now
                # may a completed Executive proposal be committed.
                if pending_future is not None and pending_future.done():
                    accept_ready_future()
                elif pending_request is not None and pending_future is not None:
                    discard_request(
                        pending_request,
                        reason="FUTURE planner was late; current prepared NEXT remains authoritative",
                    )
                    # Keep the running future only so its worker can finish. It
                    # will be discarded as stale below and no queued work is
                    # allowed to overtake it.

                if pending_future is not None and pending_future.done():
                    # A late result that was not eligible at its original
                    # boundary is now stale by definition.
                    if pending_request is not None:
                        discard_request(
                            pending_request,
                            reason="Late FUTURE result discarded after its safe commit boundary passed",
                        )
                    pending_request = None
                    pending_future = None

                submit_future_replan(planner_pool)

        final_body_command = runtime.begin_final_body(source_cursor=source_cursor)
        final = _render_segment(
            current_audio,
            sample_rate,
            source_cursor,
            plan.tracks[current_index].duration_seconds,
            rate=current_render_state.rate,
            gain=current_render_state.gain,
        )
        final = _fade_out(final, sample_rate)
        runtime.observe_audio(final, final_body_command)
        yield from emit(final)
        set_time += final.shape[-1] / sample_rate
        runtime.complete_final_body(
            final_body_command,
            source_cursor_after=plan.tracks[current_index].duration_seconds,
            set_time=set_time,
            frames=final.shape[-1],
        )
        runtime.finish(set_time=set_time)
        publish_telemetry(status="complete", detail="Live execution completed")
    except BaseException as exc:
        runtime.fail(exc, set_time=set_time)
        publish_telemetry(status="failed", detail=f"Live execution failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        # Never wait for obsolete neural/planning work on the live audio path.
        # A running worker is allowed to finish off-path; queued work is cancelled.
        planner_pool.shutdown(wait=False, cancel_futures=True)
        snapshot = runtime.to_dict()
        snapshot["live_executive"] = executive.to_dict()
        record_runtime_feedback(
            plan,
            executor_snapshot=snapshot,
            mode="continuous_live_stream" if live_replanning else "prepared_live_stream",
        )


def iter_mp3_stream(
    plan: SetPlan,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    block_frames: int = 4096,
    bitrate: str = "192k",
    telemetry_session_id: str | None = None,
) -> Iterator[bytes]:
    """Encode the prepared live set to a progressive MP3 stream with FFmpeg."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("Live HTTP streaming requires FFmpeg in the runtime image.")
    process = subprocess.Popen(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-f",
            "mp3",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    failure: list[BaseException] = []

    def feed() -> None:
        try:
            assert process.stdin is not None
            for block in iter_live_chunks(
                plan,
                sample_rate=sample_rate,
                block_frames=block_frames,
                telemetry_session_id=telemetry_session_id,
            ):
                process.stdin.write(np.asarray(block, dtype="<f4").tobytes(order="C"))
            process.stdin.close()
        except BaseException as exc:  # pragma: no cover - pipe/runtime dependent
            failure.append(exc)
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, name="djai-live-encoder-feed", daemon=True)
    feeder.start()
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(32 * 1024)
            if not chunk:
                break
            yield chunk
        return_code = process.wait(timeout=10)
        if failure:
            raise RuntimeError(f"Live stream producer failed: {failure[0]}") from failure[0]
        if return_code != 0:
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            raise RuntimeError("FFmpeg live encoder failed: " + stderr[-2000:])
    finally:
        feeder.join(timeout=2.0)
        if process.poll() is None:
            process.kill()


def play_live(
    plan: SetPlan,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    block_frames: int = 2048,
    queue_blocks: int = 24,
) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - optional host dependency
        raise RuntimeError(
            "Live playback requires the optional 'live' dependency and host audio access."
        ) from exc

    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_blocks)
    failure: list[BaseException] = []
    stop_event = threading.Event()

    def producer() -> None:
        try:
            for block in iter_live_chunks(
                plan,
                sample_rate=sample_rate,
                block_frames=block_frames,
            ):
                if stop_event.is_set():
                    break
                audio_queue.put(block)
        except BaseException as exc:  # pragma: no cover - hardware/runtime dependent
            failure.append(exc)
        finally:
            audio_queue.put(None)

    thread = threading.Thread(target=producer, name="djai-live-producer", daemon=True)
    thread.start()

    current = np.zeros((0, 2), dtype=np.float32)
    finished = False

    def callback(outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        nonlocal current, finished
        del time_info, status
        written = 0
        outdata.fill(0.0)
        while written < frames:
            if current.shape[0] == 0 and not finished:
                try:
                    item = audio_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    finished = True
                    break
                current = item
            if current.shape[0] == 0:
                break
            take = min(frames - written, current.shape[0])
            outdata[written : written + take] = current[:take]
            current = current[take:]
            written += take
        if finished and current.shape[0] == 0:
            raise sd.CallbackStop

    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=2,
            dtype="float32",
            blocksize=block_frames,
            callback=callback,
        ) as stream:
            while stream.active:
                if failure:
                    raise failure[0]
                time.sleep(0.1)
    except KeyboardInterrupt:  # pragma: no cover - interactive behavior
        stop_event.set()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
    if failure:
        raise RuntimeError(f"Live renderer failed: {failure[0]}") from failure[0]
