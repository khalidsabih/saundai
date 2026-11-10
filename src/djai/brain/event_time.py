from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

from ..models import TrackAnalysis, TransitionPlan

EVENT_TIME_VERSION = "1.0"
EventPhase = Literal[
    "opening",
    "building",
    "peak_window",
    "post_peak",
    "closing",
    "overtime",
]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True, slots=True)
class EventTimeConfig:
    target_duration_seconds: float
    planned_peak_fraction: float = 0.68
    peak_window_fraction: float = 0.12
    closing_window_seconds: float = 600.0
    target_ending_energy: float = 0.45
    target_source: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": EVENT_TIME_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class EventTimeState:
    elapsed_seconds: float
    target_duration_seconds: float
    remaining_seconds: float
    progress: float
    event_phase: EventPhase
    peak_target_seconds: float
    peak_window_start_seconds: float
    peak_window_end_seconds: float
    closing_window_start_seconds: float
    target_ending_energy: float
    close_urgency: float
    overtime_seconds: float
    track_progress_fallback: float
    timing_delta_vs_track_progress: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = EVENT_TIME_VERSION
        return payload


def auto_target_duration_seconds(
    analyses: Sequence[TrackAnalysis],
    *,
    minimum_body_seconds: float,
    minimum_body_fraction: float,
) -> float:
    """Estimate a useful event clock when the user did not specify one.

    This is intentionally a planning target, not a promise. It approximates a
    professional mixed-set body using the configured minimum body rule plus a
    modest phrase/transition allowance. The final rendered duration is still
    derived from the committed immutable programs.
    """
    if not analyses:
        return 0.0
    bodies = []
    for track in analyses:
        minimum = max(
            float(minimum_body_seconds),
            float(track.duration_seconds) * float(minimum_body_fraction),
        )
        # Avoid interpreting the minimum as the desired body. Give normal
        # tracks room to establish themselves while remaining below full-play.
        desired = max(minimum, min(float(track.duration_seconds) * 0.62, 210.0))
        bodies.append(min(float(track.duration_seconds), desired))
    median_bpm = sorted(max(60.0, float(track.bpm)) for track in analyses)[len(analyses) // 2]
    transition_allowance = max(4.0, 8.0 * 4.0 * 60.0 / median_bpm)
    return max(
        60.0,
        sum(bodies) + transition_allowance * max(0, len(analyses) - 1),
    )


def resolve_event_time_config(
    analyses: Sequence[TrackAnalysis],
    *,
    target_duration_seconds: float | None,
    planned_peak_fraction: float = 0.68,
    closing_window_seconds: float | None = None,
    target_ending_energy: float | None = None,
    minimum_body_seconds: float = 30.0,
    minimum_body_fraction: float = 0.35,
) -> EventTimeConfig:
    explicit = target_duration_seconds is not None and float(target_duration_seconds) > 0.0
    target = (
        float(target_duration_seconds)
        if explicit
        else auto_target_duration_seconds(
            analyses,
            minimum_body_seconds=minimum_body_seconds,
            minimum_body_fraction=minimum_body_fraction,
        )
    )
    target = max(60.0, target)
    peak = _clamp(planned_peak_fraction, 0.20, 0.90)
    close_window = (
        float(closing_window_seconds)
        if closing_window_seconds is not None and float(closing_window_seconds) > 0.0
        else max(180.0, min(900.0, target * 0.16))
    )
    close_window = max(60.0, min(target * 0.45, close_window))
    ending_energy = (
        _clamp(float(target_ending_energy))
        if target_ending_energy is not None
        else 0.45
    )
    return EventTimeConfig(
        target_duration_seconds=round(target, 6),
        planned_peak_fraction=round(peak, 6),
        peak_window_fraction=0.12,
        closing_window_seconds=round(close_window, 6),
        target_ending_energy=round(ending_energy, 6),
        target_source="explicit" if explicit else "auto",
    )


def event_time_config_from_mapping(payload: Any) -> EventTimeConfig | None:
    if not isinstance(payload, dict):
        return None
    try:
        target = float(payload.get("target_duration_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if target <= 0.0:
        return None
    return EventTimeConfig(
        target_duration_seconds=target,
        planned_peak_fraction=_clamp(float(payload.get("planned_peak_fraction", 0.68) or 0.68), 0.20, 0.90),
        peak_window_fraction=_clamp(float(payload.get("peak_window_fraction", 0.12) or 0.12), 0.04, 0.30),
        closing_window_seconds=max(60.0, float(payload.get("closing_window_seconds", 600.0) or 600.0)),
        target_ending_energy=_clamp(float(payload.get("target_ending_energy", 0.45) or 0.45)),
        target_source=str(payload.get("target_source") or "loaded"),
    )


def event_time_state(
    config: EventTimeConfig,
    *,
    elapsed_seconds: float,
    track_progress_fallback: float = 0.0,
) -> EventTimeState:
    elapsed = max(0.0, float(elapsed_seconds))
    target = max(1.0, float(config.target_duration_seconds))
    progress = _clamp(elapsed / target)
    remaining = max(0.0, target - elapsed)
    peak_target = target * config.planned_peak_fraction
    half_peak = target * config.peak_window_fraction * 0.5
    peak_start = max(0.0, peak_target - half_peak)
    peak_end = min(target, peak_target + half_peak)
    close_start = max(0.0, target - config.closing_window_seconds)

    if elapsed >= target:
        phase: EventPhase = "overtime"
    elif elapsed >= close_start:
        phase = "closing"
    elif peak_start <= elapsed <= peak_end:
        phase = "peak_window"
    elif elapsed < min(peak_start, target * 0.15):
        phase = "opening"
    elif elapsed < peak_start:
        phase = "building"
    else:
        phase = "post_peak"

    close_urgency = 0.0
    if elapsed >= close_start:
        close_urgency = _clamp(
            (elapsed - close_start) / max(config.closing_window_seconds, 1.0)
        )
    fallback = _clamp(track_progress_fallback)
    return EventTimeState(
        elapsed_seconds=round(elapsed, 6),
        target_duration_seconds=round(target, 6),
        remaining_seconds=round(remaining, 6),
        progress=round(progress, 6),
        event_phase=phase,
        peak_target_seconds=round(peak_target, 6),
        peak_window_start_seconds=round(peak_start, 6),
        peak_window_end_seconds=round(peak_end, 6),
        closing_window_start_seconds=round(close_start, 6),
        target_ending_energy=round(config.target_ending_energy, 6),
        close_urgency=round(close_urgency, 6),
        overtime_seconds=round(max(0.0, elapsed - target), 6),
        track_progress_fallback=round(fallback, 6),
        timing_delta_vs_track_progress=round(progress - fallback, 6),
    )


def estimate_elapsed_seconds(
    tracks: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
) -> float:
    """Estimate actual rendered time consumed before the current track body.

    With committed transition programs this mirrors set rendering: body until
    exit, transition, tempo release, then the source cursor/rate of the new
    owner. Without transitions, completed earlier tracks are treated as fully
    played and the current track is assumed to have just started.
    """
    if not tracks:
        return 0.0
    plans = list(transitions or [])
    if not plans:
        return float(sum(max(0.0, track.duration_seconds) for track in tracks[:-1]))

    source_cursor = 0.0
    current_rate = 1.0
    elapsed = 0.0
    usable = min(len(plans), max(0, len(tracks) - 1))
    for plan in plans[:usable]:
        body_source = max(0.0, float(plan.exit_time_a) - source_cursor)
        elapsed += body_source / max(current_rate, 1e-9)
        elapsed += max(0.0, float(plan.transition_seconds))
        elapsed += max(0.0, float(plan.tempo_release_seconds_b))
        source_cursor = max(0.0, float(plan.post_transition_source_end_b))
        current_rate = max(1e-9, float(plan.post_transition_rate_b or 1.0))
    return float(elapsed)


def move_timing_fit(
    state: EventTimeState,
    plan: TransitionPlan,
    *,
    source_cursor_seconds: float,
    render_rate: float,
    remaining_track_count: int,
) -> tuple[float, dict[str, float]]:
    """Small time-budget preference for already-safe complete moves."""
    if remaining_track_count <= 0:
        return 1.0, {
            "desired_segment_seconds": 0.0,
            "candidate_segment_seconds": 0.0,
            "timing_fit": 1.0,
        }
    desired = state.remaining_seconds / max(1, int(remaining_track_count))
    body = max(0.0, float(plan.exit_time_a) - float(source_cursor_seconds)) / max(float(render_rate), 1e-9)
    candidate = (
        body
        + max(0.0, float(plan.transition_seconds))
        + max(0.0, float(plan.tempo_release_seconds_b))
    )
    # Broad tolerance: the clock breaks ties; it does not force musically bad
    # exits simply to hit a stopwatch.
    spread = max(18.0, desired * 0.60)
    fit = 2.718281828459045 ** (-((candidate - desired) / spread) ** 2)
    return _clamp(fit), {
        "desired_segment_seconds": round(desired, 6),
        "candidate_segment_seconds": round(candidate, 6),
        "timing_fit": round(_clamp(fit), 6),
    }


def event_time_status(config: EventTimeConfig) -> dict[str, Any]:
    return {
        "version": EVENT_TIME_VERSION,
        "config": config.to_dict(),
        "progress_authority": "elapsed_time_when_target_known",
        "track_position_role": "fallback_and_diagnostic_only",
        "peak_window_active": True,
        "closing_window_active": True,
        "ending_energy_active": True,
        "timing_fit_authority": "small_bounded_tiebreaker",
        "hard_safety_authoritative": True,
    }
