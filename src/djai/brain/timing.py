from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..models import TrackAnalysis, TransitionPlan

SET_TIMING_VERSION = "13.0"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True, slots=True)
class SetTimingIntent:
    """User/event timing intent for a planned or live set."""

    target_duration_seconds: float | None = None
    peak_fraction: float = 0.65
    peak_window_fraction: float = 0.10
    closing_window_seconds: float = 600.0
    ending_energy: float = 0.45

    def normalized(self) -> "SetTimingIntent":
        target = None
        if self.target_duration_seconds is not None:
            target = max(60.0, float(self.target_duration_seconds))
        return SetTimingIntent(
            target_duration_seconds=target,
            peak_fraction=_clamp(self.peak_fraction, 0.15, 0.95),
            peak_window_fraction=_clamp(self.peak_window_fraction, 0.02, 0.40),
            closing_window_seconds=max(30.0, float(self.closing_window_seconds)),
            ending_energy=_clamp(self.ending_energy),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


@dataclass(frozen=True, slots=True)
class SetTimingState:
    version: str
    elapsed_seconds: float
    projected_total_seconds: float
    target_duration_seconds: float | None
    remaining_seconds: float | None
    progress: float
    planned_peak_seconds: float | None
    seconds_to_peak: float | None
    peak_window_seconds: float | None
    closing_window_seconds: float
    in_closing_window: bool
    ending_energy_target: float
    duration_error_seconds: float | None
    urgency: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_performance_duration(
    tracks: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
) -> float:
    """Estimate audible set duration with committed transition overlap removed."""

    total = sum(max(0.0, float(track.duration_seconds)) for track in tracks)
    for plan in transitions or ():
        overlap = max(0.0, float(getattr(plan, "transition_seconds", 0.0) or 0.0))
        total -= overlap
    return max(0.0, total)


def build_timing_state(
    tracks: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
    *,
    intent: SetTimingIntent | None = None,
    actual_elapsed_seconds: float | None = None,
    projected_total_seconds: float | None = None,
    fallback_progress: float | None = None,
) -> SetTimingState:
    cfg = (intent or SetTimingIntent()).normalized()
    projected = (
        max(0.0, float(projected_total_seconds))
        if projected_total_seconds is not None
        else estimate_performance_duration(tracks, transitions)
    )
    elapsed = (
        max(0.0, float(actual_elapsed_seconds))
        if actual_elapsed_seconds is not None
        else projected
    )

    target = cfg.target_duration_seconds
    if target is not None:
        progress = _clamp(elapsed / max(target, 1e-9))
        remaining = target - elapsed
        peak = target * cfg.peak_fraction
        peak_window = target * cfg.peak_window_fraction
        seconds_to_peak = peak - elapsed
        closing_window = min(cfg.closing_window_seconds, target * 0.35)
        in_closing = remaining <= closing_window
        duration_error = projected - target
        start = max(0.0, target - closing_window)
        urgency = _clamp((elapsed - start) / max(closing_window, 1e-9))
    else:
        progress = _clamp(
            fallback_progress
            if fallback_progress is not None
            else (1.0 if tracks else 0.0)
        )
        remaining = None
        peak = None
        peak_window = None
        seconds_to_peak = None
        in_closing = False
        duration_error = None
        urgency = 0.0

    return SetTimingState(
        version=SET_TIMING_VERSION,
        elapsed_seconds=round(elapsed, 6),
        projected_total_seconds=round(projected, 6),
        target_duration_seconds=(round(target, 6) if target is not None else None),
        remaining_seconds=(round(remaining, 6) if remaining is not None else None),
        progress=round(progress, 6),
        planned_peak_seconds=(round(peak, 6) if peak is not None else None),
        seconds_to_peak=(round(seconds_to_peak, 6) if seconds_to_peak is not None else None),
        peak_window_seconds=(round(peak_window, 6) if peak_window is not None else None),
        closing_window_seconds=round(cfg.closing_window_seconds, 6),
        in_closing_window=bool(in_closing),
        ending_energy_target=round(cfg.ending_energy, 6),
        duration_error_seconds=(round(duration_error, 6) if duration_error is not None else None),
        urgency=round(urgency, 6),
    )


def timing_candidate_adjustment(
    state: SetTimingState,
    candidate: TrackAnalysis,
) -> tuple[float, dict[str, float]]:
    """Return a small bounded timing/closing residual for candidate ranking."""

    if state.target_duration_seconds is None:
        return 0.0, {"timing_authority": 0.0}

    remaining = float(state.remaining_seconds or 0.0)
    playable = max(1.0, float(candidate.duration_seconds))
    overshoot = max(0.0, playable - max(0.0, remaining))
    overshoot_ratio = overshoot / max(playable, 1.0)

    adjustment = -0.12 * state.urgency * overshoot_ratio
    ending_fit = 1.0 - abs(float(candidate.energy) - state.ending_energy_target)
    if state.in_closing_window:
        adjustment += 0.10 * state.urgency * (ending_fit - 0.5)

    return (
        max(-0.15, min(0.10, adjustment)),
        {
            "timing_authority": round(state.urgency, 6),
            "overshoot_ratio": round(overshoot_ratio, 6),
            "ending_energy_fit": round(_clamp(ending_fit), 6),
        },
    )
