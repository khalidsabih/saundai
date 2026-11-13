from __future__ import annotations

import bisect
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from ..models import TrackAnalysis, TransitionCandidate, TransitionPlan
from ..music.timeline import build_musical_timeline


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _nearest_offset(values: list[float], cue: float, *, radius: float) -> float | None:
    if not values:
        return None
    nearby = [float(value) - cue for value in values if abs(float(value) - cue) <= radius]
    if not nearby:
        return None
    return min(nearby, key=abs)


def _unwrapped_beat_numbers(
    beats: list[float], *, expected_source_beat_seconds: float
) -> list[int]:
    """Account for confidently detected missing-beat gaps in the index origin.

    This is *not* interpolation through gaps. Those intervals remain invalid
    evidence, and the phase profile rejects a gap crossed by the overlap.
    """
    positions = [0]
    if not beats or expected_source_beat_seconds <= 0:
        return positions
    if not _enabled("DJAI_BEAT_GAP_AWARE", "0"):
        return list(range(len(beats)))
    for left, right in zip(beats, beats[1:]):
        interval = float(right) - float(left)
        steps = 1
        if math.isfinite(interval):
            multiple = round(interval / expected_source_beat_seconds)
            if 2 <= multiple <= 4 and abs(
                interval / multiple - expected_source_beat_seconds
            ) <= 0.15 * expected_source_beat_seconds:
                steps = multiple
        positions.append(positions[-1] + steps)
    return positions


def _overlap_crosses_unreliable_beat_interval(
    beats: list[float],
    *,
    start_seconds: float,
    end_seconds: float,
    expected_source_beat_seconds: float,
) -> bool:
    """Fail closed if an overlap traverses any unsupported beat interval."""
    if len(beats) < 2:
        return True
    first = max(1, bisect.bisect_right(beats, start_seconds))
    last = min(len(beats), bisect.bisect_left(beats, end_seconds) + 1)
    for index in range(first, last):
        left, right = float(beats[index - 1]), float(beats[index])
        interval = right - left
        if (
            left < end_seconds and right > start_seconds
            and not (
                math.isfinite(interval)
                and 0.65 * expected_source_beat_seconds <= interval
                <= 1.35 * expected_source_beat_seconds
            )
        ):
            return True
    return False


def _continuous_beat_position(
    beats: list[float],
    source_seconds: float,
    *,
    expected_source_beat_seconds: float,
    beat_numbers: list[int] | None = None,
) -> float | None:
    """Continuous, unwrapped beat index from adjacent detected beats.

    Unlike nearest-beat offsets, this does not jump by a beat when the nearest
    reference changes. Invalid/missing beats are not interpolated through:
    interpolating across a doubled interval would hide a detection error.
    """
    if len(beats) < 2 or not math.isfinite(source_seconds):
        return None
    index = bisect.bisect_right(beats, source_seconds)
    if index == 0 or index >= len(beats):
        return None
    left = float(beats[index - 1])
    right = float(beats[index])
    interval = right - left
    if not (
        math.isfinite(left)
        and math.isfinite(right)
        and 0.65 * expected_source_beat_seconds <= interval
        <= 1.35 * expected_source_beat_seconds
    ):
        return None
    numbers = beat_numbers if beat_numbers is not None else _unwrapped_beat_numbers(
        beats, expected_source_beat_seconds=expected_source_beat_seconds
    )
    return float(numbers[index - 1]) + (source_seconds - left) / interval


@dataclass(frozen=True, slots=True)
class DeckPerformanceDecision:
    phase_nudge_seconds_b: float = 0.0
    tempo_release_seconds_b: float = 0.0
    tempo_release_source_duration_b: float = 0.0
    tempo_release_start_rate_b: float = 1.0
    tempo_release_end_rate_b: float = 1.0
    post_transition_rate_b: float = 1.0
    post_transition_bpm_b: float = 0.0
    tempo_release_bars: int = 0
    tempo_release_reason: str = "native tempo already active"

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_nudge_seconds_b": round(self.phase_nudge_seconds_b, 6),
            "tempo_release_seconds_b": round(self.tempo_release_seconds_b, 6),
            "tempo_release_source_duration_b": round(self.tempo_release_source_duration_b, 6),
            "tempo_release_start_rate_b": round(self.tempo_release_start_rate_b, 6),
            "tempo_release_end_rate_b": round(self.tempo_release_end_rate_b, 6),
            "post_transition_rate_b": round(self.post_transition_rate_b, 6),
            "post_transition_bpm_b": round(self.post_transition_bpm_b, 4),
            "tempo_release_bars": int(self.tempo_release_bars),
            "tempo_release_reason": self.tempo_release_reason,
        }


def _phase_nudge(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
) -> float:
    """Small deck-style source nudge that aligns the nearest beat phase.

    Cue/phrase selection remains authoritative. This corrects only sub-beat
    analysis error and is deliberately capped so it cannot turn a good cue into
    a different musical phrase.
    """
    if not _enabled("DJAI_DECK_PHASE_NUDGE", "1") or not plan.beatmatched:
        return 0.0
    confidence = min(track_a.beat_grid_confidence, track_b.beat_grid_confidence)
    if confidence < 0.55:
        return 0.0
    beat_seconds = 60.0 / max(plan.target_bpm, 1.0)
    radius = min(0.30, beat_seconds * 0.48)
    offset_a = _nearest_offset(track_a.beats_seconds, plan.exit_time_a, radius=radius)
    offset_b = _nearest_offset(track_b.beats_seconds, plan.entry_time_b, radius=radius)
    if offset_a is None or offset_b is None:
        return 0.0
    rendered_a = offset_a / max(plan.render_rate_a, 1e-9)
    rendered_b = offset_b / max(plan.render_rate_b, 1e-9)
    output_delta = rendered_b - rendered_a
    source_nudge = -output_delta * plan.render_rate_b
    cap_ms = _clamp(float(os.environ.get("DJAI_DECK_PHASE_NUDGE_MAX_MS", "90")), 10.0, 140.0)
    cap_seconds = min(cap_ms / 1000.0, beat_seconds * 0.20 * max(plan.render_rate_b, 1e-9))
    source_nudge = _clamp(source_nudge, -cap_seconds, cap_seconds)
    source_nudge = _clamp(
        source_nudge,
        -max(0.0, plan.entry_time_b),
        max(0.0, track_b.duration_seconds - plan.entry_time_b - plan.source_duration_b),
    )
    if abs(source_nudge) < 0.004:
        return 0.0
    return float(source_nudge)


def _phase_integrity_profile(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    phase_nudge_seconds_b: float,
) -> dict[str, object]:
    """Predict bounded deck-B phase corrections across the complete overlap."""
    if not _enabled("DJAI_DECK_PHASE_WARP", "1") or not plan.beatmatched:
        return {"enabled": False, "safe": True, "reason": "not applicable", "points": [], "max_phase_error_ms": 0.0, "max_residual_ms": 0.0, "source_delta_seconds_b": 0.0}

    confidence = min(float(track_a.beat_grid_confidence), float(track_b.beat_grid_confidence))
    if confidence < 0.55 or len(track_a.beats_seconds) < 4 or len(track_b.beats_seconds) < 4:
        return {"enabled": False, "safe": False, "reason": "beat evidence is too weak for continuous phase control", "points": [], "max_phase_error_ms": 0.0, "max_residual_ms": 999.0, "source_delta_seconds_b": 0.0}

    beat_seconds = 60.0 / max(float(plan.target_bpm), 1.0)
    max_correction = _clamp(float(os.environ.get("DJAI_DECK_PHASE_WARP_MAX_MS", "45")), 10.0, 80.0) / 1000.0
    max_step = _clamp(float(os.environ.get("DJAI_DECK_PHASE_WARP_STEP_MAX_MS", "18")), 5.0, 35.0) / 1000.0
    residual_limit_ms = _clamp(float(os.environ.get("DJAI_DECK_PHASE_RESIDUAL_MAX_MS", "25")), 10.0, 45.0)
    samples = max(5, min(17, int(os.environ.get("DJAI_DECK_PHASE_WARP_SAMPLES", "9"))))

    points: list[dict[str, float]] = []
    previous = 0.0
    baseline_error: float | None = None
    max_error = 0.0
    max_residual = 0.0
    missing = 0
    rate_a = max(float(plan.render_rate_a), 1e-9)
    rate_b = max(float(plan.render_rate_b), 1e-9)
    numbers_a = _unwrapped_beat_numbers(
        track_a.beats_seconds, expected_source_beat_seconds=beat_seconds * rate_a
    )
    numbers_b = _unwrapped_beat_numbers(
        track_b.beats_seconds, expected_source_beat_seconds=beat_seconds * rate_b
    )
    overlap_a_start = float(plan.exit_time_a)
    overlap_b_start = float(plan.entry_time_b) + float(phase_nudge_seconds_b)
    overlap_a_end = overlap_a_start + float(plan.transition_seconds) * rate_a
    overlap_b_end = overlap_b_start + float(plan.transition_seconds) * rate_b
    if _enabled("DJAI_BEAT_GAP_AWARE", "0") and (
        _overlap_crosses_unreliable_beat_interval(
            track_a.beats_seconds,
            start_seconds=overlap_a_start,
            end_seconds=overlap_a_end,
            expected_source_beat_seconds=beat_seconds * rate_a,
        )
        or _overlap_crosses_unreliable_beat_interval(
            track_b.beats_seconds,
            start_seconds=overlap_b_start,
            end_seconds=overlap_b_end,
            expected_source_beat_seconds=beat_seconds * rate_b,
        )
    ):
        return {
            "enabled": False,
            "safe": False,
            "reason": "unreliable beat interval crossed by overlap",
            "points": [],
            "max_phase_error_ms": 0.0,
            "max_residual_ms": 999.0,
            "source_delta_seconds_b": 0.0,
        }

    for index in range(samples):
        fraction = index / max(1, samples - 1)
        output_seconds = float(plan.transition_seconds) * fraction
        source_a = float(plan.exit_time_a) + output_seconds * rate_a
        source_b = float(plan.entry_time_b) + float(phase_nudge_seconds_b) + output_seconds * rate_b
        phase_a = _continuous_beat_position(
            track_a.beats_seconds,
            source_a,
            expected_source_beat_seconds=beat_seconds * rate_a,
            beat_numbers=numbers_a,
        )
        phase_b = _continuous_beat_position(
            track_b.beats_seconds,
            source_b,
            expected_source_beat_seconds=beat_seconds * rate_b,
            beat_numbers=numbers_b,
        )
        if phase_a is None or phase_b is None:
            missing += 1
            continue
        # Unwrapped phase coordinates retain real accumulated drift, while
        # subtracting the initial phase removes arbitrary beat-index origins.
        phase_error = (phase_a - phase_b) * beat_seconds
        if baseline_error is None:
            baseline_error = phase_error
        drift_error = phase_error - baseline_error
        max_error = max(max_error, abs(drift_error))
        correction = _clamp(-drift_error, -max_correction, max_correction)
        correction = _clamp(correction, previous - max_step, previous + max_step)
        previous = correction
        residual = drift_error + correction
        max_residual = max(max_residual, abs(residual))
        points.append({
            "fraction": round(fraction, 6),
            "source_offset_seconds": round(correction * rate_b, 7),
            "phase_error_ms": round(drift_error * 1000.0, 3),
            "residual_ms": round(residual * 1000.0, 3),
        })

    if missing > samples // 3 or len(points) < 4:
        return {"enabled": False, "safe": False, "reason": "insufficient reliable beat intervals across overlap", "points": points, "max_phase_error_ms": round(max_error * 1000.0, 3), "max_residual_ms": 999.0, "source_delta_seconds_b": 0.0}

    source_delta = float(points[-1]["source_offset_seconds"])
    available_delta = max(0.0, float(track_b.duration_seconds) - (float(plan.entry_time_b) + float(phase_nudge_seconds_b) + float(plan.source_duration_b)))
    source_available = source_delta <= available_delta + 1e-6 if source_delta > 0.0 else True
    safe = bool(max_residual * 1000.0 <= residual_limit_ms and source_available)
    return {
        "enabled": True,
        "safe": safe,
        "reason": "bounded continuous phase lock" if safe else "required correction exceeds bounded phase-lock envelope",
        "points": points,
        "max_phase_error_ms": round(max_error * 1000.0, 3),
        "max_residual_ms": round(max_residual * 1000.0, 3),
        "residual_limit_ms": round(residual_limit_ms, 3),
        "source_delta_seconds_b": round(source_delta if safe else 0.0, 7),
    }



def _trace_rejected_phase_lock(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    profile: dict[str, object],
) -> None:
    """Opt-in JSONL evidence for unsafe phase lock; never influences decisions."""
    destination = os.environ.get("DJAI_PHASE_LOCK_TRACE_PATH", "").strip()
    if not destination or not plan.beatmatched or bool(profile.get("safe", True)):
        return
    points = list(profile.get("points", []) or [])
    residual_limit = float(profile.get("residual_limit_ms", 0.0) or 0.0)
    max_residual = float(profile.get("max_residual_ms", 0.0) or 0.0)
    sample_count = max(5, min(17, int(os.environ.get("DJAI_DECK_PHASE_WARP_SAMPLES", "9"))))
    evidence = {
        "track_a": Path(track_a.path).name,
        "track_b": Path(track_b.path).name,
        "exit_time_a_seconds": round(float(plan.exit_time_a), 6),
        "entry_time_b_seconds": round(float(plan.entry_time_b), 6),
        "transition_seconds": round(float(plan.transition_seconds), 6),
        "render_rate_a": round(float(plan.render_rate_a), 6),
        "render_rate_b": round(float(plan.render_rate_b), 6),
        "phase_nudge_seconds_b": round(float(plan.phase_nudge_seconds_b), 6),
        "beat_grid_confidence_a": round(float(track_a.beat_grid_confidence), 4),
        "beat_grid_confidence_b": round(float(track_b.beat_grid_confidence), 4),
        "beat_interval_cv_a": round(float(track_a.beat_interval_cv), 6),
        "beat_interval_cv_b": round(float(track_b.beat_interval_cv), 6),
        "variable_tempo_a": bool(track_a.variable_tempo),
        "variable_tempo_b": bool(track_b.variable_tempo),
        "reason": str(profile.get("reason", "")),
        "max_phase_error_ms": profile.get("max_phase_error_ms"),
        "max_residual_ms": profile.get("max_residual_ms"),
        "residual_limit_ms": profile.get("residual_limit_ms"),
        "phase_warp_max_correction_ms": round(_clamp(float(os.environ.get("DJAI_DECK_PHASE_WARP_MAX_MS", "45")), 10.0, 80.0), 3),
        "phase_warp_max_step_ms": round(_clamp(float(os.environ.get("DJAI_DECK_PHASE_WARP_STEP_MAX_MS", "18")), 5.0, 35.0), 3),
        "sample_count": sample_count,
        "matched_points": len(points),
        "unmatched_points": sample_count - len(points),
        "exceeds_residual_limit": residual_limit > 0.0 and max_residual > residual_limit,
        "points": points,
    }
    try:
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, sort_keys=True) + "\n")
    except OSError:
        # Diagnostic output must never alter transition selection or delivery.
        return


def _release_bars(rate_delta: float, vocal_probability: float, section: str) -> int:
    # More exposed incoming vocals get a slower tempo ride; instrumental and
    # breakdown sections can absorb a shorter release cleanly.
    percent = rate_delta * 100.0
    if percent <= 1.5:
        bars = 4
    elif percent <= 3.0:
        bars = 8
    elif percent <= 5.0:
        bars = 12
    else:
        bars = 16
    if vocal_probability >= 0.60 and section not in {"intro", "breakdown"}:
        bars += 4
    if section in {"intro", "breakdown", "outro"}:
        bars = max(4, bars - 4)
    return min(24, max(4, bars))


def decide_deck_performance(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    entry_candidate: TransitionCandidate | None = None,
) -> DeckPerformanceDecision:
    phase_nudge = _phase_nudge(track_a, track_b, plan)
    phase_integrity = _phase_integrity_profile(
        track_a, track_b, plan, phase_nudge_seconds_b=phase_nudge
    )
    phase_source_delta = float(phase_integrity.get("source_delta_seconds_b", 0.0) or 0.0)
    start_rate = max(0.01, float(plan.render_rate_b))
    native_rate = 1.0
    native_bpm = float(track_b.bpm)

    if (
        not _enabled("DJAI_DECK_TEMPO_RELEASE", "1")
        or not plan.beatmatched
        or plan.transition_mode == "emergency_cut"
        or abs(start_rate - native_rate) < 0.002
    ):
        return DeckPerformanceDecision(
            phase_nudge_seconds_b=phase_nudge,
            tempo_release_start_rate_b=start_rate,
            tempo_release_end_rate_b=start_rate,
            post_transition_rate_b=start_rate,
            post_transition_bpm_b=round(native_bpm * start_rate, 4),
            tempo_release_reason=(
                "native tempo already active"
                if abs(start_rate - 1.0) < 0.002
                else "tempo release disabled or transition is not a transparent beatmatched blend"
            ),
        )

    rate_delta = abs(start_rate - 1.0)
    max_delta = _clamp(float(os.environ.get("DJAI_DECK_TEMPO_RELEASE_MAX", "0.10")), 0.02, 0.15)
    if rate_delta > max_delta:
        return DeckPerformanceDecision(
            phase_nudge_seconds_b=phase_nudge,
            tempo_release_start_rate_b=start_rate,
            tempo_release_end_rate_b=start_rate,
            post_transition_rate_b=start_rate,
            post_transition_bpm_b=round(native_bpm * start_rate, 4),
            tempo_release_reason=f"matched-rate delta {rate_delta:.1%} exceeds bounded deck-release limit",
        )

    timeline_b = build_musical_timeline(track_b)
    entry_bar = (
        int(entry_candidate.bar_index)
        if entry_candidate is not None
        else timeline_b.bar_for_time(plan.entry_time_b)
    )
    entry_phrase = timeline_b.phrase_for_bar(entry_bar)
    vocal = float(
        entry_phrase.vocal_probability
        if entry_phrase is not None
        else (entry_candidate.vocal_probability if entry_candidate is not None else track_b.vocal_probability)
    )
    section = str(
        entry_phrase.section
        if entry_phrase is not None
        else (entry_candidate.section_label if entry_candidate is not None else "unknown")
    )
    bars = _release_bars(rate_delta, vocal, section)
    beats_per_bar = max(3, int(track_b.time_signature or track_a.time_signature or 4))
    # The release is heard after B owns the deck. Use the midpoint BPM so the
    # nominal bar duration remains stable while the rate glides to native.
    matched_bpm = max(1.0, native_bpm * start_rate)
    mean_bpm = max(1.0, 0.5 * (matched_bpm + native_bpm))
    release_seconds = bars * beats_per_bar * 60.0 / mean_bpm
    average_rate = 0.5 * (start_rate + native_rate)
    release_source = release_seconds * average_rate

    adjusted_start = max(0.0, plan.entry_time_b + phase_nudge)
    transition_end = adjusted_start + plan.source_duration_b + phase_source_delta
    available_source = max(0.0, track_b.duration_seconds - transition_end - 0.25)
    if available_source + 1e-6 < release_source:
        if available_source < beats_per_bar * 60.0 / max(native_bpm, 1.0) * 2.0:
            return DeckPerformanceDecision(
                phase_nudge_seconds_b=phase_nudge,
                tempo_release_start_rate_b=start_rate,
                tempo_release_end_rate_b=start_rate,
                post_transition_rate_b=start_rate,
                post_transition_bpm_b=round(native_bpm * start_rate, 4),
                tempo_release_reason="not enough incoming source remains for a safe tempo-ownership release",
            )
        release_source = available_source
        release_seconds = release_source / max(average_rate, 1e-9)
        bars = max(2, int(math.floor(release_seconds * mean_bpm / (beats_per_bar * 60.0))))

    return DeckPerformanceDecision(
        phase_nudge_seconds_b=phase_nudge,
        tempo_release_seconds_b=round(release_seconds, 6),
        tempo_release_source_duration_b=round(release_source, 6),
        tempo_release_start_rate_b=round(start_rate, 6),
        tempo_release_end_rate_b=1.0,
        post_transition_rate_b=1.0,
        post_transition_bpm_b=round(native_bpm, 4),
        tempo_release_bars=bars,
        tempo_release_reason="incoming deck takes tempo ownership and glides back to its native BPM after the handoff",
    )


def apply_deck_performance(
    plan: TransitionPlan,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    entry_candidate: TransitionCandidate | None = None,
) -> TransitionPlan:
    decision = decide_deck_performance(track_a, track_b, plan, entry_candidate=entry_candidate)
    plan.phase_nudge_seconds_b = decision.phase_nudge_seconds_b
    phase_integrity = _phase_integrity_profile(
        track_a, track_b, plan, phase_nudge_seconds_b=decision.phase_nudge_seconds_b
    )
    phase_source_delta = float(phase_integrity.get("source_delta_seconds_b", 0.0) or 0.0)
    _trace_rejected_phase_lock(track_a, track_b, plan, phase_integrity)
    if bool(phase_integrity.get("safe", True)) and abs(phase_source_delta) > 1e-9:
        plan.source_duration_b = round(max(0.0, plan.source_duration_b + phase_source_delta), 6)
    plan.tempo_release_seconds_b = decision.tempo_release_seconds_b
    plan.tempo_release_source_duration_b = decision.tempo_release_source_duration_b
    plan.tempo_release_start_rate_b = decision.tempo_release_start_rate_b
    plan.tempo_release_end_rate_b = decision.tempo_release_end_rate_b
    plan.post_transition_rate_b = decision.post_transition_rate_b
    plan.post_transition_bpm_b = decision.post_transition_bpm_b
    if decision.tempo_release_seconds_b > 0.0:
        plan.tempo_strategy = "deck_handoff"
    timeline_b = build_musical_timeline(track_b)
    entry_bar = (
        int(entry_candidate.bar_index)
        if entry_candidate is not None
        else timeline_b.bar_for_time(plan.entry_time_b)
    )
    entry_phrase = timeline_b.phrase_for_bar(entry_bar)
    plan.metadata = {
        **plan.metadata,
        "deck_performance": {
            **decision.to_dict(),
            "phase_lock": bool(abs(decision.phase_nudge_seconds_b) >= 0.004),
            "pitch_preserving_time_stretch": True,
            "tempo_owner_before": "deck_a",
            "tempo_owner_after": "deck_b",
            "policy": "continuous_beat_phase_lock_then_native_tempo_release",
            "phase_integrity": phase_integrity,
            "musical_timeline_version": timeline_b.version,
            "timeline_fingerprint": timeline_b.fingerprint,
            "entry_phrase_id": entry_phrase.phrase_id if entry_phrase is not None else None,
            "entry_phrase_role": entry_phrase.role if entry_phrase is not None else "unknown",
        },
    }
    if plan.beatmatched and not bool(phase_integrity.get("safe", True)):
        plan.requires_confirmation = True
        plan.warning = (
            "Continuous beat-phase lock is outside the bounded correction envelope; "
            "automatic delivery should replan or use a structural reset."
        )
        plan.rationale = [
            *plan.rationale,
            "Beat Integrity rejected this long rhythmic overlap because residual phase error remains too large.",
        ]
    elif plan.beatmatched and bool(phase_integrity.get("enabled", False)):
        plan.rationale = [
            *plan.rationale,
            (
                f"Beat Integrity applies bounded continuous correction across the overlap "
                f"(predicted residual <= {float(phase_integrity.get('max_residual_ms', 0.0)):.1f} ms)."
            ),
        ]

    if decision.tempo_release_seconds_b > 0.0:
        plan.rationale = [
            *plan.rationale,
            (
                f"Deck B is beatmatched at {track_b.bpm * plan.render_rate_b:.2f} BPM for the handoff, "
                f"then takes tempo ownership over about {decision.tempo_release_bars} bars and returns to "
                f"its native {track_b.bpm:.2f} BPM."
            ),
        ]
    if abs(decision.phase_nudge_seconds_b) >= 0.004:
        plan.rationale = [
            *plan.rationale,
            f"Deck B receives a {decision.phase_nudge_seconds_b * 1000.0:+.0f} ms bounded beat-phase nudge before the blend.",
        ]
    return plan


def deck_runtime_status() -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "tempo_release_enabled": _enabled("DJAI_DECK_TEMPO_RELEASE", "1"),
        "phase_nudge_enabled": _enabled("DJAI_DECK_PHASE_NUDGE", "1"),
        "phase_nudge_max_ms": round(_clamp(float(os.environ.get("DJAI_DECK_PHASE_NUDGE_MAX_MS", "90")), 10.0, 140.0), 1),
        "phase_warp_enabled": _enabled("DJAI_DECK_PHASE_WARP", "1"),
        "phase_warp_max_ms": round(_clamp(float(os.environ.get("DJAI_DECK_PHASE_WARP_MAX_MS", "45")), 10.0, 80.0), 1),
        "phase_residual_max_ms": round(_clamp(float(os.environ.get("DJAI_DECK_PHASE_RESIDUAL_MAX_MS", "25")), 10.0, 45.0), 1),
        "tempo_release_max_delta": round(_clamp(float(os.environ.get("DJAI_DECK_TEMPO_RELEASE_MAX", "0.10")), 0.02, 0.15), 4),
        "pitch_preserving": True,
        "tempo_ownership_transfer": True,
        "stem_vocal_exchange_profile": True,
        "hard_audio_safety_authoritative": True,
    }
