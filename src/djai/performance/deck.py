from __future__ import annotations

import math
import os
from dataclasses import dataclass

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
    transition_end = adjusted_start + plan.source_duration_b
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
            "policy": "beat_phase_lock_then_native_tempo_release",
            "musical_timeline_version": timeline_b.version,
            "timeline_fingerprint": timeline_b.fingerprint,
            "entry_phrase_id": entry_phrase.phrase_id if entry_phrase is not None else None,
            "entry_phrase_role": entry_phrase.role if entry_phrase is not None else "unknown",
        },
    }
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
        "tempo_release_max_delta": round(_clamp(float(os.environ.get("DJAI_DECK_TEMPO_RELEASE_MAX", "0.10")), 0.02, 0.15), 4),
        "pitch_preserving": True,
        "tempo_ownership_transfer": True,
        "stem_vocal_exchange_profile": True,
        "hard_audio_safety_authoritative": True,
    }
