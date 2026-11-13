from __future__ import annotations

"""Pure sequence-compilation cue and pacing helpers.

These helpers determine source-body pacing windows and rank existing cue
candidates. They do not choose transition plans, run rehearsal, mutate
execution state, or alter safety authority.
"""

import math
import os
from typing import Callable

from .brain.dj_move import (
    DJState,
    MoveBrainConfig,
    MusicalTimeline,
    build_dj_state,
    build_musical_timeline,
    entry_candidate_musical_score,
    exit_candidate_musical_score,
)
from .models import TrackAnalysis, TransitionCandidate
from .music.timeline import earliest_safe_song_form_exit_bar


def required_source_body(
    track: TrackAnalysis,
    current_cursor: float,
    current_rate: float,
    *,
    minimum_body_seconds: float,
    minimum_body_fraction: float,
) -> float:
    remaining = max(0.0, track.duration_seconds - current_cursor)
    move_cfg = MoveBrainConfig.from_environment()
    if move_cfg.enabled and track.bpm > 1.0:
        bar_source_seconds = max(
            0.20,
            (
                track.time_signature
                if track.time_signature in {3, 4, 6}
                else 4
            )
            * 60.0
            / track.bpm,
        )
        requested_source = (
            minimum_body_seconds * max(current_rate, 1e-9)
        )
        requested_bars = int(
            math.ceil(requested_source / bar_source_seconds)
        )
        requested_bars = int(
            math.ceil(
                max(
                    move_cfg.minimum_body_bars,
                    requested_bars,
                )
                / 4.0
            )
            * 4
        )
        desired = requested_bars * bar_source_seconds
    else:
        source_for_time = (
            minimum_body_seconds * max(current_rate, 1e-9)
        )
        source_for_fraction = (
            track.duration_seconds * minimum_body_fraction
        )
        desired = max(
            source_for_time,
            source_for_fraction,
        )

    if track.duration_seconds < max(
        60.0,
        minimum_body_seconds * 2.0,
    ):
        desired = min(
            desired,
            max(0.75, remaining * 0.45),
        )
    return max(
        0.75,
        min(desired, max(0.75, remaining - 0.5)),
    )


def mixing_window(
    track: TrackAnalysis,
    current_cursor: float,
    current_rate: float,
    required_source: float,
) -> tuple[float, float]:
    """Return a weak pacing target and latest window in source seconds."""

    remaining = max(
        0.0,
        track.duration_seconds - current_cursor,
    )
    if remaining <= 1.0:
        return current_cursor, current_cursor

    move_cfg = MoveBrainConfig.from_environment()
    if move_cfg.enabled and track.bpm > 1.0:
        bar_source_seconds = max(
            0.20,
            (
                track.time_signature
                if track.time_signature in {3, 4, 6}
                else 4
            )
            * 60.0
            / track.bpm,
        )
        target_cap = (
            move_cfg.soft_target_bars
            * bar_source_seconds
        )
        latest_cap = (
            move_cfg.soft_latest_bars
            * bar_source_seconds
        )
    else:
        rate = max(current_rate, 1e-9)
        target_cap = float(
            os.environ.get(
                "DJAI_TARGET_SOURCE_PLAY_SECONDS",
                "150",
            )
        ) * rate
        latest_cap = float(
            os.environ.get(
                "DJAI_SOFT_MAX_SOURCE_PLAY_SECONDS",
                "210",
            )
        ) * rate

    target_body = max(
        required_source,
        min(target_cap, remaining * 0.62),
    )
    latest_body = max(
        target_body,
        min(latest_cap, remaining * 0.80),
    )
    latest_body = min(
        latest_body,
        max(required_source, remaining - 0.5),
    )
    return (
        current_cursor + target_body,
        current_cursor + latest_body,
    )


def sequence_exit_candidates(
    track: TrackAnalysis,
    *,
    source_cursor: float,
    current_rate: float,
    required_source: float,
    candidate_diverse_by_bar: Callable[
        [list[tuple[float, TransitionCandidate]]],
        list[TransitionCandidate],
    ],
    dj_state: DJState | None = None,
    timeline: MusicalTimeline | None = None,
    limit: int = 10,
) -> tuple[list[TransitionCandidate], float, float]:
    """Rank exits primarily by musical state; elapsed time is a weak prior."""

    target_exit, soft_latest = mixing_window(
        track,
        source_cursor,
        current_rate,
        required_source,
    )
    required_exit = source_cursor + required_source
    remaining = max(
        1.0,
        track.duration_seconds - source_cursor,
    )
    spread = max(12.0, remaining * 0.24)
    section_value = {
        "breakdown": 1.00,
        "groove": 0.94,
        "build": 0.78,
        "outro": 0.86,
        "drop": 0.44,
        "intro": 0.18,
        "unknown": 0.52,
    }

    active_timeline = (
        timeline or build_musical_timeline(track)
    )
    active_state = (
        dj_state
        or build_dj_state(
            track,
            source_cursor=source_cursor,
            effective_bpm=track.bpm * current_rate,
            render_rate=current_rate,
            timeline=active_timeline,
        )
    )

    song_form_safe_bar = earliest_safe_song_form_exit_bar(
        active_timeline,
        active_state.source_bar,
    )
    song_form_safe_time = (
        active_timeline.bar_start_seconds(
            song_form_safe_bar
        )
    )
    required_exit = max(
        required_exit,
        song_form_safe_time,
    )
    target_exit = max(target_exit, required_exit)
    soft_latest = max(soft_latest, required_exit)

    scored: list[tuple[float, TransitionCandidate]] = []
    protected: list[tuple[float, TransitionCandidate]] = []

    for candidate in track.exit_candidates:
        if candidate.time_seconds < required_exit - 1e-6:
            continue

        musical, move_meta = exit_candidate_musical_score(
            track,
            candidate,
            active_state,
            timeline=active_timeline,
        )
        position = math.exp(
            -(
                (
                    (
                        candidate.time_seconds
                        - target_exit
                    )
                    / spread
                )
                ** 2
            )
        )
        late = (
            max(
                0.0,
                candidate.time_seconds - soft_latest,
            )
            / max(remaining * 0.20, 1.0)
        )
        low_vocal = 1.0 - max(
            0.0,
            min(1.0, candidate.vocal_probability),
        )
        value = (
            0.48 * musical
            + 0.18 * candidate.score
            + 0.10 * candidate.boundary_strength
            + 0.09 * candidate.phrase_strength
            + 0.06
            * section_value.get(
                candidate.section_label,
                0.5,
            )
            + 0.04 * low_vocal
            + 0.05 * position
            - 0.07 * min(1.0, late)
        )
        item = (value, candidate)
        if bool(
            move_meta.get(
                "protected_payoff_violation",
                False,
            )
        ):
            protected.append(item)
        else:
            scored.append(item)

    if not scored:
        scored = protected
    if not scored:
        fallback = [
            candidate
            for candidate in track.exit_candidates
            if candidate.time_seconds
            >= max(
                source_cursor + 0.25,
                required_exit,
            )
        ]
        scored = [
            (candidate.score, candidate)
            for candidate in fallback
        ]

    return (
        candidate_diverse_by_bar(
            scored,
            limit=limit,
        ),
        target_exit,
        soft_latest,
    )


def sequence_entry_candidates(
    track: TrackAnalysis,
    *,
    candidate_diverse_by_bar: Callable[
        [list[tuple[float, TransitionCandidate]]],
        list[TransitionCandidate],
    ],
    limit: int = 10,
) -> list[TransitionCandidate]:
    """Rank incoming musical opportunities by phrase role instead of clock time."""

    if not track.entry_candidates:
        return []

    move_cfg = MoveBrainConfig.from_environment()
    if not move_cfg.enabled:
        bar_duration = max(
            0.5,
            (
                track.time_signature
                if track.time_signature in {3, 4, 6}
                else 4
            )
            * 60.0
            / max(track.bpm, 1.0),
        )
        mature_after = min(
            4.0 * bar_duration,
            max(
                2.0,
                track.duration_seconds * 0.08,
            ),
        )
        preferred_before = min(
            135.0,
            track.duration_seconds * 0.48,
        )
        mature = [
            candidate
            for candidate in track.entry_candidates
            if candidate.time_seconds >= mature_after
            and candidate.time_seconds
            <= preferred_before
        ]
        pool = (
            mature
            or [
                candidate
                for candidate
                in track.entry_candidates
                if candidate.time_seconds
                <= min(
                    150.0,
                    track.duration_seconds * 0.55,
                )
            ]
            or list(track.entry_candidates)
        )
        section_value = {
            "groove": 1.00,
            "build": 0.96,
            "drop": 0.94,
            "intro": 0.72,
            "breakdown": 0.70,
            "outro": 0.18,
            "unknown": 0.52,
        }
        target_time = min(
            18.0 * bar_duration,
            max(
                mature_after,
                track.duration_seconds * 0.16,
            ),
        )
        spread = max(
            8.0 * bar_duration,
            track.duration_seconds * 0.18,
            12.0,
        )
        scored: list[
            tuple[float, TransitionCandidate]
        ] = []
        for candidate in pool:
            position = math.exp(
                -(
                    (
                        (
                            candidate.time_seconds
                            - target_time
                        )
                        / spread
                    )
                    ** 2
                )
            )
            low_vocal = 1.0 - max(
                0.0,
                min(
                    1.0,
                    candidate.vocal_probability,
                ),
            )
            value = (
                0.34 * candidate.score
                + 0.18 * candidate.boundary_strength
                + 0.17 * candidate.phrase_strength
                + 0.13
                * section_value.get(
                    candidate.section_label,
                    0.5,
                )
                + 0.10 * low_vocal
                + 0.08 * position
            )
            scored.append((value, candidate))
        return candidate_diverse_by_bar(
            scored,
            limit=limit,
        )

    primary = [
        candidate
        for candidate in track.entry_candidates
        if candidate.section_label != "outro"
        and candidate.time_seconds
        <= max(
            24.0,
            track.duration_seconds * 0.68,
        )
    ]
    pool = primary or list(track.entry_candidates)
    timeline = build_musical_timeline(track)
    scored: list[
        tuple[float, TransitionCandidate]
    ] = []
    for candidate in pool:
        musical, _ = entry_candidate_musical_score(
            track,
            candidate,
            timeline=timeline,
        )
        low_vocal = 1.0 - max(
            0.0,
            min(
                1.0,
                candidate.vocal_probability,
            ),
        )
        value = (
            0.62 * musical
            + 0.14 * candidate.score
            + 0.09 * candidate.boundary_strength
            + 0.08 * candidate.phrase_strength
            + 0.07 * low_vocal
        )
        scored.append((value, candidate))

    return candidate_diverse_by_bar(
        scored,
        limit=limit,
    )
