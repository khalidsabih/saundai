from __future__ import annotations

import hashlib
import threading
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from ..models import BarFeature, TrackAnalysis
from ..version import MUSICAL_TIMELINE_SCHEMA_VERSION

TIMELINE_VERSION = MUSICAL_TIMELINE_SCHEMA_VERSION
HARD_SONG_FORM_COMMITMENTS = frozenset({"first_body_payoff", "protected_hook"})


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class TimelineBeat:
    beat_index: int
    time_seconds: float
    bar_index: int
    beat_in_bar: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimelineBar:
    bar_index: int
    start_seconds: float
    end_seconds: float
    section: str
    energy: float
    rhythm: float
    brightness: float
    vocal_probability: float
    bass_activity: float
    tension: float
    boundary_strength: float
    phrase_strength: float
    beat_confidence: float
    local_bpm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MusicalCommitment:
    kind: str
    created_at_bar: int
    source_phrase: int
    target_phrase: int
    target_bar: int
    protect_until_bar: int
    confidence: float
    importance: float
    relation: str
    rationale: str

    @property
    def payoff_bar(self) -> int:
        return self.target_bar

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["payoff_bar"] = self.payoff_bar
        return payload


@dataclass(frozen=True, slots=True)
class TimelinePhrase:
    phrase_index: int
    phrase_id: str
    start_bar: int
    end_bar: int
    start_seconds: float
    end_seconds: float
    role: str
    section: str
    importance: float
    transition_readiness: float
    vocal_probability: float
    bass_activity: float
    energy: float
    rhythm: float
    brightness: float
    tension: float
    phrase_strength: float
    boundary_strength: float
    beat_confidence: float
    energy_delta: float = 0.0
    song_role: str = "unknown"
    song_role_index: int = 0
    song_form_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Compatibility name used by the 1.4 Move Brain API.
PhraseMoment = TimelinePhrase


@dataclass(frozen=True, slots=True)
class TimelineSection:
    section_id: str
    label: str
    start_bar: int
    end_bar: int
    start_seconds: float
    end_seconds: float
    phrase_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phrase_ids"] = list(self.phrase_ids)
        return payload


@dataclass(frozen=True, slots=True)
class MusicalTimeline:
    version: str
    track: str
    bpm: float
    time_signature: int
    duration_seconds: float
    beats: tuple[TimelineBeat, ...]
    bars: tuple[TimelineBar, ...]
    phrases: tuple[TimelinePhrase, ...]
    sections: tuple[TimelineSection, ...]
    commitments: tuple[MusicalCommitment, ...]
    fingerprint: str

    @property
    def track_id(self) -> str:
        return self.track

    @property
    def moments(self) -> tuple[TimelinePhrase, ...]:
        """1.4 compatibility alias. New code should use ``phrases``."""
        return self.phrases

    def bar_for_time(self, seconds: float) -> int:
        if not self.bars:
            bar_seconds = max(0.5, self.time_signature * 60.0 / max(self.bpm, 1.0))
            return max(0, int(max(0.0, seconds) / bar_seconds))
        starts = [item.start_seconds for item in self.bars]
        position = bisect_right(starts, max(0.0, float(seconds))) - 1
        position = max(0, min(position, len(self.bars) - 1))
        return int(self.bars[position].bar_index)

    def nearest_bar(self, seconds: float) -> int:
        if not self.bars:
            return self.bar_for_time(seconds)
        starts = [item.start_seconds for item in self.bars]
        position = bisect_left(starts, max(0.0, float(seconds)))
        if position <= 0:
            return int(self.bars[0].bar_index)
        if position >= len(self.bars):
            return int(self.bars[-1].bar_index)
        before = self.bars[position - 1]
        after = self.bars[position]
        selected = before if abs(before.start_seconds - seconds) <= abs(after.start_seconds - seconds) else after
        return int(selected.bar_index)

    def bar_start_seconds(self, bar_index: int) -> float:
        if self.bars:
            by_index = {item.bar_index: item for item in self.bars}
            if int(bar_index) in by_index:
                return float(by_index[int(bar_index)].start_seconds)
        bar_seconds = max(0.5, self.time_signature * 60.0 / max(self.bpm, 1.0))
        return max(0.0, int(bar_index) * bar_seconds)

    def phrase_for_bar(self, bar_index: int) -> TimelinePhrase | None:
        for phrase in self.phrases:
            if phrase.start_bar <= int(bar_index) < phrase.end_bar:
                return phrase
        if self.phrases and int(bar_index) >= self.phrases[-1].end_bar:
            return self.phrases[-1]
        return self.phrases[0] if self.phrases else None

    def phrase_for_time(self, seconds: float) -> TimelinePhrase | None:
        return self.phrase_for_bar(self.bar_for_time(seconds))

    def section_for_bar(self, bar_index: int) -> TimelineSection | None:
        for section in self.sections:
            if section.start_bar <= int(bar_index) < section.end_bar:
                return section
        if self.sections and int(bar_index) >= self.sections[-1].end_bar:
            return self.sections[-1]
        return self.sections[0] if self.sections else None

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "track": self.track,
            "bpm": self.bpm,
            "time_signature": self.time_signature,
            "duration_seconds": self.duration_seconds,
            "fingerprint": self.fingerprint,
            "phrase_count": len(self.phrases),
            "section_count": len(self.sections),
            "phrases": [item.to_dict() for item in self.phrases],
            "sections": [item.to_dict() for item in self.sections],
            "commitments": [item.to_dict() for item in self.commitments],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_summary_dict(),
            "beats": [item.to_dict() for item in self.beats],
            "bars": [item.to_dict() for item in self.bars],
        }


def is_hard_song_form_commitment(commitment: MusicalCommitment | None) -> bool:
    return bool(commitment is not None and commitment.kind in HARD_SONG_FORM_COMMITMENTS)


def earliest_safe_song_form_exit_bar(timeline: MusicalTimeline, source_bar: int = 0) -> int:
    """Return the earliest bar that is outside active/future hard song-form promises.

    A confident intro->body or body->first-hook promise is not an optional
    scoring preference. If playback has not yet completed that promise, normal
    planning and last-resort planning must both wait until its protected phrase
    finishes. Once playback is already beyond the promise this function is a
    no-op.
    """
    source = max(0, int(source_bar))
    safe = source
    for commitment in timeline.commitments:
        if not is_hard_song_form_commitment(commitment) or commitment.confidence < 0.55:
            continue
        if source < int(commitment.protect_until_bar):
            safe = max(safe, int(commitment.protect_until_bar))
    return safe


def _role_for_phrase(
    section: str,
    *,
    index: int,
    count: int,
    energy: float,
    rhythm: float,
    tension: float,
    energy_delta: float,
    boundary: float,
    phrase_strength: float,
) -> str:
    progress = index / max(1, count - 1)
    if section == "intro":
        return "intro"
    if section == "outro":
        return "outro"
    if section == "drop":
        return "drop"
    if section == "breakdown":
        return "breakdown"
    if section == "build":
        return "pre_drop" if energy_delta >= 0.10 or tension >= 0.62 else "build"
    if progress <= 0.12 and energy < 0.58:
        return "intro"
    if progress >= 0.86 and energy_delta <= 0.02:
        return "outro"
    if energy >= 0.72 and rhythm >= 0.64:
        return "drop"
    if energy <= 0.42 and rhythm <= 0.48:
        return "breakdown"
    if energy_delta >= 0.08 or (tension >= 0.62 and energy_delta > 0.015):
        return "build"
    if section == "groove":
        if 0.18 < progress < 0.82 and boundary >= 0.58 and phrase_strength >= 0.55:
            return "bridge"
        return "groove"
    return "groove"


def _phrase_importance(
    *,
    role: str,
    phrase_strength: float,
    boundary_strength: float,
    energy: float,
    vocal_probability: float,
    index: int,
    count: int,
) -> float:
    role_base = {
        "drop": 0.92,
        "pre_drop": 0.82,
        "build": 0.72,
        "breakdown": 0.62,
        "bridge": 0.52,
        "groove": 0.45,
        "intro": 0.28,
        "outro": 0.24,
        "unknown": 0.40,
    }.get(role, 0.40)
    vocal_hook = vocal_probability * (0.72 if role not in {"intro", "outro"} else 0.38)
    salience = (
        0.42 * role_base
        + 0.22 * phrase_strength
        + 0.14 * boundary_strength
        + 0.14 * energy
        + 0.08 * vocal_hook
    )
    if count > 1 and index == count - 1 and role not in {"drop", "pre_drop"}:
        salience -= 0.10
    return _clamp(salience)


def _adaptive_phrase_ranges(bars: list[BarFeature]) -> list[tuple[int, int]]:
    if not bars:
        return []
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(bars)):
        length = index - start
        current = bars[index]
        previous = bars[index - 1]
        section_change = current.section_label != previous.section_label and current.section_label != "unknown"
        strong_phrase = float(current.phrase_strength) >= 0.70 and float(current.boundary_strength) >= 0.42
        very_strong = float(current.phrase_strength) >= 0.84 or float(current.boundary_strength) >= 0.82
        natural_multiple = length in {4, 8, 12, 16}
        should_split = False
        if length >= 4 and section_change:
            should_split = True
        elif length >= 4 and strong_phrase and natural_multiple:
            should_split = True
        elif length >= 6 and very_strong:
            should_split = True
        elif length >= 16:
            should_split = True
        if should_split:
            ranges.append((start, index))
            start = index
    if start < len(bars):
        ranges.append((start, len(bars)))
    if len(ranges) >= 2:
        left_start, left_end = ranges[-2]
        tail_start, tail_end = ranges[-1]
        if tail_end - tail_start < 4 and bars[tail_start].section_label == bars[left_end - 1].section_label:
            ranges[-2:] = [(left_start, tail_end)]
    return ranges


def _mean(window: list[BarFeature], field: str) -> float:
    return float(np.mean([float(getattr(item, field)) for item in window])) if window else 0.0


def _dominant_section(window: list[BarFeature]) -> str:
    labels = [bar.section_label for bar in window if bar.section_label != "unknown"]
    return max(set(labels), key=labels.count) if labels else "unknown"


def _build_phrases(bars: list[BarFeature]) -> tuple[TimelinePhrase, ...]:
    ranges = _adaptive_phrase_ranges(bars)
    phrases: list[TimelinePhrase] = []
    for index, (start, end) in enumerate(ranges):
        window = bars[start:end]
        if not window:
            continue
        section = _dominant_section(window)
        energy = _mean(window, "energy")
        rhythm = _mean(window, "rhythm")
        brightness = _mean(window, "brightness")
        tension = _mean(window, "harmonic_tension")
        vocal = _mean(window, "vocal_probability")
        bass = _mean(window, "bass_activity")
        phrase_strength = max(float(bar.phrase_strength) for bar in window)
        boundary = max(float(bar.boundary_strength) for bar in window)
        beat_confidence = _mean(window, "beat_grid_confidence")
        energy_delta = float(window[-1].energy) - float(window[0].energy) if len(window) > 1 else 0.0
        role = _role_for_phrase(
            section,
            index=index,
            count=len(ranges),
            energy=energy,
            rhythm=rhythm,
            tension=tension,
            energy_delta=energy_delta,
            boundary=boundary,
            phrase_strength=phrase_strength,
        )
        readiness = _clamp(0.38 * boundary + 0.34 * phrase_strength + 0.28 * beat_confidence)
        importance = _phrase_importance(
            role=role,
            phrase_strength=phrase_strength,
            boundary_strength=boundary,
            energy=energy,
            vocal_probability=vocal,
            index=index,
            count=len(ranges),
        )
        start_bar = int(window[0].bar_index)
        end_bar = int(window[-1].bar_index + 1)
        phrases.append(
            TimelinePhrase(
                phrase_index=index,
                phrase_id=f"phrase_{index}",
                start_bar=start_bar,
                end_bar=end_bar,
                start_seconds=float(window[0].start_seconds),
                end_seconds=float(window[-1].end_seconds),
                role=role,
                section=str(section),
                importance=round(importance, 4),
                transition_readiness=round(readiness, 4),
                vocal_probability=round(vocal, 4),
                bass_activity=round(bass, 4),
                energy=round(energy, 4),
                rhythm=round(rhythm, 4),
                brightness=round(brightness, 4),
                tension=round(tension, 4),
                phrase_strength=round(phrase_strength, 4),
                boundary_strength=round(boundary, 4),
                beat_confidence=round(beat_confidence, 4),
                energy_delta=round(energy_delta, 4),
            )
        )
    return tuple(phrases)



def _annotate_song_form(phrases: tuple[TimelinePhrase, ...]) -> tuple[TimelinePhrase, ...]:
    """Infer human-readable song-form roles without changing coarse section labels.

    The raw analyzer intentionally uses a small, robust section vocabulary.  For
    DJ decisions we need a second layer that distinguishes the first sung body
    from the intro and identifies likely hooks.  This is deliberately
    conservative: ambiguous material remains ``instrumental``/``bridge`` rather
    than pretending to know lyrical semantics.
    """
    if not phrases:
        return ()

    body = [item for item in phrases if item.role not in {"intro", "outro", "build", "pre_drop", "breakdown"}]
    body_energy = float(np.median([item.energy for item in body])) if body else 0.5
    first_vocal_body_seen = False
    previous_role = ""
    counters: dict[str, int] = {}
    annotated: list[TimelinePhrase] = []

    for phrase in phrases:
        base = phrase.role
        vocal = float(phrase.vocal_probability)
        salience = _clamp(
            0.38 * float(phrase.importance)
            + 0.22 * float(phrase.energy)
            + 0.20 * float(phrase.boundary_strength)
            + 0.20 * float(phrase.phrase_strength)
        )

        previous_phrase = phrases[phrase.phrase_index - 1] if phrase.phrase_index > 0 else None
        vocal_lift = vocal - float(previous_phrase.vocal_probability) if previous_phrase is not None else 0.0
        intro_body_break = bool(
            base == "intro"
            and phrase.phrase_index > 0
            and vocal >= 0.38
            and (phrase.boundary_strength >= 0.48 or vocal_lift >= 0.16)
        )

        if base in {"outro", "build", "pre_drop", "breakdown"} or (base == "intro" and not intro_body_break):
            song_role = base
            confidence = _clamp(0.72 + 0.20 * phrase.boundary_strength)
        elif base == "drop":
            hookish = vocal >= 0.42 and salience >= 0.64
            song_role = "hook" if hookish else "drop"
            confidence = _clamp(0.62 + 0.20 * salience + 0.12 * vocal)
            if hookish:
                first_vocal_body_seen = True
        elif base == "bridge" and vocal < 0.38:
            song_role = "bridge"
            confidence = _clamp(0.56 + 0.24 * phrase.boundary_strength + 0.12 * phrase.phrase_strength)
        elif vocal >= 0.38:
            # The first clear sung body after an intro is much more likely to be
            # verse material than a chorus.  Later, materially more salient sung
            # phrases are tagged as hooks.
            if not first_vocal_body_seen:
                song_role = "verse"
                first_vocal_body_seen = True
            else:
                hookish = salience >= 0.66 and phrase.energy >= body_energy - 0.02
                song_role = "hook" if hookish else "verse"
            confidence = _clamp(0.52 + 0.26 * vocal + 0.18 * salience)
        else:
            song_role = "instrumental"
            confidence = _clamp(0.48 + 0.22 * phrase.rhythm + 0.16 * phrase.phrase_strength)

        if song_role != previous_role:
            counters[song_role] = counters.get(song_role, 0) + 1
        role_index = counters.get(song_role, 1)
        previous_role = song_role
        annotated.append(
            replace(
                phrase,
                song_role=song_role,
                song_role_index=role_index,
                song_form_confidence=round(confidence, 4),
            )
        )
    return tuple(annotated)

def _build_sections(bars: list[BarFeature], phrases: tuple[TimelinePhrase, ...]) -> tuple[TimelineSection, ...]:
    """Group canonical phrases into stable contiguous section spans."""
    del bars  # Bars are already represented by the phrase boundaries here.
    if not phrases:
        return ()
    sections: list[TimelineSection] = []
    start = 0
    for index in range(1, len(phrases) + 1):
        changed = index == len(phrases) or phrases[index].section != phrases[start].section
        if not changed:
            continue
        window = phrases[start:index]
        sections.append(
            TimelineSection(
                section_id=f"section_{len(sections)}",
                label=window[0].section,
                start_bar=window[0].start_bar,
                end_bar=window[-1].end_bar,
                start_seconds=window[0].start_seconds,
                end_seconds=window[-1].end_seconds,
                phrase_ids=tuple(phrase.phrase_id for phrase in window),
            )
        )
        start = index
    return tuple(sections)


def _build_commitments(phrases: tuple[TimelinePhrase, ...]) -> tuple[MusicalCommitment, ...]:
    commitments: list[MusicalCommitment] = []
    for index, (source, target) in enumerate(zip(phrases, phrases[1:])):
        resolves = target.role == "drop" and source.role in {"build", "pre_drop", "breakdown"}
        if not resolves:
            continue
        confidence = _clamp(
            0.48 * target.transition_readiness
            + 0.22 * source.transition_readiness
            + 0.30 * target.importance
        )
        commitments.append(
            MusicalCommitment(
                kind="pending_drop_payoff",
                created_at_bar=int(source.start_bar),
                source_phrase=index,
                target_phrase=index + 1,
                target_bar=int(target.start_bar),
                protect_until_bar=int(target.end_bar),
                confidence=round(confidence, 4),
                importance=round(target.importance, 4),
                relation="resolves_to_drop",
                rationale="A build/pre-drop phrase creates a musical promise that should normally resolve before mixing out.",
            )
        )

    # Song-form protection: if a record has a real intro followed by its first
    # sung/primary body, hearing only the intro is not a complete musical arc.
    # Protect from the final intro phrase through the end of that first body.
    intro_prefix = 0
    for phrase in phrases:
        if phrase.song_role != "intro":
            break
        intro_prefix += 1
    if intro_prefix and intro_prefix < len(phrases):
        target_index = intro_prefix
        target = phrases[target_index]
        if target.song_role in {"verse", "hook", "drop", "instrumental", "bridge"}:
            source_index = max(0, target_index - 1)
            source = phrases[source_index]
            confidence = _clamp(
                0.52
                + 0.16 * target.song_form_confidence
                + 0.12 * target.boundary_strength
                + 0.10 * target.phrase_strength
                + 0.10 * min(1.0, target.vocal_probability + target.energy * 0.35)
            )
            commitments.append(
                MusicalCommitment(
                    kind="first_body_payoff",
                    created_at_bar=int(source.start_bar),
                    source_phrase=source_index,
                    target_phrase=target_index,
                    target_bar=int(target.start_bar),
                    protect_until_bar=int(target.end_bar),
                    confidence=round(confidence, 4),
                    importance=round(max(target.importance, 0.62), 4),
                    relation="intro_resolves_to_body",
                    rationale=(
                        "The intro creates a song-form promise: SAUND should hear the first body phrase "
                        "before beginning an ordinary transition."
                    ),
                )
            )

    # Protect the *first* clear hook/chorus as a song payoff. Later hooks stay
    # visible to scoring but are not hard-gated, which keeps long pop records
    # mixable instead of forcing every performance to wait for the outro.
    first_hook_index = next(
        (
            index
            for index, phrase in enumerate(phrases)
            if phrase.song_role == "hook"
            and phrase.song_form_confidence >= 0.58
            and phrase.importance >= 0.58
        ),
        None,
    )
    if first_hook_index is not None:
        first_hook = phrases[first_hook_index]
        if not any(
            item.target_phrase == first_hook.phrase_index
            and item.kind in {"pending_drop_payoff", "first_body_payoff"}
            for item in commitments
        ):
            first_body_index = next(
                (index for index, phrase in enumerate(phrases) if phrase.song_role != "intro"),
                max(0, first_hook.phrase_index - 1),
            )
            first_body = phrases[min(first_body_index, first_hook.phrase_index)]
            commitments.append(
                MusicalCommitment(
                    kind="protected_hook",
                    created_at_bar=int(first_body.start_bar),
                    source_phrase=int(first_body.phrase_index),
                    target_phrase=first_hook.phrase_index,
                    target_bar=first_hook.start_bar,
                    protect_until_bar=first_hook.end_bar,
                    confidence=round(_clamp(0.54 + 0.34 * first_hook.song_form_confidence), 4),
                    importance=round(max(first_hook.importance, 0.62), 4),
                    relation="protect_first_hook",
                    rationale=(
                        "The first likely hook/chorus is a high-value song payoff. SAUND should hear "
                        "at least one complete hook phrase before considering ordinary later exits."
                    ),
                )
            )

    for phrase in phrases:
        if phrase.importance < 0.80 or phrase.role not in {"drop", "pre_drop"}:
            continue
        if any(item.target_phrase == phrase.phrase_index for item in commitments):
            continue
        commitments.append(
            MusicalCommitment(
                kind="protected_moment",
                created_at_bar=int(phrase.start_bar),
                source_phrase=max(0, phrase.phrase_index - 1),
                target_phrase=phrase.phrase_index,
                target_bar=phrase.start_bar,
                protect_until_bar=phrase.end_bar,
                confidence=round(_clamp(0.55 + 0.40 * phrase.importance), 4),
                importance=round(phrase.importance, 4),
                relation="protect",
                rationale="A high-salience drop/hook should normally be heard through its phrase before SAUND exits.",
            )
        )
    commitments.sort(key=lambda item: (item.target_bar, item.protect_until_bar, item.kind))
    return tuple(commitments)


def _build_beats(track: TrackAnalysis, bars: tuple[TimelineBar, ...]) -> tuple[TimelineBeat, ...]:
    if not track.beats_seconds:
        return ()
    starts = [bar.start_seconds for bar in bars]
    beats_per_bar = max(1, int(track.time_signature or 4))
    output: list[TimelineBeat] = []
    per_bar_counts: dict[int, int] = {}
    for beat_index, seconds in enumerate(track.beats_seconds):
        if bars:
            position = bisect_right(starts, float(seconds)) - 1
            position = max(0, min(position, len(bars) - 1))
            bar_index = int(bars[position].bar_index)
        else:
            bar_seconds = beats_per_bar * 60.0 / max(float(track.bpm), 1.0)
            bar_index = max(0, int(float(seconds) / max(bar_seconds, 1e-9)))
        beat_in_bar = per_bar_counts.get(bar_index, 0)
        per_bar_counts[bar_index] = beat_in_bar + 1
        output.append(
            TimelineBeat(
                beat_index=beat_index,
                time_seconds=round(float(seconds), 6),
                bar_index=bar_index,
                beat_in_bar=beat_in_bar,
            )
        )
    return tuple(output)


def _timeline_fingerprint(track: TrackAnalysis) -> str:
    digest = hashlib.sha256()
    digest.update(TIMELINE_VERSION.encode("utf-8"))
    digest.update(str(track.path).encode("utf-8"))
    digest.update(str(track.analysis_version).encode("utf-8"))
    digest.update(str(track.neural_evidence_fingerprint).encode("utf-8"))
    digest.update(f"{track.duration_seconds:.6f}|{track.bpm:.6f}|{track.time_signature}".encode("utf-8"))
    for bar in track.bar_features:
        digest.update(
            (
                f"{bar.bar_index}|{bar.start_seconds:.6f}|{bar.end_seconds:.6f}|{bar.section_label}|"
                f"{bar.energy:.4f}|{bar.rhythm:.4f}|{bar.vocal_probability:.4f}|{bar.bass_activity:.4f}|"
                f"{bar.harmonic_tension:.4f}|{bar.boundary_strength:.4f}|{bar.phrase_strength:.4f}|"
                f"{bar.beat_grid_confidence:.4f}"
            ).encode("utf-8")
        )
    return digest.hexdigest()[:24]


def _build_musical_timeline_uncached(track: TrackAnalysis) -> MusicalTimeline:
    raw_bars = list(track.bar_features)
    bars = tuple(
        TimelineBar(
            bar_index=int(item.bar_index),
            start_seconds=float(item.start_seconds),
            end_seconds=float(item.end_seconds),
            section=str(item.section_label),
            energy=float(item.energy),
            rhythm=float(item.rhythm),
            brightness=float(item.brightness),
            vocal_probability=float(item.vocal_probability),
            bass_activity=float(item.bass_activity),
            tension=float(item.harmonic_tension),
            boundary_strength=float(item.boundary_strength),
            phrase_strength=float(item.phrase_strength),
            beat_confidence=float(item.beat_grid_confidence),
            local_bpm=float(item.local_bpm),
        )
        for item in raw_bars
    )
    phrases = _annotate_song_form(_build_phrases(raw_bars))
    sections = _build_sections(raw_bars, phrases)
    commitments = _build_commitments(phrases)
    beats = _build_beats(track, bars)
    return MusicalTimeline(
        version=TIMELINE_VERSION,
        track=track.path,
        bpm=float(track.bpm),
        time_signature=max(1, int(track.time_signature or 4)),
        duration_seconds=float(track.duration_seconds),
        beats=beats,
        bars=bars,
        phrases=phrases,
        sections=sections,
        commitments=commitments,
        fingerprint=_timeline_fingerprint(track),
    )


_TIMELINE_CACHE: dict[str, MusicalTimeline] = {}
_TIMELINE_CACHE_LOCK = threading.RLock()
_TIMELINE_CACHE_MAX = 256


def build_musical_timeline(track: TrackAnalysis) -> MusicalTimeline:
    """Return the canonical immutable musical timeline for an analyzed track.

    Raw analysis owns beat/bar evidence. Everything above bars -- phrases,
    sections, transition readiness and protected musical commitments -- is
    derived here so planning, Studio and deterministic performance cannot
    silently disagree about musical structure.
    """
    key = _timeline_fingerprint(track)
    with _TIMELINE_CACHE_LOCK:
        cached = _TIMELINE_CACHE.get(key)
    if cached is not None:
        return cached
    timeline = _build_musical_timeline_uncached(track)
    with _TIMELINE_CACHE_LOCK:
        if len(_TIMELINE_CACHE) >= _TIMELINE_CACHE_MAX:
            _TIMELINE_CACHE.pop(next(iter(_TIMELINE_CACHE)))
        _TIMELINE_CACHE[key] = timeline
    return timeline


def clear_musical_timeline_cache() -> None:
    with _TIMELINE_CACHE_LOCK:
        _TIMELINE_CACHE.clear()


def musical_timeline_status() -> dict[str, Any]:
    with _TIMELINE_CACHE_LOCK:
        cache_entries = len(_TIMELINE_CACHE)
    return {
        "schema_version": TIMELINE_VERSION,
        "authority": "beats -> bars -> adaptive phrases -> sections -> commitments",
        "canonical_phrase_authority": True,
        "song_form_roles": ["intro", "verse", "hook", "instrumental", "bridge", "build", "pre_drop", "drop", "breakdown", "outro"],
        "first_body_protection": True,
        "hook_protection": True,
        "first_hook_pre_entry_protection": True,
        "song_form_hard_guard": True,
        "world_model_is_projection": True,
        "move_brain_consumes_canonical_timeline": True,
        "studio_summary_available": True,
        "cache_entries": cache_entries,
        "cache_limit": _TIMELINE_CACHE_MAX,
    }
