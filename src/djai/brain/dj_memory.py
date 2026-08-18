from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from ..models import TrackAnalysis, TransitionPlan

DJ_MEMORY_VERSION = "6.0"


@dataclass(frozen=True, slots=True)
class DJMemory:
    """Musical memory for the current performance, independent of ML memory."""

    version: str = DJ_MEMORY_VERSION
    recent_tracks: tuple[str, ...] = ()
    recent_transition_families: tuple[str, ...] = ()
    recent_transition_lengths: tuple[int, ...] = ()
    recent_intents: tuple[str, ...] = ()
    recent_energy_changes: tuple[float, ...] = ()
    recent_keys: tuple[str, ...] = ()
    recent_vocal_density: tuple[float, ...] = ()
    peak_count: int = 0
    creative_count: int = 0
    emergency_count: int = 0
    surprise_budget: float = 0.5
    repetition_penalties: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "recent_tracks",
            "recent_transition_families",
            "recent_transition_lengths",
            "recent_intents",
            "recent_energy_changes",
            "recent_keys",
            "recent_vocal_density",
        ):
            payload[name] = list(payload[name])
        return payload

    def transition_repetition_penalty(self, technique: str, bars: int, intent: str) -> float:
        penalty = 0.0
        recent_families = self.recent_transition_families[-4:]
        recent_lengths = self.recent_transition_lengths[-4:]
        recent_intents = self.recent_intents[-4:]
        penalty += 0.10 * recent_families.count(technique)
        penalty += 0.055 * recent_lengths.count(int(bars))
        penalty += 0.055 * recent_intents.count(intent)
        if len(recent_families) >= 3 and len(set(recent_families[-3:])) == 1 and technique == recent_families[-1]:
            penalty += 0.12
        return float(np.clip(penalty, 0.0, 0.48))


def build_dj_memory(
    tracks: Sequence[TrackAnalysis],
    transitions: Sequence[TransitionPlan] | None = None,
    *,
    memory_tracks: int = 8,
    memory_transitions: int = 10,
) -> DJMemory:
    recent_tracks = list(tracks[-memory_tracks:])
    recent_transitions = list((transitions or [])[-memory_transitions:])
    energy = [float(track.energy) for track in recent_tracks]
    changes = [b - a for a, b in zip(energy, energy[1:])]
    peaks = sum(value >= 0.82 for value in energy)
    creative = sum(plan.style == "creative" for plan in recent_transitions)
    emergency = sum(plan.style == "emergency" for plan in recent_transitions)

    # Surprise is a renewable budget: repeated creative/emergency moves spend
    # it; long stretches of foundational mixing allow it to rebuild.
    spent = min(1.0, 0.15 * creative + 0.32 * emergency)
    calm_tail = 0
    for plan in reversed(recent_transitions):
        if plan.style != "safe":
            break
        calm_tail += 1
    surprise_budget = float(np.clip(0.56 - spent + 0.08 * calm_tail, 0.08, 0.92))

    techniques = tuple(plan.technique for plan in recent_transitions)
    lengths = tuple(int(plan.transition_bars) for plan in recent_transitions)
    intents = tuple(plan.intent for plan in recent_transitions)
    penalties: dict[str, float] = {}
    for technique in set(techniques):
        penalties[f"technique:{technique}"] = round(min(0.45, 0.09 * techniques[-5:].count(technique)), 4)
    for bars in set(lengths):
        penalties[f"bars:{bars}"] = round(min(0.30, 0.05 * lengths[-5:].count(bars)), 4)

    return DJMemory(
        recent_tracks=tuple(track.path for track in recent_tracks),
        recent_transition_families=techniques,
        recent_transition_lengths=lengths,
        recent_intents=intents,
        recent_energy_changes=tuple(round(value, 4) for value in changes[-6:]),
        recent_keys=tuple(str(getattr(track.key, "camelot", "unknown")) for track in recent_tracks),
        recent_vocal_density=tuple(round(float(track.vocal_probability), 4) for track in recent_tracks),
        peak_count=int(peaks),
        creative_count=int(creative),
        emergency_count=int(emergency),
        surprise_budget=round(surprise_budget, 4),
        repetition_penalties=penalties,
    )
