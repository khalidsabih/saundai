from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .compatibility import score_transition
from .learning import load_scoring_model
from .models import CompatibilityBreakdown, TrackAnalysis, TransitionCandidate

CandidateKey = tuple[str, int, float, str]
ScoringModelPayload = Mapping[str, Any]
_DEFAULT_MODEL = object()


def scoring_model_fingerprint(model: ScoringModelPayload | None) -> str | None:
    if model is None:
        return None
    encoded = json.dumps(dict(model), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def candidate_key(candidate: TransitionCandidate) -> CandidateKey:
    """Stable cue identity that survives process serialization."""
    return (
        candidate.kind,
        int(candidate.bar_index),
        round(float(candidate.time_seconds), 6),
        candidate.section_label,
    )


@dataclass(frozen=True, slots=True)
class CandidatePairEvidence:
    exit_index: int
    entry_index: int
    exit_candidate: TransitionCandidate
    entry_candidate: TransitionCandidate
    breakdown: CompatibilityBreakdown

    @property
    def cue_signature(self) -> tuple[int, int]:
        return self.exit_candidate.bar_index, self.entry_candidate.bar_index


@dataclass(frozen=True, slots=True)
class PairEvidenceMatrix:
    track_a_path: str
    track_b_path: str
    exit_candidates: tuple[TransitionCandidate, ...]
    entry_candidates: tuple[TransitionCandidate, ...]
    pairs: tuple[CandidatePairEvidence, ...]
    scoring_model_fingerprint: str | None = None

    @property
    def compatibility_evaluations(self) -> int:
        return len(self.pairs)

    def validate_tracks(self, track_a: TrackAnalysis, track_b: TrackAnalysis) -> None:
        if self.track_a_path != track_a.path or self.track_b_path != track_b.path:
            raise ValueError("Compatibility evidence belongs to a different track pair.")

    def subset(
        self,
        *,
        exit_candidates: Sequence[TransitionCandidate] | None = None,
        entry_candidates: Sequence[TransitionCandidate] | None = None,
    ) -> PairEvidenceMatrix:
        """Return an ordered matrix view without recomputing compatibility."""
        requested_exits = list(exit_candidates) if exit_candidates is not None else list(self.exit_candidates)
        requested_entries = list(entry_candidates) if entry_candidates is not None else list(self.entry_candidates)
        exit_lookup = {candidate_key(value): value for value in self.exit_candidates}
        entry_lookup = {candidate_key(value): value for value in self.entry_candidates}
        pair_lookup = {
            (candidate_key(item.exit_candidate), candidate_key(item.entry_candidate)): item
            for item in self.pairs
        }
        selected_exits = [
            exit_lookup[key]
            for value in requested_exits
            if (key := candidate_key(value)) in exit_lookup
        ]
        selected_entries = [
            entry_lookup[key]
            for value in requested_entries
            if (key := candidate_key(value)) in entry_lookup
        ]
        selected_pairs = tuple(
            pair_lookup[(candidate_key(exit_candidate), candidate_key(entry_candidate))]
            for exit_candidate in selected_exits
            for entry_candidate in selected_entries
            if (candidate_key(exit_candidate), candidate_key(entry_candidate)) in pair_lookup
        )
        return PairEvidenceMatrix(
            track_a_path=self.track_a_path,
            track_b_path=self.track_b_path,
            exit_candidates=tuple(selected_exits),
            entry_candidates=tuple(selected_entries),
            pairs=selected_pairs,
            scoring_model_fingerprint=self.scoring_model_fingerprint,
        )


@dataclass(slots=True)
class PlanningSession:
    """Per-planning-job model snapshot, pair evidence, and performance stats."""

    scoring_model: dict[str, Any] | None = field(default_factory=load_scoring_model)
    pair_evidence: dict[tuple[int, int], PairEvidenceMatrix] = field(default_factory=dict)
    compatibility_evaluations: int = 0
    graph_workers: int = 1
    graph_seconds: float = 0.0
    graph_mode: str = "full"
    graph_pair_count: int = 0
    graph_scout_cues: int = 0

    def store(self, key: tuple[int, int], matrix: PairEvidenceMatrix) -> None:
        previous = self.pair_evidence.get(key)
        if previous is not None:
            self.compatibility_evaluations -= previous.compatibility_evaluations
        self.pair_evidence[key] = matrix
        self.compatibility_evaluations += matrix.compatibility_evaluations

    def get(self, key: tuple[int, int]) -> PairEvidenceMatrix | None:
        return self.pair_evidence.get(key)

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_contexts": len(self.pair_evidence),
            "compatibility_evaluations": self.compatibility_evaluations,
            "graph_workers": self.graph_workers,
            "graph_parallel": self.graph_workers > 1,
            "graph_seconds": round(self.graph_seconds, 6),
            "graph_mode": self.graph_mode,
            "graph_pair_count": self.graph_pair_count,
            "graph_scout_cues": self.graph_scout_cues,
            "scoring_model_fingerprint": scoring_model_fingerprint(self.scoring_model),
        }


def build_pair_evidence_matrix(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    exit_candidates: Sequence[TransitionCandidate] | None = None,
    entry_candidates: Sequence[TransitionCandidate] | None = None,
    scoring_model: ScoringModelPayload | None | object = _DEFAULT_MODEL,
) -> PairEvidenceMatrix:
    """Compute style-independent cue-pair evidence exactly once."""
    exits = tuple(exit_candidates) if exit_candidates is not None else tuple(track_a.exit_candidates)
    entries = tuple(entry_candidates) if entry_candidates is not None else tuple(track_b.entry_candidates)
    if not exits or not entries:
        raise RuntimeError("No valid transition candidates were found.")

    resolved_model: ScoringModelPayload | None
    if scoring_model is _DEFAULT_MODEL:
        resolved_model = load_scoring_model()
    else:
        resolved_model = scoring_model  # type: ignore[assignment]

    pairs = tuple(
        CandidatePairEvidence(
            exit_index=exit_index,
            entry_index=entry_index,
            exit_candidate=exit_candidate,
            entry_candidate=entry_candidate,
            breakdown=score_transition(
                track_a,
                track_b,
                exit_candidate,
                entry_candidate,
                scoring_model=resolved_model,
            ),
        )
        for exit_index, exit_candidate in enumerate(exits)
        for entry_index, entry_candidate in enumerate(entries)
    )
    return PairEvidenceMatrix(
        track_a_path=track_a.path,
        track_b_path=track_b.path,
        exit_candidates=exits,
        entry_candidates=entries,
        pairs=pairs,
        scoring_model_fingerprint=scoring_model_fingerprint(resolved_model),
    )
