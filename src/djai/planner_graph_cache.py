from __future__ import annotations

"""Graph-pair cache contracts used by set planning.

This module owns cache identity, serialization and durable storage only. Graph
construction, scoring, worker orchestration and musical policy remain in
set_planner so this extraction is behavior-preserving.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .environment import current_environment
from .models import CompatibilityBreakdown, TrackAnalysis
from .planning_context import CandidatePairEvidence, PairEvidenceMatrix
from .serialization import transition_candidate_from_dict, transition_plan_from_dict
from .set_models import TransitionEdge

GRAPH_PAIR_CACHE_VERSION = "1.0"


@dataclass(slots=True)
class GraphPairResult:
    from_index: int
    to_index: int
    label: str
    edge: TransitionEdge
    evidence_matrix: PairEvidenceMatrix


def analysis_graph_fingerprint(track: TrackAnalysis) -> str:
    payload = track.to_dict(include_beats=False)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def graph_pair_cache_key(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    *,
    tempo_policy: str,
    preferred_style: str,
    journey: str,
    scout_cues: int,
    model_fingerprint: str | None,
) -> str:
    payload = {
        "version": GRAPH_PAIR_CACHE_VERSION,
        "track_a": analysis_graph_fingerprint(track_a),
        "track_b": analysis_graph_fingerprint(track_b),
        "tempo_policy": tempo_policy,
        "preferred_style": preferred_style,
        "journey": journey,
        "scout_cues": int(scout_cues),
        "scoring_model_fingerprint": model_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def graph_pair_cache_path(key: str):
    return current_environment().cache_dir / "graph-pairs" / f"{key}.json"


def pair_evidence_to_dict(matrix: PairEvidenceMatrix) -> dict[str, Any]:
    return {
        "track_a_path": matrix.track_a_path,
        "track_b_path": matrix.track_b_path,
        "exit_candidates": [asdict(item) for item in matrix.exit_candidates],
        "entry_candidates": [asdict(item) for item in matrix.entry_candidates],
        "pairs": [
            {
                "exit_index": item.exit_index,
                "entry_index": item.entry_index,
                "exit_candidate": asdict(item.exit_candidate),
                "entry_candidate": asdict(item.entry_candidate),
                "breakdown": asdict(item.breakdown),
            }
            for item in matrix.pairs
        ],
        "scoring_model_fingerprint": matrix.scoring_model_fingerprint,
    }


def pair_evidence_from_dict(payload: dict[str, Any]) -> PairEvidenceMatrix:
    return PairEvidenceMatrix(
        track_a_path=str(payload["track_a_path"]),
        track_b_path=str(payload["track_b_path"]),
        exit_candidates=tuple(
            transition_candidate_from_dict(dict(item))
            for item in payload.get("exit_candidates", [])
        ),
        entry_candidates=tuple(
            transition_candidate_from_dict(dict(item))
            for item in payload.get("entry_candidates", [])
        ),
        pairs=tuple(
            CandidatePairEvidence(
                exit_index=int(item["exit_index"]),
                entry_index=int(item["entry_index"]),
                exit_candidate=transition_candidate_from_dict(dict(item["exit_candidate"])),
                entry_candidate=transition_candidate_from_dict(dict(item["entry_candidate"])),
                breakdown=CompatibilityBreakdown(**dict(item["breakdown"])),
            )
            for item in payload.get("pairs", [])
        ),
        scoring_model_fingerprint=payload.get("scoring_model_fingerprint"),
    )


def load_graph_pair_cache(
    key: str,
    *,
    from_index: int,
    to_index: int,
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
) -> GraphPairResult | None:
    path = graph_pair_cache_path(key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != GRAPH_PAIR_CACHE_VERSION:
            return None
        matrix = pair_evidence_from_dict(dict(payload["evidence_matrix"]))
        matrix.validate_tracks(track_a, track_b)
        edge_payload = dict(payload["edge"])
        edge = TransitionEdge(
            from_index=from_index,
            to_index=to_index,
            selected_style=str(edge_payload["selected_style"]),
            objective_score=float(edge_payload["objective_score"]),
            quality=str(edge_payload["quality"]),
            plan=transition_plan_from_dict(dict(edge_payload["plan"])),
            alternatives=[
                transition_plan_from_dict(dict(item))
                for item in edge_payload.get("alternatives", [])
            ],
            warnings=[str(item) for item in edge_payload.get("warnings", [])],
            reasons=[str(item) for item in edge_payload.get("reasons", [])],
        )
        return GraphPairResult(
            from_index=from_index,
            to_index=to_index,
            label=f"{track_a.title} -> {track_b.title}",
            edge=edge,
            evidence_matrix=matrix,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def store_graph_pair_cache(key: str, result: GraphPairResult) -> None:
    path = graph_pair_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": GRAPH_PAIR_CACHE_VERSION,
            "from_index": result.from_index,
            "to_index": result.to_index,
            "edge": result.edge.to_dict(),
            "evidence_matrix": pair_evidence_to_dict(result.evidence_matrix),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        return
