"""DJAI learned-intelligence layer.

Rules remain responsible for technical feasibility. This package adds richer
music representations and preference-based ranking on top of feasible plans.
"""

from .embeddings import EmbeddingResult, embed_audio, embedding_similarity
from .edm_cue import import_edm_cue_annotations, edm_cue_stats
from .cue_detr import CuePrediction, predict_cue_points, rerank_analysis_candidates_with_cue_model
from .transition_ranker import (
    RankerPrediction,
    load_transition_ranker,
    rank_transition_plans,
    train_transition_ranker,
)

__all__ = [
    "CuePrediction",
    "EmbeddingResult",
    "RankerPrediction",
    "edm_cue_stats",
    "embed_audio",
    "import_edm_cue_annotations",
    "predict_cue_points",
    "rerank_analysis_candidates_with_cue_model",
    "embedding_similarity",
    "load_transition_ranker",
    "rank_transition_plans",
    "train_transition_ranker",
]
