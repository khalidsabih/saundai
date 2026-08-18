from __future__ import annotations

import os
from dataclasses import dataclass

import librosa
import numpy as np

from ..intelligence.embeddings import EmbeddingResult, embed_array


@dataclass(frozen=True, slots=True)
class RenderedTransitionEmbedding:
    vector: tuple[float, ...]
    backend: str
    model: str
    fingerprint: str


def embed_rendered_transition(
    audio: np.ndarray,
    sample_rate: int,
    *,
    dimension: int = 96,
) -> RenderedTransitionEmbedding:
    """Encode the audio DJAI actually rendered, not only source metadata."""
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=0, dtype=np.float32)
    else:
        values = values.reshape(-1)
    # Statistical is intentionally the default for candidate rehearsal: it is
    # local, deterministic, fast, and does not reload a giant encoder per idea.
    # Research builds may override this once a cached batch encoder is present.
    backend = os.environ.get("DJAI_RENDER_EMBEDDING_BACKEND", "statistical")
    embedding_rate = int(sample_rate)
    # Some spectral representations (notably CQT chroma in the deterministic
    # fallback) require more high-frequency headroom than low-rate rehearsal
    # tests or speed-optimized previews provide. Resampling only the embedding
    # view keeps the rendered/measurement audio untouched.
    if embedding_rate < 16_000 and values.size:
        values = librosa.resample(values, orig_sr=embedding_rate, target_sr=22_050)
        embedding_rate = 22_050
    result: EmbeddingResult = embed_array(
        values,
        embedding_rate,
        backend=backend,
        dimension=dimension,
    )
    return RenderedTransitionEmbedding(
        vector=result.vector,
        backend=result.backend,
        model=result.model,
        fingerprint=result.fingerprint,
    )
