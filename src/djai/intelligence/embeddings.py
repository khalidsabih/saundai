from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


DEFAULT_DIMENSION = 96
DEFAULT_SEGMENT_SECONDS = 24.0
DEFAULT_HOP_SECONDS = 12.0
DEFAULT_MODEL_NAME = "m-a-p/MERT-v1-95M"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    model: str
    dimension: int
    vector: tuple[float, ...]
    backend: str
    fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "dimension": self.dimension,
            "vector": list(self.vector),
            "backend": self.backend,
            "fingerprint": self.fingerprint,
        }


def _normalized(values: np.ndarray, dimension: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size < dimension:
        vector = np.pad(vector, (0, dimension - vector.size))
    elif vector.size > dimension:
        # Deterministic signed hashing keeps this dependency-free and stable.
        folded = np.zeros(dimension, dtype=np.float64)
        for index, value in enumerate(vector):
            slot = index % dimension
            folded[slot] += value if (index // dimension) % 2 == 0 else -value
        vector = folded
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector = vector / norm
    return vector.astype(np.float32)


def _statistical_segment_embedding(y: np.ndarray, sr: int, *, dimension: int) -> np.ndarray:
    """Strong local fallback representation used when MERT is unavailable.

    This is intentionally richer than the legacy seven-scalar genre vector and
    provides a deterministic representation for tests/offline installs. It is
    not presented as a pretrained model.
    """
    if y.size == 0:
        return np.zeros(dimension, dtype=np.float32)
    hop = 512
    n_fft = 2048
    harmonic, percussive = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    perc_rms = librosa.feature.rms(y=percussive, frame_length=n_fft, hop_length=hop)

    def moments(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.size == 0:
            return np.empty(0, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        return np.concatenate(
            [
                np.mean(matrix, axis=1),
                np.std(matrix, axis=1),
                np.percentile(matrix, 25, axis=1),
                np.percentile(matrix, 75, axis=1),
            ]
        )

    features = np.concatenate(
        [
            moments(mfcc),
            moments(chroma),
            moments(contrast),
            moments(centroid),
            moments(bandwidth),
            moments(rolloff),
            moments(flatness),
            moments(onset),
            moments(rms),
            moments(perc_rms),
        ]
    )
    return _normalized(features, dimension)



@lru_cache(maxsize=4)
def _musicfm_runtime(home_value: str, stat_value: str, model_value: str, device_value: str):
    """Load one MusicFM runtime per process instead of once per track section."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MusicFM embeddings require PyTorch.") from exc
    home = Path(home_value).expanduser().resolve()
    stat = Path(stat_value).expanduser().resolve()
    model_path = Path(model_value).expanduser().resolve()
    if not home.exists() or not stat.exists() or not model_path.exists():
        raise RuntimeError("MusicFM checkout/stats/checkpoint is missing.")
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    try:
        from model.musicfm_25hz import MusicFM25Hz
    except ImportError as exc:
        raise RuntimeError("Configured MusicFM checkout could not be imported.") from exc
    device = device_value
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu"
        )
    model = MusicFM25Hz(is_flash=False, stat_path=str(stat), model_path=str(model_path)).to(device).eval()
    return model, device


def _musicfm_embedding(y: np.ndarray, sr: int, *, dimension: int) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MusicFM embeddings require PyTorch.") from exc
    home_value = os.environ.get("DJAI_MUSICFM_HOME", "").strip()
    if not home_value:
        raise RuntimeError("DJAI_MUSICFM_HOME is not configured.")
    home = Path(home_value).expanduser().resolve()
    stat_value = os.environ.get("DJAI_MUSICFM_STATS", str(home / "data" / "fma_stats.json"))
    model_value = os.environ.get("DJAI_MUSICFM_CHECKPOINT", str(home / "data" / "pretrained_fma.pt"))
    device_value = os.environ.get("DJAI_MUSICFM_DEVICE", "auto").strip().lower()
    model, device = _musicfm_runtime(home_value, stat_value, model_value, device_value)
    if sr != 24000:
        y = librosa.resample(y, orig_sr=sr, target_sr=24000)
    wav = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(1, -1)
    with torch.inference_mode():
        latent = model.get_latent(wav.to(device), layer_ix=int(os.environ.get("DJAI_MUSICFM_LAYER", "7")))
    arr = np.asarray(latent.detach().float().cpu().numpy()).squeeze(0)
    pooled = arr.mean(axis=0 if arr.ndim == 2 and arr.shape[0] >= arr.shape[1] else 1) if arr.ndim == 2 else arr.reshape(-1)
    return _normalized(np.asarray(pooled), dimension)


@lru_cache(maxsize=4)
def _mert_runtime(model_name: str, local_only: bool):
    """Load MERT once per process; section embeddings reuse the same weights."""
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MERT embeddings require the optional 'embeddings' dependencies.") from exc
    extractor = AutoFeatureExtractor.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_only
    )
    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_only
    )
    model.eval()
    return extractor, model


def _mert_embedding(y: np.ndarray, sr: int, *, model_name: str, dimension: int) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MERT embeddings require the optional 'embeddings' dependencies.") from exc

    local_only = os.environ.get("DJAI_EMBEDDING_ALLOW_DOWNLOAD", "0").strip().lower() not in {
        "1", "true", "yes", "on"
    }
    extractor, model = _mert_runtime(model_name, local_only)
    target_sr = int(getattr(extractor, "sampling_rate", 24_000) or 24_000)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states:
        # MERT documentation commonly recommends aggregating hidden layers for
        # downstream tasks rather than relying on one arbitrary layer.
        layers = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)
        pooled = layers.mean(dim=1).squeeze(0).cpu().numpy()
    else:  # pragma: no cover - compatibility with unusual model wrappers
        pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return _normalized(np.asarray(pooled), dimension)


def _fingerprint(vector: np.ndarray, model: str, backend: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(backend.encode("utf-8"))
    digest.update(np.asarray(vector, dtype=np.float32).tobytes())
    return digest.hexdigest()[:24]


def embed_array(
    y: np.ndarray,
    sr: int,
    *,
    backend: str | None = None,
    model_name: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
) -> EmbeddingResult:
    requested = (backend or os.environ.get("DJAI_EMBEDDING_BACKEND", "auto")).strip().lower()
    model_name = model_name or os.environ.get("DJAI_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    if requested not in {"auto", "musicfm", "mert", "statistical"}:
        raise ValueError("Embedding backend must be auto, musicfm, mert, or statistical.")

    chosen = requested
    vector: np.ndarray | None = None
    if requested in {"auto", "musicfm"}:
        try:
            vector=_musicfm_embedding(y,sr,dimension=dimension); chosen="musicfm"
        except (RuntimeError,OSError,ValueError,ImportError):
            if requested=="musicfm": raise
    if vector is None and requested in {"auto", "mert"}:
        try:
            vector=_mert_embedding(y,sr,model_name=model_name,dimension=dimension); chosen="mert"
        except (RuntimeError,OSError,ValueError):
            if requested=="mert": raise
    if vector is None:
        vector=_statistical_segment_embedding(y,sr,dimension=dimension); chosen="statistical"
    model=model_name if chosen=="mert" else (os.environ.get("DJAI_MUSICFM_MODEL_NAME","MusicFM-FMA") if chosen=="musicfm" else "djai-statistical-local-v1")
    return EmbeddingResult(
        model=model,
        dimension=dimension,
        vector=tuple(float(value) for value in vector),
        backend=chosen,
        fingerprint=_fingerprint(vector, model, chosen),
    )


def embed_track_array(
    y: np.ndarray,
    sr: int,
    *,
    backend: str | None = None,
    model_name: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
) -> EmbeddingResult:
    """Represent a full track from bounded, distributed listening windows.

    Foundation encoders scale with audio duration. Feeding a five-minute track
    to MERT/MusicFM and then separately encoding its transition sections made
    first-time large-crate analysis unnecessarily expensive. A DJ does not need
    every sample to form a global representation: sample several windows across
    the record, keep the same encoder, and pool their normalized embeddings.
    """
    audio = np.asarray(y, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return embed_array(audio, sr, backend=backend, model_name=model_name, dimension=dimension)
    try:
        window_seconds = max(8.0, min(45.0, float(os.environ.get("DJAI_TRACK_EMBED_WINDOW_SECONDS", "24"))))
    except ValueError:
        window_seconds = 24.0
    try:
        window_count = max(1, min(6, int(os.environ.get("DJAI_TRACK_EMBED_WINDOWS", "3"))))
    except ValueError:
        window_count = 3
    duration = audio.size / max(1, int(sr))
    if duration <= window_seconds * 1.35 or window_count == 1:
        return embed_array(audio, sr, backend=backend, model_name=model_name, dimension=dimension)

    window_samples = max(1, int(round(window_seconds * sr)))
    max_start = max(0, audio.size - window_samples)
    starts = np.linspace(0, max_start, num=window_count, dtype=np.int64)
    vectors: list[np.ndarray] = []
    chosen_backend = backend
    first: EmbeddingResult | None = None
    for raw_start in starts:
        start = int(raw_start)
        segment = audio[start : min(audio.size, start + window_samples)]
        result = embed_array(
            segment,
            sr,
            backend=chosen_backend,
            model_name=model_name,
            dimension=dimension,
        )
        if first is None:
            first = result
            chosen_backend = result.backend
        vectors.append(np.asarray(result.vector, dtype=np.float32))
    assert first is not None
    pooled = _normalized(np.mean(np.stack(vectors, axis=0), axis=0), dimension)
    return EmbeddingResult(
        model=first.model,
        dimension=dimension,
        vector=tuple(float(value) for value in pooled),
        backend=first.backend,
        fingerprint=_fingerprint(pooled, first.model, f"{first.backend}-track-windows-{window_count}x{window_seconds:g}s"),
    )


def embed_audio(
    path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    sample_rate: int = 24_000,
    backend: str | None = None,
    model_name: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
) -> EmbeddingResult:
    source = Path(path).expanduser().resolve()
    y, sr = librosa.load(
        source,
        sr=sample_rate,
        mono=True,
        offset=max(0.0, float(start_seconds)),
        duration=None if duration_seconds is None else max(0.1, float(duration_seconds)),
    )
    return embed_array(y, sr, backend=backend, model_name=model_name, dimension=dimension)


def section_embeddings(
    path: str | Path,
    sections: Iterable[tuple[str, float]],
    *,
    track_duration: float,
    backend: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
) -> dict[str, list[float]]:
    ordered = sorted((str(label), max(0.0, float(time))) for label, time in sections)
    output: dict[str, list[float]] = {}
    for index, (label, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else track_duration
        duration = min(DEFAULT_SEGMENT_SECONDS, max(4.0, end - start))
        result = embed_audio(path, start_seconds=start, duration_seconds=duration, backend=backend, dimension=dimension)
        # Keep the strongest/first representation for a repeated section label.
        output.setdefault(label, list(result.vector))
    return output


def embedding_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    left = np.asarray(list(a), dtype=np.float64)
    right = np.asarray(list(b), dtype=np.float64)
    if left.size == 0 or right.size == 0:
        return 0.5
    size = min(left.size, right.size)
    left, right = left[:size], right[:size]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.5
    cosine = float(np.dot(left, right) / denom)
    return float(np.clip((cosine + 1.0) * 0.5, 0.0, 1.0))


def section_embeddings_from_array(
    y: np.ndarray,
    sr: int,
    sections: Iterable[tuple[str, float]],
    *,
    track_duration: float,
    backend: str | None = None,
    dimension: int = DEFAULT_DIMENSION,
) -> dict[str, list[float]]:
    ordered = sorted((str(label), max(0.0, float(time))) for label, time in sections)
    output: dict[str, list[float]] = {}
    for index, (label, start) in enumerate(ordered):
        if label in output:
            continue
        end = ordered[index + 1][1] if index + 1 < len(ordered) else track_duration
        duration = min(DEFAULT_SEGMENT_SECONDS, max(4.0, end - start))
        start_sample = max(0, min(y.size, int(round(start * sr))))
        end_sample = max(start_sample + 1, min(y.size, int(round((start + duration) * sr))))
        result = embed_array(y[start_sample:end_sample], sr, backend=backend, dimension=dimension)
        output[label] = list(result.vector)
    return output
