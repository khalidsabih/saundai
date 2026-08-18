from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .environment import config_path
from .models import GenreEstimate, TrackAnalysis


FEATURE_NAMES = (
    "bpm",
    "energy",
    "brightness",
    "percussive_ratio",
    "bass_activity",
    "vocal_probability",
    "onset_density",
)

# These centroids are conservative priors, not a claim of a fully trained
# commercial genre model. A user-labelled library can replace them with
# `djai genre-train`.
DEFAULT_CENTROIDS: dict[str, list[float]] = {
    "house": [124.0, 0.72, 0.23, 0.55, 0.56, 0.36, 2.0],
    "tech_house": [126.0, 0.78, 0.25, 0.64, 0.64, 0.24, 2.3],
    "techno": [132.0, 0.82, 0.28, 0.70, 0.68, 0.12, 2.6],
    "melodic_techno": [124.0, 0.74, 0.22, 0.48, 0.52, 0.32, 1.8],
    "trance": [138.0, 0.82, 0.31, 0.57, 0.50, 0.40, 2.2],
    "drum_and_bass": [174.0, 0.86, 0.39, 0.80, 0.68, 0.28, 3.8],
    "hip_hop": [92.0, 0.67, 0.24, 0.54, 0.73, 0.67, 1.7],
    "pop": [116.0, 0.70, 0.34, 0.48, 0.45, 0.72, 1.9],
    "ambient": [82.0, 0.34, 0.17, 0.14, 0.25, 0.22, 0.5],
}

DEFAULT_SCALES = [24.0, 0.24, 0.16, 0.23, 0.23, 0.28, 1.4]


def feature_vector(analysis: TrackAnalysis) -> np.ndarray:
    return np.asarray(
        [
            analysis.bpm,
            analysis.energy,
            analysis.brightness,
            analysis.percussive_ratio,
            analysis.bass_activity,
            analysis.vocal_probability,
            analysis.onset_density,
        ],
        dtype=np.float64,
    )


@lru_cache(maxsize=8)
def _load_model_cached(
    resolved_path: str,
    modification_ns: int,
    size_bytes: int,
) -> tuple[dict[str, list[float]], list[float], str]:
    del modification_ns, size_bytes  # File metadata intentionally keys the cache.
    payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    centroids = {
        str(label): [float(value) for value in values]
        for label, values in payload["centroids"].items()
    }
    scales = [float(value) for value in payload.get("scales", DEFAULT_SCALES)]
    return centroids, scales, str(payload.get("model", "centroid-trained-v1"))


def clear_genre_model_cache() -> None:
    _load_model_cached.cache_clear()


def _load_model(path: str | Path | None) -> tuple[dict[str, list[float]], list[float], str]:
    model_path = config_path("genre-model.json") if path is None else config_path(path)
    try:
        resolved = model_path.resolve()
        stat = resolved.stat()
    except OSError:
        return DEFAULT_CENTROIDS, DEFAULT_SCALES, "centroid-prior-v1"
    try:
        return _load_model_cached(str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return DEFAULT_CENTROIDS, DEFAULT_SCALES, "centroid-prior-v1"


def classify_genre(
    analysis: TrackAnalysis,
    *,
    model_path: str | Path | None = None,
) -> GenreEstimate:
    centroids, scales, model_name = _load_model(model_path)
    vector = feature_vector(analysis)
    scale = np.maximum(np.asarray(scales, dtype=np.float64), 1e-6)

    labels: list[str] = []
    logits: list[float] = []
    for label, centroid_values in centroids.items():
        centroid = np.asarray(centroid_values, dtype=np.float64)
        if centroid.shape != vector.shape:
            continue
        distance = float(np.sqrt(np.mean(np.square((vector - centroid) / scale))))
        labels.append(label)
        logits.append(-2.2 * distance)

    if not labels:
        return GenreEstimate()
    maximum = max(logits)
    exp_values = np.exp(np.asarray(logits) - maximum)
    probabilities = exp_values / max(float(np.sum(exp_values)), 1e-12)
    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    best_probability = float(probabilities[best])

    # Low-confidence classifications are explicitly marked unknown while still
    # retaining the probability distribution for planning and UI display.
    label = labels[best] if best_probability >= 0.22 else "unknown"
    return GenreEstimate(
        label=label,
        confidence=round(best_probability, 4),
        probabilities={
            labels[index]: round(float(probabilities[index]), 4)
            for index in order
        },
        model=model_name,
    )


def fit_centroid_model(
    labelled: Iterable[tuple[TrackAnalysis, str]],
    output_path: str | Path,
) -> Path:
    groups: dict[str, list[np.ndarray]] = {}
    all_vectors: list[np.ndarray] = []
    for analysis, label in labelled:
        clean = label.strip().lower().replace(" ", "_")
        if not clean:
            continue
        vector = feature_vector(analysis)
        groups.setdefault(clean, []).append(vector)
        all_vectors.append(vector)
    if len(groups) < 2:
        raise ValueError("Genre training needs at least two labelled genres.")

    centroids = {
        label: np.mean(np.vstack(vectors), axis=0).tolist()
        for label, vectors in sorted(groups.items())
    }
    matrix = np.vstack(all_vectors)
    scales = np.std(matrix, axis=0)
    scales = np.maximum(scales, np.asarray(DEFAULT_SCALES) * 0.25)

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": "centroid-trained-v1",
                "feature_names": FEATURE_NAMES,
                "centroids": centroids,
                "scales": scales.tolist(),
                "sample_count": len(all_vectors),
                "labels": sorted(groups),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    clear_genre_model_cache()
    return path


def train_from_manifest(
    manifest: str | Path,
    output_path: str | Path,
) -> Path:
    # Local import prevents a circular dependency during normal track analysis.
    from .analysis import analyze_track

    manifest_path = Path(manifest).expanduser().resolve()
    rows: list[tuple[TrackAnalysis, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
            raise ValueError("Manifest must contain path,label columns.")
        for row in reader:
            raw_path = str(row.get("path", "")).strip()
            label = str(row.get("label", "")).strip()
            if not raw_path or not label:
                continue
            track_path = Path(raw_path)
            if not track_path.is_absolute():
                track_path = manifest_path.parent / track_path
            rows.append((analyze_track(track_path), label))
    return fit_centroid_model(rows, output_path)
