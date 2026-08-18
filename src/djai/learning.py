from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .environment import config_path


SCORING_FEATURES = (
    "tempo",
    "harmonic",
    "energy",
    "texture",
    "beat_grid",
    "phrase",
    "sections",
    "genre",
    "vocal_overlap",
    "bass_conflict",
    "boundary",
    "section_fit",
)

_DEFAULT_MODEL = object()
MINIMUM_SAMPLES = 8
FULL_TRUST_SAMPLES = 200
MAXIMUM_LEARNED_WEIGHT = 0.32
_REFIT_LOCK = threading.Lock()


def _model_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return config_path(os.environ.get("DJAI_SCORING_MODEL", "scoring-model.json"))


def _model_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=8)
def _load_scoring_model_cached(
    resolved_path: str,
    modification_ns: int,
    size_bytes: int,
) -> dict[str, Any] | None:
    del modification_ns, size_bytes
    try:
        payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("feature_names") != list(SCORING_FEATURES):
        return None
    return payload


def clear_scoring_model_cache() -> None:
    _load_scoring_model_cached.cache_clear()


def load_scoring_model(path: str | Path | None = None) -> dict[str, Any] | None:
    signature = _model_signature(_model_path(path))
    if signature is None:
        return None
    return _load_scoring_model_cached(*signature)


def scoring_model_fingerprint(model: Mapping[str, Any] | None) -> str | None:
    if model is None:
        return None
    encoded = json.dumps(dict(model), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _genre_key(genre_pair: Sequence[str] | None) -> str | None:
    if not genre_pair or len(genre_pair) < 2:
        return None
    left = str(genre_pair[0] or "unknown")
    right = str(genre_pair[1] or "unknown")
    if left == "unknown" or right == "unknown":
        return None
    return f"{left}->{right}"


def _select_model(
    model: Mapping[str, Any],
    genre_pair: Sequence[str] | None,
) -> Mapping[str, Any]:
    key = _genre_key(genre_pair)
    genre_models = model.get("genre_models")
    if key and isinstance(genre_models, Mapping):
        selected = genre_models.get(key)
        if isinstance(selected, Mapping):
            return selected
    return model


def learned_model_weight(model: Mapping[str, Any] | None) -> float:
    if model is None:
        return 0.0
    if "learned_weight" in model:
        try:
            return float(np.clip(float(model["learned_weight"]), 0.0, MAXIMUM_LEARNED_WEIGHT))
        except (TypeError, ValueError):
            return 0.0
    sample_count = int(model.get("sample_count", 0))
    sample_factor = float(
        np.clip(
            (sample_count - MINIMUM_SAMPLES) / max(1, FULL_TRUST_SAMPLES - MINIMUM_SAMPLES),
            0.0,
            1.0,
        )
    )
    cv_skill = float(model.get("cv_skill", 0.0) or 0.0)
    quality_factor = float(np.clip(cv_skill / 0.20, 0.0, 1.0))
    return MAXIMUM_LEARNED_WEIGHT * sample_factor * quality_factor


def predict_learned_score(
    features: dict[str, float],
    *,
    model_path: str | Path | None = None,
    model: Mapping[str, Any] | None | object = _DEFAULT_MODEL,
    genre_pair: Sequence[str] | None = None,
) -> float | None:
    resolved_model: Mapping[str, Any] | None
    if model is _DEFAULT_MODEL:
        resolved_model = load_scoring_model(model_path)
    else:
        resolved_model = model  # type: ignore[assignment]
    if resolved_model is None:
        return None
    selected = _select_model(resolved_model, genre_pair)
    if int(selected.get("sample_count", 0)) < MINIMUM_SAMPLES:
        return None
    weights = np.asarray(selected.get("weights", []), dtype=np.float64)
    if weights.size != len(SCORING_FEATURES):
        return None
    vector = np.asarray(
        [float(features.get(name, 0.5)) for name in SCORING_FEATURES],
        dtype=np.float64,
    )
    score = float(selected.get("intercept", 0.0) + np.dot(weights, vector))
    return float(np.clip(score, 0.0, 1.0))


def learned_prediction(
    features: dict[str, float],
    *,
    model: Mapping[str, Any] | None,
    genre_pair: Sequence[str] | None = None,
) -> tuple[float | None, float]:
    if model is None:
        return None, 0.0
    selected = _select_model(model, genre_pair)
    score = predict_learned_score(features, model=model, genre_pair=genre_pair)
    return score, learned_model_weight(selected)


def _ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    means = np.average(x, axis=0, weights=sample_weights)
    variance = np.average(np.square(x - means), axis=0, weights=sample_weights)
    scales = np.sqrt(np.maximum(variance, 1e-12))
    scales = np.where(scales < 1e-6, 1.0, scales)
    normalized = (x - means) / scales
    design = np.column_stack([np.ones(len(normalized)), normalized])
    root_weights = np.sqrt(np.maximum(sample_weights, 1e-6))[:, None]
    weighted_design = design * root_weights
    weighted_target = y * root_weights[:, 0]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    weights = coefficients[1:] / scales
    intercept = float(coefficients[0] - np.dot(coefficients[1:], means / scales))
    predictions = np.clip(intercept + x @ weights, 0.0, 1.0)
    return weights, intercept, predictions


def _grouped_folds(groups: Sequence[str], maximum_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        return []
    fold_count = min(maximum_folds, len(unique))
    buckets: list[list[str]] = [[] for _ in range(fold_count)]
    for index, group in enumerate(unique):
        digest = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16)
        buckets[digest % fold_count].append(group)
    # Hashing can leave an empty bucket for small datasets. Redistribute deterministically.
    nonempty = [bucket for bucket in buckets if bucket]
    if len(nonempty) < 2:
        nonempty = [[group] for group in unique]
    group_array = np.asarray(groups, dtype=object)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for bucket in nonempty:
        validation_mask = np.isin(group_array, np.asarray(bucket, dtype=object))
        validation = np.flatnonzero(validation_mask)
        training = np.flatnonzero(~validation_mask)
        if validation.size and training.size:
            folds.append((training, validation))
    return folds


def _model_metrics(
    x: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    groups: Sequence[str],
    *,
    ridge: float,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    weights, intercept, train_predictions = _ridge_fit(
        x, y, sample_weights, ridge=ridge
    )
    train_rmse = float(np.sqrt(np.average(np.square(train_predictions - y), weights=sample_weights)))

    folds = _grouped_folds(groups)
    cv_predictions = np.full(len(y), np.nan, dtype=np.float64)
    baseline_predictions = np.full(len(y), np.nan, dtype=np.float64)
    for training, validation in folds:
        fold_weights, fold_intercept, _ = _ridge_fit(
            x[training], y[training], sample_weights[training], ridge=ridge
        )
        cv_predictions[validation] = np.clip(
            fold_intercept + x[validation] @ fold_weights, 0.0, 1.0
        )
        baseline_predictions[validation] = np.average(
            y[training], weights=sample_weights[training]
        )

    valid = np.isfinite(cv_predictions)
    if np.any(valid):
        cv_rmse = float(
            np.sqrt(
                np.average(
                    np.square(cv_predictions[valid] - y[valid]),
                    weights=sample_weights[valid],
                )
            )
        )
        cv_mae = float(
            np.average(
                np.abs(cv_predictions[valid] - y[valid]),
                weights=sample_weights[valid],
            )
        )
        baseline_rmse = float(
            np.sqrt(
                np.average(
                    np.square(baseline_predictions[valid] - y[valid]),
                    weights=sample_weights[valid],
                )
            )
        )
        cv_skill = max(0.0, 1.0 - cv_rmse / max(baseline_rmse, 1e-9))
    else:
        cv_rmse = train_rmse
        cv_mae = float(np.average(np.abs(train_predictions - y), weights=sample_weights))
        baseline_rmse = train_rmse
        cv_skill = 0.0

    sample_factor = float(
        np.clip(
            (len(y) - MINIMUM_SAMPLES) / max(1, FULL_TRUST_SAMPLES - MINIMUM_SAMPLES),
            0.0,
            1.0,
        )
    )
    quality_factor = float(np.clip(cv_skill / 0.20, 0.0, 1.0))
    learned_weight = MAXIMUM_LEARNED_WEIGHT * sample_factor * quality_factor
    return weights, intercept, {
        "train_rmse": train_rmse,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "cv_baseline_rmse": baseline_rmse,
        "cv_skill": cv_skill,
        "fold_count": len(folds),
        "learned_weight": learned_weight,
    }


def _records(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.exists():
            continue
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def _training_row(record: dict[str, Any]) -> tuple[list[float], float, float, str, str | None] | None:
    plan = record.get("plan") if isinstance(record.get("plan"), dict) else {}
    compatibility = plan.get("compatibility") if isinstance(plan.get("compatibility"), dict) else None
    if compatibility is None and isinstance(record.get("compatibility"), dict):
        compatibility = record["compatibility"]
    if not isinstance(compatibility, dict):
        return None

    target: float | None = None
    weight = 1.0
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    if scores.get("overall") is not None:
        target = (float(scores["overall"]) - 1.0) / 4.0
        weight = 1.0
    elif record.get("measured_score") is not None:
        target = float(record["measured_score"])
        weight = 0.42 if bool(record.get("accepted")) else 0.28
    elif isinstance(record.get("measurement"), dict) and record["measurement"].get("overall") is not None:
        target = float(record["measurement"]["overall"])
        weight = 0.35
    if target is None:
        return None

    aliases = {"vocal_overlap": "vocal_safety", "bass_conflict": "bass_safety"}
    vector = [
        float(compatibility.get(name, compatibility.get(aliases.get(name, ""), 0.5)))
        for name in SCORING_FEATURES
    ]
    track_a = str(record.get("track_a") or plan.get("track_a") or "unknown-a")
    track_b = str(record.get("track_b") or plan.get("track_b") or "unknown-b")
    group = hashlib.sha256(f"{track_a}|{track_b}".encode("utf-8")).hexdigest()[:16]
    pair = plan.get("genre_pair") or record.get("genre_pair")
    genre_key = _genre_key(pair if isinstance(pair, (list, tuple)) else None)
    return vector, float(np.clip(target, 0.0, 1.0)), weight, group, genre_key


def _fit_payload(
    rows: Sequence[tuple[list[float], float, float, str, str | None]],
    *,
    ridge: float,
) -> dict[str, Any]:
    x = np.asarray([row[0] for row in rows], dtype=np.float64)
    y = np.asarray([row[1] for row in rows], dtype=np.float64)
    sample_weights = np.asarray([row[2] for row in rows], dtype=np.float64)
    groups = [row[3] for row in rows]
    effective_ridge = float(ridge) * max(0.65, MINIMUM_SAMPLES / max(len(rows), 1))
    weights, intercept, metrics = _model_metrics(
        x, y, sample_weights, groups, ridge=effective_ridge
    )
    return {
        "weights": weights.tolist(),
        "intercept": intercept,
        "sample_count": len(rows),
        "group_count": len(set(groups)),
        "ridge": effective_ridge,
        **metrics,
    }


def fit_scoring_model(
    ratings_path: str | Path | Sequence[str | Path],
    output_path: str | Path = "config/scoring-model.json",
    *,
    ridge: float = 0.35,
    measurement_path: str | Path | None = None,
) -> Path:
    sources: list[str | Path]
    if isinstance(ratings_path, (str, Path)):
        sources = [ratings_path]
    else:
        sources = list(ratings_path)
    if measurement_path is not None:
        sources.append(measurement_path)
    rows = [row for record in _records(sources) if (row := _training_row(record)) is not None]
    if len(rows) < MINIMUM_SAMPLES:
        raise ValueError(
            f"At least {MINIMUM_SAMPLES} rated or measured transitions with compatibility data are required."
        )

    global_payload = _fit_payload(rows, ridge=ridge)
    genre_models: dict[str, Any] = {}
    by_genre: dict[str, list[tuple[list[float], float, float, str, str | None]]] = {}
    for row in rows:
        if row[4]:
            by_genre.setdefault(row[4], []).append(row)
    for key, group_rows in sorted(by_genre.items()):
        if len(group_rows) >= 12 and len({row[3] for row in group_rows}) >= 3:
            genre_models[key] = _fit_payload(group_rows, ridge=ridge)

    payload = {
        "model": "ridge-listening-score-v2",
        "feature_names": list(SCORING_FEATURES),
        **global_payload,
        "genre_models": genre_models,
        "minimum_samples": MINIMUM_SAMPLES,
        "full_trust_samples": FULL_TRUST_SAMPLES,
        "maximum_learned_weight": MAXIMUM_LEARNED_WEIGHT,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    clear_scoring_model_cache()
    return output


def maybe_refit_scoring_model(
    *,
    ratings_path: str | Path = "ratings/listening-tests.jsonl",
    measurement_path: str | Path = "ratings/automatic-measurements.jsonl",
    output_path: str | Path = "config/scoring-model.json",
    minimum_new_samples: int | None = None,
) -> Path | None:
    """Refit the shared model when enough new evidence has accumulated.

    This keeps the default product learning loop automatic without refitting for
    every transition. Failures are deliberately non-fatal to set delivery.
    """
    enabled = os.environ.get("DJAI_AUTO_REFIT", "1").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return None
    interval = minimum_new_samples
    if interval is None:
        try:
            interval = int(os.environ.get("DJAI_AUTO_REFIT_INTERVAL", "20"))
        except ValueError:
            interval = 20
    interval = max(1, int(interval))
    sources = [ratings_path, measurement_path]
    rows = [
        row
        for record in _records(sources)
        if (row := _training_row(record)) is not None
    ]
    if len(rows) < MINIMUM_SAMPLES:
        return None
    existing = load_scoring_model(output_path)
    existing_count = int(existing.get("sample_count", 0)) if existing else 0
    if existing is not None and len(rows) - existing_count < interval:
        return None
    with _REFIT_LOCK:
        # Recheck after acquiring the lock in case another job already refit.
        existing = load_scoring_model(output_path)
        existing_count = int(existing.get("sample_count", 0)) if existing else 0
        if existing is not None and len(rows) - existing_count < interval:
            return None
        return fit_scoring_model(
            ratings_path,
            output_path,
            measurement_path=measurement_path,
        )

