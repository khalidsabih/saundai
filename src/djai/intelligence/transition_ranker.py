from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

from ..environment import config_path
from ..models import TransitionPlan
from .preferences import load_preferences, plan_features

RANKER_VERSION = "2.0"
MINIMUM_COMPARISONS = 24
FULL_TRUST_COMPARISONS = 400
MAXIMUM_RANKER_AUTHORITY = 0.45
RENDER_EMBEDDING_DIMS = 24

NUMERIC_FEATURES = (
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
    "cue_confidence",
    "analytical_score",
    "measured_score",
    "intent_score",
    "technique_score",
    "embedding_similarity",
    "transition_bars_scaled",
    "tempo_difference_scaled",
    "energy_delta",
    # Actual rendered-audio execution evidence.
    "loudness_continuity",
    "spectral_continuity",
    "bass_cleanliness",
    "phase_safety",
    "artifact_safety",
    "silence_safety",
    "rhythm_coherence",
    "transient_clarity",
    "groove_continuity",
    "harmonic_continuity",
    "energy_intent_fit",
    "drop_payoff",
    "musical_overall",
    # Context/memory features.
    "set_context_score",
    "rejection_penalty",
)

CATEGORICALS = {
    "style": ("safe", "creative", "emergency"),
    "intent": (
        "smooth_continuation",
        "groove_handoff",
        "energy_lift",
        "energy_release",
        "breakdown_to_drop",
        "vocal_handoff",
        "dramatic_switch",
        "emergency_escape",
    ),
    "technique": (
        "eq_bass_swap",
        "equal_power_blend",
        "filter_build",
        "echo_out_drop_in",
        "loop_roll",
        "quick_cut",
        "backspin",
        "double_drop",
        "stem_vocal_overlay",
        "stem_choreography",
        "layer_vocal_bridge",
        "layer_instrumental_handoff",
    ),
    "exit_section": ("intro", "groove", "build", "breakdown", "drop", "outro", "unknown"),
    "entry_section": ("intro", "groove", "build", "breakdown", "drop", "outro", "unknown"),
}

FEATURE_NAMES = (
    list(NUMERIC_FEATURES)
    + [f"render_embedding_{index:02d}" for index in range(RENDER_EMBEDDING_DIMS)]
    + [f"{field}={value}" for field, values in CATEGORICALS.items() for value in values]
)


@dataclass(frozen=True, slots=True)
class RankerPrediction:
    learned_score: float
    acceptability_score: float
    authority: float
    blended_score: float
    confidence: float


def _measurement_value(payload: dict[str, Any], name: str, default: float = 0.5) -> float:
    measurement = payload.get("measurement")
    if isinstance(measurement, dict) and measurement.get(name) is not None:
        try:
            return float(measurement[name])
        except (TypeError, ValueError):
            return default
    return default


def _fold_render_embedding(payload: dict[str, Any]) -> list[float]:
    raw = payload.get("rendered_transition_embedding", [])
    if not isinstance(raw, (list, tuple)) or not raw:
        return [0.0] * RENDER_EMBEDDING_DIMS
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    folded = np.zeros(RENDER_EMBEDDING_DIMS, dtype=np.float64)
    for index, value in enumerate(values):
        slot = index % RENDER_EMBEDDING_DIMS
        folded[slot] += float(value) if (index // RENDER_EMBEDDING_DIMS) % 2 == 0 else -float(value)
    norm = float(np.linalg.norm(folded))
    if norm > 1e-12:
        folded /= norm
    return folded.tolist()


def feature_vector(payload: dict[str, Any]) -> np.ndarray:
    compatibility = payload.get("compatibility") if isinstance(payload.get("compatibility"), dict) else {}
    numeric = [
        float(compatibility.get("tempo", 0.5)),
        float(compatibility.get("harmonic", 0.5)),
        float(compatibility.get("energy", 0.5)),
        float(compatibility.get("texture", 0.5)),
        float(compatibility.get("beat_grid", 0.5)),
        float(compatibility.get("phrase", 0.5)),
        float(compatibility.get("sections", 0.5)),
        float(compatibility.get("genre", 0.5)),
        float(compatibility.get("vocal_overlap", 0.5)),
        float(compatibility.get("bass_conflict", 0.5)),
        float(compatibility.get("boundary", 0.5)),
        float(compatibility.get("section_fit", 0.5)),
        float(payload.get("cue_confidence", 0.5)),
        float(payload.get("analytical_score", 0.5)),
        float(payload.get("measured_score", 0.5) or 0.5),
        float(payload.get("intent_score", 0.5)),
        float(payload.get("technique_score", 0.5)),
        float(payload.get("embedding_similarity", 0.5)),
        min(1.0, max(0.0, float(payload.get("transition_bars", 8)) / 32.0)),
        min(1.0, abs(float(payload.get("tempo_difference_percent", 0.0))) / 20.0),
        max(-1.0, min(1.0, float(payload.get("energy_delta", 0.0)))),
        _measurement_value(payload, "loudness_continuity"),
        _measurement_value(payload, "spectral_continuity"),
        _measurement_value(payload, "bass_cleanliness"),
        _measurement_value(payload, "phase_safety"),
        _measurement_value(payload, "artifact_safety"),
        _measurement_value(payload, "silence_safety"),
        _measurement_value(payload, "rhythm_coherence"),
        _measurement_value(payload, "transient_clarity"),
        _measurement_value(payload, "groove_continuity"),
        _measurement_value(payload, "harmonic_continuity"),
        _measurement_value(payload, "energy_intent_fit"),
        _measurement_value(payload, "drop_payoff"),
        _measurement_value(payload, "musical_overall"),
        float(payload.get("set_context_score", 0.5)),
        max(0.0, min(1.0, float(payload.get("rejection_penalty", 0.0)))),
    ]
    categories: list[float] = []
    for field, values in CATEGORICALS.items():
        actual = str(payload.get(field, "unknown"))
        categories.extend(1.0 if actual == value else 0.0 for value in values)
    vector = np.asarray([*numeric, *_fold_render_embedding(payload), *categories], dtype=np.float64)
    if vector.size != len(FEATURE_NAMES):
        raise RuntimeError("Transition ranker feature schema is internally inconsistent.")
    return vector


def _pair_key(candidate: dict[str, Any]) -> str:
    pair = f"{candidate.get('track_a')}|{candidate.get('track_b')}"
    return hashlib.sha256(pair.encode("utf-8")).hexdigest()[:16]


def _pair_rows(preferences: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    weights: list[float] = []
    groups: list[str] = []
    for record in preferences:
        choice = str(record.get("choice", "")).lower()
        if choice not in {"a", "b"}:
            continue
        a = record.get("candidate_a")
        b = record.get("candidate_b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        difference = feature_vector(a) - feature_vector(b)
        target = 1.0 if choice == "a" else 0.0
        x_rows.append(difference)
        y_rows.append(target)
        weights.append(float(record.get("weight", 1.0)))
        groups.append(_pair_key(a))
    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), np.empty(0), []
    return np.vstack(x_rows), np.asarray(y_rows), np.asarray(weights), groups


def _absolute_rows(preferences: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build sparse acceptability labels.

    `neither` is the strong negative signal. A selected A/B winner is only a
    weak positive because winning a pair does not imply that the transition is
    objectively good. We intentionally do not label the loser as bad.
    """
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    weights: list[float] = []
    groups: list[str] = []
    for record in preferences:
        choice = str(record.get("choice", "")).lower()
        a = record.get("candidate_a")
        b = record.get("candidate_b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        weight = max(0.05, float(record.get("weight", 1.0)))
        if choice == "neither":
            for candidate in (a, b):
                x_rows.append(feature_vector(candidate))
                y_rows.append(0.0)
                weights.append(weight * 1.25)
                groups.append(_pair_key(candidate))
        elif choice == "a":
            x_rows.append(feature_vector(a))
            y_rows.append(1.0)
            weights.append(weight * 0.45)
            groups.append(_pair_key(a))
        elif choice == "b":
            x_rows.append(feature_vector(b))
            y_rows.append(1.0)
            weights.append(weight * 0.45)
            groups.append(_pair_key(b))
    if not x_rows:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), np.empty(0), []
    return np.vstack(x_rows), np.asarray(y_rows), np.asarray(weights), groups


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, ridge: float) -> tuple[np.ndarray, float]:
    if x.size == 0:
        raise ValueError("No preference rows are available.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        coef = params[:-1]
        intercept = params[-1]
        logits = z @ coef + intercept
        probabilities = np.clip(_sigmoid(logits), 1e-8, 1.0 - 1e-8)
        losses = -(y * np.log(probabilities) + (1.0 - y) * np.log(1.0 - probabilities))
        loss = float(np.average(losses, weights=weights) + 0.5 * ridge * np.sum(coef * coef))
        error = (probabilities - y) * weights / max(float(np.sum(weights)), 1e-9)
        grad_coef = z.T @ error + ridge * coef
        grad_intercept = float(np.sum(error))
        return loss, np.concatenate([grad_coef, [grad_intercept]])

    initial = np.zeros(z.shape[1] + 1, dtype=np.float64)
    result = minimize(lambda p: objective(p)[0], initial, jac=lambda p: objective(p)[1], method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Transition ranker optimization failed: {result.message}")
    coef_z = result.x[:-1]
    intercept_z = float(result.x[-1])
    coef = coef_z / scale
    intercept = intercept_z - float(np.dot(mean / scale, coef_z))
    return coef, intercept


def _grouped_folds(groups: Sequence[str], max_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = sorted(set(groups))
    if len(unique) < 2:
        return []
    fold_count = min(max_folds, len(unique))
    buckets = [unique[index::fold_count] for index in range(fold_count)]
    array = np.asarray(groups, dtype=object)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for bucket in buckets:
        validation_mask = np.isin(array, np.asarray(bucket, dtype=object))
        train = np.flatnonzero(~validation_mask)
        valid = np.flatnonzero(validation_mask)
        if train.size and valid.size:
            output.append((train, valid))
    return output


def _metrics(x: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: Sequence[str], *, ridge: float) -> dict[str, float | int]:
    folds = _grouped_folds(groups)
    predictions = np.full(len(y), np.nan)
    baseline = np.full(len(y), np.nan)
    for train, valid in folds:
        if len(np.unique(y[train])) < 2:
            majority = float(np.average(y[train], weights=weights[train]))
            predictions[valid] = majority
            baseline[valid] = majority
            continue
        coef, intercept = _fit_logistic(x[train], y[train], weights[train], ridge=ridge)
        predictions[valid] = _sigmoid(x[valid] @ coef + intercept)
        majority = float(np.average(y[train], weights=weights[train]))
        baseline[valid] = majority
    mask = np.isfinite(predictions)
    if not np.any(mask):
        return {"cv_accuracy": 0.5, "cv_log_loss": math.log(2.0), "baseline_accuracy": 0.5, "cv_skill": 0.0, "fold_count": 0}
    predicted_labels = predictions[mask] >= 0.5
    accuracy = float(np.average(predicted_labels == (y[mask] >= 0.5), weights=weights[mask]))
    baseline_labels = baseline[mask] >= 0.5
    baseline_accuracy = float(np.average(baseline_labels == (y[mask] >= 0.5), weights=weights[mask]))
    probabilities = np.clip(predictions[mask], 1e-8, 1.0 - 1e-8)
    losses = -(y[mask] * np.log(probabilities) + (1.0 - y[mask]) * np.log(1.0 - probabilities))
    log_loss = float(np.average(losses, weights=weights[mask]))
    skill = max(0.0, accuracy - baseline_accuracy)
    return {
        "cv_accuracy": accuracy,
        "cv_log_loss": log_loss,
        "baseline_accuracy": baseline_accuracy,
        "cv_skill": skill,
        "fold_count": len(folds),
    }


def train_transition_ranker(
    preferences_path: str | Path = "ratings/transition-preferences.jsonl",
    output_path: str | Path = "config/transition-ranker.json",
    *,
    ridge: float = 0.18,
) -> Path:
    records = load_preferences(preferences_path)
    x, y, sample_weights, groups = _pair_rows(records)
    if len(y) < MINIMUM_COMPARISONS:
        raise ValueError(f"At least {MINIMUM_COMPARISONS} decisive A/B preferences are required.")
    coef, intercept = _fit_logistic(x, y, sample_weights, ridge=ridge)
    metrics = _metrics(x, y, sample_weights, groups, ridge=ridge)
    sample_factor = float(np.clip((len(y) - MINIMUM_COMPARISONS) / max(1, FULL_TRUST_COMPARISONS - MINIMUM_COMPARISONS), 0.0, 1.0))
    quality_factor = float(np.clip(float(metrics["cv_skill"]) / 0.10, 0.0, 1.0))
    authority = MAXIMUM_RANKER_AUTHORITY * sample_factor * quality_factor

    absolute_x, absolute_y, absolute_weights, absolute_groups = _absolute_rows(records)
    acceptability: dict[str, Any] = {
        "available": False,
        "sample_count": int(len(absolute_y)),
        "negative_count": int(np.sum(absolute_y < 0.5)) if len(absolute_y) else 0,
        "positive_count": int(np.sum(absolute_y >= 0.5)) if len(absolute_y) else 0,
        "authority": 0.0,
    }
    if len(absolute_y) >= 16 and len(np.unique(absolute_y)) >= 2 and len(set(absolute_groups)) >= 2:
        abs_coef, abs_intercept = _fit_logistic(absolute_x, absolute_y, absolute_weights, ridge=ridge)
        abs_metrics = _metrics(absolute_x, absolute_y, absolute_weights, absolute_groups, ridge=ridge)
        abs_sample = float(np.clip((len(absolute_y) - 16) / 384.0, 0.0, 1.0))
        abs_quality = float(np.clip(float(abs_metrics["cv_skill"]) / 0.10, 0.0, 1.0))
        acceptability = {
            **acceptability,
            "available": True,
            "weights": abs_coef.tolist(),
            "intercept": abs_intercept,
            "authority": 0.30 * abs_sample * abs_quality,
            **{f"acceptability_{key}": value for key, value in abs_metrics.items()},
        }

    payload = {
        "model": "pairwise-professional-transition-ranker-v2",
        "version": RANKER_VERSION,
        "feature_names": FEATURE_NAMES,
        "weights": coef.tolist(),
        "intercept": intercept,
        "comparison_count": int(len(y)),
        "track_pair_groups": len(set(groups)),
        "ridge": ridge,
        "authority": authority,
        "maximum_authority": MAXIMUM_RANKER_AUTHORITY,
        "acceptability": acceptability,
        **metrics,
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    load_transition_ranker.cache_clear()
    return destination


@lru_cache(maxsize=8)
def load_transition_ranker(path: str | Path | None = None) -> dict[str, Any] | None:
    source = config_path(path or os.environ.get("DJAI_TRANSITION_RANKER_MODEL", "transition-ranker.json"))
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("feature_names") != FEATURE_NAMES:
        return None
    return payload


def ranker_uses_render_embedding(model: dict[str, Any] | None = None) -> bool:
    """Return True only when an authoritative ranker actually weights render embeddings."""
    model = model if model is not None else load_transition_ranker()
    if not model or float(model.get("authority", 0.0)) <= 0.0:
        return False
    indices = [index for index, name in enumerate(FEATURE_NAMES) if name.startswith("render_embedding_")]
    if not indices:
        return False
    heads: list[dict[str, Any]] = [model]
    acceptability = model.get("acceptability")
    if isinstance(acceptability, dict) and acceptability.get("available") and float(acceptability.get("authority", 0.0)) > 0.0:
        heads.append(acceptability)
    for head in heads:
        weights = np.asarray(head.get("weights", []), dtype=np.float64).reshape(-1)
        if weights.size != len(FEATURE_NAMES):
            continue
        if any(abs(float(weights[index])) > 1e-9 for index in indices):
            return True
    return False


def _predict_with(weights_payload: Sequence[float], intercept: float, vector: np.ndarray) -> float:
    weights = np.asarray(weights_payload, dtype=np.float64)
    if weights.size != vector.size:
        return 0.5
    logit = float(intercept) + float(np.dot(vector, weights))
    return float(_sigmoid(np.asarray([logit]))[0])


def _predict_payload(payload: dict[str, Any], model: dict[str, Any]) -> float:
    return _predict_with(model.get("weights", []), float(model.get("intercept", 0.0)), feature_vector(payload))


def _predict_acceptability(payload: dict[str, Any], model: dict[str, Any]) -> tuple[float, float]:
    head = model.get("acceptability")
    if not isinstance(head, dict) or not head.get("available"):
        return 0.5, 0.0
    score = _predict_with(head.get("weights", []), float(head.get("intercept", 0.0)), feature_vector(payload))
    return score, max(0.0, min(0.30, float(head.get("authority", 0.0))))


def predict_plan(plan: TransitionPlan, model: dict[str, Any] | None = None) -> RankerPrediction:
    model = model if model is not None else load_transition_ranker()
    # Naturalness is part of the deterministic production baseline. Learned
    # taste may move that baseline only within its validated authority.
    baseline = max(
        0.0,
        min(1.0, float(plan.metadata.get("naturalness_selection_score", plan.score))),
    )
    if not model:
        return RankerPrediction(learned_score=0.5, acceptability_score=0.5, authority=0.0, blended_score=baseline, confidence=0.0)
    features = plan_features(plan)
    preference = _predict_payload(features, model)
    acceptability, accept_authority = _predict_acceptability(features, model)
    learned = 0.72 * preference + 0.28 * acceptability if accept_authority > 0.0 else preference
    pair_authority = max(0.0, min(MAXIMUM_RANKER_AUTHORITY, float(model.get("authority", 0.0))))
    authority = max(0.0, min(MAXIMUM_RANKER_AUTHORITY, pair_authority + 0.25 * accept_authority))
    blended = (1.0 - authority) * baseline + authority * learned
    data_confidence = authority / max(MAXIMUM_RANKER_AUTHORITY, 1e-9)
    decisiveness = min(1.0, abs(learned - 0.5) * 2.0)
    confidence = max(0.0, min(1.0, 0.72 * data_confidence + 0.28 * decisiveness))
    return RankerPrediction(
        learned_score=float(learned),
        acceptability_score=float(acceptability),
        authority=float(authority),
        blended_score=float(blended),
        confidence=float(confidence),
    )


def rank_transition_plans(plans: Sequence[TransitionPlan], model: dict[str, Any] | None = None) -> list[TransitionPlan]:
    enabled = os.environ.get("DJAI_TRANSITION_RANKER", "1").strip().lower() not in {"0", "false", "off", "no"}
    if not enabled:
        return list(plans)
    model = model if model is not None else load_transition_ranker()
    if not model or float(model.get("authority", 0.0)) <= 0.0:
        return list(plans)
    ranked: list[tuple[float, TransitionPlan]] = []
    for plan in plans:
        prediction = predict_plan(plan, model)
        plan.metadata = {
            **plan.metadata,
            "transition_ranker_score": round(prediction.learned_score, 6),
            "transition_ranker_acceptability": round(prediction.acceptability_score, 6),
            "transition_ranker_authority": round(prediction.authority, 6),
            "transition_ranker_confidence": round(prediction.confidence, 6),
            "transition_ranker_blended_score": round(prediction.blended_score, 6),
            "transition_ranker_version": RANKER_VERSION,
        }
        ranked.append((prediction.blended_score, plan))
    ranked.sort(key=lambda item: (item[0], item[1].score, item[1].execution_signature), reverse=True)
    return [item[1] for item in ranked]
