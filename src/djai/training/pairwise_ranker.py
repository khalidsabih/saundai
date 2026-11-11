from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PAIRWISE_RANKER_SCHEMA_VERSION = "1.0"


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("features"), Mapping):
                raise ValueError(f"Next-track row {line_number} must contain a features object.")
            split = str(row.get("split") or "").strip().lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError("Pairwise ranker requires explicit train/validation/test splits.")
            if "target" not in row:
                raise ValueError(f"Next-track row {line_number} is missing target.")
            rows.append(row)
    if not rows:
        raise ValueError("Pairwise ranker dataset is empty.")
    return rows


def _matrix(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    names = sorted({str(key) for row in rows for key in row["features"]})
    matrix = np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in names] for row in rows],
        dtype=float,
    )
    return names, matrix


def _indices(rows: list[dict[str, Any]], split: str) -> np.ndarray:
    return np.asarray(
        [index for index, row in enumerate(rows) if str(row.get("split")).lower() == split],
        dtype=int,
    )


def _normalize(
    names: list[str],
    matrix: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    if len(train_indices) == 0:
        raise ValueError("Pairwise ranker requires training rows.")
    reference = matrix[train_indices]
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (matrix - mean) / scale
    return (
        normalized,
        {name: float(value) for name, value in zip(names, mean, strict=True)},
        {name: float(value) for name, value in zip(names, scale, strict=True)},
    )


def _group_indices(rows: list[dict[str, Any]], indices: np.ndarray) -> list[list[int]]:
    selected = set(int(index) for index in indices)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index not in selected:
            continue
        group = str(row.get("group") or row.get("context_id") or row.get("id") or index)
        groups[group].append(index)
    return [items for items in groups.values() if len(items) >= 2]


def _pair_matrix(
    rows: list[dict[str, Any]],
    features: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    diffs: list[np.ndarray] = []
    weights: list[float] = []
    targets = np.asarray([float(row["target"]) for row in rows], dtype=float)
    row_weights = np.asarray(
        [max(0.05, min(1.0, float(row.get("weight", 1.0)))) for row in rows],
        dtype=float,
    )
    for items in _group_indices(rows, train_indices):
        ordered = sorted(items, key=lambda index: targets[index], reverse=True)
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                preferred = ordered[left]
                other = ordered[right]
                gap = targets[preferred] - targets[other]
                if gap <= 1e-12:
                    continue
                diffs.append(features[preferred] - features[other])
                confidence = 0.5 * (row_weights[preferred] + row_weights[other])
                weights.append(max(0.05, confidence * gap))
    if not diffs:
        raise ValueError("Pairwise ranker could not build any ordered candidate pairs.")
    return np.asarray(diffs, dtype=float), np.asarray(weights, dtype=float)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        ranks[order[index:end]] = rank
        index = end
    return ranks


def ranking_metrics(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    targets = np.asarray([float(row["target"]) for row in rows], dtype=float)
    groups = _group_indices(rows, indices)
    if not groups:
        return {
            "groups": 0,
            "top1_accuracy": None,
            "top2_recall": None,
            "pairwise_accuracy": None,
            "spearman": None,
            "ndcg": None,
            "mrr": None,
        }

    top1: list[float] = []
    top2: list[float] = []
    pairwise: list[float] = []
    spearman: list[float] = []
    ndcg: list[float] = []
    mrr: list[float] = []

    for items in groups:
        predicted = np.asarray([scores[index] for index in items], dtype=float)
        truth = np.asarray([targets[index] for index in items], dtype=float)
        best = int(np.argmax(truth))
        order = np.argsort(-predicted, kind="mergesort")

        top1.append(float(order[0] == best))
        top2.append(float(best in {int(index) for index in order[: min(2, len(order))]}))

        pair_total = 0
        pair_correct = 0.0
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                if truth[left] == truth[right]:
                    continue
                pair_total += 1
                actual = truth[left] > truth[right]
                if predicted[left] == predicted[right]:
                    pair_correct += 0.5
                elif (predicted[left] > predicted[right]) == actual:
                    pair_correct += 1.0
        if pair_total:
            pairwise.append(pair_correct / pair_total)

        if np.std(predicted) > 1e-12 and np.std(truth) > 1e-12:
            spearman.append(float(np.corrcoef(_rankdata(predicted), _rankdata(truth))[0, 1]))

        gains = np.power(2.0, truth) - 1.0
        discounts = 1.0 / np.log2(np.arange(2, len(items) + 2, dtype=float))
        dcg = float(np.sum(gains[order] * discounts))
        ideal = np.argsort(-truth, kind="mergesort")
        idcg = float(np.sum(gains[ideal] * discounts))
        ndcg.append(dcg / idcg if idcg > 0 else 0.0)

        rank_position = int(np.where(order == best)[0][0]) + 1
        mrr.append(1.0 / rank_position)

    def average(values: list[float]) -> float | None:
        return None if not values else round(float(np.mean(values)), 6)

    return {
        "groups": len(groups),
        "top1_accuracy": average(top1),
        "top2_recall": average(top2),
        "pairwise_accuracy": average(pairwise),
        "spearman": average(spearman),
        "ndcg": average(ndcg),
        "mrr": average(mrr),
    }


def train_pairwise_next_track_ranker(
    dataset: str | Path,
    output: str | Path,
    *,
    iterations: int = 700,
    learning_rate: float = 0.04,
    l2: float = 0.004,
) -> Path:
    rows = _load_rows(dataset)
    names, raw_features = _matrix(rows)
    train_indices = _indices(rows, "train")
    validation_indices = _indices(rows, "validation")
    test_indices = _indices(rows, "test")
    features, means, scales = _normalize(names, raw_features, train_indices)
    pair_diffs, pair_weights = _pair_matrix(rows, features, train_indices)

    weights = np.zeros(features.shape[1], dtype=float)
    denominator = max(float(pair_weights.sum()), 1e-9)
    iterations = max(50, int(iterations))
    for _ in range(iterations):
        margins = np.clip(pair_diffs @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-margins))
        gradient = -((1.0 - probabilities) * pair_weights) @ pair_diffs / denominator
        gradient += l2 * weights
        weights -= learning_rate * gradient

    scores = features @ weights
    train_ranking = ranking_metrics(rows, train_indices, scores)
    validation_ranking = ranking_metrics(rows, validation_indices, scores)
    test_ranking = ranking_metrics(rows, test_indices, scores)

    payload = {
        "schema_version": PAIRWISE_RANKER_SCHEMA_VERSION,
        "model_id": "next_track_pairwise_ranker",
        "task": "next_track_policy",
        "kind": "pairwise_ranker",
        "objective": "weighted_pairwise_logistic_ranknet",
        "feature_names": names,
        "feature_mean": means,
        "feature_scale": scales,
        "weights": weights.tolist(),
        "bias": 0.0,
        "split_policy": "explicit",
        "training_pairs": int(len(pair_diffs)),
        "training_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "test_examples": int(len(test_indices)),
        "examples": len(rows),
        "metrics": {
            "train_ranking": train_ranking,
            "validation_ranking": validation_ranking,
            "test_ranking": test_ranking,
        },
    }
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def train_mlp_pairwise_next_track_ranker(
    dataset: str | Path,
    output: str | Path,
    *,
    hidden_sizes: tuple[int, int] = (64, 32),
    epochs: int = 120,
    learning_rate: float = 0.004,
    l2: float = 0.001,
    seed: int = 1337,
) -> Path:
    rows = _load_rows(dataset)
    names, raw_features = _matrix(rows)
    train_indices = _indices(rows, "train")
    validation_indices = _indices(rows, "validation")
    test_indices = _indices(rows, "test")
    features, means, scales = _normalize(names, raw_features, train_indices)

    targets = np.asarray([float(row["target"]) for row in rows], dtype=float)
    row_weights = np.asarray(
        [max(0.05, min(1.0, float(row.get("weight", 1.0)))) for row in rows],
        dtype=float,
    )
    pairs: list[tuple[int, int, float]] = []
    for items in _group_indices(rows, train_indices):
        ordered = sorted(items, key=lambda index: targets[index], reverse=True)
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                preferred = ordered[left]
                other = ordered[right]
                gap = targets[preferred] - targets[other]
                if gap <= 1e-12:
                    continue
                confidence = 0.5 * (row_weights[preferred] + row_weights[other])
                pairs.append((preferred, other, max(0.05, confidence * gap)))
    if not pairs:
        raise ValueError("MLP pairwise ranker could not build any ordered candidate pairs.")

    h1, h2 = hidden_sizes
    rng = np.random.default_rng(seed)
    input_dim = features.shape[1]
    w1 = rng.normal(0.0, np.sqrt(2.0 / max(1, input_dim)), size=(input_dim, h1))
    b1 = np.zeros(h1, dtype=float)
    w2 = rng.normal(0.0, np.sqrt(2.0 / max(1, h1)), size=(h1, h2))
    b2 = np.zeros(h2, dtype=float)
    w3 = rng.normal(0.0, np.sqrt(2.0 / max(1, h2)), size=(h2, 1))
    b3 = np.zeros(1, dtype=float)

    def forward(x: np.ndarray):
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2
        a2 = np.maximum(z2, 0.0)
        score = a2 @ w3 + b3
        return z1, a1, z2, a2, score[:, 0]

    pair_left = np.asarray([item[0] for item in pairs], dtype=int)
    pair_right = np.asarray([item[1] for item in pairs], dtype=int)
    pair_weight = np.asarray([item[2] for item in pairs], dtype=float)
    pair_weight_sum = max(float(pair_weight.sum()), 1e-9)

    batch_size = min(2048, len(pairs))
    epochs = max(10, int(epochs))
    for _ in range(epochs):
        permutation = rng.permutation(len(pairs))
        for start in range(0, len(pairs), batch_size):
            batch = permutation[start : start + batch_size]
            left_idx = pair_left[batch]
            right_idx = pair_right[batch]
            sample_weights = pair_weight[batch]
            unique = np.unique(np.concatenate([left_idx, right_idx]))
            local = {int(index): pos for pos, index in enumerate(unique)}
            x = features[unique]
            z1, a1, z2, a2, scores = forward(x)

            local_left = np.asarray([local[int(index)] for index in left_idx], dtype=int)
            local_right = np.asarray([local[int(index)] for index in right_idx], dtype=int)
            margins = np.clip(scores[local_left] - scores[local_right], -40.0, 40.0)
            probs = 1.0 / (1.0 + np.exp(-margins))
            coeff = -(1.0 - probs) * sample_weights / max(float(sample_weights.sum()), 1e-9)

            dscore = np.zeros(len(unique), dtype=float)
            np.add.at(dscore, local_left, coeff)
            np.add.at(dscore, local_right, -coeff)
            dscore = dscore[:, None]

            grad_w3 = a2.T @ dscore + l2 * w3
            grad_b3 = dscore.sum(axis=0)
            da2 = dscore @ w3.T
            dz2 = da2 * (z2 > 0.0)
            grad_w2 = a1.T @ dz2 + l2 * w2
            grad_b2 = dz2.sum(axis=0)
            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0.0)
            grad_w1 = x.T @ dz1 + l2 * w1
            grad_b1 = dz1.sum(axis=0)

            for grad in (grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3):
                np.clip(grad, -5.0, 5.0, out=grad)

            w1 -= learning_rate * grad_w1
            b1 -= learning_rate * grad_b1
            w2 -= learning_rate * grad_w2
            b2 -= learning_rate * grad_b2
            w3 -= learning_rate * grad_w3
            b3 -= learning_rate * grad_b3

    _, _, _, _, scores = forward(features)
    payload = {
        "schema_version": PAIRWISE_RANKER_SCHEMA_VERSION,
        "model_id": "next_track_mlp_pairwise_ranker",
        "task": "next_track_policy",
        "kind": "mlp_pairwise_ranker",
        "objective": "weighted_pairwise_logistic_ranknet",
        "architecture": {
            "input_dim": int(input_dim),
            "hidden_sizes": [int(h1), int(h2)],
            "activation": "relu",
            "output_dim": 1,
        },
        "feature_names": names,
        "feature_mean": means,
        "feature_scale": scales,
        "parameters": {
            "w1": w1.tolist(),
            "b1": b1.tolist(),
            "w2": w2.tolist(),
            "b2": b2.tolist(),
            "w3": w3.tolist(),
            "b3": b3.tolist(),
        },
        "split_policy": "explicit",
        "seed": int(seed),
        "training_pairs": int(len(pairs)),
        "training_examples": int(len(train_indices)),
        "validation_examples": int(len(validation_indices)),
        "test_examples": int(len(test_indices)),
        "examples": len(rows),
        "metrics": {
            "train_ranking": ranking_metrics(rows, train_indices, scores),
            "validation_ranking": ranking_metrics(rows, validation_indices, scores),
            "test_ranking": ranking_metrics(rows, test_indices, scores),
        },
    }
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
