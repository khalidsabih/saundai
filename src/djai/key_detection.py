from __future__ import annotations

import numpy as np

from .models import KeyEstimate


PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Kessler key profiles, starting at C.
MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

MAJOR_CAMELOT = {
    0: "8B",
    1: "3B",
    2: "10B",
    3: "5B",
    4: "12B",
    5: "7B",
    6: "2B",
    7: "9B",
    8: "4B",
    9: "11B",
    10: "6B",
    11: "1B",
}
MINOR_CAMELOT = {
    0: "5A",
    1: "12A",
    2: "7A",
    3: "2A",
    4: "9A",
    5: "4A",
    6: "11A",
    7: "6A",
    8: "1A",
    9: "8A",
    10: "3A",
    11: "10A",
}


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denominator)


def estimate_key_from_chroma(chroma: np.ndarray) -> KeyEstimate:
    """Estimate one global key from a 12-bin chroma vector or chromagram."""
    values = np.asarray(chroma, dtype=np.float64)
    if values.ndim == 2:
        values = np.median(values, axis=1)
    values = np.maximum(values.reshape(12), 0.0)

    if float(np.sum(values)) <= 1e-12:
        return KeyEstimate("C", 0, "major", "8B", 0.0)

    values /= np.sum(values)
    candidates: list[tuple[float, int, str]] = []
    for root in range(12):
        candidates.append((_correlation(values, np.roll(MAJOR_PROFILE, root)), root, "major"))
        candidates.append((_correlation(values, np.roll(MINOR_PROFILE, root)), root, "minor"))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, root, mode = candidates[0]
    second_score = candidates[1][0]

    # Confidence combines the absolute fit and separation from the runner-up.
    fit = np.clip((best_score + 1.0) / 2.0, 0.0, 1.0)
    margin = np.clip((best_score - second_score) / 0.25, 0.0, 1.0)
    confidence = float(np.clip(0.55 * fit + 0.45 * margin, 0.0, 1.0))
    camelot = MAJOR_CAMELOT[root] if mode == "major" else MINOR_CAMELOT[root]

    return KeyEstimate(
        tonic=PITCH_NAMES[root],
        pitch_class=root,
        mode=mode,  # type: ignore[arg-type]
        camelot=camelot,
        confidence=confidence,
    )
