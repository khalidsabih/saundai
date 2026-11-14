#!/usr/bin/env python3
"""Validation-only shadow: fix production meter, infer downbeat phase independently.

The existing 3/4/6 meter and beat times are held fixed. A new autonomous
phase selector measures onset/RMS accent and chroma-bar recurrence, demands
agreement between first and second half of the recording, then abstains on
weak/unstable evidence. Also evaluate prior hierarchical meter+phase shadow.

Read annotation labels strictly AFTER predictions have been calculated.
No audio/model/cache writes and no production analysis changes. Benchmark
windows are not the protected 25ms cross-deck phase residual threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import librosa
import numpy as np

from djai.analysis import ANALYSIS_SAMPLE_RATE, analyze_track
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    DOWNBEAT_WINDOW_MS, match_events, read_timestamp_labels, valid_times,
)
from scripts.diagnose_beat_error_decomposition import reference_meter
from scripts.shadow_hierarchical_meter import (
    METERS, MIN_BEATS, beat_features, choose_downbeats, infer_meter,
)

# Fixed, exploratory structural thresholds; NOT fitted from the 70 validation labels.
MIN_PHASE_CONTRAST = 0.15
MIN_PHASE_MARGIN = 0.07
MIN_HALF_BEATS = 12


def _phase_scores(features: np.ndarray, meter: int) -> list[dict[str, Any]]:
    """Rank accent and bar recurrence at each phase of a FIXED meter."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or np.any(~np.isfinite(x)):
        raise ValueError("finite beat-level onset/RMS/chroma features required")
    if meter not in METERS:
        raise ValueError("unsupported fixed meter")
    log_accent = np.log1p(np.maximum(x[:, :2], 0))
    scale = np.maximum(np.percentile(log_accent, 90, axis=0), 1e-6)
    strength = np.clip(.65*log_accent[:, 0]/scale[0] +
                       .35*log_accent[:, 1]/scale[1], 0, 3)
    chroma = np.maximum(x[:, 2:], 0)
    chroma /= np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    result = []
    for phase in range(meter):
        # Include all beats and retain original global phase indices.
        offsets = np.arange(phase, len(x), meter)
        others = [i for i in range(meter) if i != phase]
        phase_values = strength[offsets]
        accent = max(0.0, (float(np.mean(phase_values)) -
                           float(np.mean([np.mean(strength[np.arange(i, len(x), meter)])
                                          for i in others if i < len(x)]))) /
                     max(float(np.mean(np.abs(strength))), 1e-8))
        # Restrict neighboring bar harmonic comparison to valid full bars.
        bars = (len(x)-phase)//meter
        recurrence = 0.0
        if bars >= 3:
            cc = chroma[phase:phase+bars*meter].reshape(bars, meter, -1)
            same = np.sum(cc[1:]*cc[:-1], axis=2)
            shifted = np.sum(cc[1:]*np.roll(cc[:-1], 1, axis=1), axis=2)
            recurrence = max(0.0, float(np.mean(same-shifted)))
        score = .75*accent + .25*recurrence
        result.append({
            "phase": phase,
            "score": float(score),
            "accent": float(accent),
            "bar_repetition": float(recurrence),
        })
    return result


def infer_phase_fixed_meter(features: np.ndarray, meter: int) -> dict[str, Any]:
    """Select a phase without changing the existing meter, or abstain."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or np.any(~np.isfinite(x)):
        raise ValueError("finite beat-level features required")
    if meter not in METERS:
        return {"status": "unsupported_production_meter", "abstained": True,
                "meter": meter, "phase": None}
    if len(x) < max(MIN_BEATS, meter*6):
        return {"status": "insufficient_beats", "abstained": True,
                "meter": meter, "phase": None}
    candidates = _phase_scores(x, meter)
    rank = sorted(candidates, key=lambda r: r["score"], reverse=True)
    win, runner_up = rank[:2]
    mid = len(x)//2
    # Each half has its own evidence, but must retain indices of the original
    # beat grid. Offset the local phase by the half's start index.
    halves = []
    for start, end in ((0, mid), (mid, len(x))):
        local = x[start:end]
        if len(local) < max(MIN_HALF_BEATS, meter*3):
            halves.append(None)
        else:
            local_scores = _phase_scores(local, meter)
            best_local = max(local_scores, key=lambda r: r["score"])
            halves.append((best_local["phase"]+start) % meter)
    agreed = all(p is not None and p == win["phase"] for p in halves)
    margin = win["score"]-runner_up["score"]
    permitted = (win["score"] >= MIN_PHASE_CONTRAST and
                 margin >= MIN_PHASE_MARGIN and agreed)
    return {
        "status": "hypothesis" if permitted else "ambiguous",
        "meter": meter,
        "phase": win["phase"] if permitted else None,
        "abstained": not permitted,
        "leading_phase": win["phase"],
        "half_winners": halves,
        "half_agreement": agreed,
        "score": round(win["score"], 5),
        "score_margin": round(margin, 5),
        "scores_by_phase": [{
            "phase": r["phase"],
            "score": round(r["score"], 5),
            "accent": round(r["accent"], 5),
            "bar_repetition": round(r["bar_repetition"], 5),
        } for r in candidates],
        "score_is_not_probability": True,
        "no_reference_labels_used": True,
    }


def downbeats_for_phase(beats: list[float], phase: dict[str, Any]) -> list[float]:
    timestamps = valid_times(beats, allow_empty=True)
    if phase["abstained"]:
        return []
    meter, p = phase["meter"], phase["phase"]
    if meter not in METERS or not isinstance(p, int) or not 0 <= p < meter:
        raise ValueError("invalid phase-only hypothesis")
    return timestamps[p::meter]


def _score(ref: list[float] | None, predicted: list[float]) -> dict[str, Any] | None:
    return match_events(ref, predicted, tolerance_ms=DOWNBEAT_WINDOW_MS) if ref else None


def evaluate_one(item: dict[str, Any], root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"id": str(item["id"]), "status": "pending"}
    try:
        audio = root/str(item["audio"])
        original = analyze_track(audio)
        beats = valid_times(original.beats_seconds or [], allow_empty=True)
        baseline = valid_times(original.downbeats_seconds or [], allow_empty=True)
        meter = int(original.time_signature)
        if len(beats) >= MIN_BEATS:
            source, sr = load_audio(audio, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
            x = beat_features(librosa.to_mono(source), sr, beats)
        else:
            x = np.empty((0, 14), dtype=float)
        start = time.perf_counter()
        fixed = infer_phase_fixed_meter(x, meter)
        hierarchical = infer_meter(x)
        inference_seconds = round(time.perf_counter()-start, 5)
        fixed_raw = downbeats_for_phase(beats, fixed)
        hierarchy_raw = choose_downbeats(beats, hierarchical)
        fixed_selected = baseline if fixed["abstained"] else fixed_raw
        hierarchy_selected = baseline if hierarchical["abstained"] else hierarchy_raw
        # The reference is accessed only once both autonomous predictions exist.
        ref_beats, _ = read_timestamp_labels(root/str(item["beats"]))
        ref_downbeats = None
        if item.get("downbeats"):
            ref_downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        reference = reference_meter(ref_downbeats, ref_beats)
        row.update({
            "status": "evaluated",
            "production_meter": meter,
            "reference_meter": reference["meter"],
            "production_meter_matches_reference": (
                meter == reference["meter"] if reference["meter"] in METERS else None
            ),
            "fixed_phase": fixed,
            "hierarchical_meter_phase": hierarchical,
            "inference_seconds_excluding_features": inference_seconds,
            "baseline": _score(ref_downbeats, baseline),
            "fixed_phase_only_accepted": (
                _score(ref_downbeats, fixed_raw) if not fixed["abstained"] else None
            ),
            "fixed_phase_only_with_fallback": _score(ref_downbeats, fixed_selected),
            "hierarchical_with_fallback": _score(ref_downbeats, hierarchy_selected),
            "hierarchical_accepted": (
                _score(ref_downbeats, hierarchy_raw)
                if not hierarchical["abstained"] else None
            ),
        })
        if ref_downbeats:
            base = row["baseline"]["f_measure"]
            row["fixed_phase_f1_delta"] = round(
                row["fixed_phase_only_with_fallback"]["f_measure"]-base, 5
            )
            row["hierarchical_f1_delta"] = round(
                row["hierarchical_with_fallback"]["f_measure"]-base, 5
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    known = [r for r in evaluated if r["reference_meter"] in METERS]
    def macro(field: str, subset: list[dict[str, Any]]) -> float | None:
        vals = [r[field]["f_measure"] for r in subset if r.get(field) is not None]
        return round(float(np.mean(vals)), 5) if vals else None
    out: dict[str, Any] = {
        "evaluated": len(evaluated),
        "meter_reference_known_count": len(known),
        "baseline_meter_correct": sum(r["production_meter_matches_reference"] for r in known),
        "production_meter_counts": dict(sorted(Counter(str(r["production_meter"]) for r in evaluated).items())),
        "baseline_downbeat_macro_f1": macro("baseline", evaluated),
        "fixed_phase_fallback_macro_f1": macro("fixed_phase_only_with_fallback", evaluated),
        "hierarchical_fallback_macro_f1": macro("hierarchical_with_fallback", evaluated),
        "by_reference_meter": {},
        "per_track_fixed_phase_improved": sum(r.get("fixed_phase_f1_delta", 0) > 0 for r in evaluated),
        "per_track_fixed_phase_regressed": sum(r.get("fixed_phase_f1_delta", 0) < 0 for r in evaluated),
        "per_track_hierarchical_improved": sum(r.get("hierarchical_f1_delta", 0) > 0 for r in evaluated),
        "per_track_hierarchical_regressed": sum(r.get("hierarchical_f1_delta", 0) < 0 for r in evaluated),
        "regressions": [{
            "id": r["id"], "reference_meter": r["reference_meter"],
            "production_meter": r["production_meter"],
            "fixed_phase_f1_delta": r["fixed_phase_f1_delta"],
            "hierarchical_f1_delta": r["hierarchical_f1_delta"],
        } for r in evaluated if r.get("fixed_phase_f1_delta", 0) < 0
                               or r.get("hierarchical_f1_delta", 0) < 0],
    }
    for name, col, selected in (
        ("fixed_phase_only", "fixed_phase", "fixed_phase_only_accepted"),
        ("hierarchical", "hierarchical_meter_phase", "hierarchical_accepted"),
    ):
        accepted = [r for r in evaluated if not r[col]["abstained"]]
        out[name] = {
            "accepted": len(accepted),
            "abstained": len(evaluated)-len(accepted),
            "baseline_f1_on_accepted": macro("baseline", accepted),
            "candidate_f1_on_accepted": macro(selected, accepted),
            "accepted_reference_meter_counts": dict(sorted(Counter(
                str(r["reference_meter"]) for r in accepted).items())),
        }
    for meter in (3, 4, 6):
        subset = [r for r in evaluated if r["reference_meter"] == meter]
        out["by_reference_meter"][str(meter)] = {
            "tracks": len(subset),
            "baseline_downbeat_macro_f1": macro("baseline", subset),
            "fixed_phase_fallback_macro_f1": macro("fixed_phase_only_with_fallback", subset),
            "hierarchical_fallback_macro_f1": macro("hierarchical_with_fallback", subset),
            "fixed_phase_accepted": sum(not r["fixed_phase"]["abstained"] for r in subset),
            "hierarchical_accepted": sum(not r["hierarchical_meter_phase"]["abstained"] for r in subset),
        }
    return out


def compare(manifest: dict[str, Any], root: Path, *, limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    seen = set()
    validation = []
    for item in tracks:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            raise ValueError("unique nonempty track IDs required")
        seen.add(tid)
        if item.get("split") not in ("train", "validation", "test"):
            raise ValueError("explicit train/validation/test split required")
        if item["split"] == "validation":
            validation.append(item)
    chosen = validation[:limit] if limit is not None else validation
    rows = [evaluate_one(item, root) for item in chosen]
    return {
        "schema": "saund-validation-phase-only-shadow-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(chosen),
        "summary": _summary(rows),
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "auto_promotion_allowed": False,
        "reference_labels_used_for_inference": False,
        "limitations": [
            "Exploratory fixed-meter phase scoring; confidence scores are not probabilities.",
            "Production meter and beat positions remain unchanged. Wrong meter cannot be repaired by phase-only inference.",
            "Half-track stability is necessary but not sufficient for correct downbeat phase.",
            "Validation-only: previously inspected test split is never evaluated.",
            "Original references are used only after autonomous predictions for scoring.",
            "100ms downbeat scoring window is NOT the 25ms production cross-deck integrity limit.",
            "This is an offline fallback simulation; no live transitions are authorized.",
            "No real EDM or independent unseen evaluation has been performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/beat-benchmark/manifest.json"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/ballroom-phase-only-shadow.json",
    ))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"Manifest missing: {args.manifest}")
    report = compare(json.loads(args.manifest.read_text(encoding="utf-8")),
                     args.root or args.manifest.parent, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "summary": report["summary"],
    }, indent=2))
    if not report["selected_count"] or report["summary"]["evaluated"] != report["selected_count"]:
        parser.exit(2, "Incomplete validation evaluation; see detailed report.\n")


if __name__ == "__main__":
    main()
