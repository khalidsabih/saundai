#!/usr/bin/env python3
"""Offline hierarchical meter/downbeat shadow experiment (VALIDATION only).

Uses unchanged SAUND beat times, audio-derived onset accents, RMS dynamics,
and chroma repetition. Scores 3/4/6 pulse grouping, tests whether six
pulses are two groups of three, and abstains when evidence is ambiguous.
Reference labels are opened only in the scoring layer, never the candidate.
An abstention leaves production predictions untouched in fallback metrics.

This is exploratory and not an approved production model; no beat grids,
caches, policy thresholds, fallbacks, or audio files are changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import librosa
import numpy as np

from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH, analyze_track
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    DOWNBEAT_WINDOW_MS, match_events, read_timestamp_labels, valid_times,
)
from scripts.diagnose_beat_error_decomposition import reference_meter

METERS = (3, 4, 6)
# Exploratory fixed constants, not tuned to reference labels.
MIN_BEATS = 24
MIN_CONTRAST = 0.08
MIN_MARGIN = 0.12
MIN_SIX_GROUP_ASYMMETRY = 0.22
FEATURE_HOP = HOP_LENGTH


def beat_features(y: np.ndarray, sr: int, beat_times: list[float]) -> np.ndarray:
    """One feature vector per beat: onset + RMS + normalized chroma.

    Audio derived features alone (not annotated meter or downbeat). The
    localized median of spectrogram-derived features reduces single-frame
    sampling noise.
    """
    times = valid_times(beat_times, allow_empty=True)
    if len(times) == 0:
        return np.empty((0, 14), dtype=float)
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=FEATURE_HOP, aggregate=np.median,
    )
    rms = librosa.feature.rms(y=y, hop_length=FEATURE_HOP)[0]
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=FEATURE_HOP,
    )
    frames = librosa.time_to_frames(np.asarray(times), sr=sr, hop_length=FEATURE_HOP)
    output = np.zeros((len(times), 14), dtype=float)
    for index, frame in enumerate(frames):
        a = max(0, int(frame)-1)
        b = min(len(onset), int(frame)+3)
        if a < b:
            output[index, 0] = float(np.max(onset[a:b]))
        a = max(0, int(frame)-2)
        b = min(len(rms), int(frame)+3)
        if a < b:
            output[index, 1] = float(np.mean(rms[a:b]))
        a = max(0, int(frame)-2)
        b = min(chroma.shape[1], int(frame)+3)
        if a < b:
            output[index, 2:] = np.mean(chroma[:, a:b], axis=1)
    return output


def infer_meter(features: np.ndarray) -> dict[str, Any]:
    """Beat-synchronous hierarchical meter hypothesis; may abstain.

    For each meter and each possible downbeat phase, measure group-level
    accent separation and adjacent-bar feature recurrence. Six-beat
    hypotheses require extra evidence distinguishing two three-beat groups.
    Scores are not trained probabilities.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or np.any(~np.isfinite(x)):
        raise ValueError("features must be finite [beats, onset, rms, chroma...]")
    if len(x) < MIN_BEATS:
        return {"status": "insufficient_beats", "meter": None, "phase": None,
                "abstained": True, "scores": {}}
    # Robust within-track normalization without annotations.
    accent = np.log1p(np.maximum(x[:, :2], 0))
    scale = np.percentile(accent, 90, axis=0)
    accent = accent / np.maximum(scale, 1e-6)
    strength = np.clip(.65*accent[:, 0] + .35*accent[:, 1], 0, 3)
    chroma = np.maximum(x[:, 2:], 0)
    chroma /= np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    scores = {}
    for meter in METERS:
        options = []
        for phase in range(meter):
            seq = strength[phase:]
            bars = len(seq)//meter
            if bars < 3:
                continue
            grouped = seq[:bars*meter].reshape(bars, meter)
            # Relative accent at bar start, accounting for overall power.
            contrast = float(np.mean(grouped[:, 0] -
                np.mean(grouped[:, 1:], axis=1)))
            norm = float(np.mean(np.abs(grouped))) + 1e-8
            accent_score = max(0, contrast/norm)
            # Repetition in harmonic profiles on corresponding bar positions.
            cc = chroma[phase:phase+bars*meter].reshape(bars, meter, -1)
            recurrence = float(np.mean(np.sum(cc[1:]*cc[:-1], axis=2)))
            # Compare recurrence to mismatched within-bar positions (control).
            wrong = float(np.mean(np.sum(cc[1:]*np.roll(cc[:-1], 1, axis=1), axis=2)))
            repetition = max(0.0, recurrence-wrong)
            # For six-pulse bars: two groups of three with similar emphasis
            # are ambiguous; not positive evidence for genuine six-beat meter.
            half = (float(np.mean(np.abs(grouped[:, 0]-grouped[:, 3])))
                    / norm if meter == 6 else None)
            structural = .65*accent_score + .35*repetition
            if meter == 6:
                structural *= min(1.0, half/max(MIN_SIX_GROUP_ASYMMETRY, 1e-8))
            options.append({
                "phase": phase, "score": structural,
                "accent_contrast": accent_score,
                "bar_repetition": repetition,
                "six_group_asymmetry": half,
            })
        if not options:
            continue
        best = max(options, key=lambda o: o["score"])
        scores[str(meter)] = {
            "phase": best["phase"],
            "score": round(best["score"], 5),
            "accent_contrast": round(best["accent_contrast"], 5),
            "bar_repetition": round(best["bar_repetition"], 5),
            "six_group_asymmetry": (
                round(best["six_group_asymmetry"], 5)
                if best["six_group_asymmetry"] is not None else None
            ),
        }
    ranked = sorted(scores.items(), key=lambda z: z[1]["score"], reverse=True)
    if len(ranked) < 2:
        return {"status": "insufficient_grouping_evidence", "meter": None,
                "phase": None, "abstained": True, "scores": scores}
    top, second = ranked[0], ranked[1]
    margin = top[1]["score"]-second[1]["score"]
    accepted = top[1]["score"] >= MIN_CONTRAST and margin >= MIN_MARGIN
    if top[0] == "6" and (top[1]["six_group_asymmetry"] is None or
                           top[1]["six_group_asymmetry"] < MIN_SIX_GROUP_ASYMMETRY):
        accepted = False
    return {
        "status": "hypothesis" if accepted else "ambiguous",
        "meter": int(top[0]) if accepted else None,
        "phase": top[1]["phase"] if accepted else None,
        "abstained": not accepted,
        "top_meter_before_abstention": int(top[0]),
        "score_margin": round(margin, 5),
        "scores": scores,
        "score_is_not_probability": True,
    }


def choose_downbeats(beat_times: list[float], hypothesis: dict[str, Any]) -> list[float]:
    times = valid_times(beat_times, allow_empty=True)
    if hypothesis.get("abstained", True):
        return []
    meter, phase = hypothesis["meter"], hypothesis["phase"]
    if meter not in METERS or not isinstance(phase, int) or not 0 <= phase < meter:
        raise ValueError("invalid shadow meter/phase")
    return times[phase::meter]


def evaluate_one(item: dict[str, Any], root: Path) -> dict[str, Any]:
    row = {"id": str(item["id"]), "status": "pending"}
    try:
        # Production predictions and candidate evidence are computed without
        # passing any reference labels to the autonomous inference functions.
        audio = root/str(item["audio"])
        prod = analyze_track(audio)
        beats = valid_times(prod.beats_seconds or [], allow_empty=True)
        baseline_downbeats = valid_times(prod.downbeats_seconds or [], allow_empty=True)
        if len(beats) >= MIN_BEATS:
            signal, sr = load_audio(audio, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
            features = beat_features(librosa.to_mono(signal), sr, beats)
            started = time.perf_counter()
            candidate = infer_meter(features)
            row["inference_seconds_excluding_feature_extraction"] = round(
                time.perf_counter()-started, 5)
        else:
            candidate = infer_meter(np.empty((0, 14)))
        candidate_downbeats = choose_downbeats(beats, candidate)
        selected_downbeats = baseline_downbeats if candidate["abstained"] else candidate_downbeats
        # Only scoring has access to original reference annotations.
        ref_beats, _ = read_timestamp_labels(root/str(item["beats"]))
        ref_downbeats = None
        if item.get("downbeats"):
            ref_downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        reference = reference_meter(ref_downbeats, ref_beats)
        row.update({
            "status": "evaluated",
            "production_meter": prod.time_signature,
            "reference_meter": reference["meter"],
            "candidate": candidate,
            "autonomous_meter_correct": (
                candidate["meter"] == reference["meter"]
                if not candidate["abstained"] and reference["meter"] in METERS
                else None
            ),
            "production_meter_correct": (
                prod.time_signature == reference["meter"]
                if reference["meter"] in METERS else None
            ),
            "production_downbeat": match_events(
                ref_downbeats, baseline_downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS,
            ) if ref_downbeats else None,
            "candidate_accepted_downbeat": match_events(
                ref_downbeats, candidate_downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS,
            ) if ref_downbeats and not candidate["abstained"] else None,
            "shadow_with_abstention_fallback": match_events(
                ref_downbeats, selected_downbeats, tolerance_ms=DOWNBEAT_WINDOW_MS,
            ) if ref_downbeats else None,
        })
    except (ValueError, OSError, KeyError, TypeError) as exc:
        row["status"] = "unavailable_or_invalid_data"
        row["reason"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    accepted = [r for r in evaluated if not r["candidate"]["abstained"]]
    ref_meter = [r for r in evaluated if r["production_meter_correct"] is not None]
    def macro(key: str, subset: list[dict[str, Any]]) -> float | None:
        values = [r[key]["f_measure"] for r in subset if r.get(key) is not None]
        return round(float(np.mean(values)), 5) if values else None
    return {
        "evaluated": len(evaluated),
        "accepted_count": len(accepted),
        "abstained_count": len(evaluated)-len(accepted),
        "candidate_meter_counts": dict(sorted(Counter(
            str(r["candidate"]["meter"]) for r in accepted).items())),
        "reference_meter_known_count": len(ref_meter),
        "baseline_meter_correct": sum(r["production_meter_correct"] for r in ref_meter),
        "accepted_reference_meter_known_count": sum(
            r["autonomous_meter_correct"] is not None for r in accepted),
        "accepted_candidate_meter_correct": sum(
            r["autonomous_meter_correct"] is True for r in accepted),
        "baseline_downbeat_macro_f1_all": macro("production_downbeat", evaluated),
        "fallback_shadow_downbeat_macro_f1_all": macro(
            "shadow_with_abstention_fallback", evaluated),
        "baseline_downbeat_macro_f1_accepted_subset": macro(
            "production_downbeat", accepted),
        "candidate_downbeat_macro_f1_accepted_subset": macro(
            "candidate_accepted_downbeat", accepted),
    }


def compare(manifest: dict[str, Any], root: Path, *,
            limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    ids = [str(x.get("id") or "") for x in tracks]
    if len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("unique nonempty IDs required")
    if any(t.get("split") not in ("train", "validation", "test") for t in tracks):
        raise ValueError("explicit split required")
    validation = [x for x in tracks if x["split"] == "validation"]
    selected = validation[:limit] if limit is not None else validation
    rows = [evaluate_one(item, root) for item in selected]
    return {
        "schema": "saund-shadow-hierarchical-meter-v1",
        "dataset": manifest.get("dataset"),
        "selected_split": "validation",
        "selected_count": len(selected),
        "summary": summarize(rows),
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "auto_promotion_allowed": False,
        "reference_labels_used_for_candidate": False,
        "limitations": [
            "This is an exploratory fixed-rule candidate, not independently validated.",
            "Validation-only: previously inspected Ballroom test partition is never loaded.",
            "Six beats vs two groups of three can be musically and notationally ambiguous.",
            "Abstention fallback F1 is diagnostic simulation, not a production deployment.",
            "The candidate uses production beat positions and cannot fix wrong pulse or timestamp bias.",
            "Reference meter is inferred from annotations solely for scoring, NEVER inference.",
            "100ms downbeat matching tolerance is not the protected 25ms cross-deck limit.",
            "Real EDM and holdout testing required before production consideration.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path("data/beat-benchmark/manifest.json"))
    p.add_argument("--root", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/ballroom-hierarchical-meter-shadow.json",
    ))
    args = p.parse_args()
    if not args.manifest.is_file():
        p.error(f"Missing manifest: {args.manifest}")
    report = compare(json.loads(args.manifest.read_text(encoding="utf-8")),
                     args.root or args.manifest.parent, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "summary": report["summary"],
    }, indent=2))
    if (not report["selected_count"] or
            report["summary"]["evaluated"] != report["selected_count"]):
        p.exit(2, "Incomplete validation run; inspect JSON report.\n")


if __name__ == "__main__":
    main()
