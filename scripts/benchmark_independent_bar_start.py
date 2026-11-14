#!/usr/bin/env python3
"""Frozen bar-start comparison on newly collected independent audio ONLY.

Four policies run on identical unchanged SAUND beats/meter:
  production, existing three-cue fusion, bass alone, repetition alone,
  and rhythm consensus (bass AND repetition must independently vote
  strongly for the same phase). Harmonic novelty cannot authorize a
  rhythm-consensus override. All thresholds/feature extraction are
  imported unchanged from the frozen multicue shadow.

REQUIRES an external explicit independent manifest and complete annotation
files. Refuses Ballroom and the existing validation/test dataset.
No tuning, promotion, production cache writes or live DJ changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import librosa
import numpy as np

from djai.analysis import ANALYSIS_SAMPLE_RATE, analyze_track
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    DOWNBEAT_WINDOW_MS, match_events, read_timestamp_labels, valid_times,
)
from scripts.shadow_multicue_bar_start import (
    CUES, MIN_BEATS, audio_cues, infer_bar_start, select_downbeats,
)

METHODS = ("production", "frozen_fusion", "bass_onset",
           "bar_repetition", "rhythm_consensus")
MIN_INDEPENDENT_TRACKS = 1


def validate_manifest(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    """Reject known Ballroom leakage and any non-independent split.

    The operator must ALSO verify ownership, provenance, recording overlap,
    annotation alignment and dataset independence externally.
    """
    if "ballroom" in str(manifest_path).lower():
        raise ValueError("Ballroom benchmark cannot serve as independent holdout")
    name = str(manifest.get("dataset") or "")
    if not name or "ballroom" in name.lower():
        raise ValueError("independent dataset name required; Ballroom is excluded")
    origin = str(manifest.get("provenance") or "").strip()
    if not origin:
        raise ValueError("document independent dataset provenance")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or len(tracks) < MIN_INDEPENDENT_TRACKS:
        raise ValueError("independent manifest must contain tracks")
    ids = [str(t.get("id") or "") for t in tracks]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("track IDs must be unique and nonempty")
    for item in tracks:
        if item.get("split") != "independent":
            raise ValueError("ONLY split=independent is permitted; no Ballroom validation/test")
        for key in ("audio", "beats", "downbeats"):
            path = str(item.get(key) or "")
            if not path or "ballroom" in path.lower():
                raise ValueError(f"{key} must be explicit, independent, and not Ballroom")
        if "ballroom" in str(item["id"]).lower():
            raise ValueError("Ballroom track IDs are excluded")
    return tracks


def rhythm_consensus(candidate: dict[str, Any]) -> int | None:
    """Two rhythmic feature votes must be strong and agree; no chroma vote."""
    votes = candidate.get("cue_votes", {})
    bass = votes.get("bass_onset", {})
    repeat = votes.get("bar_repetition", {})
    if not (bass.get("strong") and repeat.get("strong")):
        return None
    if bass["leading_phase"] != repeat["leading_phase"]:
        return None
    return int(bass["leading_phase"])


def predicted_alternatives(
    beats: list[float], baseline_downbeats: list[float], meter: int,
    features: np.ndarray,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    candidate = infer_bar_start(features, meter)
    by_method: dict[str, list[float]] = {"production": baseline_downbeats}
    by_method["frozen_fusion"] = (
        select_downbeats(beats, candidate["phase"], meter)
        if not candidate["abstained"] else baseline_downbeats
    )
    for cue in ("bass_onset", "bar_repetition"):
        vote = candidate.get("cue_votes", {}).get(cue, {})
        by_method[cue] = (
            select_downbeats(beats, vote["leading_phase"], meter)
            if vote.get("strong") else baseline_downbeats
        )
    rhythm_phase = rhythm_consensus(candidate)
    by_method["rhythm_consensus"] = (
        select_downbeats(beats, rhythm_phase, meter)
        if rhythm_phase is not None else baseline_downbeats
    )
    candidate["rhythm_consensus_phase"] = rhythm_phase
    return by_method, candidate


def evaluate_track(item: dict[str, Any], root: Path) -> dict[str, Any]:
    row = {"id": str(item["id"]), "status": "pending"}
    try:
        # Infer ALL candidate phases before even opening annotations.
        audio_path = root / str(item["audio"])
        production = analyze_track(audio_path)
        beats = valid_times(production.beats_seconds or [], allow_empty=True)
        baseline = valid_times(production.downbeats_seconds or [], allow_empty=True)
        meter = int(production.time_signature)
        if len(beats) >= MIN_BEATS:
            audio, sr = load_audio(audio_path, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
            features = audio_cues(librosa.to_mono(audio), sr, beats)
        else:
            features = np.empty((0, 3), dtype=float)
        alternatives, candidate = predicted_alternatives(beats, baseline, meter, features)
        reference_beats, _ = read_timestamp_labels(root/str(item["beats"]))
        reference_downbeats, _ = read_timestamp_labels(root/str(item["downbeats"]))
        # Annotation beat labels are read for provenance/validity only;
        # none are supplied to phase inference.
        if not reference_beats:
            raise ValueError("missing beat annotations")
        scored = {method: match_events(
            reference_downbeats, alternatives[method],
            tolerance_ms=DOWNBEAT_WINDOW_MS,
        ) for method in METHODS}
        baseline_f1 = scored["production"]["f_measure"]
        row.update({
            "status": "evaluated", "production_meter": meter,
            "candidate": candidate,
            "methods": {
                method: {
                    "f1": scored[method]["f_measure"],
                    "delta_vs_production": round(scored[method]["f_measure"]-baseline_f1, 5),
                    "fallback_used": alternatives[method] is baseline,
                    "matched": scored[method]["matched"],
                    "reference_count": scored[method]["reference_count"],
                    "predicted_count": scored[method]["predicted_count"],
                } for method in METHODS
            },
        })
    except (OSError, TypeError, ValueError, KeyError) as exc:
        row.update(status="unavailable_or_invalid_data",
                   reason=f"{type(exc).__name__}: {exc}")
    return row


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["status"] == "evaluated"]
    out: dict[str, Any] = {
        "evaluated": len(valid),
        "failed": len(rows)-len(valid),
        "failures": [{"id": r["id"], "reason": r.get("reason")}
                     for r in rows if r["status"] != "evaluated"],
        "method_comparison": {},
        "meter_groups": dict(sorted(Counter(
            str(r["production_meter"]) for r in valid
        ).items())),
    }
    for method in METHODS:
        vals = [r["methods"][method]["f1"] for r in valid]
        deltas = [r["methods"][method]["delta_vs_production"] for r in valid]
        out["method_comparison"][method] = {
            "macro_f1": round(float(np.mean(vals)), 5) if vals else None,
            "mean_delta": round(float(np.mean(deltas)), 5) if deltas else None,
            "improved": sum(d > 0 for d in deltas),
            "regressed": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas),
            "overrides": sum(not r["methods"][method]["fallback_used"] for r in valid),
            "regressions": [
                {"id": r["id"], "delta": r["methods"][method]["delta_vs_production"]}
                for r in valid if r["methods"][method]["delta_vs_production"] < 0
            ],
        }
    return out


def benchmark(manifest: dict[str, Any], root: Path, manifest_path: Path) -> dict[str, Any]:
    tracks = validate_manifest(manifest, manifest_path)
    rows = [evaluate_track(t, root) for t in tracks]
    return {
        "schema": "saund-independent-bar-start-comparison-v1",
        "dataset": manifest["dataset"], "provenance": manifest["provenance"],
        "selected_split": "independent",
        "selected_count": len(tracks),
        "summary": summary(rows),
        "tracks": rows,
        "parameters_frozen": True,
        "reference_labels_used_for_phase_inference": False,
        "production_changes": False,
        "beat_grid_edit_authorized": False,
        "phase_integrity_limit_changed": False,
        "auto_promotion_allowed": False,
        "limitations": [
            "Manifest provenance and no recording overlap must be independently verified by operator.",
            "No ground-truth data is used for phase inference; annotations are read only after alternatives are fixed.",
            "This is fixed-meter/unchanged-beat-grid offline evaluation, not transition quality.",
            "Rhythm consensus uses dependent bass-derived signals; not statistically independent votes.",
            "70ms beat / 100ms downbeat annotation windows never change the 25ms production cross-deck safety gate.",
            "All methods are fixed before evaluation; do not tune thresholds on independent holdout.",
            "No promotion based solely on downbeat F1; EDM genre, phase safety and listening checks remain necessary.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/independent-bar-start-comparison.json"
    ))
    args = p.parse_args()
    if not args.manifest.is_file():
        p.error("Independent manifest missing: provide a new, annotated dataset")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = benchmark(manifest, args.root or args.manifest.parent, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output.resolve()),
        "selected": report["selected_count"],
        "summary": report["summary"],
    }, indent=2))
    if report["summary"]["failed"]:
        p.exit(2, "Incomplete independent benchmark; see per-track errors.\n")


if __name__ == "__main__":
    main()
