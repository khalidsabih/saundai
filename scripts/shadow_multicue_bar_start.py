#!/usr/bin/env python3
"""Offline, annotation-blind bar-start evidence from three independent cues.

Bass onset, harmonic change, and repeated bar-level accent stability each
vote for a phase at the UNCHANGED SAUND meter and beat positions. The fused
candidate requires cue agreement and agreement across recording halves;
otherwise it abstains. This is a fixed-rule exploratory shadow, not a
trained model or production feature. Ground-truth labels are opened only
after audio inference, for retrospective scoring and cohort assignment.

The existing rhythm model, 25ms cross-deck phase gate, caches, bounded
warp, fallbacks and mixing engine are NEVER modified by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import librosa
import numpy as np

from djai.analysis import ANALYSIS_SAMPLE_RATE, HOP_LENGTH, analyze_track
from djai.audio import load_audio
from scripts.benchmark_beat_integrity_dataset import (
    BEAT_WINDOW_MS, DOWNBEAT_WINDOW_MS, match_events, read_timestamp_labels,
    valid_times,
)
from scripts.diagnose_beat_error_decomposition import (
    event_rate_diagnosis, reference_meter, diagnostic_downbeat_phase_search,
)
from scripts.shadow_hierarchical_meter import METERS

MIN_BEATS = 32
MIN_BARS_PER_HALF = 3
MIN_CUE_MARGIN = 0.08
MIN_FUSION_MARGIN = 0.12
MIN_CUE_VOTES = 2
CUES = ("bass_onset", "harmonic_change", "bar_repetition")


def audio_cues(y: np.ndarray, sr: int, beat_times: list[float]) -> np.ndarray:
    """Extract [bass spectral flux, chroma novelty, low-band accent] per beat.

    Uses ONE STFT for all three signals, avoiding repeated high-cost
    transforms. The third channel is only the beat strength used by the
    bar-repetition stability test, not a third independent physical sensor.
    """
    times = valid_times(beat_times, allow_empty=True)
    signal = np.asarray(y, dtype=float)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)) or sr < 1:
        raise ValueError("audio must be finite mono at positive sample rate")
    if not times:
        return np.zeros((0, 3), dtype=float)
    stft = np.abs(librosa.stft(signal, n_fft=2048, hop_length=HOP_LENGTH))
    freq = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freq >= 35) & (freq <= 190)
    if not np.any(band):
        raise ValueError("no low-frequency bins in input sample rate")
    low = np.log1p(stft[band])
    positive = np.maximum(np.diff(low, axis=1, prepend=low[:, :1]), 0)
    bass_flux = np.mean(positive, axis=0)
    bass_power = np.mean(low, axis=0)
    chroma = librosa.feature.chroma_stft(
        S=stft**2, sr=sr, hop_length=HOP_LENGTH,
    )
    chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
    novelty = np.zeros(chroma.shape[1], dtype=float)
    if len(novelty) > 1:
        # Temporal chroma novelty: compare adjacent FRAMES (axis 1), not pitch bins.
        # The result must contain T-1 values for novelty[1:].
        novelty[1:] = np.maximum(0, 1-np.sum(chroma[:, 1:]*chroma[:, :-1], axis=0))
    frames = librosa.time_to_frames(np.asarray(times), sr=sr, hop_length=HOP_LENGTH)
    out = np.zeros((len(times), 3), dtype=float)
    for i, frame in enumerate(frames):
        a = max(0, int(frame)-1)
        b = min(len(bass_flux), int(frame)+3)
        if b > a:
            out[i, 0] = float(np.max(bass_flux[a:b]))
            out[i, 1] = float(np.max(novelty[a:b]))
            out[i, 2] = float(np.mean(bass_power[a:b]))
    return out


def _normalized(x: np.ndarray) -> np.ndarray:
    z = np.log1p(np.maximum(x, 0))
    q = float(np.percentile(z, 90))
    return np.clip(z/max(q, 1e-8), 0, 3)


def phase_scores(features: np.ndarray, meter: int, *, start: int = 0,
                 end: int | None = None) -> dict[str, list[float]]:
    """Fixed global beat-index phase comparison, no reference annotations."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("finite [beats,3] audio features required")
    if meter not in METERS:
        raise ValueError("unsupported meter")
    stop = len(x) if end is None else end
    if start < 0 or stop > len(x) or stop < start:
        raise ValueError("invalid beat interval")
    if stop-start < meter*MIN_BARS_PER_HALF:
        raise ValueError("insufficient bars")
    # Normalize across entire recording, not per phase or per half.
    strength = {CUES[i]: _normalized(x[:, i]) for i in range(3)}
    # Repetition is not a distinct sensor: evaluate stability of the bass
    # accent pattern across adjacent complete bars at each possible phase.
    output: dict[str, list[float]] = {}
    for cue in CUES[:2]:
        values = strength[cue]
        means = [
            float(np.mean(values[np.arange(start, stop)[np.arange(start, stop)%meter == p]]))
            for p in range(meter)
        ]
        center = float(np.mean(means))
        output[cue] = [
            max(0.0, (v-center)/max(abs(center), 1e-8))
            for v in means
        ]
    repeats = []
    values = strength["bar_repetition"]
    for phase in range(meter):
        # Preserve original beat indexing, including half-track slicing.
        first = start + ((phase-start) % meter)
        bars = (stop-first)//meter
        if bars < MIN_BARS_PER_HALF:
            repeats.append(0.0)
            continue
        pattern = values[first:first+bars*meter].reshape(bars, meter)
        # Emphasis of beat 1 plus consistency across adjacent bars.
        head = pattern[:, 0]
        other = np.mean(pattern[:, 1:], axis=1)
        contrast = np.maximum(head-other, 0)
        consistency = 1/(1+float(np.std(head))/max(float(np.mean(head)), 1e-8))
        repeats.append(max(0.0, float(np.mean(contrast)))*consistency /
                       max(float(np.mean(np.abs(pattern))), 1e-8))
    output["bar_repetition"] = repeats
    return output


def infer_bar_start(features: np.ndarray, meter: int) -> dict[str, Any]:
    """Accept a fixed-meter phase only with independently scored cue agreement.

    Fixed conservative thresholds are hypotheses, not calibrated probabilities.
    For transparent diagnosis, each cue always has a leading phase even
    when consensus abstains.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise ValueError("finite [beats,3] audio features required")
    if meter not in METERS:
        return {"status": "unsupported_meter", "meter": meter,
                "phase": None, "abstained": True, "cue_votes": {}}
    if len(x) < max(MIN_BEATS, 2*meter*MIN_BARS_PER_HALF+meter):
        return {"status": "insufficient_beats", "meter": meter,
                "phase": None, "abstained": True, "cue_votes": {}}
    scores = phase_scores(x, meter)
    mid = len(x)//2
    half_scores = [
        phase_scores(x, meter, start=0, end=mid),
        phase_scores(x, meter, start=mid, end=len(x)),
    ]
    votes = {}
    for cue in CUES:
        vals = scores[cue]
        order = sorted(range(meter), key=lambda p: vals[p], reverse=True)
        leader, runner = order[:2]
        margin = vals[leader]-vals[runner]
        # Cue does not vote if weak or half-track winners disagree.
        halves = [int(np.argmax(s[cue])) for s in half_scores]
        stable = all(p == leader for p in halves)
        votes[cue] = {
            "leading_phase": leader,
            "half_phases": halves,
            "half_agreement": stable,
            "margin": round(float(margin), 5),
            "strong": margin >= MIN_CUE_MARGIN and stable,
            "scores_by_phase": [round(float(v), 5) for v in vals],
        }
    count = Counter(v["leading_phase"] for v in votes.values() if v["strong"])
    top_votes = max(count.values(), default=0)
    winners = [p for p, n in count.items() if n == top_votes and n >= MIN_CUE_VOTES]
    accepted = len(winners) == 1
    selected = winners[0] if accepted else None
    # Fusion contrast prevents accepting near-tied evidence even with 2 votes.
    fused = np.zeros(meter, dtype=float)
    for cue in CUES:
        v = np.asarray(scores[cue], dtype=float)
        fused += v/max(float(np.max(v)), 1e-8)
    ranked = np.argsort(-fused)
    fusion_margin = float(fused[ranked[0]]-fused[ranked[1]])
    if not accepted or int(ranked[0]) != selected or fusion_margin < MIN_FUSION_MARGIN:
        selected, accepted = None, False
    return {
        "status": "hypothesis" if accepted else "ambiguous",
        "meter": meter,
        "phase": selected,
        "abstained": not accepted,
        "cue_votes": votes,
        "vote_counts": {str(k): v for k, v in sorted(count.items())},
        "fused_leading_phase": int(ranked[0]),
        "fusion_margin": round(fusion_margin, 5),
        "scores_are_not_probabilities": True,
        "no_reference_labels_used": True,
    }


def select_downbeats(beats: list[float], phase: int, meter: int) -> list[float]:
    t = valid_times(beats, allow_empty=True)
    if meter not in METERS or not isinstance(phase, int) or not 0 <= phase < meter:
        raise ValueError("invalid meter/phase")
    return t[phase::meter]


def evaluate_one(item: dict[str, Any], root: Path) -> dict[str, Any]:
    row = {"id": str(item["id"]), "status": "pending"}
    try:
        # All autonomous choices are made BEFORE reading annotations.
        source = root/str(item["audio"])
        prod = analyze_track(source)
        beats = valid_times(prod.beats_seconds or [], allow_empty=True)
        baseline = valid_times(prod.downbeats_seconds or [], allow_empty=True)
        meter = int(prod.time_signature)
        if len(beats) >= MIN_BEATS:
            signal, sr = load_audio(source, sample_rate=ANALYSIS_SAMPLE_RATE, mono=False)
            x = audio_cues(librosa.to_mono(signal), sr, beats)
        else:
            x = np.empty((0, 3), dtype=float)
        candidate = infer_bar_start(x, meter)
        variants = {}
        for cue in CUES:
            vote = candidate["cue_votes"].get(cue)
            variants[cue] = (
                select_downbeats(beats, vote["leading_phase"], meter)
                if vote and vote["strong"] else None
            )
        fusion = (select_downbeats(beats, candidate["phase"], meter)
                  if not candidate["abstained"] else None)
        # Reference labels and reference-assisted oracle are used for
        # retrospective scoring/triage ONLY.
        ref_beats, _ = read_timestamp_labels(root/str(item["beats"]))
        ref_down = None
        if item.get("downbeats"):
            ref_down, _ = read_timestamp_labels(root/str(item["downbeats"]))
        ref_meter = reference_meter(ref_down, ref_beats).get("meter")
        beat = match_events(ref_beats, beats, tolerance_ms=BEAT_WINDOW_MS)
        pulse = event_rate_diagnosis(ref_beats, beats)
        oracle = diagnostic_downbeat_phase_search(
            beats, ref_down, reference_meter_value=meter
        )
        def measure(t: list[float] | None) -> dict[str, Any] | None:
            return (match_events(ref_down, t, tolerance_ms=DOWNBEAT_WINDOW_MS)
                    if ref_down is not None and t is not None else None)
        production_metric = measure(baseline)
        fusion_metric = measure(fusion if fusion is not None else baseline)
        cues_metrics = {
            cue: measure(pred if pred is not None else baseline)
            for cue, pred in variants.items()
        }
        base_f1 = production_metric["f_measure"] if production_metric else None
        oracle_gain = (
            oracle["best_phase_f_measure"]-base_f1
            if base_f1 is not None and oracle.get("status") == "reference_assisted_oracle"
            else None
        )
        # Cohorts are for interpretation, NOT for model selection.
        clean = (
            ref_meter == meter
            and pulse["class"] == "similar_pulse_period"
            and beat["f_measure"] >= .80
        )
        if clean and base_f1 is not None and base_f1 <= .10 and oracle_gain is not None and oracle_gain >= .10:
            cohort = "clean_phase_failure"
        elif clean and base_f1 is not None and base_f1 >= .80:
            cohort = "clean_phase_success"
        else:
            cohort = "other"
        row.update({
            "status": "evaluated",
            "production_meter": meter, "reference_meter": ref_meter,
            "pulse_class": pulse["class"], "beat_f1": beat["f_measure"],
            "cohort": cohort, "oracle": oracle,
            "candidate": candidate,
            "baseline_downbeat": production_metric,
            "fusion_with_fallback": fusion_metric,
            "cue_with_fallback": cues_metrics,
            "fusion_f1_delta": round(fusion_metric["f_measure"]-base_f1, 5)
                if base_f1 is not None else None,
            "cue_f1_deltas": {
                cue: round(metric["f_measure"]-base_f1, 5)
                if metric is not None and base_f1 is not None else None
                for cue, metric in cues_metrics.items()
            },
        })
    except (OSError, ValueError, TypeError, KeyError) as exc:
        row.update(status="unavailable_or_invalid_data",
                   reason=f"{type(exc).__name__}: {exc}")
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "evaluated"]
    def avg(values: list[float]) -> float | None:
        return round(float(np.mean(values)), 5) if values else None
    def mean_f1(field: str, group: list[dict[str, Any]]) -> float | None:
        return avg([r[field]["f_measure"] for r in group if r[field] is not None])
    out = {
        "evaluated": len(ok),
        "failed": len(rows)-len(ok),
        "failure_reasons": dict(sorted(Counter(
            r.get("reason", r["status"]) for r in rows if r["status"] != "evaluated"
        ).items())),
        "failed_examples": [
            {"id": r["id"], "reason": r.get("reason", r["status"])}
            for r in rows if r["status"] != "evaluated"
        ][:10],
        "cohort_counts": dict(sorted(Counter(r["cohort"] for r in ok).items())),
        "baseline_macro_f1": mean_f1("baseline_downbeat", ok),
        "fusion_fallback_macro_f1": mean_f1("fusion_with_fallback", ok),
        "fusion_accepted": sum(not r["candidate"]["abstained"] for r in ok),
        "fusion_abstained": sum(r["candidate"]["abstained"] for r in ok),
        "fusion_improved_tracks": sum((r["fusion_f1_delta"] or 0) > 0 for r in ok),
        "fusion_regressed_tracks": sum((r["fusion_f1_delta"] or 0) < 0 for r in ok),
        "regressions": [
            {"id": r["id"], "cohort": r["cohort"],
             "fusion_f1_delta": r["fusion_f1_delta"]}
            for r in ok if (r["fusion_f1_delta"] or 0) < 0
        ],
        "cues": {},
        "by_reference_meter": {},
        "by_cohort": {},
    }
    for cue in CUES:
        out["cues"][cue] = {
            "accepted": sum(r["candidate"]["cue_votes"].get(cue, {}).get("strong", False) for r in ok),
            "fallback_macro_f1": avg([
                r["cue_with_fallback"][cue]["f_measure"] for r in ok
                if r["cue_with_fallback"][cue] is not None
            ]),
            "improved_tracks": sum((r["cue_f1_deltas"][cue] or 0) > 0 for r in ok),
            "regressed_tracks": sum((r["cue_f1_deltas"][cue] or 0) < 0 for r in ok),
        }
    for kind, values in (
        ("by_reference_meter", (3, 4, 6, None)),
        ("by_cohort", ("clean_phase_failure", "clean_phase_success", "other")),
    ):
        for value in values:
            group = [r for r in ok if (r["reference_meter"] if kind == "by_reference_meter" else r["cohort"]) == value]
            out[kind][str(value)] = {
                "count": len(group),
                "baseline_macro_f1": mean_f1("baseline_downbeat", group),
                "fusion_fallback_macro_f1": mean_f1("fusion_with_fallback", group),
                "fusion_accepted": sum(not r["candidate"]["abstained"] for r in group),
                "fusion_improved": sum((r["fusion_f1_delta"] or 0) > 0 for r in group),
                "fusion_regressed": sum((r["fusion_f1_delta"] or 0) < 0 for r in group),
            }
    return out


def compare(manifest: dict[str, Any], root: Path, *,
            limit: int | None = None) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("manifest requires tracks array")
    seen = set()
    selected = []
    for item in tracks:
        tid = str(item.get("id") or "")
        if not tid or tid in seen:
            raise ValueError("all IDs must be unique and nonempty")
        seen.add(tid)
        if item.get("split") not in ("train", "validation", "test"):
            raise ValueError("all tracks require explicit split")
        if item["split"] == "validation":
            selected.append(item)
    if limit is not None:
        selected = selected[:limit]
    rows = [evaluate_one(item, root) for item in selected]
    return {
        "schema": "saund-multicue-bar-start-shadow-v1",
        "selected_split": "validation",
        "dataset": manifest.get("dataset"),
        "selected_count": len(selected),
        "summary": summarize(rows),
        "tracks": rows,
        "production_changes": False,
        "phase_integrity_limit_changed": False,
        "beat_grid_edit_authorized": False,
        "autonomous_inference_uses_reference": False,
        "auto_promotion_allowed": False,
        "limitations": [
            "Fixed-rule multi-cue candidate with uncalibrated score thresholds.",
            "Repetition reuses bass features, so the three cues are NOT fully independent.",
            "Clean failure/success cohorts are defined using labels after inference only.",
            "Cohorts are diagnostic and must not be used to tune and claim unbiased accuracy.",
            "This evaluates Ballroom validation ONLY, never previously inspected test split.",
            "Wrong production pulse or meter are not fixed by this phase-only candidate.",
            "100ms downbeat annotation matching is not the protected 25ms cross-deck phase-integrity threshold.",
            "Candidate and per-cue fallback are simulated OFFLINE; no live correction occurs.",
            "EDM/independent fresh evaluation and robustness studies required before promotion.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path("data/beat-benchmark/manifest.json"))
    p.add_argument("--root", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--output", type=Path, default=Path(
        "output/beat-integrity/ballroom-multicue-bar-start-shadow.json"
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
        "evaluated": report["summary"]["evaluated"],
        "failed": report["summary"]["failed"],
        "failure_reasons": report["summary"]["failure_reasons"],
        "failed_examples": report["summary"]["failed_examples"],
        "summary": report["summary"],
    }, indent=2))
    if not report["selected_count"] or report["summary"]["evaluated"] != report["selected_count"]:
        p.exit(2, "Incomplete validation; inspect individual track failures.\n")


if __name__ == "__main__":
    main()
