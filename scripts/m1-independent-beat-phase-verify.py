#!/usr/bin/env python3
"""Research-only independent beat-reference comparison for a private M1 render.

Essentia multifeature and locally pinned Beat This! propose source-track beat
events. The signed renderer's source-time warp maps them into output time.
Candidate agreement / timing are *not* independent rendered-audio certification:
the final mix cannot isolate beats by deck. Ambiguous coverage fails closed.

No production threshold, neural authority, source audio, or signed plan is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ["DJAI_NEURAL_BEATS"] = "1"
os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"


def load_acoustic_verifier():
    spec = importlib.util.spec_from_file_location(
        "m1_acoustic_verifier", ROOT / "scripts" / "m1-rendered-phase-acoustic-check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _beats(values, name):
    import numpy as np
    if values is None:
        return []
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not np.isfinite(a).all() or (len(a) > 1 and np.any(np.diff(a) <= 0)):
        raise ValueError(f"{name} must be finite, strictly increasing beat times")
    return a.tolist()


def essentia_beats(path):
    """Full-track 44.1 kHz Essentia reference independent of SAUND selected grid."""
    try:
        import essentia.standard as es
    except (ImportError, AttributeError, RuntimeError) as exc:
        return {"available": False, "reason": f"essentia unavailable or binary incompatible: {type(exc).__name__}"}
    try:
        signal = es.MonoLoader(filename=str(path), sampleRate=44100)()
        bpm, ticks, confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(signal)
        return {
            "available": True, "bpm": round(float(bpm), 5),
            "confidence": round(float(confidence), 5),
            "beats_seconds": _beats(ticks, "essentia beats"),
            "method": "essentia_rhythmextractor2013_multifeature",
        }
    except Exception as exc:
        return {"available": False, "reason": f"essentia inference failed: {type(exc).__name__}"}



def external_essentia_predictions(predictions_path, original):
    """Read SHA-bound beats produced in a separate NumPy 1.x environment."""
    data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "saund-m1-essentia-sidecar-v1":
        raise ValueError("unsupported Essentia predictions schema")
    if data.get("method") != "essentia_rhythmextractor2013_multifeature":
        raise ValueError("unexpected Essentia reference method")
    tracks = data.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != {"A", "B"}:
        raise ValueError("Essentia predictions must contain exactly decks A and B")
    output = {}
    for deck, path in original.items():
        item = tracks[deck]
        if not isinstance(item, dict):
            raise ValueError(f"invalid Essentia sidecar entry for {deck}")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if item.get("audio_sha256") != checksum:
            raise ValueError(f"Essentia audio SHA256 mismatch for deck {deck}")
        if item.get("method") != data["method"]:
            raise ValueError(f"Essentia method mismatch for deck {deck}")
        beats = _beats(item.get("beats_seconds"), f"Essentia {deck} beats")
        if len(beats) < 4:
            output[deck] = {
                "available": False, "reason": "fewer than four independent beat predictions",
                "source": "numpy1_sidecar",
            }
        else:
            output[deck] = {
                "available": True, "beats_seconds": beats,
                "bpm": float(item["bpm"]),
                "confidence": float(item["confidence"]),
                "method": data["method"],
                "source": "numpy1_sidecar",
            }
    return output


def neural_candidate(track):
    """Evidence from the already-pinned advisory model; never use selected grid."""
    evidence = (getattr(track, "neural_evidence", None) or {}).get("beat_downbeat", {})
    if evidence.get("available") is not True:
        return {"available": False, "reason": evidence.get("reason", "neural candidate unavailable")}
    return {
        "available": True,
        "beats_seconds": _beats(evidence.get("beats_seconds"), "neural beats"),
        "bpm": evidence.get("bpm"),
        "confidence": evidence.get("confidence"),
        "used": evidence.get("used"),
        "method": "beat_this_local_checkpoint_advisory",
    }


def consensus(reference, candidate, start, end, *, tolerance=0.045):
    """Local one-to-one beat matching. No tempo octave or phase-shift rescue."""
    if not (math.isfinite(start) and math.isfinite(end) and
            math.isfinite(tolerance) and end > start and tolerance > 0):
        raise ValueError("invalid beat matching window or tolerance")
    # Validate each complete reference before filtering. Filtering first can
    # silently discard NaN events or conceal out-of-order timestamps.
    reference = _beats(reference, "essentia reference beats")
    candidate = _beats(candidate, "neural candidate beats")
    a = [x for x in reference if start <= x <= end]
    b = [x for x in candidate if start <= x <= end]
    i = j = 0
    matches = []
    while i < len(a) and j < len(b):
        delta = b[j] - a[i]
        if abs(delta) <= tolerance:
            matches.append((a[i], b[j], delta))
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    count = len(matches)
    coverage = count / max(len(a), len(b), 1)
    intervals = [matches[k + 1][0] - matches[k][0] for k in range(count - 1)]
    max_gap = max(intervals) if intervals else float("inf")
    time_span = matches[-1][0] - matches[0][0] if count > 1 else 0.0
    reliable = (
        count >= 12 and coverage >= 0.75 and time_span >= 0.75 * (end - start)
        and max_gap <= 1.5
        and matches[0][0] <= start + 1.5 and matches[-1][0] >= end - 1.5
    ) if count else False
    return {
        "status": "reference_agreement" if reliable else "inconclusive",
        "essentia_count": len(a), "neural_count": len(b), "matched": count,
        "match_fraction": round(coverage, 5),
        "max_gap_seconds": round(max_gap, 5) if math.isfinite(max_gap) else None,
        "max_disagreement_ms": round(max(abs(x[2]) for x in matches) * 1000, 3) if matches else None,
        "matched_essentia_source_seconds": [round(x[0], 7) for x in matches],
        "matches_are_human_annotated": False,
    }


def map_source_beats_to_render(beats, start, duration, rate, *, offsets=None):
    """Invert source_position(t) = source_start + rate*t + source_offset(t).

    Mirrors the renderer's linear correction interpolation. Strict monotonicity
    is required; otherwise the correspondence is inconclusive.
    """
    import numpy as np
    if not math.isfinite(rate) or rate <= 0 or duration <= 0:
        raise ValueError("invalid render rate or duration")
    samples = max(20001, int(math.ceil(duration * 1000)) + 1)
    times = np.linspace(0, duration, samples)
    if offsets:
        x = np.array([float(p["fraction"]) for p in offsets])
        y = np.array([float(p["source_offset_seconds"]) for p in offsets])
        if not (np.isfinite(x).all() and np.isfinite(y).all() and
                np.all(np.diff(x) > 0) and x[0] == 0 and x[-1] == 1):
            raise ValueError("invalid phase-warp correction points")
        shifts = np.interp(times / duration, x, y)
    else:
        shifts = np.zeros_like(times)
    source_positions = start + rate * times + shifts
    if not np.all(np.diff(source_positions) > 0):
        raise ValueError("non-monotonic source-to-render map")
    beats = _beats(beats, "source beat annotations")
    included = [v for v in beats if source_positions[0] <= v <= source_positions[-1]]
    return np.interp(included, source_positions, times).tolist()


def relative_phase_from_events(a_events, b_events, duration, *, tolerance=0.12):
    """Proxy inter-deck drift relative to initial phase; no 25ms certification.

    Matching uses a permissive nearest-event window to *detect ambiguity*.
    A high apparent error must not be interpreted as a measured violation
    unless event identities are adjudicated.
    """
    import numpy as np
    a = _beats(a_events, "rendered A beats")
    b = _beats(b_events, "rendered B beats")
    i = j = 0
    pairs = []
    while i < len(a) and j < len(b):
        delta = b[j] - a[i]
        if abs(delta) <= tolerance:
            pairs.append((a[i], delta))
            i += 1
            j += 1
        elif delta < -tolerance:
            j += 1
        else:
            i += 1
    if len(pairs) < 12:
        return {"status": "inconclusive", "reason": "fewer than 12 interdeck beat correspondences",
                "matched": len(pairs), "proxy_within_25ms": None}
    times = np.array([x[0] for x in pairs])
    raw = np.array([x[1] for x in pairs]) * 1000.0
    first = raw[times <= 2.0]
    gaps = np.diff(times)
    coverage = len(pairs) / max(len(a), len(b), 1)
    admissible = (
        len(first) >= 2 and coverage >= 0.7
        and times[0] <= 1.5 and times[-1] >= duration - 1.5
        and (len(gaps) > 0 and max(gaps) <= 1.5)
    )
    if not admissible:
        return {"status": "inconclusive", "reason": "insufficient phase coverage",
                "matched": len(pairs), "match_fraction": round(coverage, 5),
                "proxy_within_25ms": None}
    baseline = float(np.median(first))
    residual = raw - baseline
    return {
        "status": "beat_reference_proxy_measured",
        "matched": len(pairs), "match_fraction": round(coverage, 5),
        "initial_offset_ms": round(baseline, 3),
        "max_absolute_relative_drift_ms": round(float(np.max(abs(residual))), 3),
        "p95_absolute_relative_drift_ms": round(float(np.percentile(abs(residual), 95)), 3),
        "proxy_within_25ms": bool(float(np.max(abs(residual))) <= 25.0),
        "phase_samples": [
            {"render_seconds": round(float(t), 4), "relative_drift_ms": round(float(v), 3)}
            for t, v in zip(times, residual)
        ],
    }


def run(args):
    from djai.analysis import analyze_track
    from djai.planner import plan_transition
    from djai.performance.program import transition_program_from_dict
    module = load_acoustic_verifier()
    # This gate independently re-renders the signed program and checks the WAV.
    wav_validation = module.verify_private_render(
        args.manifest, args.audio_root, args.canary_report, args.rendered_wav, args.phase_samples
    )
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("m1_canary_manifest", ROOT / "scripts" / "m1-real-transition-render-canary.py")
    canary = module_from_spec(spec)
    spec.loader.exec_module(canary)
    pairs = canary.validate_manifest(json.loads(args.manifest.read_text()), args.audio_root.resolve())
    pair_id, a_path, b_path = pairs[0]
    a = analyze_track(str(a_path))
    b = analyze_track(str(b_path))
    plan = plan_transition(a, b, style="safe", tempo_policy="safe", force=False)
    program = transition_program_from_dict(dict(plan.performance_program or {}))
    phase = plan.metadata.get("deck_performance", {}).get("phase_integrity", {})
    if (plan.forced or plan.requires_confirmation or not phase.get("safe") or
            program.source_mode != "master" or
            program.annotations.get("incoming_alignment", "full") != "full"):
        raise ValueError("signed plan does not meet isolated full-alignment verification preconditions")
    original = {"A": a_path, "B": b_path}
    tracks = {"A": a, "B": b}
    start_a = float(plan.exit_time_a)
    start_b = float(plan.source_start_b)
    duration = float(plan.transition_seconds)
    end_a = start_a + duration * plan.render_rate_a
    end_b = start_b + duration * plan.render_rate_b + float(phase["points"][-1]["source_offset_seconds"])
    windows = {"A": (start_a, end_a), "B": (start_b, end_b)}
    sidecar = (external_essentia_predictions(args.essentia_predictions, original)
               if getattr(args, "essentia_predictions", None) else None)
    references = {}
    for deck in ("A", "B"):
        ess = sidecar[deck] if sidecar is not None else essentia_beats(original[deck])
        neural = neural_candidate(tracks[deck])
        if ess.get("available") and neural.get("available"):
            agreement = consensus(
                ess["beats_seconds"], neural["beats_seconds"], *windows[deck]
            )
        else:
            agreement = {"status": "inconclusive", "reason": "one or both independent candidate trackers unavailable"}
        references[deck] = {
            "essentia": {k: v for k, v in ess.items() if k != "beats_seconds"},
            "beat_this": {k: v for k, v in neural.items() if k != "beats_seconds"},
            "agreement": agreement,
        }
    if all(references[d]["agreement"].get("status") == "reference_agreement" for d in ("A", "B")):
        events_a = map_source_beats_to_render(
            references["A"]["agreement"]["matched_essentia_source_seconds"],
            start_a, duration, plan.render_rate_a,
        )
        events_b = map_source_beats_to_render(
            references["B"]["agreement"]["matched_essentia_source_seconds"],
            start_b, duration, plan.render_rate_b, offsets=phase["points"],
        )
        comparison = relative_phase_from_events(events_a, events_b, duration)
    else:
        comparison = {
            "status": "inconclusive", "reason": "insufficient independent source-beat reference agreement",
            "proxy_within_25ms": None,
        }
    return {
        "schema_version": "saund-m1-independent-beat-phase-research-v1",
        "status": comparison["status"],
        "qualification": (
            "Research-only: model agreement is not human-verified beat truth. Source beat "
            "events mapped via *the intended signed warp* do not independently measure "
            "post-render event displacement. Additional verified annotations and isolated "
            "rendered-deck event measurements are needed for 25ms certification."
        ),
        "private_wav_reproduced": wav_validation["rendered_wav_reproduced"],
        "private_wav_max_sample_difference": wav_validation["rendered_wav_max_sample_difference"],
        "pair": pair_id, "phase_samples": args.phase_samples,
        "plan": {
            "exit_time_a": start_a, "entry_time_b": plan.entry_time_b,
            "duration_seconds": duration, "technique": plan.technique,
        },
        "source_windows": windows,
        "references": references,
        "phase_comparison": comparison,
        "planner_predicted_residual_ms": phase["max_residual_ms"],
        "independent_beat_annotations_present": False,
        "independent_rendered_phase_certified": False,
        "production_changed": False, "model_promoted": False,
        "rights_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--canary-report", required=True, type=Path)
    parser.add_argument("--rendered-wav", required=True, type=Path)
    parser.add_argument("--phase-samples", type=int, choices=(9, 17), default=17)
    parser.add_argument("--essentia-predictions", type=Path,
                        help="SHA-bound JSON from isolated Essentia NumPy 1.x sidecar")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-render-rights", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_render_rights:
        parser.error("explicit --confirm-render-rights required")
    if os.environ.get("DJAI_TEMPO_CONSISTENCY") != "1":
        parser.error("DJAI_TEMPO_CONSISTENCY=1 is required for M1 plan reproduction")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; refusing overwrite")
    result = run(args)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({
        "status": result["status"],
        "private_wav_reproduced": result["private_wav_reproduced"],
        "essentia_beat_this_agreement": {
            deck: result["references"][deck]["agreement"]["status"]
            for deck in ("A", "B")
        },
        "phase_comparison": {
            key: value for key, value in result["phase_comparison"].items()
            if key != "phase_samples"
        },
        "independent_rendered_phase_certified": False,
        "report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
