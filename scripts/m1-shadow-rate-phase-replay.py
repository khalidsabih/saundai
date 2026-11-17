#!/usr/bin/env python3
"""Research-only bounded phase replay at shadow tempo rate on synthetic demo.

Reuses existing read-only counterfactual phase_profile. Never changes a plan,
production beat grid, reconciliation threshold, or the phase envelope. A
numerical result is NOT a feasibility gate or render authorization.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

DEMO_NAMES = ("01_opening_122bpm_Aminor.wav", "02_groove_124bpm_Eminor.wav")


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def compare(trace, assessment, tracks, profile):
    """Profile at existing versus hypothetical rates on unchanged beat grids.

    The recorded overlap duration and cues stay constant. Therefore this
    counterfactual deliberately does not establish phrase/cue/source bounds
    feasibility and may differ from a freshly planned transition.
    """
    comparison = assessment.get("comparison") or {}
    proposed = comparison.get("hypothetical_incoming_source_rate_to_match_outgoing")
    if (
        not assessment.get("shadow_only")
        or comparison.get("rate_is_research_counterfactual_only") is not True
        or not _numeric(proposed) or float(proposed) <= 0
    ):
        raise ValueError("requires an explicitly research-only shadow rate")
    proposed = float(proposed)
    outcomes = []
    skipped = {}
    for row in trace:
        a = tracks.get(row.get("track_a"))
        b = tracks.get(row.get("track_b"))
        if a is None or b is None:
            skipped["unknown_track"] = skipped.get("unknown_track", 0) + 1
            continue
        fields = ("exit_time_a_seconds", "entry_time_b_seconds", "transition_seconds",
                  "render_rate_a", "render_rate_b", "phase_nudge_seconds_b")
        if any(not _numeric(row.get(key)) for key in fields):
            skipped["incomplete_trace"] = skipped.get("incomplete_trace", 0) + 1
            continue
        ea, eb, duration, ra, rb, nudge = [float(row[key]) for key in fields]
        if min(duration, ra, rb) <= 0:
            skipped["invalid_trace"] = skipped.get("invalid_trace", 0) + 1
            continue
        # This standalone demo's baseline target is scalar outgoing tempo.
        # Hold the original target fixed to isolate B's hypothetical rate;
        # changing the target would require fresh planning and duration logic.
        target = a["bpm"] * ra
        params = dict(
            exit_a=ea, entry_b=eb, duration=duration, bpm=target,
            rate_a=ra, nudge_b=nudge, samples=len(row.get("points") or []) or 9,
        )
        baseline = profile(a["beats"], b["beats"], rate_b=rb, **params)
        hypothetical = profile(a["beats"], b["beats"], rate_b=proposed, **params)
        base_residual = baseline.get("max_bounded_residual_ms") if baseline.get("status") == "replayed" else None
        hyp_residual = hypothetical.get("max_bounded_residual_ms") if hypothetical.get("status") == "replayed" else None
        outcomes.append({
            "exit_time_a_seconds": ea,
            "entry_time_b_seconds": eb,
            "duration_seconds": duration,
            "baseline_rate_b": rb,
            "hypothetical_rate_b": proposed,
            "baseline_profile_status": baseline.get("status"),
            "hypothetical_profile_status": hypothetical.get("status"),
            "baseline_numeric_residual_ms": base_residual,
            "hypothetical_numeric_residual_ms": hyp_residual,
            "numeric_reduction_ms": round(base_residual - hyp_residual, 3)
            if base_residual is not None and hyp_residual is not None else None,
            "hypothetical_below_existing_25ms_numeric_limit": (
                hypothetical.get("numeric_residual_below_25ms") is True
                if hyp_residual is not None else None
            ),
        })
    paired = [x for x in outcomes if x["numeric_reduction_ms"] is not None]
    def med(values):
        return round(median(values), 3) if values else None
    return {
        "schema": "saund-m1-shadow-rate-phase-replay-v1",
        "trace_rows": len(trace),
        "evaluated_profiles": len(outcomes),
        "paired_numeric_profiles": len(paired),
        "skipped": skipped,
        "hypothetical_rate_b": proposed,
        "median_original_numeric_residual_ms": med([x["baseline_numeric_residual_ms"] for x in paired]),
        "median_hypothetical_numeric_residual_ms": med([x["hypothetical_numeric_residual_ms"] for x in paired]),
        "median_numeric_reduction_ms": med([x["numeric_reduction_ms"] for x in paired]),
        "hypothetical_numeric_below_25ms_count": sum(x["hypothetical_below_existing_25ms_numeric_limit"] is True for x in paired),
        "examples": outcomes[:8],
        "counterfactual_only": True,
        "original_grid_unchanged": True,
        "new_plan_created": False,
        "transition_rendered": False,
        "production_changed": False,
        "notes": [
            "Hypothetical numerical phase pass does NOT establish phrase, source bounds, downbeat, quality or render feasibility.",
            "Rate comes from the shadow assessment, not from an authorized tempo reconciliation.",
            "Existing sample count, cues, duration and target BPM are retained to isolate only the rate effect.",
            "Do not set any planning option or edit a signed TransitionPlan based on this report.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite existing output")
    if os.environ.get("DJAI_BEAT_GAP_AWARE", "0").strip().lower() not in {"0", "false", "off", "no", ""}:
        raise ValueError("the original trace used default sequential beat indices; disable DJAI_BEAT_GAP_AWARE")
    os.environ["DJAI_MODEL_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_BEAT_THIS_ALLOW_DOWNLOAD"] = "0"
    os.environ["DJAI_CUE_MODEL_ENABLED"] = "0"
    os.environ["DJAI_EMBEDDINGS_ENABLED"] = "0"
    from djai.analysis import analyze_track
    from scripts.counterfactual_crossdeck_phase import phase_profile

    tracks = {}
    for name in DEMO_NAMES:
        path = args.audio_root.expanduser().resolve() / name
        if not path.is_file():
            raise ValueError(f"missing generated demo fixture: {name}")
        item = analyze_track(path)
        tracks[name] = {"beats": item.beats_seconds, "bpm": item.bpm}
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("trace must contain JSON objects")
    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    report = compare(rows, assessment, tracks, phase_profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"M1 shadow rate replay failed: {exc}") from exc
