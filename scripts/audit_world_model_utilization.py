#!/usr/bin/env python3
"""Read-only trace of cached musical knowledge versus saved DJ decisions.

This script does NOT rerun analysis, replan, modify the model registry, or
assert that a saved transition used a particular phrase. It joins cached
world-model records to the four-track plan by source filename and reports
uncertainty when the join is ambiguous. Selected cue proximity is descriptive
rather than proof of causal model use.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return round(x, 6) if math.isfinite(x) else None


def load_world_records(root: Path) -> tuple[dict[str, list[dict[str, Any]]], int]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors = 0
    for path in sorted(root.glob("*/world_model/world-model__6.0.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            payload = data.get("payload")
            if not isinstance(payload, dict) or not payload.get("path"):
                errors += 1
                continue
            name = Path(str(payload["path"])).name
            records[name].append({
                "cache_path": str(path),
                "source_path": str(payload["path"]),
                "fingerprint": data.get("fingerprint"),
                "version": data.get("version"),
                "created_at_utc": data.get("created_at_utc"),
                "payload": payload,
            })
        except (OSError, ValueError, TypeError):
            errors += 1
    return records, errors


def _phrase_at(phrases: list[dict[str, Any]], timestamp: Any) -> dict[str, Any] | None:
    time = _safe_float(timestamp)
    if time is None:
        return None
    for i, phrase in enumerate(phrases):
        start = _safe_float(phrase.get("start_seconds"))
        end = _safe_float(phrase.get("end_seconds"))
        if start is not None and end is not None and start <= time < end:
            return {
                "index": i,
                "start_seconds": start,
                "end_seconds": end,
                "role": phrase.get("role"),
                "section": phrase.get("section"),
                "readiness": _safe_float(phrase.get("transition_readiness")),
                "beat_confidence": _safe_float(phrase.get("beat_confidence")),
                "boundary_strength": _safe_float(phrase.get("boundary_strength")),
                "phrase_strength": _safe_float(phrase.get("phrase_strength")),
                "vocal_probability": _safe_float(phrase.get("vocal_probability")),
                "bass_activity": _safe_float(phrase.get("bass_activity")),
                "energy": _safe_float(phrase.get("energy")),
                "tension": _safe_float(phrase.get("tension")),
            }
    return None


def _phrase_brief(phrase: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "index": idx,
        "role": phrase.get("role"),
        "section": phrase.get("section"),
        "start_seconds": _safe_float(phrase.get("start_seconds")),
        "end_seconds": _safe_float(phrase.get("end_seconds")),
        "transition_readiness": _safe_float(phrase.get("transition_readiness")),
        "beat_confidence": _safe_float(phrase.get("beat_confidence")),
        "vocal_probability": _safe_float(phrase.get("vocal_probability")),
        "bass_activity": _safe_float(phrase.get("bass_activity")),
    }


def _track_summary(row: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    name = Path(str(row.get("path") or row.get("title") or "")).name
    item: dict[str, Any] = {
        "track": name,
        "matching_cache_records": len(records),
        "join_status": "missing" if not records else "ambiguous" if len(records) > 1 else "unique_filename_match",
    }
    if not records:
        return item
    # Never silently choose one model when multiple cached versions exist.
    selected = records[0] if len(records) == 1 else None
    item["matches"] = [{
        "cache_path": x["cache_path"],
        "source_path": x["source_path"],
        "fingerprint": x["fingerprint"],
        "created_at_utc": x["created_at_utc"],
    } for x in records]
    if selected is None:
        item["warning"] = "Multiple caches share this basename; cue-to-phrase mapping withheld."
        return item
    payload = selected["payload"]
    phrases = list(payload.get("phrases") or [])
    item.update({
        "world_model_version": payload.get("version"),
        "bpm_world": _safe_float(payload.get("bpm")),
        "bpm_plan": _safe_float((row.get("analysis") or {}).get("bpm") or row.get("bpm")),
        "phrase_count": len(phrases),
        "phrase_roles": dict(Counter(str(x.get("role", "unknown")) for x in phrases)),
        "readiness_ge_062": sum((_safe_float(x.get("transition_readiness")) or 0) >= 0.62 for x in phrases),
        "top_ready_phrases": [
            _phrase_brief(phrase, i)
            for i, phrase in sorted(
                enumerate(phrases),
                key=lambda x: -(_safe_float(x[1].get("transition_readiness")) or 0),
            )[:5]
        ],
        "neural_beat_used": bool((payload.get("neural_evidence") or {}).get("beat_downbeat", {}).get("used")),
        "neural_encoder_available": bool((payload.get("neural_evidence") or {}).get("music_encoder", {}).get("available")),
        "world_has_stem_activity": isinstance(payload.get("stem_activity"), dict),
        "world_stem_activity_is_separated_sources": None,
        "phrase_graph_links": len(payload.get("phrase_graph") or []),
    })
    # A stem_activity dictionary can contain master-based fallbacks; do not
    # misrepresent it as evidence that actual separated stems were available.
    item["warning"] = "Filename match alone does not prove cached analysis matches saved plan's audio fingerprint."
    return item


def analyze_plan(
    plan: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    *,
    cache_errors: int = 0,
) -> dict[str, Any]:
    tracks = list(plan.get("tracks") or [])
    summaries: list[dict[str, Any]] = []
    record_for_index: dict[int, dict[str, Any] | None] = {}
    for i, track in enumerate(tracks):
        name = Path(str(track.get("path") or track.get("title") or "")).name
        matches = records.get(name, [])
        summaries.append(_track_summary(track, matches))
        record_for_index[i] = matches[0] if len(matches) == 1 else None

    decisions: list[dict[str, Any]] = []
    for index, edge in enumerate(plan.get("transitions") or []):
        a = int(edge.get("from_index") or 0)
        b = int(edge.get("to_index") or 0)
        p = dict(edge.get("plan") or {})
        meta = dict(p.get("metadata") or {})
        a_record, b_record = record_for_index.get(a), record_for_index.get(b)
        a_phrases = list(a_record["payload"].get("phrases") or []) if a_record else []
        b_phrases = list(b_record["payload"].get("phrases") or []) if b_record else []
        exit_phrase = _phrase_at(a_phrases, p.get("exit_time_a")) if a_record else None
        entry_phrase = _phrase_at(b_phrases, p.get("entry_time_b")) if b_record else None
        decisions.append({
            "transition_number": index + 1,
            "from": summaries[a]["track"] if 0 <= a < len(summaries) else None,
            "to": summaries[b]["track"] if 0 <= b < len(summaries) else None,
            "technique": p.get("technique"),
            "beatmatched": p.get("beatmatched"),
            "fallback_level": p.get("fallback_level"),
            "fallback_reason": p.get("fallback_reason"),
            "selected_exit_time": _safe_float(p.get("exit_time_a")),
            "selected_entry_time": _safe_float(p.get("entry_time_b")),
            "cached_exit_phrase": exit_phrase,
            "cached_entry_phrase": entry_phrase,
            "cache_pair_resolved": bool(a_record and b_record),
            "selected_metadata_keys": sorted(meta.keys()),
            "transition_router_recorded": meta.get("transition_router"),
            "learned_ranker_authority_recorded": meta.get("transition_ranker_authority"),
            "selection_utility_recorded": meta.get("selection_utility"),
            "source_of_cue_choice_proven": False,
            "reason": "Saved winner and cached phrases cannot establish which phrase signals caused candidate selection.",
        })

    return {
        "schema": "saund-world-model-utilization-audit-v1",
        "plan_id": plan.get("plan_id"),
        "cache_files_read": sum(len(x) for x in records.values()),
        "cache_read_errors": cache_errors,
        "track_knowledge": summaries,
        "decisions": decisions,
        "architecture_findings": [
            {
                "type": "canonical_timeline",
                "finding": "World-model phrases are projected from build_musical_timeline(track), not a separately trained world-model network.",
                "source": "src/djai/perception/world_model.py:build_track_world_model",
            },
            {
                "type": "readiness_formula",
                "finding": "Transition readiness is a fixed weighted combination of boundary strength, phrase strength, and beat confidence (0.38/0.34/0.28).",
                "source": "src/djai/music/timeline.py:_build_phrases",
            },
            {
                "type": "planning_consumption",
                "finding": "Sequence compilation consumes the canonical timeline for song-form protection and exit cue selection. It does not load a world-model JSON as a separate inference input.",
                "source": "src/djai/set_planner.py:_compile_sequence_edges",
            },
            {
                "type": "learned_family",
                "finding": "Transition-family neural override is bounded and authority-gated; a shadow model with zero authority cannot replace heuristic family.",
                "source": "src/djai/brain/transition_router.py:route_transition",
            },
            {
                "type": "beat_integrity",
                "finding": "The final phase-lock safety gate uses beat timestamps, beat-grid confidence and render timing; cached world-model phrase readiness cannot override it.",
                "source": "src/djai/performance/deck.py:_phase_integrity_profile",
            },
        ],
        "unresolved": [
            "Accuracy of detected phrases/downbeats against reference annotations.",
            "Whether source cached features correspond to the same bytes/version as the saved plan.",
            "Which candidate phrase scores influenced each selected cue; no full candidate decision trace is preserved.",
            "Whether beat-grid rejection reflects detector errors, source tempo drift, cue alignment, or execution math.",
            "Whether a preferred musical technique sounds good to expert listeners.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-four-track/balanced/set-plan.json"))
    parser.add_argument("--world-cache", type=Path, default=Path(".djai-cache/features"))
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-four-track/world-model-utilization-report.json"))
    args = parser.parse_args(argv)
    plan = json.loads(args.set_plan.expanduser().read_text(encoding="utf-8"))
    records, errors = load_world_records(args.world_cache.expanduser())
    result = analyze_plan(plan, records, cache_errors=errors)
    destination = args.output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(destination.resolve()),
        "track_matches": {row["track"]: row["join_status"] for row in result["track_knowledge"]},
        "transitions_with_resolved_world_phrase_context": sum(d["cache_pair_resolved"] for d in result["decisions"]),
        "decisions": len(result["decisions"]),
        "notes": "Read-only observational audit; not proof of causal model influence.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
