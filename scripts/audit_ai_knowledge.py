#!/usr/bin/env python3
"""Read-only inventory of SAUND's AI knowledge and decision authority.

Uses the saved Phase 16 set plan, never renders or replans. It distinguishes
registered, installed, loadable, and influential learned models, and records
the source features and selected transition decisions. The saved plan exposes
only a 16-beat preview; this tool never treats it as a full timing grid.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _number(value: Any) -> float | None:
    try:
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _track(row: dict[str, Any]) -> dict[str, Any]:
    a = dict(row.get("analysis") or {})
    if not a:
        return {"title": row.get("title"), "analysis_saved": False,
                "limitation": "No serialized TrackAnalysis in set plan"}
    keys = ("bpm", "bpm_confidence", "beat_grid_confidence", "beat_interval_cv",
            "downbeat_confidence", "structure_confidence", "perception_confidence",
            "time_signature", "time_signature_confidence", "variable_tempo",
            "energy", "vocal_probability", "bass_activity", "percussive_ratio",
            "loudness_dbfs", "integrated_lufs", "true_peak_dbtp",
            "embedding_model", "embedding_backend", "cue_model")
    facts = {name: a.get(name) for name in keys}
    facts.update({
        "title": row.get("title"),
        "analysis_saved": True,
        "key": a.get("key"),
        "genre": a.get("genre"),
        "beat_count_recorded": a.get("beat_count"),
        "beats_serialized": len(a.get("beats_seconds") or []),
        "downbeat_count_recorded": a.get("downbeat_count"),
        "tempo_curve_points": len(a.get("tempo_curve") or []),
        "entry_candidates": len(a.get("entry_candidates") or []),
        "exit_candidates": len(a.get("exit_candidates") or []),
        "section_embedding_count": len(a.get("section_embeddings") or {}),
        "music_embedding_dimensions": len(a.get("music_embedding") or []),
        "neural_evidence": a.get("neural_evidence") or {},
        "stem_analysis": a.get("stem_analysis") or {},
        "performance_layer_analysis": a.get("performance_layer_analysis") or {},
        "cue_model_predictions_count": len(a.get("cue_model_predictions") or []),
        "timing_grid_complete": bool(a.get("beat_count") is not None and len(a.get("beats_seconds") or []) >= int(a.get("beat_count") or 0)),
        "limitation": "Saved plan contains only a beat preview; cannot independently validate full phase drift.",
    })
    return facts


def _decision(edge: dict[str, Any], tracks: list[dict[str, Any]]) -> dict[str, Any]:
    p = dict(edge.get("plan") or {})
    meta = dict(p.get("metadata") or {})
    a = int(edge.get("from_index") or 0)
    b = int(edge.get("to_index") or 0)
    name = lambda i: str(tracks[i].get("title")) if 0 <= i < len(tracks) else "unknown"
    keys = ("technique", "transition_mode", "transition_bars", "beatmatched",
            "exit_time_a", "entry_time_b", "target_bpm", "render_rate_a",
            "render_rate_b", "cue_confidence", "score", "measured_score",
            "quality_checked", "measurement_passed", "fallback_level",
            "fallback_reason")
    return {
        "from": name(a),
        "to": name(b),
        "selected": {k: p.get(k) for k in keys},
        "professional_mix": meta.get("professional_mix"),
        "professional_mix_contract": meta.get("professional_mix_contract"),
        "neural_routing": meta.get("neural_policy"),
        "topk_selection": meta.get("topk_selection"),
        "ranker_authority": meta.get("transition_ranker_authority"),
        "unified_resolution": meta.get("unified_transition_resolution"),
        "warnings": edge.get("warnings") or [],
        "limitation": "Saved winning plan does not preserve every rejected candidate or complete model activation trace.",
    }


def summarize_saved_knowledge(plan: dict[str, Any]) -> dict[str, Any]:
    tracks = list(plan.get("tracks") or [])
    edges = list(plan.get("transitions") or [])
    return {
        "schema": "saund-ai-knowledge-audit-v1",
        "plan_id": plan.get("plan_id"),
        "saved_tracks": len(tracks),
        "saved_transitions": len(edges),
        "track_knowledge": [_track(t) for t in tracks],
        "chosen_decisions": [_decision(e, tracks) for e in edges],
        "recorded_set_evidence": {
            "full_set_benchmark": (plan.get("metadata") or {}).get("full_set_benchmark"),
            "stem_enriched_tracks": (plan.get("metadata") or {}).get("stem_enriched_tracks"),
            "performance_layer_enriched_tracks": (plan.get("metadata") or {}).get("performance_layer_enriched_tracks"),
            "large_crate_planning": (plan.get("metadata") or {}).get("large_crate_planning"),
        },
    }


def inspect_models() -> dict[str, Any]:
    """Inspect actual current local runtime; do not load/download heavyweight models."""
    from djai.intelligence.model_registry import registry_status
    from djai.intelligence.neural_policy import _load
    from djai.intelligence.transition_ranker import load_transition_ranker
    models: list[dict[str, Any]] = []
    for row in registry_status()["models"]:
        ident = str(row["id"])
        item = {
            "id": ident,
            "configured_authority": row.get("authority"),
            "effective_weight": row.get("effective_weight"),
            "checkpoint_exists": row.get("checkpoint_exists"),
            "checkpoint_verified": row.get("checkpoint_verified"),
            "training_data_provenance_claim": row.get("training_data_provenance"),
            "production_approved": row.get("production_approved"),
            "minimum_confidence": row.get("minimum_confidence"),
        }
        if ident in {
            "transition_family_policy", "duration_policy", "stem_choreography_policy",
            "rendered_dj_judge", "next_track_policy",
        }:
            payload, _, reason = _load(ident)
            item.update({
                "checkpoint_loadable": payload is not None,
                "load_reason": reason or ("loaded" if payload is not None else "unknown"),
                "checkpoint_feature_count": len(payload.get("feature_names") or []) if payload else None,
                "checkpoint_feature_names": payload.get("feature_names") if payload else None,
                "checkpoint_training_metrics": payload.get("metrics") if payload else None,
            })
        else:
            item.update({
                "checkpoint_loadable": None,
                "load_reason": "Adapter model: runtime invocation and local dependency availability not tested",
            })
        models.append(item)
    ranker = load_transition_ranker()
    return {
        "registry_models": models,
        "separate_transition_ranker": {
            "checkpoint_loadable": ranker is not None,
            "authority": _number(ranker.get("authority")) if ranker else 0.0,
            "feature_names": ranker.get("feature_names") if ranker else None,
            "metrics": ranker.get("metrics") if ranker else None,
        },
        "caution": "Registry authority describes permission, not proof that a model influenced the saved plan. Checkpoint provenance fields are claims until independently verified.",
    }


def build_report(plan: dict[str, Any], model_status: dict[str, Any]) -> dict[str, Any]:
    result = summarize_saved_knowledge(plan)
    result["current_local_model_inventory"] = model_status
    result["known_gaps"] = [
        "Full source beat arrays are not saved in set-plan.json; use original track cache or fresh audio diagnostics for phase validation.",
        "Registered models and shadow predictions are not proof of learned artistic judgment.",
        "The report describes current local model inventory; it may differ from the inventory when the set was planned.",
        "Only committed transitions are present; rejected candidates and ranking history require separate logs.",
        "High rendered safety scores do not establish musical taste or creative quality.",
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-plan", type=Path, default=Path("output/milestone1-four-track/balanced/set-plan.json"))
    parser.add_argument("--output", type=Path, default=Path("output/milestone1-four-track/ai-knowledge-report.json"))
    args = parser.parse_args(argv)
    plan_path = args.set_plan.expanduser().resolve()
    if not plan_path.is_file():
        parser.error(f"Set plan not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = build_report(plan, inspect_models())
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(destination),
        "tracks": report["saved_tracks"],
        "transitions": report["saved_transitions"],
        "models_with_nonzero_configured_weight": [
            m["id"] for m in report["current_local_model_inventory"]["registry_models"]
            if float(m.get("effective_weight") or 0) > 0
        ],
        "learned_dj_heads_loadable": [
            m["id"] for m in report["current_local_model_inventory"]["registry_models"]
            if m.get("checkpoint_loadable") is True
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
