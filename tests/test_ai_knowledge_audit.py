from __future__ import annotations

from scripts.audit_ai_knowledge import build_report, summarize_saved_knowledge


def test_saved_plan_knowledge_distinguishes_preview_from_complete_beat_grid():
    plan = {
        "plan_id": "example",
        "tracks": [
            {"title": "Alpha", "analysis": {
                "bpm": 124.0,
                "beat_grid_confidence": 0.88,
                "beats_seconds": [0.0, 0.484],
                "beat_count": 650,
                "neural_evidence": {"beat_downbeat": {
                    "available": False, "used": False, "reason": "checkpoint not bootstrapped"
                }},
                "entry_candidates": [{"time_seconds": 20.0}],
                "exit_candidates": [],
            }},
            {"title": "Beta", "analysis": {}},
        ],
        "transitions": [{
            "from_index": 0, "to_index": 1,
            "plan": {
                "technique": "echo_out_drop_in",
                "beatmatched": False,
                "fallback_level": 5,
                "metadata": {"topk_selection": {"ranking_source": "deterministic"}},
            },
        }],
    }
    result = summarize_saved_knowledge(plan)
    assert result["saved_tracks"] == 2
    assert result["track_knowledge"][0]["beat_count_recorded"] == 650
    assert result["track_knowledge"][0]["timing_grid_complete"] is False
    assert result["track_knowledge"][0]["neural_evidence"]["beat_downbeat"]["used"] is False
    assert result["track_knowledge"][1]["analysis_saved"] is False
    assert result["chosen_decisions"][0]["selected"]["fallback_level"] == 5
    assert result["chosen_decisions"][0]["topk_selection"]["ranking_source"] == "deterministic"


def test_audit_does_not_claim_shadow_model_caused_transition():
    result = build_report(
        {"tracks": [], "transitions": []},
        {"registry_models": [{
            "id": "transition_family_policy",
            "configured_authority": "shadow",
            "effective_weight": 0.0,
            "checkpoint_loadable": True,
        }], "separate_transition_ranker": {"authority": 0.0}},
    )
    assert result["current_local_model_inventory"]["registry_models"][0]["effective_weight"] == 0.0
    assert any("not proof" in item for item in result["known_gaps"])
