from __future__ import annotations

from dataclasses import replace

from djai.brain.set_state import build_set_state
from djai.intelligence.hybrid_next_track import score_hybrid_next_track


def test_hybrid_next_track_has_deterministic_cold_target_path(track_pair) -> None:
    a, b = track_pair
    b.music_embedding = []
    b.perception_confidence = 0.45
    state = build_set_state([a])
    prediction = score_hybrid_next_track(a, b, edge_score=0.72, set_state=state, journey="build")
    assert 0.0 <= prediction.score <= 1.0
    assert prediction.deterministic_score > 0.0
    assert prediction.target_evidence_confidence > 0.0
    assert prediction.components["embedding_present"] == 0.0


def test_hybrid_next_track_hard_render_failure_is_veto(track_pair) -> None:
    a, b = track_pair
    from djai.planner import plan_transition

    plan = plan_transition(a, b, tempo_policy="safe")
    assert plan is not None
    if plan.measurement is None:
        from djai.models import TransitionMeasurement

        plan.measurement = TransitionMeasurement(version="test", sample_rate=44100, duration_seconds=1.0)
    plan.measurement.hard_failures = ["clipping"]
    prediction = score_hybrid_next_track(a, b, edge_score=0.99, plan=plan)
    assert prediction.score == 0.0
    assert prediction.safety["hard_failures"] == ["clipping"]
