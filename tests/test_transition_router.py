from __future__ import annotations

from pathlib import Path

from conftest import analysis
from djai.brain.transition_router import apply_transition_route, route_transition
from djai.models import GenreEstimate
from djai.planner import rank_transition_candidates


def test_similar_dance_tracks_route_to_direct_blend(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.66)
    b = analysis(tmp_path / "b.wav", bpm=128.0, camelot="9A", energy=0.69)
    route = route_transition(a, b)
    assert route.family == "direct_blend"
    assert route.direct_transition_confidence > 0.70
    assert route.bridge_preference < 0.20


def test_large_bpm_and_genre_jump_prefers_bridge_track(tmp_path: Path) -> None:
    a = analysis(tmp_path / "house.wav", bpm=126.0, camelot="8A", energy=0.78)
    b = analysis(tmp_path / "hiphop.wav", bpm=94.0, camelot="10A", energy=0.62)
    b.genre = GenreEstimate(label="hip_hop", confidence=0.92, probabilities={"hip_hop": 0.92})
    b.percussive_ratio = 0.28
    b.bass_activity = 0.78
    b.brightness = 0.16
    b.onset_density = 0.9
    route = route_transition(a, b)
    assert route.family == "bridge_preferred"
    assert route.bridge_preference >= 0.68
    assert route.direct_transition_confidence < 0.50


def test_clean_layers_unlock_layer_handoff_for_distant_style(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=124.0, camelot="8A", energy=0.70)
    b = analysis(tmp_path / "b.wav", bpm=125.0, camelot="8A", energy=0.55)
    b.genre = GenreEstimate(label="hip_hop", confidence=0.90, probabilities={"hip_hop": 0.90})
    b.percussive_ratio = 0.18
    b.bass_activity = 0.85
    b.brightness = 0.08
    b.onset_density = 0.45
    b.performance_layer_analysis = {
        "available": 1.0,
        "complete": 1.0,
        "quality_confidence": 0.92,
    }
    route = route_transition(a, b)
    assert route.family == "layer_handoff"
    assert route.layer_capability >= 0.90


def test_router_metadata_is_attached_to_candidate_plans(tmp_path: Path) -> None:
    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.66)
    b = analysis(tmp_path / "b.wav", bpm=128.0, camelot="9A", energy=0.69)
    plans = rank_transition_candidates(a, b, force=True, limit_per_style=2)
    assert plans
    route = plans[0].metadata.get("transition_router")
    assert isinstance(route, dict)
    assert route["family"] == "direct_blend"
    assert 0.0 <= float(route["route_fit"]) <= 1.0


def test_bridge_preference_materially_penalizes_direct_set_edge(tmp_path: Path) -> None:
    import copy

    from djai.set_planner import _edge_objective

    a = analysis(tmp_path / "a.wav", bpm=126.0, camelot="8A", energy=0.70)
    b = analysis(tmp_path / "b.wav", bpm=126.5, camelot="8A", energy=0.70)
    plan = rank_transition_candidates(a, b, force=True, limit_per_style=1)[0]
    neutral = copy.deepcopy(plan)
    bridge = copy.deepcopy(plan)
    neutral.metadata["transition_router"] = {"route_fit": 0.5, "bridge_preference": 0.0}
    bridge.metadata["transition_router"] = {"route_fit": 0.5, "bridge_preference": 0.9}

    neutral_score = _edge_objective(neutral, a, b, journey="balanced")
    bridge_score = _edge_objective(bridge, a, b, journey="balanced")
    assert neutral_score - bridge_score >= 0.33
