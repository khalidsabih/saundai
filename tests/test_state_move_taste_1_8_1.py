from __future__ import annotations

from pathlib import Path

from djai.brain.dj_move import DJMove, DJState
from djai.intelligence.preferences import append_preference
from djai.training.state_move_taste import (
    STATE_MOVE_FEATURES,
    move_taste_payload,
    shadow_state_move_prediction,
    state_context_id,
    state_move_features,
    state_taste_payload,
    train_state_move_taste_shadow,
)


def _state(*, bar: int, energy: float) -> DJState:
    return DJState(
        version="1.4",
        current_track=f"/private/music/current-{bar}.wav",
        source_cursor_seconds=32.0,
        source_bar=bar,
        phrase_index=2,
        phrase_role="build",
        effective_bpm=126.0,
        native_bpm=126.0,
        render_rate=1.0,
        tempo_owner="A",
        current_energy=energy,
        current_tension=0.72,
        current_vocal_probability=0.32,
        current_bass_activity=0.68,
        set_energy=energy - 0.04,
        set_tension=0.61,
        set_groove=0.78,
        set_vocal_density=0.29,
        pending_commitments=(),
    )


def _move(state: DJState, *, good: bool) -> DJMove:
    projected = DJState(
        version="1.4",
        current_track="/private/music/next.wav",
        source_cursor_seconds=24.0,
        source_bar=9,
        phrase_index=1,
        phrase_role="drop" if good else "breakdown",
        effective_bpm=127.0 if good else 131.0,
        native_bpm=127.0,
        render_rate=1.0,
        tempo_owner="B",
        current_energy=0.82 if good else 0.42,
        current_tension=0.55 if good else 0.32,
        current_vocal_probability=0.18 if good else 0.78,
        current_bass_activity=0.76 if good else 0.38,
        set_energy=0.72,
        set_tension=0.60,
        set_groove=0.80,
        set_vocal_density=0.28,
        pending_commitments=(),
    )
    return DJMove(
        version="1.4",
        action="wait_then_transition" if good else "transition",
        current_track=state.current_track,
        next_track=projected.current_track,
        exit_bar=state.source_bar + (8 if good else 2),
        entry_bar=1 if good else 17,
        exit_phrase=2,
        entry_phrase=0 if good else 3,
        transition_bars=8 if good else 24,
        wait_bars=8 if good else 2,
        matched_bpm=126.5 if good else 131.0,
        incoming_native_bpm=127.0,
        post_transition_bpm=127.0,
        tempo_release_seconds=4.0 if good else 18.0,
        phase_nudge_seconds=0.01 if good else 0.07,
        technique="breakdown_to_drop" if good else "long_vocal_blend",
        intent="breakdown_to_drop" if good else "blend",
        tempo_strategy="match_then_release",
        phase_strategy="phase_nudge",
        stem_strategy="vocal_handoff" if good else "master",
        stem_profile="vocal_handoff" if good else None,
        structural_operations=("WAIT", "STEM_HANDOFF") if good else (),
        deck_operations=(),
        protected_payoff_violation=not good,
        payoff_satisfied=good,
        landing_phrase=1,
        future_optionality=0.88 if good else 0.22,
        projected_state=projected,
        score=0.91 if good else 0.36,
        components={},
        rationale=(),
    )


def _candidate(state: DJState, move: DJMove, signature: str) -> dict:
    return {
        "execution_signature": signature,
        "track_a": "track-a",
        "track_b": "track-b",
        "pair_id": "pair-a-b",
        "dj_state": state_taste_payload(state),
        "dj_move": move_taste_payload(move),
        "state_context_id": state_context_id(state),
    }


def test_state_move_payload_is_path_free_and_feature_complete() -> None:
    state = _state(bar=33, energy=0.70)
    move = _move(state, good=True)
    state_payload = state_taste_payload(state)
    move_payload = move_taste_payload(move)
    text = str({"state": state_payload, "move": move_payload})
    assert "/private/music" not in text
    features = state_move_features(state_payload, move_payload)
    assert tuple(features) == STATE_MOVE_FEATURES
    assert features["payoff_satisfied"] == 1.0
    assert features["protected_payoff_violation"] == 0.0


def test_state_move_shadow_trains_from_blind_preferences(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    preferences = tmp_path / "preferences.jsonl"
    for index in range(16):
        state = _state(bar=17 if index % 2 == 0 else 49, energy=0.62 if index % 2 == 0 else 0.78)
        good = _candidate(state, _move(state, good=True), f"good-{index}")
        bad = _candidate(state, _move(state, good=False), f"bad-{index}")
        if index % 2 == 0:
            candidate_a, candidate_b, choice = good, bad, "a"
        else:
            candidate_a, candidate_b, choice = bad, good, "b"
        append_preference(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            choice=choice,
            listener="expert",
            context={"reasons": ["great_payoff"]},
            path=preferences,
        )

    result = train_state_move_taste_shadow(
        tmp_path,
        preferences_path=preferences,
        dataset_id="dataset-test",
        ridge=0.18,
    )
    assert result["comparison_count"] == 16
    assert result["state_group_count"] >= 2
    assert result["shadow_only"] is True
    assert result["authority"] == 0.0

    state = _state(bar=17, energy=0.62)
    good_prediction = shadow_state_move_prediction(state, _move(state, good=True), workspace=tmp_path)
    bad_prediction = shadow_state_move_prediction(state, _move(state, good=False), workspace=tmp_path)
    assert good_prediction["available"] is True
    assert good_prediction["authority"] == 0.0
    assert good_prediction["score"] > bad_prediction["score"]


def test_taste_learning_endpoint_trains_state_move_head_independently(tmp_path: Path, monkeypatch) -> None:
    import json
    from djai.training.taste_learning import train_taste_learning_heads

    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    snapshot = tmp_path / "datasets" / "training-lab" / "dataset-state-move"
    snapshot.mkdir(parents=True)
    preferences = snapshot / "preferences.jsonl"
    for index in range(16):
        state = _state(bar=17 if index % 2 == 0 else 49, energy=0.62 if index % 2 == 0 else 0.78)
        good = _candidate(state, _move(state, good=True), f"endpoint-good-{index}")
        bad = _candidate(state, _move(state, good=False), f"endpoint-bad-{index}")
        candidate_a, candidate_b, choice = (good, bad, "a") if index % 2 == 0 else (bad, good, "b")
        append_preference(
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            choice=choice,
            listener="expert",
            context={"reasons": ["great_payoff"]},
            path=preferences,
        )
    (snapshot / "rendered_judge.jsonl").write_text("", encoding="utf-8")
    (snapshot / "manifest.json").write_text(
        json.dumps({"dataset_id": snapshot.name, "counts": {"decisive_preferences": 16}}),
        encoding="utf-8",
    )

    result = train_taste_learning_heads(tmp_path, snapshot_id=snapshot.name)
    assert result["global_taste_ranker"]["status"] == "blocked"
    assert result["state_move_taste"]["status"] == "trained"
    assert result["state_move_taste"]["authority"] == 0.0
    assert result["rendered_dj_judge"]["status"] == "blocked"
