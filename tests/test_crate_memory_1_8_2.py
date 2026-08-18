from __future__ import annotations

from pathlib import Path

from djai.brain.crate_memory import CrateMemoryStore, crate_memory_status
from djai.intelligence.preferences import append_preference
from djai.models import BarFeature, KeyEstimate, TrackAnalysis, TransitionCandidate


def _track(path: Path, *, bpm: float = 126.0) -> TrackAnalysis:
    bars = [
        BarFeature(
            bar_index=index,
            start_seconds=index * 2.0,
            end_seconds=(index + 1) * 2.0,
            section_label="groove",
            energy=0.65,
            vocal_probability=0.55 if index <= 12 else 0.10,
            boundary_strength=0.55 if index in {8, 16, 24} else 0.15,
        )
        for index in range(32)
    ]
    exits = [
        TransitionCandidate(
            kind="exit",
            time_seconds=bar * 2.0,
            bar_index=bar,
            score=0.75,
            local_energy=0.60,
            stability=0.80,
            available_bars=8,
            boundary_strength=0.70,
            vocal_probability=0.12,
        )
        for bar in (8, 16, 24)
    ]
    return TrackAnalysis(
        path=str(path),
        duration_seconds=128.0,
        sample_rate=44100,
        bpm=bpm,
        bpm_confidence=0.95,
        beats_seconds=[float(index) * 0.5 for index in range(256)],
        downbeats_seconds=[float(index) * 2.0 for index in range(64)],
        key=KeyEstimate(tonic="C", pitch_class=0, mode="minor", camelot="5A", confidence=0.9),
        loudness_dbfs=-10.0,
        energy=0.68,
        brightness=0.5,
        percussive_ratio=0.72,
        bar_features=bars,
        exit_candidates=exits,
    )


def _candidate(a: str, b: str, *, technique: str = "eq_bass_swap", exit_bar: int = 24, entry_bar: int = 1, matched_bpm: float = 127.0, stem: str = "bass_handoff", loop: bool = False) -> dict:
    operations = ["LOOP_OUTGOING"] if loop else []
    deck_operations = [{"operation": "LOOP_OUTGOING", "start_bar": exit_bar - 4}] if loop else []
    return {
        "execution_signature": f"{Path(a).stem}-{Path(b).stem}-{technique}-{exit_bar}",
        "track_a": a,
        "track_b": b,
        "dj_move": {
            "version": "1.8.2",
            "action": "wait_then_transition",
            "exit_bar": exit_bar,
            "entry_bar": entry_bar,
            "transition_bars": 16,
            "wait_bars": 8,
            "matched_bpm": matched_bpm,
            "incoming_native_bpm": 126.0,
            "technique": technique,
            "intent": "groove_handoff",
            "stem_strategy": stem,
            "structural_operations": operations,
            "deck_operations": deck_operations,
        },
    }


def test_crate_memory_learns_directional_record_and_pair_preferences(tmp_path: Path) -> None:
    store = CrateMemoryStore(tmp_path / "ratings" / "crate-memory.sqlite3")
    track_x = str(tmp_path / "audio" / "x.wav")
    track_y = str(tmp_path / "audio" / "y.wav")
    winner = _candidate(track_x, track_y, loop=True)
    loser = _candidate(track_x, str(tmp_path / "audio" / "z.wav"), technique="stem_vocal_overlay")

    for _ in range(4):
        store.record_preference(
            candidate_a=winner,
            candidate_b=loser,
            choice="a",
            reasons=["great_payoff"],
            listener="expert",
        )

    good = store.hint(track_x, track_y, winner["dj_move"])
    reverse = store.hint(track_y, track_x, winner["dj_move"])
    assert good.available is True
    assert good.adjustment > 0.0
    assert good.learned_exit_bar == 24.0
    assert good.learned_transition_bars == 16.0
    assert good.learned_loop_bar == 20.0
    assert "eq_bass_swap" in good.preferred_techniques
    assert "bass_handoff" in good.preferred_stem_strategies
    assert reverse.pair_observations == 0.0
    assert reverse.adjustment == 0.0


def test_crate_memory_remembers_known_pair_failures_conditionally(tmp_path: Path) -> None:
    store = CrateMemoryStore(tmp_path / "ratings" / "crate-memory.sqlite3")
    x = str(tmp_path / "x.wav")
    z = str(tmp_path / "z.wav")
    bad = _candidate(x, z, technique="stem_vocal_overlay", matched_bpm=131.0, stem="vocal_handoff")
    other = _candidate(x, str(tmp_path / "other.wav"))
    for _ in range(4):
        store.record_preference(
            candidate_a=bad,
            candidate_b=other,
            choice="neither",
            reasons=["vocal_clash", "tempo_felt_forced"],
            listener="expert",
        )
    hint = store.hint(x, z, bad["dj_move"])
    assert hint.available is True
    assert hint.adjustment < 0.0
    assert "vocal_clash" in hint.known_failure_reasons
    assert "tempo_felt_forced" in hint.known_failure_reasons


def test_track_structure_facts_do_not_create_preference_authority(tmp_path: Path) -> None:
    store = CrateMemoryStore(tmp_path / "ratings" / "crate-memory.sqlite3")
    a = _track(tmp_path / "a.wav")
    b = _track(tmp_path / "b.wav")
    store.observe_track_analysis(a)
    store.observe_track_analysis(b)
    hint = store.hint(a, b, _candidate(a.path, b.path)["dj_move"])
    assert hint.available is False
    assert hint.adjustment == 0.0
    assert hint.vocal_end_bar == 12
    assert set(hint.structural_loop_bars) == {8, 16, 24}


def test_standard_taste_preference_automatically_updates_crate_memory(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings"
    preferences = ratings / "transition-preferences.jsonl"
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    c = str(tmp_path / "c.wav")
    append_preference(
        candidate_a=_candidate(a, b),
        candidate_b=_candidate(a, c, technique="quick_cut"),
        choice="a",
        listener="expert",
        context={"reasons": ["great_payoff"]},
        path=preferences,
    )
    status = CrateMemoryStore(ratings / "crate-memory.sqlite3").status()
    assert status["preference_events"] == 2
    assert status["pair_records"] == 2


def test_listener_purge_removes_aggregate_influence_not_only_events(tmp_path: Path) -> None:
    store = CrateMemoryStore(tmp_path / "ratings" / "crate-memory.sqlite3")
    a = str(tmp_path / "a.wav")
    b = str(tmp_path / "b.wav")
    c = str(tmp_path / "c.wav")
    store.record_preference(
        candidate_a=_candidate(a, b),
        candidate_b=_candidate(a, c),
        choice="a",
        listener="listener-delete",
    )
    before = store.hint(a, b, _candidate(a, b)["dj_move"])
    assert before.adjustment > 0.0
    assert store.purge_listener("listener-delete") == 2
    after = store.hint(a, b, _candidate(a, b)["dj_move"])
    assert after.available is False
    assert after.adjustment == 0.0


def test_crate_memory_status_is_honest_when_store_is_absent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    status = crate_memory_status(workspace=tmp_path)
    assert status["version"] == "1.8.2"
    assert status["available"] is False
    assert status["maximum_live_adjustment"] <= 0.065


def test_complete_move_ranking_consumes_bounded_crate_memory_prior(tmp_path: Path, monkeypatch) -> None:
    from conftest import analysis
    from djai.brain.dj_move import build_dj_state
    from djai.brain.move_generator import CompleteMoveConfig, generate_complete_moves
    from djai.brain.set_state import build_set_state
    from djai.set_planner import build_transition_graph

    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_CRATE_MEMORY", "1")
    paths = [tmp_path / f"track-{index}.wav" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(f"memory-track-{index}".encode())
    tracks = [
        analysis(paths[0], bpm=124.0, camelot="8A", energy=0.50),
        analysis(paths[1], bpm=125.0, camelot="9A", energy=0.60),
        analysis(paths[2], bpm=127.0, camelot="10A", energy=0.70),
    ]
    graph = build_transition_graph(tracks, workers=1)
    set_state = build_set_state([tracks[0]])
    state = build_dj_state(tracks[0], source_cursor=0.0, set_state=set_state)
    cfg = CompleteMoveConfig(mode="move_shadow")
    baseline = generate_complete_moves(
        current_index=0,
        candidate_indices=[1, 2],
        analyses=tracks,
        graph=graph,
        dj_state=state,
        set_state=set_state,
        journey="build",
        progress=0.0,
        config=cfg,
    )
    target = next(item for item in baseline if item.next_index == 1)
    store = CrateMemoryStore(tmp_path / "ratings" / "crate-memory.sqlite3")
    payload = {
        "track_a": tracks[0].path,
        "track_b": tracks[1].path,
        "dj_move": target.move.to_dict(),
    }
    for _ in range(6):
        store.record_candidate(payload, outcome=1.0, listener="expert")

    remembered = generate_complete_moves(
        current_index=0,
        candidate_indices=[1, 2],
        analyses=tracks,
        graph=graph,
        dj_state=state,
        set_state=set_state,
        journey="build",
        progress=0.0,
        config=cfg,
    )
    same = next(item for item in remembered if item.plan.execution_signature == target.plan.execution_signature)
    assert same.crate_memory is not None
    assert same.crate_memory["available"] is True
    assert 0.0 < same.crate_memory["adjustment"] <= 0.065
    assert same.selection_score > target.selection_score
    assert same.plan.metadata["crate_memory"]["hard_safety_authoritative"] is True
