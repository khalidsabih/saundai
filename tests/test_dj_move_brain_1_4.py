from __future__ import annotations

from dataclasses import replace

from djai.brain.dj_move import (
    build_dj_state,
    build_musical_timeline,
    move_brain_status,
    move_opportunity_score,
    rank_transition_plans_as_moves,
    score_transition_plan_as_move,
)
from djai.models import BarFeature
from djai.planner import plan_transition_alternatives
from djai.set_planner import _required_source_body, _sequence_exit_candidates
from djai.music.timeline import earliest_safe_song_form_exit_bar


def _bar(index: int, *, section: str, energy: float, phrase: float = 0.30, boundary: float = 0.24, vocal: float = 0.10) -> BarFeature:
    return BarFeature(
        bar_index=index,
        start_seconds=index * 2.0,
        end_seconds=(index + 1) * 2.0,
        section_label=section,  # type: ignore[arg-type]
        energy=energy,
        rhythm=0.72,
        brightness=0.5,
        percussive_activity=0.72,
        vocal_probability=vocal,
        bass_activity=0.52,
        harmonic_tension=min(1.0, 0.35 + energy * 0.45),
        boundary_strength=boundary,
        phrase_strength=phrase,
        beat_grid_confidence=0.94,
        local_bpm=120.0,
    )


def _human_phrase_track(track):
    bars = []
    for index in range(24):
        if index < 4:
            bars.append(_bar(index, section="intro", energy=0.34, phrase=0.86 if index == 0 else 0.30, boundary=0.86 if index == 0 else 0.22))
        elif index < 12:
            # Build creates an explicit promise into the drop at bar 12.
            bars.append(_bar(index, section="build", energy=0.45 + 0.035 * (index - 4), phrase=0.82 if index in {4, 8} else 0.32, boundary=0.78 if index in {4, 8} else 0.24))
        elif index < 20:
            bars.append(_bar(index, section="drop", energy=0.90, phrase=0.88 if index in {12, 16} else 0.34, boundary=0.84 if index in {12, 16} else 0.24, vocal=0.25))
        else:
            bars.append(_bar(index, section="groove", energy=0.62, phrase=0.90 if index == 20 else 0.30, boundary=0.88 if index == 20 else 0.22))
    track.bar_features = bars
    track.duration_seconds = 48.0
    return track


def test_musical_timeline_is_adaptive_and_protects_drop(track_pair):
    a, _ = track_pair
    _human_phrase_track(a)
    timeline = build_musical_timeline(a)
    lengths = [moment.end_bar - moment.start_bar for moment in timeline.moments]
    assert lengths
    assert any(length == 4 for length in lengths)
    assert len(set(lengths)) >= 2 or lengths[0] == 4
    assert any(item.kind == "pending_drop_payoff" and item.target_bar == 12 for item in timeline.commitments)


def test_sequence_exit_waits_for_protected_payoff(track_pair):
    a, _ = track_pair
    _human_phrase_track(a)
    # A tempting build exit at bar 8 must lose to the post-drop groove at 20.
    early = replace(a.exit_candidates[0], time_seconds=16.0, bar_index=8, section_label="build", score=0.99, phrase_strength=0.95, boundary_strength=0.95)
    late = replace(a.exit_candidates[1], time_seconds=40.0, bar_index=20, section_label="groove", score=0.78, phrase_strength=0.90, boundary_strength=0.90)
    a.exit_candidates = [early, late]
    state = build_dj_state(a, source_cursor=8.0, effective_bpm=120.0)
    selected, _, _ = _sequence_exit_candidates(
        a,
        source_cursor=8.0,
        current_rate=1.0,
        required_source=1.0,
        dj_state=state,
        limit=2,
    )
    assert selected[0].bar_index == 20


def test_complete_dj_move_records_wait_tempo_and_landing(track_pair):
    a, b = track_pair
    _human_phrase_track(a)
    _human_phrase_track(b)
    # Keep planner cues inside the synthetic musical timeline.
    a.exit_candidates[0] = replace(a.exit_candidates[0], time_seconds=40.0, bar_index=20, section_label="groove")
    b.entry_candidates[0] = replace(b.entry_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro")
    plan = plan_transition_alternatives(a, b, force=True, tempo_policy="safe")[0]
    state = build_dj_state(a, source_cursor=8.0, effective_bpm=a.bpm)
    ranked = rank_transition_plans_as_moves(a, b, [plan], state=state)
    move = ranked[0].metadata["dj_move"]
    assert move["action"] in {"wait_then_transition", "transition"}
    assert move["exit_bar"] == int(ranked[0].metadata["exit_bar_index"])
    assert move["entry_bar"] == int(ranked[0].metadata["entry_bar_index"])
    assert move["incoming_native_bpm"] == b.bpm
    assert 0.0 <= move["future_optionality"] <= 1.0
    assert move["projected_state"]["tempo_owner"] == "B"
    assert "components" in move


def test_move_opportunity_is_a_bounded_ordering_signal(track_pair):
    a, b = track_pair
    _human_phrase_track(a)
    _human_phrase_track(b)
    score, metadata = move_opportunity_score(a, b)
    assert 0.0 <= score <= 1.0
    assert metadata["available"] is True
    assert metadata["best_exit_bar"] >= 0
    assert metadata["best_entry_bar"] >= 0


def test_move_brain_status_describes_human_dj_contract():
    status = move_brain_status()
    assert status["enabled"] is True
    assert status["planning_unit"] == "DJState -> DJMove -> DJState"
    assert status["coordinates"] == "bar_phrase_native"
    assert status["explicit_wait_actions"] is True
    assert status["musical_commitments"] is True
    assert status["first_hook_pre_entry_protection"] is True
    assert status["song_form_hard_guard"] is True
    assert status["hard_audio_safety_authoritative"] is True


def test_set_policy_prefers_track_with_better_complete_move(track_pair, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from conftest import analysis
    from djai.brain.set_policy import SetPolicyConfig, choose_next_track

    a, b = track_pair
    c_path = Path(tmp_path) / "c.wav"
    c_path.write_bytes(b"c")
    c = analysis(c_path, bpm=126.0, camelot="9A", energy=0.62)
    for track in (a, b, c):
        _human_phrase_track(track)

    # B has a clean early groove/intro opportunity; C only exposes a late,
    # vocal-heavy outro cue. With move authority at 100%, the set policy must
    # choose the musically executable move rather than an equal track edge.
    b.entry_candidates = [replace(b.entry_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro", vocal_probability=0.05)]
    c.entry_candidates = [replace(c.entry_candidates[0], time_seconds=44.0, bar_index=22, section_label="outro", vocal_probability=0.95, score=0.25)]
    graph = {
        (0, 1): SimpleNamespace(objective_score=0.5, plan=None),
        (0, 2): SimpleNamespace(objective_score=0.5, plan=None),
    }
    selected, trace = choose_next_track(
        [0],
        [1, 2],
        graph,
        [a, b, c],
        journey="balanced",
        config=SetPolicyConfig(horizon=1, beam=2, future_discount=0.0, context_weight=0.0, hybrid_weight=0.0, move_weight=1.0),
    )
    assert selected == 1
    assert trace["move_opportunity_score"] > 0.5


def test_required_source_body_is_bar_native_not_track_fraction(track_pair, monkeypatch):
    a, _ = track_pair
    a.duration_seconds = 300.0
    a.bpm = 120.0
    a.time_signature = 4
    monkeypatch.setenv("DJAI_MOVE_BRAIN", "1")
    required = _required_source_body(
        a,
        current_cursor=0.0,
        current_rate=1.0,
        minimum_body_seconds=30.0,
        minimum_body_fraction=0.35,
    )
    # 120 BPM / 4/4 => 2 seconds per bar. 30 seconds becomes 15 bars,
    # rounded to the next four-bar boundary: 16 bars / 32 seconds. The old
    # 35% rule would have forced 105 seconds on this record.
    assert required == 32.0


def test_legacy_body_fraction_remains_available_when_move_brain_disabled(track_pair, monkeypatch):
    a, _ = track_pair
    a.duration_seconds = 300.0
    a.bpm = 120.0
    a.time_signature = 4
    monkeypatch.setenv("DJAI_MOVE_BRAIN", "0")
    required = _required_source_body(
        a,
        current_cursor=0.0,
        current_rate=1.0,
        minimum_body_seconds=30.0,
        minimum_body_fraction=0.35,
    )
    assert required == 105.0


def test_future_build_exit_is_protected_even_outside_current_short_horizon(track_pair):
    a, _ = track_pair
    bars = []
    for index in range(64):
        if 40 <= index < 48:
            bars.append(_bar(index, section="build", energy=0.58 + 0.03 * (index - 40), phrase=0.90 if index in {40, 44} else 0.34, boundary=0.84 if index in {40, 44} else 0.25))
        elif 48 <= index < 56:
            bars.append(_bar(index, section="drop", energy=0.94, phrase=0.90 if index in {48, 52} else 0.34, boundary=0.86 if index in {48, 52} else 0.25))
        else:
            bars.append(_bar(index, section="groove", energy=0.55, phrase=0.82 if index % 8 == 0 else 0.30, boundary=0.76 if index % 8 == 0 else 0.22))
    a.bar_features = bars
    a.duration_seconds = 128.0
    early_build = replace(a.exit_candidates[0], time_seconds=88.0, bar_index=44, section_label="build", score=0.99, phrase_strength=0.95, boundary_strength=0.95)
    post_drop = replace(a.exit_candidates[1], time_seconds=112.0, bar_index=56, section_label="groove", score=0.76, phrase_strength=0.84, boundary_strength=0.82)
    a.exit_candidates = [early_build, post_drop]
    # Current state is at the beginning of the track, so the build/drop promise
    # is well beyond the normal 16-bar pending-commitment horizon. The future
    # exit must still understand its own local musical promise.
    state = build_dj_state(a, source_cursor=0.0, effective_bpm=120.0)
    selected, _, _ = _sequence_exit_candidates(
        a, source_cursor=0.0, current_rate=1.0, required_source=1.0, dj_state=state, limit=2
    )
    assert selected[0].bar_index == 56


def test_sequence_exit_does_not_mix_out_when_first_verse_begins(track_pair):
    a, _ = track_pair
    bars = []
    for index in range(20):
        strong = index in {0, 4, 8, 12, 16}
        if index < 4:
            section, energy, vocal = "intro", 0.28, 0.05
        elif index < 8:
            # This intentionally mimics a coarse cached analysis that still
            # calls the first verse "intro".
            section, energy, vocal = "intro", 0.56, 0.66
        elif index < 12:
            section, energy, vocal = "groove", 0.78, 0.72
        elif index < 16:
            section, energy, vocal = "groove", 0.58, 0.14
        else:
            section, energy, vocal = "outro", 0.40, 0.08
        bars.append(
            _bar(
                index,
                section=section,
                energy=energy,
                phrase=0.90 if strong else 0.28,
                boundary=0.88 if strong else 0.20,
                vocal=vocal,
            )
        )
    a.bar_features = bars
    a.duration_seconds = 40.0
    early = replace(
        a.exit_candidates[0],
        time_seconds=8.0,
        bar_index=4,
        section_label="intro",
        score=0.99,
        phrase_strength=0.95,
        boundary_strength=0.95,
        vocal_probability=0.66,
    )
    later = replace(
        a.exit_candidates[1],
        time_seconds=24.0,
        bar_index=12,
        section_label="groove",
        score=0.78,
        phrase_strength=0.88,
        boundary_strength=0.88,
        vocal_probability=0.14,
    )
    a.exit_candidates = [early, later]
    state = build_dj_state(a, source_cursor=0.0, effective_bpm=120.0)
    selected, _, _ = _sequence_exit_candidates(
        a,
        source_cursor=0.0,
        current_rate=1.0,
        required_source=1.0,
        dj_state=state,
        limit=2,
    )
    assert selected[0].bar_index == 12


def _song_form_track(track):
    bars = []
    for index in range(24):
        strong = index in {0, 4, 8, 12, 16, 20}
        if index < 4:
            section, energy, vocal = "intro", 0.28, 0.04
        elif index < 8:
            # Coarse analyzer still says intro, but vocals clearly establish Verse 1.
            section, energy, vocal = "intro", 0.56, 0.66
        elif index < 12:
            section, energy, vocal = "groove", 0.90, 0.88
        elif index < 16:
            section, energy, vocal = "groove", 0.58, 0.62
        elif index < 20:
            section, energy, vocal = "groove", 0.80, 0.76
        else:
            section, energy, vocal = "outro", 0.38, 0.08
        bars.append(
            _bar(
                index,
                section=section,
                energy=energy,
                phrase=0.94 if strong else 0.28,
                boundary=0.92 if strong else 0.20,
                vocal=vocal,
            )
        )
    track.bar_features = bars
    track.duration_seconds = 48.0
    return track


def test_song_form_guard_requires_first_hook_to_complete_before_exit(track_pair):
    a, _ = track_pair
    _song_form_track(a)
    timeline = build_musical_timeline(a)
    roles = [(p.song_role, p.song_role_index, p.start_bar, p.end_bar) for p in timeline.phrases]
    assert roles[:4] == [
        ("intro", 1, 0, 4),
        ("verse", 1, 4, 8),
        ("hook", 1, 8, 12),
        ("verse", 2, 12, 16),
    ]
    hook = next(item for item in timeline.commitments if item.kind == "protected_hook")
    assert timeline.phrases[hook.source_phrase].song_role == "verse"
    assert hook.target_bar == 8
    assert hook.protect_until_bar == 12
    assert earliest_safe_song_form_exit_bar(timeline, 0) == 12

    verse_start = replace(
        a.exit_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro",
        score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.66,
    )
    hook_start = replace(
        a.exit_candidates[1], time_seconds=16.0, bar_index=8, section_label="groove",
        score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.88,
    )
    verse_two = replace(
        a.exit_candidates[0], time_seconds=24.0, bar_index=12, section_label="groove",
        score=0.62, phrase_strength=0.78, boundary_strength=0.76, vocal_probability=0.62,
    )
    a.exit_candidates = [verse_start, hook_start, verse_two]
    state = build_dj_state(a, source_cursor=0.0, effective_bpm=120.0, timeline=timeline)
    selected, _, _ = _sequence_exit_candidates(
        a, source_cursor=0.0, current_rate=1.0, required_source=1.0,
        dj_state=state, timeline=timeline, limit=3,
    )
    assert [item.bar_index for item in selected] == [12]


def test_song_form_guard_cannot_be_bypassed_by_dramatic_switch(track_pair):
    a, b = track_pair
    _song_form_track(a)
    _human_phrase_track(b)
    early = replace(
        a.exit_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro",
        score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.66,
    )
    a.exit_candidates = [early]
    plans = plan_transition_alternatives(a, b, force=True, exit_time_a=8.0, tempo_policy="safe")
    assert plans
    plan = plans[0]
    plan.intent = "dramatic_switch"
    plan.metadata = {**plan.metadata, "exit_bar_index": 4}
    state = build_dj_state(a, source_cursor=0.0, effective_bpm=a.bpm)
    move = score_transition_plan_as_move(a, b, plan, state=state)
    assert move.protected_payoff_violation is True
    assert move.protected_commitment_kind in {"first_body_payoff", "protected_hook"}
    assert move.components["song_form_hard_guard"] == 1.0
    ranked = rank_transition_plans_as_moves(a, b, [plan], state=state)
    assert ranked[0].metadata["selection_utility"] == 0.0
    assert ranked[0].metadata["dj_move_brain"]["song_form_hard_guard"] is True


def test_song_form_guard_returns_no_early_fallback_candidate(track_pair):
    a, _ = track_pair
    _song_form_track(a)
    timeline = build_musical_timeline(a)
    a.exit_candidates = [
        replace(
            a.exit_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro",
            score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.66,
        ),
        replace(
            a.exit_candidates[1], time_seconds=16.0, bar_index=8, section_label="groove",
            score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.88,
        ),
    ]
    state = build_dj_state(a, source_cursor=0.0, effective_bpm=120.0, timeline=timeline)
    selected, _, _ = _sequence_exit_candidates(
        a, source_cursor=0.0, current_rate=1.0, required_source=1.0,
        dj_state=state, timeline=timeline, limit=3,
    )
    assert selected == []



def test_compiled_automatic_edge_cannot_start_inside_first_song_form_promise(track_pair):
    from djai import set_planner

    a, b = track_pair
    _song_form_track(a)
    _song_form_track(b)
    a.exit_candidates = [
        replace(
            a.exit_candidates[0], time_seconds=8.0, bar_index=4, section_label="intro",
            score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.66,
        ),
        replace(
            a.exit_candidates[1], time_seconds=16.0, bar_index=8, section_label="groove",
            score=0.999, phrase_strength=0.99, boundary_strength=0.99, vocal_probability=0.88,
        ),
        replace(
            a.exit_candidates[0], time_seconds=24.0, bar_index=12, section_label="groove",
            score=0.62, phrase_strength=0.78, boundary_strength=0.76, vocal_probability=0.62,
        ),
    ]
    compiled = set_planner._compile_sequence_edges(
        [0, 1],
        [a, b],
        {},
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        minimum_body_seconds=1.0,
        minimum_body_fraction=0.01,
        quality_loop=None,
        automatic_delivery=True,
        replan_on_review=False,
    )
    assert len(compiled) == 1
    chosen = compiled[0].plan
    assert chosen.exit_time_a >= 24.0
    assert int(chosen.metadata["exit_bar_index"]) >= 12
    assert chosen.metadata["exit_song_role"] == "verse"
    assert not bool((chosen.metadata.get("dj_move_brain", {}) or {}).get("song_form_hard_guard", False))
