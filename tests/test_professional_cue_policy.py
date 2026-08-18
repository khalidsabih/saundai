from __future__ import annotations

from djai.compatibility import section_compatibility
from djai.fallback import build_last_resort_plan
from djai.set_planner import _mixing_window, _sequence_entry_candidates, _sequence_exit_candidates

from conftest import candidate


def test_section_compatibility_no_longer_makes_outro_intro_universal_optimum():
    assert section_compatibility("groove", "groove") > section_compatibility("outro", "intro")
    assert section_compatibility("breakdown", "drop") > section_compatibility("outro", "intro")


def test_sequence_exit_policy_targets_mix_window_not_track_end(track_pair):
    a, _ = track_pair
    local = a.key
    a.exit_candidates = [
        candidate("exit", 108.0, 54, "groove", bpm=a.bpm, local_key=local, score=0.80),
        candidate("exit", 150.0, 75, "breakdown", bpm=a.bpm, local_key=local, score=0.84),
        candidate("exit", 285.0, 142, "outro", bpm=a.bpm, local_key=local, score=0.99),
    ]
    required_source = 105.0
    selected, target, soft_latest = _sequence_exit_candidates(
        a,
        source_cursor=0.0,
        current_rate=1.0,
        required_source=required_source,
        limit=3,
    )
    assert target < a.duration_seconds * 0.75
    assert soft_latest < a.duration_seconds * 0.90
    assert selected[0].time_seconds < 285.0
    assert selected[0].section_label in {"groove", "breakdown"}


def test_entry_policy_skips_literal_start_when_move_brain_disabled(track_pair, monkeypatch):
    monkeypatch.setenv("DJAI_MOVE_BRAIN", "0")
    _, b = track_pair
    local = b.key
    b.entry_candidates = [
        candidate("entry", 0.0, 0, "intro", bpm=b.bpm, local_key=local, score=0.99),
        candidate("entry", 16.0, 8, "groove", bpm=b.bpm, local_key=local, score=0.82),
        candidate("entry", 48.0, 24, "build", bpm=b.bpm, local_key=local, score=0.84),
        candidate("entry", 64.0, 32, "drop", bpm=b.bpm, local_key=local, score=0.86),
    ]
    selected = _sequence_entry_candidates(b, limit=4)
    assert selected
    assert all(item.time_seconds > 0.0 for item in selected)
    assert selected[0].section_label in {"groove", "build", "drop"}


def test_last_resort_cut_uses_mature_incoming_cue(track_pair):
    a, b = track_pair
    local = b.key
    b.entry_candidates = [
        candidate("entry", 0.0, 0, "intro", bpm=b.bpm, local_key=local, score=0.99),
        candidate("entry", 24.0, 12, "groove", bpm=b.bpm, local_key=local, score=0.78),
        candidate("entry", 56.0, 28, "drop", bpm=b.bpm, local_key=local, score=0.82),
    ]
    plan = build_last_resort_plan(a, b, source_cursor=0.0, preferred_exit_time=140.0)
    assert plan.entry_time_b > 0.0
    assert plan.entry_section in {"groove", "drop"}


def test_mixing_window_has_preferred_upper_bound(track_pair):
    a, _ = track_pair
    target, latest = _mixing_window(a, 0.0, 1.0, 105.0)
    assert 105.0 <= target < latest < a.duration_seconds
    assert latest <= 210.0 + 1e-6


def test_move_brain_allows_a_strong_real_intro_at_time_zero(track_pair, monkeypatch):
    monkeypatch.setenv("DJAI_MOVE_BRAIN", "1")
    _, b = track_pair
    local = b.key
    b.entry_candidates = [
        candidate("entry", 0.0, 0, "intro", bpm=b.bpm, local_key=local, score=0.99),
        candidate("entry", 16.0, 8, "groove", bpm=b.bpm, local_key=local, score=0.65),
        candidate("entry", 64.0, 32, "outro", bpm=b.bpm, local_key=local, score=0.95),
    ]
    selected = _sequence_entry_candidates(b, limit=3)
    assert selected
    assert selected[0].time_seconds == 0.0
    assert selected[0].section_label == "intro"
