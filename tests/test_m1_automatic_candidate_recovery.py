"""Regression tests for fail-closed automatic cue recovery.

No model weights or real audio are required. These tests exercise the
existing gates and preserve explicit/forced planning behavior.
"""
from types import SimpleNamespace

import pytest

from djai import planner


def candidate(*, score=0.76, residual=0.0, phase_safe=True,
              confirmation=False, forced=False, mode="beatmatched_blend",
              grid=0.85, bars=8, warning=None):
    return SimpleNamespace(
        score=score, requires_confirmation=confirmation, forced=forced,
        transition_mode=mode, beatmatched=(mode == "beatmatched_blend"),
        compatibility=SimpleNamespace(beat_grid=grid),
        metadata={"deck_performance": {
            "phase_integrity": {
                "enabled": True, "safe": phase_safe,
                "max_residual_ms": residual,
                "residual_limit_ms": 25.0,
            },
        }},
        transition_bars=bars, warning=warning,
    )


def search(monkeypatch, ranked, *, bars=None, force=False, style="safe", policy="safe"):
    seen = []

    def stub(*args, **kwargs):
        requested = kwargs["requested_bars"]
        seen.append(requested)
        return ranked[requested]

    monkeypatch.setattr(planner, "_rank_style_plans", stub)
    result = planner._search_style(
        object(), object(),
        evidence_matrix=object(), style=style, tempo_policy=policy,
        force=force, requested_bars=bars, effective_bpm_a=None,
        render_rate_a=1.0, stems_available=False, excluded_pairs=None,
        enforce_safety=True,
    )
    return result, seen


def test_skips_higher_ranked_unsafe_plan(monkeypatch):
    rejected = candidate(confirmation=True, phase_safe=False, residual=999)
    accepted = candidate(score=0.76)
    selected, seen = search(monkeypatch, {4: [rejected, accepted]}, bars=4)
    assert selected is accepted
    assert seen == [4]


def test_shorter_four_bar_fallback_when_default_candidates_unsafe(monkeypatch):
    bad = candidate(confirmation=True, phase_safe=False, residual=999)
    valid = candidate(score=0.7602, bars=4, residual=0.0)
    selected, seen = search(monkeypatch, {None: [bad], 4: [valid]})
    assert selected is valid
    assert seen == [None, 4]


def test_explicit_request_does_not_search_other_bar_lengths(monkeypatch):
    bad = candidate(confirmation=True, phase_safe=False)
    with pytest.raises(planner.DifficultTransitionError):
        search(monkeypatch, {8: [bad], 4: [candidate()]}, bars=8)


def test_forced_planning_keeps_original_ranked_plan(monkeypatch):
    bad = candidate(confirmation=True, phase_safe=False)
    valid = candidate(score=0.85)
    chosen, seen = search(monkeypatch, {None: [bad, valid]}, force=True)
    assert chosen is bad
    assert seen == [None]


def test_missing_phase_profile_is_not_automatically_eligible(monkeypatch):
    uncertain = candidate(confirmation=False)
    uncertain.metadata = {}
    valid = candidate(score=0.76)
    chosen, _ = search(monkeypatch, {4: [uncertain, valid]}, bars=4)
    assert chosen is valid


def test_confirmation_and_score_gates_remain_active(monkeypatch):
    bad_confirmation = candidate(confirmation=True, warning="review required")
    low_score = candidate(score=0.30)
    valid = candidate(score=0.76)
    chosen, _ = search(monkeypatch, {4: [bad_confirmation, low_score, valid]}, bars=4)
    assert chosen is valid


def test_emergency_cut_never_becomes_safe_style_fallback(monkeypatch):
    cut = candidate(mode="emergency_cut", confirmation=False)
    good = candidate()
    chosen, _ = search(monkeypatch, {4: [cut, good]}, bars=4)
    assert chosen is good


def test_no_shortening_for_cut_policy(monkeypatch):
    bad = candidate(confirmation=True, phase_safe=False)
    with pytest.raises(planner.DifficultTransitionError):
        search(monkeypatch, {None: [bad]}, policy="cut")


def test_no_shortening_for_emergency_style(monkeypatch):
    bad = candidate(confirmation=True, phase_safe=False)
    with pytest.raises(planner.DifficultTransitionError):
        search(monkeypatch, {None: [bad]}, style="emergency")
