from types import SimpleNamespace

from djai.evaluation.full_set import compare_full_sets, evaluate_full_set


def _edge(score=0.8, *, hard=False, emergency=False, fallback=0, technique="blend"):
    measurement = SimpleNamespace(hard_failures=["clipping"] if hard else [])
    plan = SimpleNamespace(
        measured_score=score,
        score=score,
        quality_checked=True,
        measurement_passed=not hard,
        measurement=measurement,
        fallback_level=fallback,
        metadata={},
        transition_mode="emergency_cut" if emergency else "beatmatched_blend",
        technique=technique,
    )
    return SimpleNamespace(plan=plan, objective_score=score, quality="good" if not hard else "poor")


def _plan(edges):
    return SimpleNamespace(
        transitions=edges,
        metadata={"journey_arc": {"curve_fit_score": 0.82}},
    )


def test_full_set_benchmark_reports_safe_quality():
    report = evaluate_full_set(_plan([_edge(0.85, technique="blend"), _edge(0.79, technique="echo")]))
    assert report.production_safe
    assert report.hard_failure_count == 0
    assert report.overall > 0.7


def test_hard_failure_can_never_win_full_set_comparison():
    champion = _plan([_edge(0.65)])
    challenger = _plan([_edge(0.99, hard=True)])
    result = compare_full_sets(champion, challenger)
    assert result["challenger"]["overall"] == 0.0
    assert result["challenger_wins"] is False
