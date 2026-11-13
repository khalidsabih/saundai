from __future__ import annotations

from pathlib import Path

from conftest import analysis
from djai.brain.candidate_generator import diversify_transition_candidates
from djai.models import CompatibilityBreakdown, TransitionPlan


def _plan(tmp_path: Path, tag: str, analytical: float, learned: float) -> TransitionPlan:
    a = analysis(tmp_path / f"a-{tag}.wav", bpm=124)
    b = analysis(tmp_path / f"b-{tag}.wav", bpm=124)
    plan = TransitionPlan(
        track_a=a.path,
        track_b=b.path,
        exit_time_a=200,
        entry_time_b=20,
        transition_bars=16,
        transition_seconds=30,
        target_bpm=124,
        time_stretch_rate_b=1,
        time_stretch_applied=False,
        technique="equal_power_blend",
        score=analytical,
        compatibility=CompatibilityBreakdown(
            tempo=.9, harmonic=.9, energy=.9, texture=.9, phrase=.9,
            sections=.9, overall=.9, tempo_difference_percent=0,
            energy_delta=.05, beat_grid=.9, vocal_overlap=.9,
            bass_conflict=.9,
        ),
        cue_confidence=.9,
        intent_score=.8,
        technique_score=.8,
    )
    plan.metadata = {
        **plan.metadata,
        "selection_utility": analytical,
        "transition_ranker_blended_score": learned,
    }
    return plan


def test_validated_model_score_can_survive_diversity_selection(tmp_path: Path) -> None:
    analytical_winner = _plan(tmp_path, "analytical", .91, .35)
    model_winner = _plan(tmp_path, "model", .78, .94)
    candidates = [analytical_winner, model_winner]

    default = diversify_transition_candidates(candidates, count=1)
    model = diversify_transition_candidates(
        candidates, count=1, score_key="transition_ranker_blended_score"
    )
    assert default[0] is analytical_winner
    assert model[0] is model_winner


def test_ranker_score_fallback_preserves_default_selection(tmp_path: Path) -> None:
    strong = _plan(tmp_path, "strong", .90, .10)
    weak = _plan(tmp_path, "weak", .70, .99)
    for plan in (strong, weak):
        plan.metadata.pop("transition_ranker_blended_score")
    selected = diversify_transition_candidates(
        [strong, weak], count=1, score_key="transition_ranker_blended_score"
    )
    assert selected[0] is strong


def test_authorized_model_rank_is_kept_at_rehearsal_boundary(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from djai.brain import professional

    plan = _plan(tmp_path, "validated", .7, .95)
    plan.metadata["topk_selection"] = {"ranking_source": "validated_model"}
    plan.metadata["transition_ranker_authority"] = .3
    score_keys: list[str] = []
    original_diversify = professional.diversify_transition_candidates

    def capture(plans, **kwargs):
        score_keys.append(kwargs["score_key"])
        return original_diversify(plans, **kwargs)

    class StopBeforeAudio(Exception):
        pass

    def no_audio(*args, **kwargs):
        raise StopBeforeAudio()

    monkeypatch.setattr(professional, "diversify_transition_candidates", capture)
    monkeypatch.setattr(professional, "_attach_cached_stems", lambda *args: None)
    monkeypatch.setattr(professional, "_rehearse", no_audio)
    with pytest.raises(StopBeforeAudio):
        professional.rehearse_candidate_pool(
            analysis(tmp_path / "left.wav", bpm=124),
            analysis(tmp_path / "right.wav", bpm=124),
            [plan],
            config=professional.ProfessionalBrainConfig(adaptive_rehearsal=False),
            context={"unified_transition": True},
        )
    assert score_keys == ["transition_ranker_blended_score"]


def test_zero_authority_cannot_control_rehearsal_shortlist(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from djai.brain import professional

    plan = _plan(tmp_path, "shadow", .8, .95)
    plan.metadata["topk_selection"] = {"ranking_source": "validated_model"}
    plan.metadata["transition_ranker_authority"] = 0.0
    score_keys = []

    def capture(plans, **kwargs):
        score_keys.append(kwargs["score_key"])
        return plans

    class StopBeforeAudio(Exception):
        pass

    monkeypatch.setattr(professional, "diversify_transition_candidates", capture)
    monkeypatch.setattr(professional, "_attach_cached_stems", lambda *args: None)
    monkeypatch.setattr(professional, "_rehearse", lambda *args, **kwargs: (_ for _ in ()).throw(StopBeforeAudio()))
    with pytest.raises(StopBeforeAudio):
        professional.rehearse_candidate_pool(
            analysis(tmp_path / "left-shadow.wav", bpm=124),
            analysis(tmp_path / "right-shadow.wav", bpm=124),
            [plan],
            config=professional.ProfessionalBrainConfig(adaptive_rehearsal=False),
            context={"unified_transition": True},
        )
    assert score_keys == ["selection_utility"]
