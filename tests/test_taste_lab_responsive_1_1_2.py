from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from djai.cache import CachedAnalysis
from djai.intelligence import lab
from djai.models import CompatibilityBreakdown, TransitionMeasurement, TransitionPlan


def _plan(track_a: str, track_b: str, signature: str) -> TransitionPlan:
    measurement = TransitionMeasurement(
        version="3.0.0",
        sample_rate=44_100,
        duration_seconds=2.0,
        overall=0.8,
    )
    return TransitionPlan(
        track_a=track_a,
        track_b=track_b,
        exit_time_a=20.0,
        entry_time_b=4.0,
        transition_bars=4,
        transition_seconds=2.0,
        target_bpm=120.0,
        time_stretch_rate_b=1.0,
        time_stretch_applied=False,
        technique="equal_power_blend",
        score=0.8,
        analytical_score=0.8,
        measured_score=0.8,
        measurement=measurement,
        measurement_passed=True,
        quality_checked=True,
        cue_confidence=0.8,
        intent_score=0.8,
        technique_score=0.8,
        execution_signature=signature,
        compatibility=CompatibilityBreakdown(
            tempo=0.8,
            harmonic=0.8,
            energy=0.8,
            texture=0.8,
            phrase=0.8,
            sections=0.8,
            overall=0.8,
            tempo_difference_percent=1.0,
            energy_delta=0.01,
            beat_grid=0.8,
            vocal_overlap=0.8,
            bass_conflict=0.8,
        ),
        metadata={"professional_judge_score": 0.8},
    )


def test_listening_session_reuses_analysis_cache_and_reports_generation(tmp_path: Path, monkeypatch) -> None:
    track_a = tmp_path / "a.wav"
    track_b = tmp_path / "b.wav"
    track_a.write_bytes(b"a")
    track_b.write_bytes(b"b")
    calls: list[tuple[str, str, bool]] = []

    def fake_cached(path, *, cache_dir, use_cache):
        resolved = str(Path(path).resolve())
        calls.append((resolved, str(Path(cache_dir).resolve()), use_cache))
        return CachedAnalysis(
            analysis=SimpleNamespace(path=resolved),
            cache_hit=True,
            fingerprint="f",
            cache_path=Path(cache_dir) / "f.json",
        )

    plans = [_plan(str(track_a.resolve()), str(track_b.resolve()), "one"), _plan(str(track_a.resolve()), str(track_b.resolve()), "two")]
    monkeypatch.setattr(lab, "analyze_track_cached", fake_cached)
    monkeypatch.setattr(lab, "generate_transition_ideas", lambda *args, **kwargs: plans)
    monkeypatch.setattr(lab, "rehearse_candidate_pool", lambda *args, **kwargs: SimpleNamespace(rehearsed=plans))

    def fake_render(plan, output, **kwargs):
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    monkeypatch.setattr(lab, "render_transition", fake_render)
    monkeypatch.setattr(lab, "plan_features", lambda plan: {"execution_signature": plan.execution_signature})

    cache_dir = tmp_path / "cache"
    manifest = lab.create_listening_session(
        track_a,
        track_b,
        output_root=tmp_path / "lab",
        candidates=2,
        cache_dir=cache_dir,
        use_cache=True,
    )

    assert len(calls) == 2
    assert all(call[1] == str(cache_dir.resolve()) for call in calls)
    assert all(call[2] is True for call in calls)
    assert manifest["generation"]["analysis_cache_hits"] == 2
    assert manifest["generation"]["analysis_cache_total"] == 2
    assert manifest["generation"]["total_seconds"] >= 0.0
    assert len(manifest["candidates"]) == 2
