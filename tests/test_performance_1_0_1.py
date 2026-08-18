from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from conftest import analysis
from djai.brain import professional as professional_brain
from djai.brain.professional import BrainSelection, ProfessionalBrainConfig
from djai.fallback import build_last_resort_plan


def _tracks_and_plans(tmp_path: Path, count: int = 6):
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    track_a = analysis(a_path, bpm=124.0, energy=0.55)
    track_b = analysis(b_path, bpm=126.0, energy=0.62)
    base = build_last_resort_plan(track_a, track_b, reason="performance-test")
    plans = [copy.deepcopy(base) for _ in range(count)]
    for index, plan in enumerate(plans):
        plan.score = max(0.0, 0.90 - index * 0.03)
        plan.metadata = {**plan.metadata, "candidate_index": index}
    return track_a, track_b, plans


def _fake_selection(measured, *, adaptive, uncertainty: float = 0.10, accepted: bool = True):
    selected = measured.evaluated[0]
    selected.metadata = {
        **selected.metadata,
        "professional_brain": {"adaptive_rehearsal": dict(adaptive or {})},
    }
    judgement = SimpleNamespace(
        accepted=accepted,
        feasible=True,
        overall=0.82 if accepted else 0.30,
        uncertainty=uncertainty,
    )
    return BrainSelection(
        selected=selected,
        rehearsed=list(measured.evaluated),
        judgements=[judgement for _ in measured.evaluated],
        attempts=1,
    )


def test_adaptive_rehearsal_stops_after_confident_two_candidate_pilot(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b, plans = _tracks_and_plans(tmp_path)
    rendered_sizes: list[int] = []

    monkeypatch.setattr(professional_brain, "diversify_transition_candidates", lambda values, **kwargs: list(values))
    monkeypatch.setattr(professional_brain, "_attach_cached_stems", lambda *args, **kwargs: None)

    def fake_rehearse(values, **kwargs):
        rendered_sizes.append(len(values))
        return SimpleNamespace(evaluated=list(values))

    monkeypatch.setattr(professional_brain, "_rehearse", fake_rehearse)
    monkeypatch.setattr(
        professional_brain,
        "_selection_from_measured",
        lambda measured, adaptive=None, **kwargs: _fake_selection(measured, adaptive=adaptive),
    )

    result = professional_brain.rehearse_candidate_pool(
        track_a,
        track_b,
        plans,
        config=ProfessionalBrainConfig(finalists=6, adaptive_pilot_finalists=2),
    )

    assert rendered_sizes == [2]
    adaptive = result.selected.metadata["professional_brain"]["adaptive_rehearsal"]
    assert adaptive["early_accept"] is True
    assert adaptive["expanded"] is False


def test_adaptive_rehearsal_expands_when_pilot_is_uncertain(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b, plans = _tracks_and_plans(tmp_path)
    rendered_sizes: list[int] = []

    monkeypatch.setattr(professional_brain, "diversify_transition_candidates", lambda values, **kwargs: list(values))
    monkeypatch.setattr(professional_brain, "_attach_cached_stems", lambda *args, **kwargs: None)

    def fake_rehearse(values, **kwargs):
        rendered_sizes.append(len(values))
        return SimpleNamespace(evaluated=list(values))

    selection_calls = 0

    def fake_selection(measured, adaptive=None, **kwargs):
        nonlocal selection_calls
        selection_calls += 1
        return _fake_selection(
            measured,
            adaptive=adaptive,
            uncertainty=0.55 if selection_calls == 1 else 0.10,
        )

    monkeypatch.setattr(professional_brain, "_rehearse", fake_rehearse)
    monkeypatch.setattr(professional_brain, "_selection_from_measured", fake_selection)

    result = professional_brain.rehearse_candidate_pool(
        track_a,
        track_b,
        plans,
        config=ProfessionalBrainConfig(finalists=6, adaptive_pilot_finalists=2),
    )

    assert rendered_sizes == [2, 6]
    adaptive = result.selected.metadata["professional_brain"]["adaptive_rehearsal"]
    assert adaptive["expanded"] is True
    assert adaptive["early_accept"] is False


def test_taste_lab_keeps_full_rehearsal_evidence(tmp_path: Path, monkeypatch) -> None:
    track_a, track_b, plans = _tracks_and_plans(tmp_path)
    rendered_sizes: list[int] = []

    monkeypatch.setattr(professional_brain, "diversify_transition_candidates", lambda values, **kwargs: list(values))
    monkeypatch.setattr(professional_brain, "_attach_cached_stems", lambda *args, **kwargs: None)

    def fake_rehearse(values, **kwargs):
        rendered_sizes.append(len(values))
        return SimpleNamespace(evaluated=list(values))

    monkeypatch.setattr(professional_brain, "_rehearse", fake_rehearse)
    monkeypatch.setattr(
        professional_brain,
        "_selection_from_measured",
        lambda measured, adaptive=None, **kwargs: _fake_selection(measured, adaptive=adaptive),
    )

    result = professional_brain.rehearse_candidate_pool(
        track_a,
        track_b,
        plans,
        config=ProfessionalBrainConfig(finalists=6, adaptive_pilot_finalists=2),
        context={"mode": "taste_lab"},
    )

    assert rendered_sizes == [6]
    assert result.selected.metadata["professional_brain"]["adaptive_rehearsal"]["enabled"] is False


def test_two_track_order_uses_exact_search_without_full_set_planners(tmp_path: Path, monkeypatch) -> None:
    from djai import set_planner

    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    analyses = [analysis(a, bpm=124.0), analysis(b, bpm=126.0)]
    graph = {(0, 1): object(), (1, 0): object()}

    monkeypatch.setattr(
        set_planner,
        "build_receding_horizon_order",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("two-track fast path should not use receding horizon")),
    )
    monkeypatch.setattr(
        set_planner,
        "build_unified_ai_order",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("two-track fast path should not use unified beam search")),
    )
    monkeypatch.setattr(
        set_planner,
        "_local_improve",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("two-track fast path should not use local search")),
    )
    monkeypatch.setattr(set_planner, "_order_score", lambda order, *args, **kwargs: 1.0 if order == [1, 0] else 0.5)

    assert set_planner.optimize_order(analyses, graph) == [1, 0]


def test_rehearsal_audio_decode_is_reused_across_retry_families(tmp_path: Path, monkeypatch) -> None:
    from djai import candidate_renderer

    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    calls: list[tuple[str, int]] = []

    candidate_renderer._load_rehearsal_audio_cached.cache_clear()

    def fake_load(path, *, sample_rate, mono):
        calls.append((str(path), int(sample_rate)))
        return "decoded", sample_rate

    monkeypatch.setattr(candidate_renderer, "load_audio", fake_load)

    first = candidate_renderer._load_rehearsal_audio(source, 44_100)
    second = candidate_renderer._load_rehearsal_audio(source, 44_100)

    assert first == second == ("decoded", 44_100)
    assert len(calls) == 1

    source.write_bytes(b"audio-changed")
    candidate_renderer._load_rehearsal_audio(source, 44_100)
    assert len(calls) == 2


def test_auto_render_embedding_capture_skips_unused_live_features(monkeypatch) -> None:
    from djai import candidate_renderer
    from djai.intelligence import neural_policy, transition_ranker

    monkeypatch.setenv("DJAI_CAPTURE_RENDER_EMBEDDINGS", "auto")
    monkeypatch.setattr(neural_policy, "scalar_head_uses_feature_prefix", lambda *args, **kwargs: False)
    monkeypatch.setattr(transition_ranker, "ranker_uses_render_embedding", lambda *args, **kwargs: False)

    assert candidate_renderer._capture_render_embeddings({"professional_brain": True}) is False
    assert candidate_renderer._capture_render_embeddings({"mode": "taste_lab"}) is True


def test_mert_runtime_is_process_cached(monkeypatch) -> None:
    from djai.intelligence import embeddings

    calls = {"extractor": 0, "model": 0}
    fake_transformers = types.ModuleType("transformers")

    class FakeFeatureExtractor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["extractor"] += 1
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["model"] += 1
            return cls()

        def eval(self):
            return self

    fake_transformers.AutoFeatureExtractor = FakeFeatureExtractor
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    embeddings._mert_runtime.cache_clear()

    first = embeddings._mert_runtime("fake-mert", True)
    second = embeddings._mert_runtime("fake-mert", True)

    assert first is second
    assert calls == {"extractor": 1, "model": 1}
