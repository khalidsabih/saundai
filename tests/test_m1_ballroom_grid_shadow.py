from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-grid-shadow.py"
    spec = importlib.util.spec_from_file_location("m1_ballroom_grid_shadow", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_scores_rejected_grid_without_replacing_selected():
    mod = load()
    ref = {
        "dataset_id": "ballroom",
        "split": "development",
        "items": [
            {"id": "one", "bpm": None,
             "beats_seconds": [0.5, 1.0, 1.5, 2.0],
             "downbeats_seconds": [0.5, 2.0]},
        ],
    }
    pred = {"one": {
        "selected": {
            "bpm": 120.0,
            "beats_seconds": [0.25, 0.75, 1.25, 1.75],
            "downbeats_seconds": [0.25, 1.75],
        },
        "candidate": {
            "available": True,
            "used": False,
            "confidence": 0.5,
            "authority": 0.1,
            "reason": "gate",
            "bpm": 120.0,
            "beats_seconds": [0.5, 1.0, 1.5, 2.0],
            "downbeats_seconds": [0.5, 2.0],
        },
    }}
    result = mod.score(ref, pred)
    assert result["coverage"]["neural_grid_rejected_with_candidate"] == 1
    assert result["selected_metrics"]["beat_f1"] == 0.0
    assert result["selected_metrics"]["downbeat_f1"] == 0.0
    assert result["neural_candidate_metrics"]["beat_f1"] == 1.0
    assert result["neural_candidate_metrics"]["downbeat_f1"] == 1.0
    assert result["per_track"][0]["selected_bpm"] == 120.0


def test_unavailable_candidate_excluded_from_coverage():
    mod = load()
    ref = {
        "dataset_id": "ballroom", "split": "development",
        "items": [{"id": "one", "bpm": None,
                   "beats_seconds": [0.5, 1.0],
                   "downbeats_seconds": [0.5]}],
    }
    pred = {"one": {
        "selected": {"bpm": 120.0, "beats_seconds": [0.5, 1.0], "downbeats_seconds": [0.5]},
        "candidate": {"available": False, "used": False, "reason": "missing"},
    }}
    result = mod.score(ref, pred)
    assert result["coverage"]["neural_unavailable"] == 1
    assert result["neural_candidate_metrics"] is None
    assert result["selected_metrics"]["beat_f1"] == 1.0


def test_grid_bpm_uses_median_interval():
    mod = load()
    assert mod.grid_bpm([0.5, 1.0, 1.5, 2.0]) == 120.0
    assert mod.grid_bpm([]) is None


def test_confidence_shadow_removes_baseline_agreement_without_changing_selection():
    mod = load()
    selected = [0.25, 0.75, 1.25, 1.75]
    evidence = {
        "available": True, "used": False, "confidence": 0.68,
        "beats_seconds": [0.0, 0.5, 1.0, 1.5],
        "downbeats_seconds": [0.0],
    }
    diagnostic = mod.confidence_shadow(selected, evidence)
    assert diagnostic["grid_regularity"] == 1.0
    assert diagnostic["deterministic_agreement"] == 0.0
    assert diagnostic["downbeat_support"] == 1.0
    assert diagnostic["legacy_reconstructed"] == 0.68
    assert diagnostic["no_baseline_agreement_shadow"] == 1.0
    assert diagnostic["adoption_authorized"] is False
    assert evidence["used"] is False


def test_confidence_shadow_unavailable_is_not_scored():
    mod = load()
    assert mod.confidence_shadow([], {"available": False}) is None


def test_shadow_score_reports_independent_delta_without_adopting_candidate():
    mod = load()
    ref = {
        "dataset_id": "ballroom", "split": "development",
        "items": [{"id": "one", "bpm": None,
                   "beats_seconds": [0.5, 1.0, 1.5, 2.0],
                   "downbeats_seconds": [0.5, 2.0]}],
    }
    outputs = {"one": {
        "selected": {"bpm": 120.0, "beats_seconds": [0.25, 0.75, 1.25, 1.75],
                     "downbeats_seconds": [0.25, 1.75]},
        "candidate": {"available": True, "used": False, "confidence": 0.5,
                      "authority": 0.0, "reason": "retained",
                      "bpm": 120.0, "beats_seconds": [0.5, 1.0, 1.5, 2.0],
                      "downbeats_seconds": [0.5, 2.0]},
    }}
    result = mod.score(ref, outputs)
    row = result["per_track"][0]
    assert row["grid_accepted"] is False
    assert row["independent_accuracy_delta"] == {"beat_f1": 1.0, "downbeat_f1": 1.0}
    assert row["confidence_shadow"]["shadow_only"] is True
    assert result["confidence_challenger"]["changes_grid_selection"] is False


def test_adopted_grid_uses_pre_adoption_agreement_evidence():
    mod = load()
    beats = [0.0, 0.5, 1.0, 1.5]
    evidence = {
        "available": True, "used": True, "agreement": 0.25,
        "confidence": 0.44, "beats_seconds": beats, "downbeats_seconds": [0.0],
    }
    shadow = mod.confidence_shadow(beats, evidence)
    assert shadow["deterministic_agreement"] == 0.25
    assert shadow["deterministic_agreement_source"] == "pre_adoption_inference_evidence"
    assert shadow["legacy_reconstructed"] == 0.76
    assert shadow["adoption_authorized"] is False


def test_adopted_grid_without_pre_adoption_agreement_avoids_false_reconstruction():
    mod = load()
    beats = [0.0, 0.5, 1.0, 1.5]
    evidence = {
        "available": True, "used": True, "confidence": 0.9,
        "beats_seconds": beats, "downbeats_seconds": [0.0],
    }
    shadow = mod.confidence_shadow(beats, evidence)
    assert shadow["deterministic_agreement"] is None
    assert shadow["legacy_reconstructed"] is None
    assert shadow["no_baseline_agreement_shadow"] == 1.0


def test_counterfactual_recovers_low_agreement_with_existing_authority_and_safe_bars():
    mod = load()
    from djai.intelligence.model_registry import ModelRegistry, default_manifests
    registry = ModelRegistry({m.id: m for m in default_manifests()})
    beats = [float(i) * 0.5 for i in range(48)]
    downbeats = beats[::4]
    evidence = {
        "available": True, "used": False, "agreement": 0.0,
        "confidence": 0.68, "beats_seconds": beats, "downbeats_seconds": downbeats,
    }
    shadow = mod.confidence_shadow([x + 0.25 for x in beats], evidence)
    assert shadow["no_baseline_agreement_shadow"] >= 0.70
    result = mod.governance_counterfactual(evidence, shadow, registry)
    assert result["status"] == "eligible"
    assert result["hypothetical_authority"] > 0
    assert result["adoption_authorized"] is False
    assert evidence["used"] is False


def test_counterfactual_preserves_structural_bar_veto():
    mod = load()
    from djai.intelligence.model_registry import ModelRegistry, default_manifests
    registry = ModelRegistry({m.id: m for m in default_manifests()})
    beats = [float(i) * 0.5 for i in range(48)]
    downbeats = [beats[i] for i in (0, 4, 8, 13, 17, 21, 25, 29, 33, 37, 41, 45)]
    evidence = {
        "available": True, "used": False, "agreement": 0.0,
        "confidence": 0.68, "beats_seconds": beats, "downbeats_seconds": downbeats,
    }
    shadow = mod.confidence_shadow([x + 0.25 for x in beats], evidence)
    result = mod.governance_counterfactual(evidence, shadow, registry)
    assert result["status"] == "blocked_bar_integrity"
    assert result["eligible"] is False
    assert result["hard_bar_integrity_preserved"] is True


def test_counterfactual_respects_shadow_model_authority_and_unavailable_grid():
    mod = load()
    from djai.intelligence.model_registry import ModelRegistry, default_manifests, ModelManifest
    manifests = {m.id: m for m in default_manifests()}
    manifests["beat_this"] = ModelManifest(
        id="beat_this", task="beat_downbeat", provider="test", model="test",
        version="1", authority="shadow", weight=0.0,
    )
    registry = ModelRegistry(manifests)
    beats = [float(i) * 0.5 for i in range(48)]
    evidence = {
        "available": True, "used": False, "agreement": 0.0,
        "confidence": 0.68, "beats_seconds": beats, "downbeats_seconds": beats[::4],
    }
    shadow = mod.confidence_shadow([x + 0.25 for x in beats], evidence)
    result = mod.governance_counterfactual(evidence, shadow, registry)
    assert result["status"] == "blocked_authority"
    assert result["hypothetical_authority"] == 0
    assert mod.governance_counterfactual(
        {"available": False}, None, registry
    )["status"] == "unavailable"


def test_counterfactual_requires_two_downbeats():
    mod = load()
    from djai.intelligence.model_registry import ModelRegistry, default_manifests
    registry = ModelRegistry({m.id: m for m in default_manifests()})
    beats = [float(i) * 0.5 for i in range(48)]
    evidence = {
        "available": True, "used": False, "agreement": 0.0,
        "confidence": 0.5, "beats_seconds": beats, "downbeats_seconds": [0.0],
    }
    shadow = mod.confidence_shadow([x + 0.25 for x in beats], evidence)
    # The low support score might itself block confidence; explicitly isolate
    # the downbeat count gate with a hypothetical high diagnostic score.
    shadow["no_baseline_agreement_shadow"] = 0.99
    result = mod.governance_counterfactual(evidence, shadow, registry)
    assert result["status"] == "blocked_downbeat_evidence"
