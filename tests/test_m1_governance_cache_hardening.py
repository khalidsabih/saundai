"""Release blockers: neural model authority and config-aware analysis caching."""
from __future__ import annotations

import json
import types

import pytest

from djai import cache
from djai.intelligence import neural_bar_phase


def evidence(confidence=.90, authority=.30):
    return {"available": True, "confidence": confidence, "authority": authority}


def registry(*, role="advisory", current=.30):
    return types.SimpleNamespace(
        get=lambda name: types.SimpleNamespace(authority=role),
        authority=lambda name, confidence: current,
    )


def test_trial_rejects_production_and_hosted_even_with_healthy_candidate(monkeypatch):
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry())
    monkeypatch.setenv("SAUND_ENV", "production")
    allowed, reason = neural_bar_phase.trial_governance(evidence())
    assert not allowed and "production" in reason
    monkeypatch.setenv("SAUND_ENV", "development")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "1")
    allowed, reason = neural_bar_phase.trial_governance(evidence())
    assert not allowed and "hosted" in reason


@pytest.mark.parametrize("role", ["shadow", "disabled"])
def test_trial_respects_model_registry_denial(monkeypatch, role):
    monkeypatch.setenv("SAUND_ENV", "development")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "0")
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry(role=role))
    allowed, _ = neural_bar_phase.trial_governance(evidence())
    assert not allowed


def test_trial_rejects_zero_authority_even_if_acoustically_confident(monkeypatch):
    monkeypatch.setenv("SAUND_ENV", "development")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "0")
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry())
    assert not neural_bar_phase.trial_governance(evidence(authority=0))[0]
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry(current=0))
    assert not neural_bar_phase.trial_governance(evidence())[0]


def test_trial_does_not_allow_reduced_threshold_to_override_release_floor(monkeypatch):
    monkeypatch.setenv("SAUND_ENV", "development")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "0")
    monkeypatch.setenv("DJAI_NEURAL_BEAT_USE_THRESHOLD", "0.1")
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry())
    assert not neural_bar_phase.trial_governance(evidence(confidence=.6599))[0]
    assert neural_bar_phase.trial_governance(evidence(confidence=.82))[0]


def test_neural_flags_invalidate_monolithic_cache_fingerprint(tmp_path, monkeypatch):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"test bytes, no decoding required for fingerprint")
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    monkeypatch.setenv("DJAI_NEURAL_BAR_PHASE_TRIAL", "0")
    baseline = cache.fingerprint_file(audio)
    assert cache.fingerprint_file(audio) == baseline
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "1")
    enabled = cache.fingerprint_file(audio)
    assert enabled != baseline
    monkeypatch.setenv("DJAI_NEURAL_BAR_PHASE_TRIAL", "1")
    assert cache.fingerprint_file(audio) != enabled
    monkeypatch.setenv("DJAI_BEAT_THIS_SHA256", "a" * 64)
    assert cache.fingerprint_file(audio) != enabled


def test_checkpoint_replacement_invalidates_cache_without_pin(tmp_path, monkeypatch):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"audio")
    model = tmp_path / "model.ckpt"
    model.write_bytes(b"first")
    monkeypatch.setenv("DJAI_BEAT_THIS_CHECKPOINT", str(model))
    first = cache.fingerprint_file(audio)
    model.write_bytes(b"second, different size")
    second = cache.fingerprint_file(audio)
    assert first != second


def test_neural_enabled_never_uses_unversioned_modular_projections(monkeypatch):
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    monkeypatch.setenv("DJAI_NEURAL_BAR_PHASE_TRIAL", "0")
    monkeypatch.setenv("DJAI_TEMPO_CONSISTENCY", "0")
    assert cache._modular_feature_cache_safe()
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "auto")
    assert not cache._modular_feature_cache_safe()
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    monkeypatch.setenv("DJAI_NEURAL_BAR_PHASE_TRIAL", "1")
    assert not cache._modular_feature_cache_safe()


def test_trial_evidence_never_treats_approval_as_data(monkeypatch):
    monkeypatch.setenv("SAUND_ENV", "development")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "0")
    monkeypatch.setattr(neural_bar_phase, "load_model_registry", lambda: registry(current=0))
    forged = {**evidence(), "used": True, "production_approved": True}
    allowed, _ = neural_bar_phase.trial_governance(forged)
    assert not allowed


def test_actual_cached_analysis_reanalyzes_when_neural_mode_changes(tmp_path, monkeypatch):
    """Two real cache loads must never cross the neural on/off boundary."""
    from conftest import analysis

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"synthetic placeholder; analysis is stubbed")
    cache_dir = tmp_path / "tracks"
    calls = []

    def fake_analysis(path):
        mode = "off" if cache._modular_feature_cache_safe() else "auto"
        calls.append(mode)
        result = analysis(audio, bpm=121 if mode == "off" else 125)
        result.analysis_version = cache.ANALYSIS_VERSION
        return result

    monkeypatch.setattr(cache, "analyze_track", fake_analysis)
    monkeypatch.setattr(cache, "_sync_feature_store", lambda *args, **kwargs: None)
    monkeypatch.setenv("DJAI_TEMPO_CONSISTENCY", "0")
    monkeypatch.setenv("DJAI_NEURAL_BAR_PHASE_TRIAL", "0")
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")

    old = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    old_again = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    assert old.analysis.bpm == 121
    assert not old.cache_hit and old_again.cache_hit
    assert calls == ["off"]

    # Changing the same audio to neural mode must trigger new analysis.
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "auto")
    new = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    new_again = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    assert new.analysis.bpm == 125
    assert not new.cache_hit and new_again.cache_hit
    assert calls == ["off", "auto"]

    # Rollback must select the original deterministic cache, not the neural one.
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    rollback = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    assert rollback.cache_hit and rollback.analysis.bpm == 121
    assert calls == ["off", "auto"]


def test_stem_storage_fingerprint_does_not_depend_on_neural_mode(tmp_path, monkeypatch):
    """Neural analysis cache isolation must not relocate existing stem assets."""
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"stable stem identity fixture")
    monkeypatch.setenv("DJAI_TEMPO_CONSISTENCY", "0")
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "off")
    off_stems = cache.fingerprint_file(
        audio, analysis_version="stems-v1", cache_schema_version="stems-v1"
    )
    off_analysis = cache.fingerprint_file(audio)
    monkeypatch.setenv("DJAI_NEURAL_BEATS", "auto")
    assert cache.fingerprint_file(
        audio, analysis_version="stems-v1", cache_schema_version="stems-v1"
    ) == off_stems
    assert cache.fingerprint_file(audio) != off_analysis
