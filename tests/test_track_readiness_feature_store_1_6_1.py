from __future__ import annotations

import shutil
from pathlib import Path

from djai import cache
from djai.feature_store import (
    DJ_READY_FEATURES,
    FEATURE_VERSIONS,
    FeatureStore,
    load_analysis_from_features,
    persist_analysis_features,
)


def test_source_fingerprint_follows_audio_content_not_path(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "renamed.wav"
    first.write_bytes(b"same-audio-content")
    shutil.copy2(first, second)
    store = FeatureStore(tmp_path / "features")
    assert store.source_fingerprint(first) == store.source_fingerprint(second)


def test_feature_versions_invalidate_independently(tmp_path: Path, track_pair) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"feature-store-audio")
    analysis, _ = track_pair
    analysis.path = str(audio.resolve())
    analysis.analysis_version = cache.ANALYSIS_VERSION
    store = FeatureStore(tmp_path / "features")

    written = persist_analysis_features(analysis, store=store)
    assert set(DJ_READY_FEATURES).issubset(set(written))
    assert store.has(audio, "embeddings")
    assert store.has(audio, "timeline")

    assert store.delete_feature_version(audio, "timeline", FEATURE_VERSIONS["timeline"])
    assert not store.has(audio, "timeline")
    # Timeline invalidation must not evict expensive unrelated intelligence.
    assert store.has(audio, "embeddings")
    assert store.has(audio, "rhythm")
    assert store.has(audio, "structure")


def test_feature_store_reconstructs_analysis_without_legacy_blob(
    tmp_path: Path,
    monkeypatch,
    track_pair,
) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"reconstructable-audio")
    fixture, _ = track_pair
    fixture.path = str(audio.resolve())
    fixture.analysis_version = cache.ANALYSIS_VERSION
    calls = {"count": 0}

    def fake_analyze(path):
        calls["count"] += 1
        fixture.path = str(Path(path).resolve())
        fixture.analysis_version = cache.ANALYSIS_VERSION
        return fixture

    cache_dir = tmp_path / "tracks"
    monkeypatch.setattr(cache, "analyze_track", fake_analyze)
    first = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    assert first.cache_hit is False
    assert calls["count"] == 1

    first.cache_path.unlink()

    def fail_analyze(_path):
        raise AssertionError("raw audio analysis should not run when modular features are complete")

    monkeypatch.setattr(cache, "analyze_track", fail_analyze)
    second = cache.analyze_track_cached(audio, cache_dir=cache_dir)
    assert second.cache_hit is True
    assert second.cache_source == "feature_store"
    assert second.analysis.bpm == fixture.bpm
    assert second.analysis.entry_candidates[0].local_key is not None
    assert second.cache_path.exists()  # rehydrates the compatibility cache


def test_readiness_becomes_dj_ready_when_core_features_exist(tmp_path: Path, track_pair) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"ready-audio")
    analysis, _ = track_pair
    analysis.path = str(audio.resolve())
    analysis.analysis_version = cache.ANALYSIS_VERSION
    store = FeatureStore(tmp_path / "features")

    before = store.track_readiness(audio, stem_status="disabled", stem_required=False)
    assert before["dj_ready"] is False
    assert before["state"] == "pending"

    persist_analysis_features(analysis, store=store)
    after = store.track_readiness(audio, stem_status="disabled", stem_required=False)
    assert after["dj_ready"] is True
    assert after["state"] == "ready"
    assert after["progress"] == 1.0
    assert all(after["features"][name]["available"] for name in DJ_READY_FEATURES)


def test_analysis_contract_change_prevents_stale_reconstruction(tmp_path: Path, track_pair) -> None:
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"contract-audio")
    analysis, _ = track_pair
    analysis.path = str(audio.resolve())
    analysis.analysis_version = cache.ANALYSIS_VERSION
    store = FeatureStore(tmp_path / "features")
    persist_analysis_features(analysis, store=store)

    assert load_analysis_from_features(
        audio,
        store=store,
        expected_analysis_version="future-analysis-contract",
    ) is None
