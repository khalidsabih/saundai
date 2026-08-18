from __future__ import annotations

from pathlib import Path

from djai import cache
from djai.serialization import track_analysis_from_dict


def test_nested_local_key_round_trip(track_pair):
    track, _ = track_pair
    restored = track_analysis_from_dict(track.to_dict())
    assert restored.entry_candidates[0].local_key is not None
    assert restored.entry_candidates[0].local_key.camelot == track.entry_candidates[0].local_key.camelot
    assert restored.entry_candidates[0].local_chroma == track.entry_candidates[0].local_chroma


def test_cache_is_versioned_and_round_trips_local_keys(tmp_path: Path, monkeypatch, track_pair):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"synthetic-audio-bytes")
    fixture, _ = track_pair
    fixture.path = str(audio.resolve())
    fixture.analysis_version = cache.ANALYSIS_VERSION
    calls = {"count": 0}

    def fake_analyze(path):
        calls["count"] += 1
        fixture.path = str(Path(path).resolve())
        fixture.analysis_version = cache.ANALYSIS_VERSION
        return fixture

    monkeypatch.setattr(cache, "analyze_track", fake_analyze)
    root = tmp_path / "cache"
    first = cache.analyze_track_cached(audio, cache_dir=root)
    second = cache.analyze_track_cached(audio, cache_dir=root)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls["count"] == 1
    assert second.analysis.exit_candidates[0].local_key is not None

    monkeypatch.setattr(cache, "ANALYSIS_VERSION", "1.2.1-test")
    third = cache.analyze_track_cached(audio, cache_dir=root)
    assert third.cache_hit is False
    assert calls["count"] == 2
    assert third.cache_path != first.cache_path
