from __future__ import annotations

import json
from pathlib import Path

from djai.intelligence.cue_detr import CuePrediction, rerank_analysis_candidates_with_cue_model
from djai.intelligence.edm_cue import edm_cue_stats, import_edm_cue_annotations, normalize_edm_cue_record
from djai.models import GenreEstimate, KeyEstimate, TrackAnalysis, TransitionCandidate


def _analysis() -> TrackAnalysis:
    key = KeyEstimate("C", 0, "minor", "5A", 0.8)
    base = dict(
        local_energy=0.5,
        stability=0.7,
        available_bars=16,
        boundary_strength=0.6,
        phrase_strength=0.4,
        beat_grid_confidence=0.9,
        local_bpm=120.0,
    )
    return TrackAnalysis(
        path="track.wav",
        duration_seconds=240.0,
        sample_rate=44100,
        bpm=120.0,
        bpm_confidence=0.9,
        beats_seconds=[],
        downbeats_seconds=[],
        key=key,
        loudness_dbfs=-12.0,
        energy=0.6,
        brightness=0.5,
        percussive_ratio=0.7,
        genre=GenreEstimate(),
        entry_candidates=[
            TransitionCandidate("entry", 16.0, 8, 0.40, section_label="intro", **base),
            TransitionCandidate("entry", 64.0, 32, 0.62, section_label="groove", **base),
        ],
        exit_candidates=[
            TransitionCandidate("exit", 160.0, 80, 0.50, section_label="outro", **base),
            TransitionCandidate("exit", 192.0, 96, 0.63, section_label="outro", **base),
        ],
    )


def test_edm_cue_normalizer_handles_common_export_shapes():
    record = normalize_edm_cue_record(
        {
            "artist_name": "Expert Artist",
            "track_name": "Expert Track",
            "deezer_id": "1234",
            "tempo": "128",
            "grid_offset": "0.125",
            "cue_points": "[16.0, 48.0, 96.0]",
        }
    )
    assert record.deezer_id == "1234"
    assert record.bpm == 128.0
    assert record.cue_points_seconds == [16.0, 48.0, 96.0]
    assert record.split in {"train", "validation", "test"}


def test_edm_cue_import_and_stats(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            [
                {"id": "a", "tempo": 125, "cue_points": [8, 24, 40]},
                {"id": "b", "tempo": 130, "cue_points": [16, 32]},
                {"id": "empty", "tempo": 120, "cue_points": []},
            ]
        )
    )
    output = import_edm_cue_annotations(source, tmp_path / "annotations.jsonl")
    stats = edm_cue_stats(output)
    assert stats["tracks"] == 2
    assert stats["cue_points"] == 5
    assert stats["bpm_min"] == 125.0
    assert stats["bpm_max"] == 130.0


def test_cue_model_evidence_reranks_existing_safe_candidates():
    analysis = _analysis()
    predictions = [CuePrediction(time_seconds=16.1, confidence=0.99), CuePrediction(time_seconds=160.1, confidence=0.98)]
    rerank_analysis_candidates_with_cue_model(analysis, predictions, learned_weight=0.60)
    assert analysis.entry_candidates[0].time_seconds == 16.0
    assert analysis.exit_candidates[0].time_seconds == 160.0
    assert analysis.entry_candidates[0].score > analysis.entry_candidates[1].score
    assert analysis.exit_candidates[0].score > analysis.exit_candidates[1].score


def test_official_cue_window_geometry_and_score_scaling():
    import numpy as np
    from djai.intelligence.cue_detr import (
        CUE_DETR_N_MELS,
        CUE_DETR_WINDOW_FRAMES,
        _scale_scores,
        _sliding_windows,
    )

    image = np.zeros((CUE_DETR_N_MELS, 900, 3), dtype=np.uint8)
    windows, borders = _sliding_windows(image)
    assert windows
    assert len(windows) == len(borders)
    assert borders[0] == -266
    assert all(window.shape == (CUE_DETR_N_MELS, CUE_DETR_WINDOW_FRAMES, 3) for window in windows)
    scaled = _scale_scores([2.0, 4.0, 6.0])
    assert np.allclose(scaled, [0.0, 0.5, 1.0])


def test_official_cue_fingerprint_changed_from_provisional_runtime():
    from djai.intelligence.cue_detr import cue_model_fingerprint

    assert len(cue_model_fingerprint()) == 20
