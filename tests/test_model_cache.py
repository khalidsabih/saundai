from __future__ import annotations

import json
from pathlib import Path

from djai import genre, learning


def test_scoring_model_file_is_parsed_once_per_version(tmp_path: Path):
    model = tmp_path / "scoring.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": list(learning.SCORING_FEATURES),
                "weights": [0.0] * len(learning.SCORING_FEATURES),
                "intercept": 0.5,
                "sample_count": 20,
            }
        ),
        encoding="utf-8",
    )
    learning.clear_scoring_model_cache()
    first = learning.load_scoring_model(model)
    info_after_first = learning._load_scoring_model_cached.cache_info()
    second = learning.load_scoring_model(model)
    info_after_second = learning._load_scoring_model_cached.cache_info()

    assert first == second
    assert info_after_first.misses == 1
    assert info_after_second.hits == 1


def test_genre_model_file_is_parsed_once_per_version(tmp_path: Path):
    model = tmp_path / "genre.json"
    model.write_text(
        json.dumps(
            {
                "centroids": {"house": genre.DEFAULT_CENTROIDS["house"]},
                "scales": genre.DEFAULT_SCALES,
                "model": "test-centroid",
            }
        ),
        encoding="utf-8",
    )
    genre.clear_genre_model_cache()
    first = genre._load_model(model)
    info_after_first = genre._load_model_cached.cache_info()
    second = genre._load_model(model)
    info_after_second = genre._load_model_cached.cache_info()

    assert first == second
    assert info_after_first.misses == 1
    assert info_after_second.hits == 1
