from __future__ import annotations

import json
from pathlib import Path

from djai.feedback import append_feedback_event
from djai.learning import (
    fit_scoring_model,
    learned_model_weight,
    load_scoring_model,
    maybe_refit_scoring_model,
)
from djai.personalization import get_listener_profile


def compatibility(value: float) -> dict[str, float]:
    names = (
        "tempo", "harmonic", "energy", "texture", "beat_grid", "phrase",
        "sections", "genre", "vocal_overlap", "bass_conflict", "boundary", "section_fit",
    )
    return {name: max(0.0, min(1.0, value + index * 0.002)) for index, name in enumerate(names)}


def write_records(path: Path, count: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            value = 0.2 + 0.7 * (index % 10) / 9.0
            record = {
                "track_a": f"a-{index % 10}",
                "track_b": f"b-{index % 10}",
                "plan": {
                    "compatibility": compatibility(value),
                    "genre_pair": ["house", "techno"],
                },
                "scores": {"overall": 1 + 4 * value},
            }
            handle.write(json.dumps(record) + "\n")


def test_scoring_model_reports_grouped_cross_validation(tmp_path: Path) -> None:
    ratings = tmp_path / "ratings.jsonl"
    model_path = tmp_path / "model.json"
    write_records(ratings, 60)
    fit_scoring_model(ratings, model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 60
    assert payload["group_count"] == 10
    assert payload["fold_count"] >= 2
    assert payload["cv_rmse"] >= 0.0
    assert payload["cv_baseline_rmse"] >= payload["cv_rmse"]
    assert 0.0 < payload["learned_weight"] <= 0.32
    loaded = load_scoring_model(model_path)
    assert loaded is not None
    assert learned_model_weight(loaded) == payload["learned_weight"]


def test_listener_profile_grows_from_implicit_feedback(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    profiles = tmp_path / "profiles.json"
    append_feedback_event(
        event="transition_like",
        listener="listener-a",
        value=1.0,
        journey="warmup",
        plan={"style": "creative", "technique": "filter_build"},
        feedback_path=feedback,
        profiles_path=profiles,
    )
    append_feedback_event(
        event="set_like",
        listener="listener-a",
        value=1.0,
        journey="warmup",
        feedback_path=feedback,
        profiles_path=profiles,
    )
    profile = get_listener_profile("listener-a", profiles)
    assert profile is not None
    assert profile.sample_count == 2
    assert profile.style_bias["creative"] > 0.0
    assert profile.technique_bias["filter_build"] > 0.0
    assert profile.journey_bias["warmup"] > 0.0


def test_automatic_refit_waits_for_new_evidence(tmp_path: Path, monkeypatch) -> None:
    ratings = tmp_path / "ratings.jsonl"
    measurements = tmp_path / "measurements.jsonl"
    model_path = tmp_path / "model.json"
    write_records(ratings, 12)
    measurements.write_text("", encoding="utf-8")
    monkeypatch.setenv("DJAI_AUTO_REFIT", "1")
    first = maybe_refit_scoring_model(
        ratings_path=ratings,
        measurement_path=measurements,
        output_path=model_path,
        minimum_new_samples=5,
    )
    assert first == model_path.resolve()
    assert model_path.exists()
    second = maybe_refit_scoring_model(
        ratings_path=ratings,
        measurement_path=measurements,
        output_path=model_path,
        minimum_new_samples=5,
    )
    assert second is None
