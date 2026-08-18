from __future__ import annotations

from pathlib import Path

import json

import soundfile as sf

from djai.demo import generate_demo_track
from djai.pipeline import run_pair_pipeline


def test_pair_pipeline_renders_preview_and_full_mix(tmp_path: Path):
    a = generate_demo_track(
        tmp_path / "a.wav",
        bpm=124.0,
        root_hz=55.0,
        minor_third_hz=65.406,
        fifth_hz=82.407,
        variant="a",
        bars=8,
    )
    b = generate_demo_track(
        tmp_path / "b.wav",
        bpm=126.0,
        root_hz=41.203,
        minor_third_hz=48.999,
        fifth_hz=61.735,
        variant="b",
        bars=8,
    )
    result = run_pair_pipeline(
        a,
        b,
        output_dir=tmp_path / "output",
        tempo_policy="safe",
        style="safe",
        force=True,
        use_cache=False,
        render_preview=True,
        render_full=True,
    )
    assert result.preview_path and result.preview_path.exists()
    assert result.full_mix_path and result.full_mix_path.exists()
    preview_info = sf.info(result.preview_path)
    full_info = sf.info(result.full_mix_path)
    assert preview_info.channels == 2
    assert full_info.channels == 2
    assert full_info.duration > result.plan.transition_seconds
    assert result.plan.quality_checked is True
    assert result.plan.measurement_passed is True
    assert result.plan.measurement is not None
    assert result.plan.measured_score == result.plan.measurement.overall


def test_set_pipeline_measures_final_junctions_only(tmp_path: Path):
    from djai.pipeline import run_set_pipeline

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for index, (bpm, root, third, fifth, variant) in enumerate(
        (
            (124.0, 55.0, 65.406, 82.407, "a"),
            (124.0, 41.203, 48.999, 61.735, "b"),
            (124.0, 49.0, 58.270, 73.416, "c"),
        )
    ):
        generate_demo_track(
            audio_dir / f"track-{index}.wav",
            bpm=bpm,
            root_hz=root,
            minor_third_hz=third,
            fifth_hz=fifth,
            variant=variant,
            bars=20,
        )

    measurement_log = tmp_path / "ratings" / "automatic-measurements.jsonl"
    result = run_set_pipeline(
        audio_dir,
        output_dir=tmp_path / "set-output",
        use_cache=False,
        minimum_body_seconds=8.0,
        minimum_body_fraction=0.10,
        render_previews=False,
        render_audio=False,
        analysis_workers=1,
        graph_workers=1,
        measure_transitions=True,
        measurement_top_k=3,
        measurement_log=str(measurement_log),
    )

    assert len(result.plan.transitions) == 2
    assert all(edge.plan.quality_checked for edge in result.plan.transitions)
    assert all(edge.plan.measurement_passed for edge in result.plan.transitions)
    assert all(edge.plan.measurement is not None for edge in result.plan.transitions)
    quality = result.plan.metadata["quality_loop"]
    assert quality["checked_transitions"] == 2
    assert quality["passed_transitions"] == 2

    records = [json.loads(line) for line in measurement_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= 2
    assert sum(bool(record["accepted"]) for record in records) == 2
