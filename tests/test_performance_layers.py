from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import analysis
from djai.audio import load_audio
from djai.decision import validate_execution_plan
from djai.layers import (
    classify_layer_filename,
    find_performance_layers,
    register_performance_layers,
)
from djai.planner import rank_transition_candidates
from djai.renderer import _render_transition_window


def _write_bundle(root: Path, name: str, *, sr: int = 22_050, duration: float = 16.0) -> tuple[Path, Path, Path]:
    frames = int(sr * duration)
    t = np.arange(frames, dtype=np.float32) / sr
    # A stable instrumental plus a phrase-shaped vocal source. The full mix is
    # deliberately not sample-identical to either alternate master.
    instrumental = (
        0.16 * np.sin(2 * np.pi * 110.0 * t)
        + 0.06 * np.sign(np.sin(2 * np.pi * 2.0 * t))
    ).astype(np.float32)
    vocal_gate = (np.sin(2 * np.pi * 0.25 * t) > -0.15).astype(np.float32)
    vocals = (0.09 * np.sin(2 * np.pi * 330.0 * t) * vocal_gate).astype(np.float32)
    full = np.clip(instrumental + vocals, -0.95, 0.95).astype(np.float32)
    full_path = root / f"{name}.wav"
    inst_path = root / f"{name} Instrumental.wav"
    vocal_path = root / f"{name} Acapella.wav"
    sf.write(full_path, full, sr)
    sf.write(inst_path, instrumental, sr)
    sf.write(vocal_path, vocals, sr)
    return full_path, inst_path, vocal_path


def test_supplied_three_part_bundle_is_registered_as_one_track(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    full, instrumental, vocals = _write_bundle(tmp_path, "Solar Skin")
    registry = tmp_path / "layers.json"

    bundles = register_performance_layers(
        [full, instrumental, vocals],
        registry_path=registry,
        validate=True,
    )

    assert classify_layer_filename(full) == ("solar skin", "full")
    assert classify_layer_filename(instrumental) == ("solar skin", "instrumental")
    assert classify_layer_filename(vocals) == ("solar skin", "vocals")
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.complete is True
    assert bundle.verified is True
    assert bundle.source_quality == "supplied_original"
    assert Path(bundle.instrumental.path) == instrumental.resolve()  # type: ignore[union-attr]
    assert Path(bundle.vocals.path) == vocals.resolve()  # type: ignore[union-attr]


def test_layer_aware_candidates_render_real_audio(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a_full, a_inst, a_vocal = _write_bundle(tmp_path, "Tussy")
    b_full, b_inst, b_vocal = _write_bundle(tmp_path, "Solar Skin")
    register_performance_layers([a_full, a_inst, a_vocal, b_full, b_inst, b_vocal])

    track_a = analysis(a_full, bpm=120.0, camelot="8A", energy=0.58, exit_section="breakdown")
    track_b = analysis(b_full, bpm=120.0, camelot="8A", energy=0.62, entry_section="groove")
    track_a.duration_seconds = 16.0
    track_b.duration_seconds = 16.0
    for candidate in track_a.exit_candidates:
        candidate.time_seconds = 6.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.72
    for candidate in track_b.entry_candidates:
        candidate.time_seconds = 1.0
        candidate.available_bars = 4
        candidate.vocal_probability = 0.65

    bundle_a = find_performance_layers(a_full)
    bundle_b = find_performance_layers(b_full)
    assert bundle_a and bundle_a.complete
    assert bundle_b and bundle_b.complete

    plans = rank_transition_candidates(
        track_a,
        track_b,
        styles=("safe", "creative"),
        force=True,
        requested_bars=2,
        limit_per_style=4,
    )
    layer_plans = [
        plan for plan in plans
        if plan.technique in {"layer_vocal_bridge", "layer_instrumental_handoff"}
    ]
    assert {plan.technique for plan in layer_plans} == {
        "layer_vocal_bridge",
        "layer_instrumental_handoff",
    }

    selected = next(plan for plan in layer_plans if plan.technique == "layer_vocal_bridge")
    validate_execution_plan(selected)
    audio_a, _ = load_audio(a_full, sample_rate=22_050, mono=False)
    audio_b, _ = load_audio(b_full, sample_rate=22_050, mono=False)
    mixed, consumed_a, consumed_b, _ = _render_transition_window(
        audio_a,
        audio_b,
        selected,
        sample_rate=22_050,
    )
    assert mixed.shape == (2, int(round(selected.transition_seconds * 22_050)))
    assert np.isfinite(mixed).all()
    assert float(np.max(np.abs(mixed))) > 0.01
    assert consumed_a > 0.0
    assert consumed_b > 0.0
    assert selected.performance_program["annotations"]["source_priority"] == "supplied_original"


def test_api_upload_groups_three_part_bundle_into_one_library_track(tmp_path: Path, monkeypatch) -> None:
    import importlib
    import sys

    from fastapi.testclient import TestClient

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    full, instrumental, vocals = _write_bundle(source_dir, "Tussy")
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("DJAI_WORKSPACE", str(workspace))
    monkeypatch.setenv("DJAI_AUTO_STEMS", "off")
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)

    handles = [path.open("rb") for path in (full, instrumental, vocals)]
    try:
        response = client.post(
            "/api/tracks/upload",
            files=[
                ("files", (path.name, handle, "audio/wav"))
                for path, handle in zip((full, instrumental, vocals), handles)
            ],
        )
    finally:
        for handle in handles:
            handle.close()

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["files"]) == 1
    assert len(payload["auxiliary_files"]) == 2
    track = payload["files"][0]
    assert track["name"] == "Tussy.wav"
    assert track["performance_layers"]["complete"] is True
    assert track["stem_status"] == "supplied_layers"

    listed = client.get("/api/tracks", params={"folder": "audio", "recursive": True})
    assert listed.status_code == 200
    tracks = listed.json()["tracks"]
    assert len(tracks) == 1
    assert tracks[0]["performance_layers"]["complete"] is True


def test_folder_discovery_collapses_alternate_masters(tmp_path: Path, monkeypatch) -> None:
    from djai.cache import discover_tracks

    monkeypatch.chdir(tmp_path)
    full, instrumental, vocals = _write_bundle(tmp_path, "Bundle Song")
    other = tmp_path / "Other Song.wav"
    sf.write(other, np.zeros(22_050, dtype=np.float32), 22_050)

    tracks = discover_tracks(tmp_path)
    assert tracks == sorted([full.resolve(), other.resolve()], key=lambda path: path.name.casefold())
    bundle = find_performance_layers(full)
    assert bundle is not None and bundle.complete


def test_mismatched_alternate_master_is_not_hidden(tmp_path: Path, monkeypatch) -> None:
    from djai.cache import discover_tracks

    monkeypatch.chdir(tmp_path)
    full, instrumental, vocals = _write_bundle(tmp_path, "Mismatch Song", duration=16.0)
    # Make the supposed acapella a materially different edit. It must remain a
    # separate library item rather than being silently folded into the master.
    sf.write(vocals, np.zeros(int(22_050 * 8.0), dtype=np.float32), 22_050)

    tracks = discover_tracks(tmp_path)
    assert full.resolve() in tracks
    assert instrumental.resolve() in tracks
    assert vocals.resolve() in tracks


def test_supplied_acapella_enriches_local_vocal_evidence(tmp_path: Path, monkeypatch) -> None:
    from djai.layers import enrich_analysis_with_performance_layers

    monkeypatch.chdir(tmp_path)
    full, instrumental, vocals = _write_bundle(tmp_path, "Vocal Truth")
    assert classify_layer_filename(full) == ("vocal truth", "full")
    register_performance_layers([full, instrumental, vocals])
    track = analysis(full, bpm=120.0, camelot="8A", energy=0.55)
    track.duration_seconds = 16.0
    for candidate in [*track.entry_candidates, *track.exit_candidates]:
        candidate.time_seconds = 2.0
        candidate.available_bars = 2
        candidate.vocal_probability = 0.0

    enrich_analysis_with_performance_layers(track)

    assert track.performance_layer_analysis["source_quality"] == "supplied_original"
    assert track.performance_layer_analysis["vocal_activity_fraction"] > 0.0
    assert any(candidate.vocal_probability > 0.0 for candidate in track.entry_candidates)
