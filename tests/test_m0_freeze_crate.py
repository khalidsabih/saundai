from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_tool():
    source = Path(__file__).resolve().parents[1] / "scripts" / "m0_freeze_crate.py"
    spec = importlib.util.spec_from_file_location("m0_freeze_crate", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_hashes_audio_without_modifying_content(tmp_path):
    tool = load_tool()
    root = tmp_path / "audio"
    root.mkdir()
    a = root / "house.wav"
    a.write_bytes(b"test-music-1")
    folder = root / "crate"
    folder.mkdir()
    b = folder / "pop.MP3"
    b.write_bytes(b"test-music-2")
    before = [a.read_bytes(), b.read_bytes()]
    payload = tool.inventory(root)
    assert len(payload["tracks"]) == 2
    assert set(payload["scenarios"][i]["id"] for i in range(12)) == set(tool.SCENARIOS)
    assert all(item["genre"] is None and item["rights_status"] == "unreviewed" for item in payload["tracks"])
    assert [a.read_bytes(), b.read_bytes()] == before
    assert payload["freeze_status"] == "draft"
    errors, stats = tool.check(payload, strict=False)
    assert errors == []
    assert stats["unassigned_scenarios"] == 12
    strict_errors, _ = tool.check(payload, strict=True)
    assert "both_development_and_test_tracks_required" in strict_errors


def test_validate_detects_unapproved_rights_and_split_leakage(tmp_path):
    tool = load_tool()
    root = tmp_path / "audio"
    root.mkdir()
    (root / "track-1.wav").write_bytes(b"one")
    (root / "track-2.wav").write_bytes(b"two")
    payload = tool.inventory(root)
    tracks = payload["tracks"]
    for track in tracks:
        track.update(artist="same artist", genre="house", recording_group="same recording")
    tracks[0]["split"] = "development"
    tracks[1]["split"] = "test"
    payload["scenarios"][0].update(
        status="ready",
        track_ids_in_order=[tracks[0]["id"], tracks[1]["id"]],
    )
    errors, _ = tool.check(payload, strict=True)
    assert any(s.startswith("development_test_leakage:artist:") for s in errors)
    assert any(s.startswith("unapproved_rights:") for s in errors)
    assert any(s.startswith("mixed_split_in_scenario:") for s in errors)


def test_unavailable_scenario_does_not_imply_coverage(tmp_path):
    tool = load_tool()
    root = tmp_path / "audio"
    root.mkdir()
    (root / "one.wav").write_bytes(b"one")
    (root / "two.wav").write_bytes(b"two")
    payload = tool.inventory(root)
    payload["scenarios"][0]["status"] = "unavailable"
    errors, stats = tool.check(payload, strict=False)
    assert errors == []
    assert stats["scenario_unavailable"] == 1
    assert stats["scenario_ready"] == 0
