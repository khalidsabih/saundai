from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-transition-timing-shadow.py"
    spec = importlib.util.spec_from_file_location("m1_transition_timing_shadow", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rhythm(meter=4, bars=8, offset=0):
    beat_interval = .5
    beats = [round(offset + n * beat_interval, 6) for n in range(meter * bars)]
    downbeats = beats[::meter]
    return {
        "bpm": 120.0,
        "beats_seconds": beats,
        "downbeats_seconds": downbeats,
    }


def test_four_four_cue_window():
    m = load()
    r = rhythm()
    v = m.grid_health(r["bpm"], r["beats_seconds"], r["downbeats_seconds"], 4)
    assert v["passed"] and v["cue_window_seconds"] == 8


def test_three_four_cue_window():
    m = load()
    r = rhythm(meter=3)
    ref = {"beats_seconds": r["beats_seconds"], "downbeats_seconds": r["downbeats_seconds"]}
    assert m.meter_from_reference(ref) == 3
    v = m.grid_health(r["bpm"], r["beats_seconds"], r["downbeats_seconds"], 3)
    assert v["passed"] and v["cue_window_seconds"] == 6


def test_wrong_meter_rejected():
    m = load()
    r = rhythm(meter=3)
    v = m.grid_health(r["bpm"], r["beats_seconds"], r["downbeats_seconds"], 4)
    assert "irregular_bar_duration" in v["failures"]


def test_offbeat_downbeats_rejected():
    m = load()
    r = rhythm()
    downs = [t + .2 for t in r["downbeats_seconds"]]
    v = m.grid_health(r["bpm"], r["beats_seconds"], downs, 4)
    assert "offbeat_downbeats" in v["failures"]


def test_frozen_threshold_and_whole_grid_policy():
    m = load()
    selected = rhythm(offset=.1)
    candidate = dict(rhythm(), available=True, used=False, confidence=.5)
    grid, accepted = m.choose(selected, candidate)
    assert accepted
    assert grid is candidate
    candidate["confidence"] = .49
    grid, accepted = m.choose(selected, candidate)
    assert not accepted
    assert grid is selected


def test_evaluate_report_never_claims_rendered_mix():
    m = load()
    reference = rhythm(meter=3)
    selected = rhythm(meter=4)
    neural = dict(reference, available=True, used=False, confidence=.6)
    manifest = {"items": [{"id": "waltz", "genre": "Waltz",
                           "beats_seconds": reference["beats_seconds"],
                           "downbeats_seconds": reference["downbeats_seconds"]}]}
    result = m.evaluate(manifest, {"waltz": {"selected": selected, "candidate": neural}})
    assert result["metrics"]["challenger_grid_preflight_failures"] == 0
    assert result["metrics"]["selected_grid_preflight_failures"] == 1
    assert result["rendered_transitions_tested"] is False
    assert result["cross_domain_tested"] is False
