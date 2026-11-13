import ast
import inspect

import numpy as np

from djai.music.tempo_consistency import _acf_tempo, reconcile_bpm


def _pulse_track(bpm: float, *, seconds: float = 180.0, fps: float = 43.06640625) -> np.ndarray:
    n = int(round(seconds * fps))
    x = np.zeros(n, dtype=float)
    for t in np.arange(0.0, seconds - 0.5, 60.0 / bpm):
        i = round(t * fps)
        if i < n:
            x[i] = 1.0
    return x


def test_reconciles_when_three_audio_windows_and_beat_span_agree():
    fps = 22050 / 512
    envelope = _pulse_track(120.0, fps=fps)
    beats = np.arange(0, 177, 0.5, dtype=float)
    selected, trace = reconcile_bpm(
        117.4538, beats, envelope, sample_rate=22050, hop_length=512,
    )
    assert trace["decision"] == "reconcile"
    assert abs(selected - 120.0) < 0.4
    assert len(trace["audio_windows"]) == 3


def test_retains_when_audio_disagrees_with_grid():
    fps = 22050 / 512
    envelope = _pulse_track(125.0, fps=fps)
    beats = np.arange(0, 177, 0.5, dtype=float)
    selected, trace = reconcile_bpm(
        117.4538, beats, envelope, sample_rate=22050, hop_length=512,
    )
    assert selected == 117.4538
    assert trace["decision"] == "retain"


def test_retains_when_audio_flat():
    selected, trace = reconcile_bpm(
        117.4538, np.arange(0, 177, 0.5, dtype=float),
        np.zeros(9000), sample_rate=22050, hop_length=512,
    )
    assert selected == 117.4538
    assert trace["reason"] == "insufficient_independent_audio_confidence"


def test_retains_when_grid_matches_global():
    selected, trace = reconcile_bpm(
        120.0, np.arange(0, 177, 0.5, dtype=float),
        _pulse_track(120.0), sample_rate=22050, hop_length=512,
    )
    assert selected == 120.0
    assert trace["reason"] == "no_material_global_disagreement"


def test_acf_flat_has_no_candidate():
    assert _acf_tempo(np.zeros(2000), frames_per_second=43.0, low=105, high=140) is None


def test_analyze_track_does_not_shadow_os_after_tempo_gate():
    from djai.analysis import analyze_track
    body = ast.parse(inspect.getsource(analyze_track)).body[0]
    assert all(
        not (isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names))
        and not (isinstance(node, ast.ImportFrom) and node.module == "os")
        for node in ast.walk(body)
    )
