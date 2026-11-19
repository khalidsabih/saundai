from __future__ import annotations

import importlib.util
from pathlib import Path


def load():
    src = Path(__file__).resolve().parents[1] / "scripts" / "m1-ballroom-bar-interval-diagnostic.py"
    spec = importlib.util.spec_from_file_location("m1_bar_interval_diagnostic", src)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normal_three_four_bars_are_not_outliers():
    m = load()
    beats = [i * .5 for i in range(24)]
    downs = [i * 1.5 for i in range(8)]
    r = m.interval_diagnostic(beats, downs, 3, downs)
    assert r["outlier_count"] == 0
    assert r["predicted_bar_count"] == 7


def test_missing_downbeat_is_double_bar_outlier():
    m = load()
    beats = [i * .5 for i in range(32)]
    downs = [0, 2, 4, 8, 10, 12, 14]
    r = m.interval_diagnostic(beats, downs, 4, downs)
    assert r["outlier_count"] == 1
    assert r["outliers"][0]["index"] == 2
    assert r["outliers"][0]["bar_ratio"] == 2.0


def test_outlier_position_counts_track_edges():
    m = load()
    beats = [i * .5 for i in range(32)]
    downs = [0, 4, 6, 8, 10, 12, 14]
    r = m.interval_diagnostic(beats, downs, 4)
    assert r["outliers_in_first_two"] == 1
    assert r["outliers_in_last_two"] == 0


def test_reject_bad_timestamps():
    m = load()
    try:
        m.interval_diagnostic([0, .5, 1], [0, 0], 4)
    except ValueError as exc:
        assert "strictly increase" in str(exc)
    else:
        assert False, "invalid timestamps must fail closed"
