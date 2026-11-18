"""M1 offline Beat This benchmark: strict TSV parsing, no grid changes."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "m1-beat-this-benchmark.py"
    spec = importlib.util.spec_from_file_location("m1_beat_this_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beat_this_tsv_reads_beat_numbers_and_downbeats(tmp_path):
    module = load()
    source = tmp_path / "demo.beats"
    source.write_text("0.5\t1\n1.0\t2\n1.5\t3\n2.0\t1\n", encoding="utf-8")
    result = module.read_beat_tsv(source)
    assert result["beats"] == [.5, 1., 1.5, 2.]
    assert result["downbeats"] == [.5, 2.]
    assert result["beat_numbers"] == [1, 2, 3, 1]


@pytest.mark.parametrize("text", [
    "0.2\t1\n0.2\t2\n", "0.6\t1\n0.4\t2\n",
    "nan\t1\n0.5\t2\n", "-1\t1\n0.5\t2\n",
    "0.2\t0\n0.5\t2\n", "0.2\t1\nfoo\t2\n",
    "0.2\t1\n0.5\n", "0.2\t1\n",
])
def test_invalid_external_tsv_fails_closed(tmp_path, text):
    module = load()
    source = tmp_path / "bad.beats"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        module.read_beat_tsv(source)


def test_one_to_one_matching_rejects_offbeat_shift():
    module = load()
    events = [
        {"status": "measured", "audio_attack_seconds": .5},
        {"status": "measured", "audio_attack_seconds": 1.},
        {"status": "measured", "audio_attack_seconds": 1.5},
    ]
    good = module.match_grid([.51, 1.01, 1.51], events)
    assert good["one_to_one_matched_count"] == 3
    assert good["f1_at_80ms"] == 1.
    wrong = module.match_grid([.75, 1.25, 1.75], events)
    assert wrong["one_to_one_matched_count"] == 0
    assert wrong["f1_at_80ms"] is None
    duplicate = module.match_grid([.49, .51, 1.02], events)
    assert duplicate["one_to_one_matched_count"] == 2


def test_outgoing_region_is_predeclared_and_separate_from_whole_track():
    module = load()
    assert module.MIDDLE_REGION == (
        "between_compressions", 9.636281, 58.305306
    )
    events = [
        {"status": "measured", "audio_attack_seconds": .5},
        {"status": "measured", "audio_attack_seconds": 12.},
    ]
    whole = module.match_grid([.5, 12.], events)
    middle = module.match_grid([.5, 12.], events, start=9.636281, end=58.305306)
    assert whole["one_to_one_matched_count"] == 2
    assert middle["one_to_one_matched_count"] == 1


def test_non_fixture_audio_is_rejected_before_inference(tmp_path):
    module = load()
    target = tmp_path / "stranger.wav"
    target.write_bytes(b"file")
    with pytest.raises(ValueError, match="only permits"):
        module.evaluate_track(target, tmp_path / "fake.beats", audit=None)


def test_upstream_output_parser_does_not_import_or_run_beat_this():
    module = load()
    assert not hasattr(module, "File2Beats")
    assert module.EVAL_TOLERANCE_MS == 80.0
    assert len(module.FIXTURES) == 2
