from __future__ import annotations

import gzip
import json

from djai.training.real_retrieval import _assign_crate_splits, import_aotm_sequences


def test_import_aotm_playlist_json_preserves_contiguous_order(tmp_path):
    source = tmp_path / "aotm2011_playlists.json.gz"
    playlists = [
        {
            "mix_id": 42,
            "category": "Dance-House",
            "filtered_lists": [
                ["SOA", "SOB", "SOC"],
                ["SOD"],
                ["SOE", "SOF"],
            ],
        }
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        json.dump(playlists, handle)

    output = tmp_path / "aotm-sequences.jsonl"
    report = import_aotm_sequences(source, output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["rows"] == 3
    assert [(row["source_track_id"], row["target_track_id"]) for row in rows] == [
        ("SOA", "SOB"),
        ("SOB", "SOC"),
        ("SOE", "SOF"),
    ]
    assert all(row["human_ordered_positive"] is True for row in rows)
    assert all(row["category"] == "Dance-House" for row in rows)
    assert report["policy"]["technical_safety_signal"] is False
    assert report["policy"]["production_authority"] is False


def test_crate_split_assignment_never_leaks_same_crate():
    rows = [
        {"group": "a1", "split_unit": "crate-a", "split": "train"},
        {"group": "a2", "split_unit": "crate-a", "split": "validation"},
        {"group": "b1", "split_unit": "crate-b", "split": "test"},
        {"group": "c1", "split_unit": "crate-c", "split": "train"},
    ]

    assignment = _assign_crate_splits(rows)

    assert set(assignment) == {"crate-a", "crate-b", "crate-c"}
    by_crate = {}
    for row in rows:
        by_crate.setdefault(row["split_unit"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_crate.values())
    assert set(assignment.values()) == {"train", "validation", "test"}
