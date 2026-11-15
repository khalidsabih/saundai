"""Read-only paired deterministic/Beat This! beat and downbeat evaluation.

Manual timestamp labels are the only reference source. No generated grid,
BPM label or model output may silently become "ground truth". This script
never runs analyze_track, modifies playback, or writes audio/cue/grid data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from scripts.benchmark_beat_integrity_dataset import (
    match_events, read_timestamp_labels, valid_times,
)
from djai.intelligence.neural_grid_gate_audit import deterministic_fallback_grid
from djai.intelligence.tempo_reference_workflow import EXTENSIONS

BEAT_TOLERANCE_MS = 70.0
DOWNBEAT_TOLERANCE_MS = 100.0


def prepare_manifest(audio_root: Path, references: list[dict], output: Path) -> dict:
    """Describe requested audio and future label paths; never create labels."""
    from djai.intelligence.tempo_validation import validate_references

    references = validate_references(references)
    available: dict[str, list[Path]] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            available.setdefault(path.name, []).append(path)
    tracks = []
    for index, item in enumerate(references, 1):
        filename = item["filename"]
        matches = available.get(filename, [])
        if len(matches) != 1:
            raise ValueError(f"{filename!r}: expected one audio basename, found {len(matches)}")
        # Manual labels are intentionally absent: prepared names are destinations,
        # not assertions that annotations exist.
        tracks.append({
            "id": f"track-{index:02d}",
            "filename": filename,
            "audio": str(matches[0].resolve()),
            "beats": f"labels/track-{index:02d}.beats.txt",
            "downbeats": f"labels/track-{index:02d}.downbeats.txt",
            "split": "validation",
            "reference_quality": "unannotated_timestamps",
            "annotation_method": "",
        })
    manifest = {
        "schema": "saund-neural-beat-downbeat-manifest-v1",
        "dataset": "local-manual-annotations",
        "annotation_root": str(output.parent.resolve()),
        "tracks": tracks,
        "warning": "No beat or downbeat annotations have been created; fill files manually using independent listening/DAW inspection before scoring. Validation split is for development, not an untouched test set.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def _score(reference: list[float], predictions: list[float], tolerance_ms: float) -> dict:
    return match_events(reference, predictions, tolerance_ms=tolerance_ms)


def _average(rows: list[dict], key: str, model: str) -> dict:
    values = [r["results"][model][key] for r in rows
              if r.get("status") == "evaluated"
              and r["results"][model].get(key) is not None]
    return {
        "evaluated_tracks": len(values),
        "macro_f_measure": round(mean(v["f_measure"] for v in values), 5) if values else None,
        "mean_median_abs_error_ms": round(mean(v["median_abs_error_ms"] for v in values
            if v["median_abs_error_ms"] is not None), 3)
            if any(v["median_abs_error_ms"] is not None for v in values) else None,
    }


def evaluate(manifest: dict, manifest_root: Path, *, split: str = "validation") -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("An explicit validation or test split is required")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Manifest needs a tracks array")
    from djai.intelligence.neural_ears import predict_beats
    seen = set()
    results = []
    for item in tracks:
        if not isinstance(item, dict):
            raise ValueError("Invalid track entry")
        id_ = item.get("id")
        if not isinstance(id_, str) or not id_ or id_ in seen:
            raise ValueError("Track IDs must be unique and nonempty")
        seen.add(id_)
        if item.get("split") not in {"validation", "test"}:
            raise ValueError(f"{id_}: explicit validation/test split required")
        if item["split"] != split:
            continue
        row = {"id": id_, "filename": item.get("filename"), "status": "unavailable",
               "split": split}
        beats_path = manifest_root / str(item["beats"])
        downbeats_path = manifest_root / str(item["downbeats"])
        if not beats_path.is_file() or not downbeats_path.is_file() or not str(item.get("annotation_method", "")).strip():
            row["status"] = "missing_independent_annotations"
            row["missing"] = [name for name, exists in (
                ("beats", beats_path.is_file()), ("downbeats", downbeats_path.is_file()),
                ("annotation_method", bool(str(item.get("annotation_method", "")).strip())),
            ) if not exists]
            results.append(row)
            continue
        try:
            reference_beats, _ = read_timestamp_labels(beats_path)
            reference_downbeats, _ = read_timestamp_labels(downbeats_path)
            audio = Path(item["audio"])
            if not audio.is_file():
                raise ValueError("Audio file unavailable")
            baseline = deterministic_fallback_grid(audio)
            neural = predict_beats(
                audio, fallback_beats=baseline["beat_times"],
                fallback_downbeats=baseline["downbeat_times"],
            )
            if not neural.available:
                row["status"] = "neural_unavailable"
                row["reason"] = neural.reason
                results.append(row)
                continue
            row["results"] = {}
            for model, beat_values, downbeat_values in (
                ("deterministic", baseline["beat_times"], baseline["downbeat_times"]),
                ("neural", neural.beats_seconds, neural.downbeats_seconds),
            ):
                beats = valid_times(list(beat_values), allow_empty=True)
                downbeats = valid_times(list(downbeat_values), allow_empty=True)
                row["results"][model] = {
                    "beat": _score(reference_beats, beats, BEAT_TOLERANCE_MS),
                    "downbeat": _score(reference_downbeats, downbeats, DOWNBEAT_TOLERANCE_MS),
                }
            row["existing_grid_gate_would_pass"] = bool(neural.used)
            row["status"] = "evaluated"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["status"] = "unavailable_or_invalid_data"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
    evaluated = [r for r in results if r["status"] == "evaluated"]
    return {
        "schema": "saund-neural-beat-downbeat-paired-report-v1",
        "mode": "read_only",
        "selected_split": split,
        "selected_count": len(results),
        "evaluated_count": len(evaluated),
        "missing_annotation_count": sum(
            r["status"] == "missing_independent_annotations" for r in results),
        "deterministic": {
            key: _average(results, key, "deterministic")
            for key in ("beat", "downbeat")
        },
        "neural": {
            key: _average(results, key, "neural")
            for key in ("beat", "downbeat")
        },
        "tracks": results,
        "live_adoptions": 0,
        "ready_for_live_adoption": False,
        "limitations": [
            "Timestamp references are manually supplied; this tool does not independently certify annotator quality.",
            "70ms beat and 100ms downbeat windows are benchmarking tolerances, not live phase limits.",
            "Timestamp matching does not independently verify musical meter or transition quality.",
            "Do not tune the screen on the test split; use separately held-out labeled audio.",
            "No live analysis, grid replacement, cue or playback changes occur.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--audio-root", type=Path, required=True)
    prep.add_argument("--references", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = subs.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split", choices=("validation", "test"), default="validation")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    try:
        if args.command == "prepare":
            if not args.audio_root.is_dir():
                raise ValueError(f"Audio root unavailable: {args.audio_root}")
            references = json.loads(args.references.read_text(encoding="utf-8"))
            manifest = prepare_manifest(args.audio_root, references, args.output)
            summary = {"prepared": len(manifest["tracks"]), "manifest": str(args.output),
                       "annotations_created": 0}
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            report = evaluate(manifest, args.manifest.parent, split=args.split)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            summary = {
                "selected": report["selected_count"],
                "evaluated": report["evaluated_count"],
                "missing_annotations": report["missing_annotation_count"],
                "deterministic": report["deterministic"],
                "neural": report["neural"],
                "report": str(args.output),
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
