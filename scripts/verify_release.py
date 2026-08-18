from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from djai.live import iter_mp3_stream, live_manifest
from djai.set_planner import load_set_plan


def main() -> None:
    root = Path("output/demo-pipeline").resolve()
    plan_path = root / "set-plan.json"
    report_path = root / "full-set.json"
    preview_report_path = root / "previews" / "index.json"
    audio_path = root / "full-set.wav"
    measurement_log_path = root / "automatic-measurements.jsonl"
    for path in (plan_path, report_path, preview_report_path, audio_path, measurement_log_path):
        if not path.exists():
            raise SystemExit(f"Missing verification artifact: {path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    typed_plan = load_set_plan(plan_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    previews = json.loads(preview_report_path.read_text(encoding="utf-8"))
    full_audio, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
    info = sf.info(audio_path)
    render = report["render"]

    assert plan.get("automatic_delivery") is True
    assert plan.get("requires_confirmation") is False
    assert plan.get("listener") == "verification-listener"
    journey = plan.get("metadata", {}).get("journey_arc", {})
    assert journey.get("mode") == "balanced"
    assert len(journey.get("target_energy", [])) == len(plan.get("order", []))
    assert 0.0 <= float(journey.get("curve_fit_score", 0.0)) <= 1.0

    assert render["immutable_plan_execution"] is True
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    assert info.samplerate == sample_rate == 44_100
    assert abs(render["duration_seconds"] - plan["estimated_duration_seconds"]) < 0.001
    assert abs(info.duration - render["duration_seconds"]) < 1.0 / sample_rate + 1e-6
    assert float(np.max(np.abs(full_audio))) <= 0.986

    performance = plan.get("metadata", {}).get("performance", {})
    planning_performance = performance.get("planning", {})
    analysis_performance = performance.get("analysis", {})
    track_count = len(plan["ordered_tracks"])
    assert planning_performance.get("pair_contexts") == track_count * (track_count - 1)
    assert int(planning_performance.get("compatibility_evaluations", 0)) > 0
    assert int(analysis_performance.get("workers", 0)) == 1
    assert analysis_performance.get("parallel") is False

    transitions = [item for item in report["timeline"] if item["type"] == "transition"]
    plan_edges = plan["transitions"]
    preview_items = previews["previews"]
    assert len(transitions) == len(plan_edges) == len(preview_items)
    assert all(edge["plan"]["metadata"].get("compatibility_matrix_reused") for edge in plan_edges)
    assert all(edge["plan"]["metadata"].get("sequence_evidence_reused") for edge in plan_edges)

    quality_loop = plan.get("metadata", {}).get("quality_loop", {})
    assert quality_loop.get("enabled") is True
    assert quality_loop.get("checked_transitions") == len(plan_edges)
    assert quality_loop.get("passed_transitions") == len(plan_edges)
    assert int(quality_loop.get("top_k", 0)) == 3
    assert all(edge["plan"].get("quality_checked") is True for edge in plan_edges)
    assert all(edge["plan"].get("measurement_passed") is True for edge in plan_edges)
    assert all(edge["plan"].get("measurement") for edge in plan_edges)
    assert all(edge["plan"].get("auto_approved") is True for edge in plan_edges)
    assert all(edge["plan"].get("requires_confirmation") is False for edge in plan_edges)
    assert all(0.0 <= float(edge["plan"]["measured_score"]) <= 1.0 for edge in plan_edges)

    measurement_records = [
        json.loads(line)
        for line in measurement_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(measurement_records) >= len(plan_edges)
    assert sum(bool(record.get("accepted")) for record in measurement_records) == len(plan_edges)
    assert all(record.get("measurement") is not None for record in measurement_records)

    maximum_difference = 0.0
    for edge, timeline, preview_item in zip(plan_edges, transitions, preview_items, strict=True):
        transition_plan = edge["plan"]
        signature = transition_plan["execution_signature"]
        assert timeline["execution_signature"] == signature
        assert preview_item["execution_signature"] == signature

        preview_audio, preview_rate = sf.read(preview_item["output_wav"], always_2d=True, dtype="float32")
        assert preview_rate == sample_rate
        assert sf.info(preview_item["output_wav"]).subtype == "PCM_24"

        transition_frames = round(transition_plan["transition_seconds"] * sample_rate)
        preview_start = round(preview_item["transition_start_seconds"] * sample_rate)
        set_start = round(timeline["set_start"] * sample_rate)
        preview_segment = preview_audio[preview_start : preview_start + transition_frames]
        full_segment = full_audio[set_start : set_start + transition_frames]
        assert len(preview_segment) == len(full_segment) == transition_frames
        difference = float(np.max(np.abs(preview_segment - full_segment)))
        maximum_difference = max(maximum_difference, difference)
        assert difference <= 1.5e-5

    for previous, current in zip(report["timeline"], report["timeline"][1:]):
        assert abs(float(previous["set_end"]) - float(current["set_start"])) < 2e-5

    manifest = live_manifest(typed_plan)
    assert manifest["plan_id"] == plan["plan_id"]
    assert len(manifest["ordered_tracks"]) == len(plan["ordered_tracks"])
    assert manifest["timeline"]
    live_stream = iter_mp3_stream(typed_plan, bitrate="96k")
    live_bytes = bytearray()
    try:
        for chunk in live_stream:
            live_bytes.extend(chunk)
            if len(live_bytes) >= 512:
                break
        assert len(live_bytes) >= 512
        assert live_bytes.startswith(b"ID3") or live_bytes[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}
    finally:
        live_stream.close()

    print("SAUND.AI release verification passed")
    print(f"Tracks: {len(plan['ordered_tracks'])}")
    print(f"Transitions: {len(plan_edges)}")
    print(f"Duration: {render['duration_seconds']:.6f}s")
    print(f"Peak: {render['peak']:.6f}")
    print(f"Timeline segments: {len(report['timeline'])}")
    print(f"Analysis workers: {analysis_performance['workers']}")
    print(f"Pair evidence matrices: {planning_performance['pair_contexts']}")
    print(f"Compatibility evaluations: {planning_performance['compatibility_evaluations']}")
    print(f"Measured candidates: {len(measurement_records)}")
    accepted_count = sum(bool(record.get("accepted")) for record in measurement_records)
    print(f"Accepted measured transitions: {accepted_count}")
    print(f"Preview/full maximum difference: {maximum_difference:.8f}")
    print(f"Journey curve fit: {float(journey['curve_fit_score']):.6f}")
    print(f"Live MP3 verification bytes: {len(live_bytes)}")


if __name__ == "__main__":
    main()
