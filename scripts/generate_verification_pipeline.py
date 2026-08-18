#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

from djai.demo import generate_demo_track
from djai.set_planner import plan_folder, save_set_plan
from djai.set_renderer import render_set, render_set_previews


def main() -> None:
    """Generate a compact, no-stretch set for deterministic release verification."""
    root = Path("output/demo-pipeline").resolve()
    shutil.rmtree(root, ignore_errors=True)
    os.environ["DJAI_SCORING_MODEL"] = str(root / "verification-scoring-model.json")
    os.environ["DJAI_AUTO_REFIT"] = "0"
    os.environ["DJAI_AUTO_STEMS"] = "off"
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    # Equal native tempos keep this verification focused on the immutable-plan,
    # preview-identity, timeline and audio-format contracts instead of spending
    # most of CI time in long-body phase-vocoder stretching.
    specifications = (
        ("01_opening_124bpm_Aminor.wav", 55.0, 65.406, 82.407, "a"),
        ("02_groove_124bpm_Eminor.wav", 41.203, 48.999, 61.735, "b"),
        ("03_close_124bpm_Bminor.wav", 61.735, 73.416, 92.499, "a"),
    )
    for filename, root_hz, third_hz, fifth_hz, variant in specifications:
        generate_demo_track(
            audio_root / filename,
            bpm=124.0,
            root_hz=root_hz,
            minor_third_hz=third_hz,
            fifth_hz=fifth_hz,
            variant=variant,
            bars=20,
        )

    plan = plan_folder(
        audio_root,
        use_cache=False,
        tempo_policy="safe",
        preferred_style="auto",
        journey="balanced",
        start_track="01_opening_124bpm_Aminor",
        end_track="03_close_124bpm_Bminor",
        minimum_body_seconds=8.0,
        minimum_body_fraction=0.15,
        analysis_workers=1,
        graph_workers=1,
        measure_transitions=True,
        measurement_top_k=3,
        measurement_log=str(root / "automatic-measurements.jsonl"),
        automatic_delivery=True,
        listener="verification-listener",
    )
    plan_path = save_set_plan(plan, root / "set-plan.json")
    preview_paths, preview_report = render_set_previews(
        plan,
        root / "previews",
        pre_roll_seconds=1.0,
        post_roll_seconds=1.0,
    )
    output, report = render_set(plan, root / "full-set.wav")

    print(f"Verification plan: {plan_path}")
    print(f"Verification previews: {preview_report} ({len(preview_paths)} files)")
    print(f"Verification set: {output}")
    print(f"Verification report: {report}")
    print(f"Verification measurements: {root / 'automatic-measurements.jsonl'}")


if __name__ == "__main__":
    main()
