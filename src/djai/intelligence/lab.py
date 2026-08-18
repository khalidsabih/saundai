from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any

from ..cache import analyze_track_cached
from ..brain.candidate_generator import CandidateGenerationConfig, generate_transition_ideas
from ..brain.dj_move import build_dj_state, score_transition_plan_as_move
from ..brain.set_state import build_set_state
from ..brain.professional import ProfessionalBrainConfig, rehearse_candidate_pool
from ..renderer import render_transition
from .preferences import plan_features
from ..training.state_move_taste import (
    move_taste_payload,
    shadow_state_move_prediction,
    state_context_id,
    state_taste_payload,
)


def create_listening_session(
    track_a: str | Path,
    track_b: str | Path,
    *,
    output_root: str | Path = "output/listening-lab",
    candidates: int = 4,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Create anonymized transition previews for blind A/B preference collection.

    Taste Lab deliberately reuses the same immutable track-analysis cache as the
    normal set planner. A blind-listening request should not decode and analyze
    two full songs again merely because the user moved from Create Set to Lab.
    """
    started = time.perf_counter()
    analysis_started = started
    cached_a = analyze_track_cached(track_a, cache_dir=cache_dir, use_cache=use_cache)
    cached_b = analyze_track_cached(track_b, cache_dir=cache_dir, use_cache=use_cache)
    analysis_a = cached_a.analysis
    analysis_b = cached_b.analysis
    analysis_seconds = time.perf_counter() - analysis_started
    candidate_count = max(2, min(8, int(candidates)))
    generation_started = time.perf_counter()
    ideas = generate_transition_ideas(
        analysis_a,
        analysis_b,
        config=CandidateGenerationConfig(
            analytical_pool=max(32, candidate_count * 8),
            finalists=candidate_count,
            diversity_weight=0.42,
            minimum_analytical_score=0.10,
        ),
    )
    generation_seconds = time.perf_counter() - generation_started
    if len(ideas) < 2:
        raise RuntimeError("At least two feasible candidates are required for a listening session.")
    rehearsal_started = time.perf_counter()
    brain = rehearse_candidate_pool(
        analysis_a,
        analysis_b,
        ideas,
        config=ProfessionalBrainConfig.from_environment(
            finalists=len(ideas),
            minimum_judge_score=0.0,
            measurement_minimum=0.0,
        ),
        measurement_log_path="ratings/automatic-measurements.jsonl",
        context={"mode": "taste_lab", "blind": True},
    )
    rehearsal_seconds = time.perf_counter() - rehearsal_started
    pool = [plan for plan in brain.rehearsed if not (plan.measurement and plan.measurement.hard_failures)]
    if len(pool) < 2:
        raise RuntimeError("At least two technically valid rehearsed candidates are required for a listening session.")
    pool = pool[:candidate_count]

    # SAUND.AI 1.8.1: every human preference is expressed in the same
    # DJState + DJMove vocabulary the live planner chooses.  The payload is
    # intentionally sanitized: it contains musical state/decision features but
    # no local track paths or raw audio.
    try:
        lab_set_state = build_set_state([analysis_a])
        lab_state = build_dj_state(
            analysis_a,
            source_cursor=0.0,
            effective_bpm=analysis_a.bpm,
            render_rate=1.0,
            set_state=lab_set_state,
        )
        state_payload = state_taste_payload(lab_state)
        state_id = state_context_id(lab_state)
        for plan in pool:
            move = score_transition_plan_as_move(analysis_a, analysis_b, plan, state=lab_state)
            plan.metadata = {
                **plan.metadata,
                "dj_state_taste": state_payload,
                "dj_move_taste": move_taste_payload(move),
                "state_context_id": state_id,
                "state_move_taste_shadow": shadow_state_move_prediction(lab_state, move),
            }
    except (AttributeError, TypeError, ValueError):
        # Lightweight test adapters and older imported manifests may not expose
        # the full TrackAnalysis surface.  Keep the legacy Taste Lab usable;
        # 1.8.1 simply omits state-move evidence for those rows.
        pass

    # Active-learning presentation: compare candidates with similar current
    # professional-judge scores because those labels carry more information.
    # Randomize only within each pair so A/B position stays blind without
    # destroying the uncertainty pairing.
    pool.sort(key=lambda plan: float(plan.metadata.get("professional_judge_score", plan.score)), reverse=True)
    paired: list[TransitionPlan] = []
    rng = secrets.SystemRandom()
    for index in range(0, len(pool), 2):
        pair = pool[index : index + 2]
        rng.shuffle(pair)
        paired.extend(pair)
    pool = paired
    session_id = f"lab-{secrets.token_hex(6)}"
    root = Path(output_root).expanduser().resolve() / session_id
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    labels = [chr(ord("A") + index) for index in range(len(pool))]
    preview_pre_roll = 6.0
    preview_post_roll = 6.0
    preview_started = time.perf_counter()
    for label, plan in zip(labels, pool, strict=True):
        path = render_transition(
            plan,
            root / f"candidate-{label.lower()}.wav",
            pre_roll_seconds=preview_pre_roll,
            post_roll_seconds=preview_post_roll,
        )
        items.append({
            "label": label,
            "audio_path": str(path),
            "pre_roll_seconds": preview_pre_roll,
            "post_roll_seconds": preview_post_roll,
            "plan": plan_features(plan),
            # Kept only in the local listening-session manifest so synthetic
            # rendered-judge examples can be re-measured against the exact
            # executable transition. Immutable training snapshots continue to
            # export sanitized feature rows rather than full local paths/assets.
            "plan_full": plan.to_dict(),
        })
    preview_seconds = time.perf_counter() - preview_started
    total_seconds = time.perf_counter() - started
    manifest = {
        "session_id": session_id,
        "track_a": str(Path(track_a).expanduser().resolve()),
        "track_b": str(Path(track_b).expanduser().resolve()),
        "preview": {
            "pre_roll_seconds": preview_pre_roll,
            "post_roll_seconds": preview_post_roll,
        },
        "candidates": items,
        "generation": {
            "analysis_seconds": round(analysis_seconds, 3),
            "candidate_generation_seconds": round(generation_seconds, 3),
            "rehearsal_seconds": round(rehearsal_seconds, 3),
            "preview_render_seconds": round(preview_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "analysis_cache_hits": int(cached_a.cache_hit) + int(cached_b.cache_hit),
            "analysis_cache_total": 2,
        },
        "active_learning": {
            "strategy": "adjacent_uncertainty_pairing",
            "blind_position_randomization": True
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
