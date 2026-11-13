from __future__ import annotations

"""Large-crate planner configuration and runtime-status helpers.

These helpers only read bounded environment configuration and expose the
current large-crate strategy. They do not perform musical scoring or planning.
"""

import os
from typing import Any


def env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def large_set_threshold() -> int:
    return env_int("DJAI_LARGE_SET_THRESHOLD", 8, minimum=4, maximum=64)


def graph_scout_cue_limit(track_count: int) -> int:
    if track_count < large_set_threshold():
        return 0
    return env_int("DJAI_GRAPH_SCOUT_CUES", 5, minimum=3, maximum=8)


def large_crate_runtime_status() -> dict[str, Any]:
    try:
        embed_window = max(
            8.0,
            min(
                45.0,
                float(os.environ.get("DJAI_TRACK_EMBED_WINDOW_SECONDS", "24")),
            ),
        )
    except ValueError:
        embed_window = 24.0

    embed_windows = env_int(
        "DJAI_TRACK_EMBED_WINDOWS",
        3,
        minimum=1,
        maximum=6,
    )
    return {
        "schema_version": "1.5.1",
        "threshold_tracks": large_set_threshold(),
        "graph_strategy": "bounded_scout_then_full_final_edges",
        "graph_scout_cues_per_side": env_int(
            "DJAI_GRAPH_SCOUT_CUES",
            5,
            minimum=3,
            maximum=8,
        ),
        "receding_horizon_opener_limit": env_int(
            "DJAI_LARGE_SET_POLICY_STARTS",
            4,
            minimum=2,
            maximum=8,
        ),
        "local_improve_passes": env_int(
            "DJAI_LARGE_SET_LOCAL_PASSES",
            2,
            minimum=1,
            maximum=4,
        ),
        "track_embedding_windows": embed_windows,
        "track_embedding_window_seconds": round(embed_window, 3),
        "full_cue_pool_restored_for_final_edges": True,
        "rendered_quality_gate_unchanged": True,
        "hard_audio_safety_authoritative": True,
    }
