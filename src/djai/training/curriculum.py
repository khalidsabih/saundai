from __future__ import annotations

from typing import Any


def training_curriculum() -> list[dict[str, Any]]:
    return [
        {"level": 1, "name": "perception", "targets": ["beats", "downbeats", "tempo", "structure", "harmony", "stems", "cue_affordance", "music_embeddings"]},
        {"level": 2, "name": "technical_quality", "targets": ["beat_alignment", "transient_flams", "bass_collision", "vocal_collision", "clipping", "phase", "tempo_artifacts"]},
        {"level": 3, "name": "transition_taste", "targets": ["A_vs_B", "tie", "neither", "reason_tags"]},
        {"level": 4, "name": "expert_imitation", "targets": ["next_track", "cue", "intent", "technique", "duration", "tempo_trajectory", "stem_choreography"]},
        {"level": 5, "name": "set_strategy", "targets": ["journey", "multi_track_future", "narrative", "controlled_rule_breaking"]},
        {"level": 6, "name": "personalization", "targets": ["listener_residual", "bounded_preferences"]},
        {"level": 7, "name": "live_adaptation", "targets": ["rolling_state", "primary_backup_emergency", "crowd_feedback"]},
    ]
