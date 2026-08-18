"""DJAI: local-first DJ set intelligence and deterministic rendering."""

from .analysis import analyze_track
from .decision import assess_transition, validate_execution_plan
from .brain.professional import plan_professional_transition
from .pipeline import run_pair_alternatives, run_pair_pipeline, run_set_pipeline
from .planner import (
    DifficultTransitionError,
    plan_transition,
    plan_transition_alternatives,
    plan_transition_measured,
    rank_transition_candidates,
)
from .measurement import measure_transition_audio
from .renderer import render_full_mix, render_transition
from .set_planner import make_set_plan, plan_folder, plan_manual_paths
from .set_renderer import render_set, render_set_previews
from .version import DJAI_VERSION

__all__ = [
    "DifficultTransitionError",
    "analyze_track",
    "assess_transition",
    "make_set_plan",
    "measure_transition_audio",
    "plan_folder",
    "plan_manual_paths",
    "plan_transition",
    "plan_transition_alternatives",
    "plan_transition_measured",
    "plan_professional_transition",
    "rank_transition_candidates",
    "render_full_mix",
    "render_set",
    "render_set_previews",
    "render_transition",
    "run_pair_alternatives",
    "run_pair_pipeline",
    "run_set_pipeline",
    "validate_execution_plan",
]

__version__ = DJAI_VERSION
