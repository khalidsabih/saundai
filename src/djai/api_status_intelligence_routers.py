from __future__ import annotations

"""Register operational status and intelligence endpoints without moving handlers.

The existing api.py functions remain authoritative during the migration.
Using their actual decorated signatures preserves FastAPI query validation.
"""

from collections.abc import Mapping
from typing import Any, Callable
from fastapi import APIRouter, FastAPI


def register_status_and_intelligence_routers(
    app: FastAPI,
    handlers: Mapping[str, Callable[..., Any]],
) -> None:
    status = APIRouter()
    for method, path, name in (
        ("GET", "/api/health", "health"),
        ("GET", "/api/hosting/status", "hosting_status"),
        ("GET", "/api/runtime/status", "runtime_status"),
        ("GET", "/api/validation/phase16", "phase16_validation_status"),
        ("GET", "/api/ready", "readiness"),
        ("GET", "/api/release/status", "release_status"),
        ("GET", "/api/release/readiness", "release_readiness"),
        ("GET", "/api/models", "model_status"),
        ("GET", "/api/brain/status", "professional_brain_status"),
        ("GET", "/api/benchmarks/status", "release_benchmark_status"),
    ):
        status.add_api_route(path, handlers[name], methods=[method])

    intelligence = APIRouter()
    for method, path, name in (
        ("POST", "/api/models/scoring/refit", "refit_scoring"),
        ("POST", "/api/models/genre/train", "train_genre"),
        ("POST", "/api/intelligence/listening-session", "create_intelligence_listening_session"),
        ("POST", "/api/intelligence/preference", "save_intelligence_preference"),
        ("GET", "/api/training-lab/human-review/status", "get_human_review_status"),
        ("POST", "/api/training-lab/human-review/session", "create_training_lab_human_review_session"),
        ("POST", "/api/training-lab/human-review/vote", "save_training_lab_human_review_vote"),
        ("POST", "/api/intelligence/ranker/train", "train_intelligence_ranker"),
        ("GET", "/api/intelligence/ranker/status", "intelligence_ranker_status"),
    ):
        intelligence.add_api_route(path, handlers[name], methods=[method])

    app.include_router(status)
    app.include_router(intelligence)
