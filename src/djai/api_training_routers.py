from __future__ import annotations

"""Dataset-builder and Training Lab route registration.

Keep endpoint functions in api.py during the behavior-preserving migration.
Router registration is grouped here without changing authentication,
request schemas, response payloads, or legacy import/patch boundaries.
"""

from collections.abc import Mapping
from typing import Any, Callable

from fastapi import APIRouter, FastAPI


def register_training_routers(
    app: FastAPI,
    handlers: Mapping[str, Callable[..., Any]],
) -> None:
    dataset_builder = APIRouter()
    dataset_builder.add_api_route(
        "/api/dataset-builder/sources",
        handlers["get_dataset_builder_sources"],
        methods=["GET"],
    )
    dataset_builder.add_api_route(
        "/api/dataset-builder/status",
        handlers["get_dataset_builder_status"],
        methods=["GET"],
    )
    dataset_builder.add_api_route(
        "/api/dataset-builder/build",
        handlers["build_dataset_builder_corpus"],
        methods=["POST"],
    )
    dataset_builder.add_api_route(
        "/api/dataset-builder/train",
        handlers["train_dataset_builder_shadows"],
        methods=["POST"],
    )

    training_lab = APIRouter()
    routes = (
        ("GET", "/api/training-lab/status", "get_training_lab_status"),
        ("GET", "/api/training-lab/sessions", "get_training_lab_sessions"),
        ("POST", "/api/training-lab/snapshot", "snapshot_training_lab_dataset"),
        ("POST", "/api/training-lab/train", "train_training_lab_shadow_model"),
        ("GET", "/api/training-lab/taste-status", "get_taste_learning_status"),
        ("POST", "/api/training-lab/real-taste/train", "train_real_taste"),
        ("GET", "/api/training-lab/real-taste/status", "get_real_taste_status"),
        ("POST", "/api/training-lab/human-release/lock", "lock_training_lab_human_release"),
        ("POST", "/api/training-lab/human-release/session", "create_training_lab_human_release_session"),
        ("POST", "/api/training-lab/human-release/vote", "save_training_lab_human_release_vote"),
        ("GET", "/api/training-lab/human-release/report", "get_training_lab_human_release_report"),
    )
    for method, path, name in routes:
        training_lab.add_api_route(
            path,
            handlers[name],
            methods=[method],
        )

    app.include_router(dataset_builder)
    app.include_router(training_lab)
