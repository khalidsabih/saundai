from __future__ import annotations

"""Live streaming, job lifecycle, and private media route registration.

Handlers remain in api.py, preserving account ownership checks, event stream
semantics, headers, security and public endpoint signatures.
"""

from collections.abc import Callable, Mapping
from typing import Any
from fastapi import APIRouter, FastAPI


def register_live_job_media_routers(
    app: FastAPI,
    handlers: Mapping[str, Callable[..., Any]],
) -> None:
    live = APIRouter()
    for method, path, name in (
        ("GET", "/api/live/manifest", "get_live_manifest"),
        ("GET", "/api/live/events", "live_events"),
        ("GET", "/api/live/stream", "stream_live_set"),
    ):
        live.add_api_route(path, handlers[name], methods=[method])

    jobs = APIRouter()
    for method, path, name in (
        ("GET", "/api/jobs/{job_id}", "get_job"),
        ("POST", "/api/jobs/{job_id}/cancel", "cancel_job"),
        ("GET", "/api/jobs/{job_id}/events", "job_events"),
    ):
        jobs.add_api_route(path, handlers[name], methods=[method])

    media = APIRouter()
    media.add_api_route(
        "/api/media/{relative_path:path}",
        handlers["media"],
        methods=["GET"],
    )

    app.include_router(live)
    app.include_router(jobs)
    app.include_router(media)
