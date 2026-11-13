from __future__ import annotations

"""Account and production workflow route registration.

Handlers remain defined in api.py, preserving endpoint signatures, dependency
injection, authentication, request validation, and compatibility imports.
"""

from collections.abc import Callable, Mapping
from typing import Any
from fastapi import APIRouter, FastAPI


def register_account_and_production_routers(
    app: FastAPI,
    handlers: Mapping[str, Callable[..., Any]],
) -> None:
    account = APIRouter()
    for method, path, name in (
        ("GET", "/api/auth/me", "auth_me"),
        ("POST", "/api/auth/register", "auth_register"),
        ("POST", "/api/auth/login", "auth_login"),
        ("GET", "/api/account/export", "account_export"),
        ("DELETE", "/api/account", "account_delete"),
        ("POST", "/api/auth/logout", "auth_logout"),
        ("PATCH", "/api/account/preferences", "account_preferences"),
        ("PATCH", "/api/account/profile", "account_profile"),
        ("GET", "/api/account/projects", "account_projects"),
    ):
        account.add_api_route(path, handlers[name], methods=[method])

    tracks = APIRouter()
    for method, path, name, status in (
        ("POST", "/api/tracks/upload", "upload_tracks", 200),
        ("POST", "/api/tracks/upload-intents", "create_upload_intents", 200),
        ("POST", "/api/tracks/upload-complete", "complete_direct_uploads", 200),
        ("GET", "/api/tracks", "list_tracks", 200),
        ("POST", "/api/tracks/readiness/prepare", "prepare_track_readiness", 202),
    ):
        tracks.add_api_route(path, handlers[name], methods=[method], status_code=status)

    production = APIRouter()
    for method, path, name, status in (
        ("POST", "/api/set/plan", "create_set_plan", 202),
        ("POST", "/api/sets/automatic", "create_automatic_set", 202),
        ("POST", "/api/set/render", "create_render", 202),
        ("POST", "/api/set/previews", "create_previews", 202),
        ("POST", "/api/studio/transition-preview", "create_studio_transition_preview", 202),
        ("POST", "/api/stems/separate", "create_stems", 202),
        ("POST", "/api/ratings", "save_rating", 200),
        ("POST", "/api/feedback", "save_feedback", 200),
    ):
        production.add_api_route(path, handlers[name], methods=[method], status_code=status)

    app.include_router(account)
    app.include_router(tracks)
    app.include_router(production)
