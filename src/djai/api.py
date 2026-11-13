from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from . import __version__
from .api_training_routers import register_training_routers
from .api_status_intelligence_routers import register_status_and_intelligence_routers
from .accounts import AccountExistsError, AccountStore
from .hosting import HostingConfig, SlidingWindowRateLimiter, invite_code_allowed
from .object_store import ObjectStore, remove_materialized_tree
from .cache import AnalysisBatchStats, SUPPORTED_AUDIO_EXTENSIONS, analyze_track_cached, analyze_tracks, discover_tracks
from .candidate_renderer import QualityLoopConfig
from .confirmation import verify_confirmation_token
from .genre import train_from_manifest
from .feedback import append_feedback_event
from .privacy import PRIVACY_VERSION, pseudonymous_listener_id, purge_jsonl_listener, purge_listener_profile
from .live import iter_mp3_stream, live_manifest
from .live_telemetry import ensure_live_telemetry_session, live_telemetry_terminal, wait_for_live_telemetry
from .jobs import JobCancelled, JobContext, JobManager, JobRecord
from .build_info import build_identity
from .release_candidate import release_candidate_status
from .feature_store import DJ_READY_FEATURES, FEATURE_VERSIONS, FeatureStore
from .music.timeline import musical_timeline_status
from .resources import resolve_resource_budget
from .learning import fit_scoring_model, maybe_refit_scoring_model
from .intelligence.lab import create_listening_session
from .intelligence.bootstrap_runtime import bootstrap_runtime_status, clear_bootstrap_runtime_cache
from .intelligence.preferences import append_preference
from .brain.memory import append_feature_rejection, load_rejections
from .brain.crate_memory import CrateMemoryStore, crate_memory_status
from .brain.professional import ProfessionalBrainConfig
from .brain.timing import SetTimingIntent
from .brain.professional_mixing import professional_mixing_status
from .brain.structural_operations import structural_operation_status
from .brain.dj_move import move_brain_status
from .brain.executive import executive_status
from .brain.layer_intelligence import LayerBrainConfig
from .intelligence.transition_ranker import load_transition_ranker, train_transition_ranker
from .intelligence.benchmark import benchmark_brain
from .intelligence.model_registry import registry_status
from .intelligence.governance import MODEL_GOVERNANCE_VERSION, audit_model_governance
from .training.provenance import audit_dataset_registry
from .training.curriculum import training_curriculum
from .training.lab_dataset import (
    create_training_snapshot,
    list_training_sessions,
    train_shadow_ranker,
    training_lab_status,
)
from .training.taste_learning import train_taste_learning_heads, taste_learning_status
from .training.real_taste import real_taste_status, train_real_taste_from_snapshot
from .training.human_preference import (
    create_human_review_session,
    human_preference_status,
    record_human_review_vote,
)
from .performance.deck import deck_runtime_status
from .performance.mix_contract import professional_mix_contract_status
from .training.source_catalog import source_catalog
from .training.dataset_builder import create_bootstrap_corpus, dataset_builder_status, train_bootstrap_shadow_models
from .evaluation.arena import arena_report
from .evaluation.release_benchmark import benchmark_runtime_status, combine_crate_fingerprints, extract_release_benchmark
from .evaluation.human_release_gate import (
    DEFAULT_MANIFEST as HUMAN_RELEASE_DEFAULT_MANIFEST,
    create_human_release_session,
    human_release_report,
    human_release_status,
    lock_human_release_benchmark,
    record_human_release_vote,
)
from .ratings import append_listening_rating
from .schemas import (
    AccountLoginRequest,
    AccountPreferencesRequest,
    AccountProfileRequest,
    AccountRegisterRequest,
    AutomaticSetRequest,
    FeedbackRequest,
    GenreTrainRequest,
    HostedUploadCompleteRequest,
    HostedUploadIntentRequest,
    PreviewRequest,
    RatingRequest,
    RefitRequest,
    RenderRequest,
    SetPlanRequest,
    StudioTransitionPreviewRequest,
    StemRequest,
    ListeningSessionRequest,
    PreferenceVoteRequest,
    HumanReviewSessionRequest,
    HumanReviewVoteRequest,
    HumanReleaseLockRequest,
    HumanReleaseSessionRequest,
    HumanReleaseVoteRequest,
    RankerTrainRequest,
    TrainingSnapshotRequest,
    TrainingLabTrainRequest,
)
from .set_planner import (
    confirmation_payload,
    large_crate_runtime_status,
    load_set_plan,
    make_set_plan,
    plan_folder,
    plan_manual_paths,
    save_set_plan,
)
from .set_renderer import render_set, render_set_previews
from .renderer import render_transition
from .stems import (
    automatic_stem_mode,
    demucs_available,
    enrich_analyses_with_cached_stems,
    find_cached_stems,
    prepare_stems_for_tracks,
    separate_stems,
)
from .layers import (
    bundle_summary,
    classify_layer_filename,
    enrich_analyses_with_performance_layers,
    find_performance_layers,
    register_performance_layers,
    resolve_performance_layers,
)


WORKSPACE = Path(os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()
TRAINING_LAB_ENABLED = os.environ.get("DJAI_TRAINING_LAB_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
ACCOUNT_COOKIE_NAME = os.environ.get("SAUND_AUTH_COOKIE_NAME", "saund_session").strip() or "saund_session"
ACCOUNT_SESSION_DAYS = max(1, int(os.environ.get("SAUND_AUTH_SESSION_DAYS", "14")))
ACCOUNT_SESSION_SECONDS = ACCOUNT_SESSION_DAYS * 24 * 60 * 60
SAUND_ENV = os.environ.get("SAUND_ENV", "development").strip().lower() or "development"
HOSTING = HostingConfig.from_mapping()
_cookie_secure_default = "1" if (SAUND_ENV == "production" or HOSTING.hosted_mode) else "0"
ACCOUNT_COOKIE_SECURE = os.environ.get("SAUND_AUTH_COOKIE_SECURE", _cookie_secure_default).strip().lower() in {"1", "true", "yes", "on"}
DATABASE_URL = os.environ.get("SAUND_DATABASE_URL", "").strip() or None
_account_db_value = Path(os.environ.get("SAUND_ACCOUNT_DB", "accounts/accounts.sqlite3")).expanduser()
ACCOUNT_DB = (_account_db_value if _account_db_value.is_absolute() else WORKSPACE / _account_db_value).resolve()
if DATABASE_URL is None:
    try:
        ACCOUNT_DB.relative_to(WORKSPACE)
    except ValueError as exc:
        raise RuntimeError("SAUND_ACCOUNT_DB must remain inside DJAI_WORKSPACE.") from exc
account_store = AccountStore(ACCOUNT_DB if DATABASE_URL is None else None, database_url=DATABASE_URL)
object_store = ObjectStore()
UPLOADS = WORKSPACE / "audio" / "uploads"
OUTPUT = WORKSPACE / "output"
_job_root_value = Path(os.environ.get("SAUND_JOB_ROOT", "output/jobs")).expanduser()
JOBS = (_job_root_value if _job_root_value.is_absolute() else WORKSPACE / _job_root_value).resolve()
_stem_job_root_value = Path(os.environ.get("SAUND_STEM_JOB_ROOT", "output/stem-jobs")).expanduser()
STEM_JOBS = (_stem_job_root_value if _stem_job_root_value.is_absolute() else WORKSPACE / _stem_job_root_value).resolve()
for _runtime_path in (JOBS, STEM_JOBS):
    try:
        _runtime_path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise RuntimeError("Hosted job roots must remain inside DJAI_WORKSPACE.") from exc
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.environ.get("DJAI_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
RESOURCE_BUDGET = resolve_resource_budget()
os.environ["SAUND_EFFECTIVE_JOB_WORKERS"] = str(RESOURCE_BUDGET.job_workers)
manager = JobManager(JOBS, workers=RESOURCE_BUDGET.job_workers)
stem_manager = JobManager(STEM_JOBS, workers=RESOURCE_BUDGET.stem_workers)
FEATURE_STORE = FeatureStore(WORKSPACE / ".djai-cache" / "features")
BACKGROUND_TRACK_READINESS = os.environ.get("DJAI_BACKGROUND_TRACK_READINESS", "1").strip().lower() in {"1", "true", "yes", "on"}
_READINESS_LOCK = threading.RLock()
auth_rate_limiter = SlidingWindowRateLimiter(limit=HOSTING.auth_rate_limit_per_minute)


app = FastAPI(
    title="SAUND.AI API",
    version=__version__,
    description="SAUND.AI local-first set planning, rendered-audio validation, performance, and live playback backend.",
)
origins = [value.strip() for value in os.environ.get("DJAI_CORS_ORIGINS", "http://localhost:3000").split(",") if value.strip()]
if "*" in origins:
    raise RuntimeError("Wildcard CORS origins are forbidden when SAUND.AI uses credential cookies.")
_hosting_errors = HOSTING.validation_errors(
    database_backend=account_store.backend,
    object_store_enabled=object_store.enabled,
    cookie_secure=ACCOUNT_COOKIE_SECURE,
    cors_origins=origins,
    training_lab_enabled=TRAINING_LAB_ENABLED,
    confirmation_secret=os.environ.get("DJAI_CONFIRMATION_SECRET", ""),
    workspace=WORKSPACE,
)
if _hosting_errors:
    raise RuntimeError("Hosted SAUND.AI startup validation failed: " + " | ".join(_hosting_errors))
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if HOSTING.hosted_mode:
    configured_hosts = [value.strip() for value in os.environ.get("SAUND_ALLOWED_HOSTS", "").split(",") if value.strip()]
    if not configured_hosts:
        configured_hosts = [
            parsed.hostname
            for parsed in (urlparse(HOSTING.api_origin or ""), urlparse(HOSTING.public_origin or ""))
            if parsed.hostname
        ]
    if configured_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(set(configured_hosts)))


@app.middleware("http")
async def _production_security_headers(request: Request, call_next):
    if HOSTING.hosted_mode and request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
        origin = (request.headers.get("origin") or "").rstrip("/")
        allowed_origins = set(origins) | {value for value in (HOSTING.public_origin, HOSTING.api_origin) if value}
        if origin and origin not in allowed_origins:
            return JSONResponse(status_code=403, content={"detail": "Origin is not allowed for this hosted SAUND.AI deployment."})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if SAUND_ENV == "production":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response



def _process_peak_rss_mb() -> float | None:
    """Best-effort process peak RSS for benchmark telemetry."""
    try:
        import resource
        import sys

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
        return round(raw / divisor, 3)
    except Exception:
        return None

def _workspace_path(value: str, *, must_exist: bool = True) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = WORKSPACE / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path must remain inside the SAUND.AI workspace.") from exc
    if must_exist and not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {value}")
    return resolved


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(path.resolve())


def _safe_filename(name: str) -> str:
    base = Path(name).name
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return clean or "track.bin"


def _require_training_lab() -> None:
    if not TRAINING_LAB_ENABLED:
        raise HTTPException(status_code=404, detail="SAUND.AI Training Lab is disabled for this deployment.")


def _auth_snapshot(http_request: Request) -> dict[str, Any] | None:
    return account_store.get_session(http_request.cookies.get(ACCOUNT_COOKIE_NAME))


def _require_account(http_request: Request) -> dict[str, Any]:
    snapshot = _auth_snapshot(http_request)
    if snapshot is None:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "login_required",
                "message": "Log in to build your SAUND.AI set.",
            },
        )
    return snapshot


def _account_user_id(account: dict[str, Any]) -> str:
    return str(account["user"]["id"])


def _account_listener(account: dict[str, Any], *, require_personalization: bool = True) -> str:
    preferences = dict(account.get("preferences", {}))
    if require_personalization and not bool(preferences.get("personalization_enabled", False)):
        return "anonymous"
    return pseudonymous_listener_id(_account_user_id(account))


def _owned_job(job_id: str, account: dict[str, Any]) -> tuple[JobManager, Any]:
    try:
        job_manager = _manager_for_job(job_id)
        record = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Resource not found") from exc
    if not record.owner_user_id or record.owner_user_id != _account_user_id(account):
        # Deliberately return 404 instead of 403 so job identifiers cannot be
        # used to discover another account's work.
        raise HTTPException(status_code=404, detail="Resource not found")
    return job_manager, record


def _job_for_workspace_path(path: Path) -> tuple[JobManager, Any] | None:
    resolved = path.resolve()
    for root, job_manager in ((JOBS, manager), (STEM_JOBS, stem_manager)):
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            return None
        job_id = relative.parts[0]
        try:
            return job_manager, job_manager.get(job_id)
        except KeyError:
            return None
    return None


def _require_owned_workspace_path(
    value: str,
    account: dict[str, Any],
    *,
    must_exist: bool = True,
) -> Path:
    path = _workspace_path(value, must_exist=must_exist)
    resolved_job = _job_for_workspace_path(path)
    if resolved_job is None:
        raise HTTPException(status_code=404, detail="Resource not found")
    _, record = resolved_job
    if not record.owner_user_id or record.owner_user_id != _account_user_id(account):
        raise HTTPException(status_code=404, detail="Resource not found")
    return path


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _account_storage_token(account: dict[str, Any]) -> str:
    return hashlib.sha256(_account_user_id(account).encode("utf-8")).hexdigest()[:24]


def _account_audio_root(account: dict[str, Any] | None) -> Path:
    if not HOSTING.hosted_mode:
        return WORKSPACE / "audio"
    if account is None:
        raise HTTPException(status_code=401, detail={"code": "login_required", "message": "Log in to access your music library."})
    root = WORKSPACE / "audio" / "users" / _account_storage_token(account)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _account_upload_root(account: dict[str, Any] | None) -> Path:
    root = (UPLOADS if not HOSTING.hosted_mode else _account_audio_root(account) / "uploads")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _audio_library_path(
    value: str,
    *,
    account: dict[str, Any] | None = None,
    must_exist: bool = True,
) -> Path:
    path = _workspace_path(value, must_exist=must_exist)
    root = _account_audio_root(account) if HOSTING.hosted_mode else WORKSPACE / "audio"
    if not _path_is_within(path, root):
        if HOSTING.hosted_mode:
            raise HTTPException(status_code=404, detail="Track not found")
        raise HTTPException(status_code=400, detail="Track paths must remain inside the SAUND.AI audio library.")
    return path


def _media_url(path: Path) -> str:
    hosted = object_store.media_url_for_path(path)
    return hosted or f"/api/media/{_relative(path)}"


def _publish_job_media(path: Path, *, account: dict[str, Any], job_id: str) -> str:
    if not object_store.enabled:
        return f"/api/media/{_relative(path)}"
    published = object_store.publish_file(path, user_id=_account_user_id(account), job_id=job_id)
    if published is None:
        return f"/api/media/{_relative(path)}"
    return str(published["media_url"])


def _client_rate_key(request: Request, action: str) -> str:
    address = request.client.host if request.client else "unknown"
    if HOSTING.trust_proxy_headers:
        forwarded = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
        if forwarded:
            address = forwarded.split(",", 1)[0].strip() or address
    return f"{action}:{address}"


def _enforce_auth_rate_limit(request: Request, action: str) -> None:
    if not auth_rate_limiter.allow(_client_rate_key(request, action)):
        raise HTTPException(status_code=429, detail="Too many authentication attempts. Please try again shortly.")


def _ratings_workspace_path(value: str, *, must_exist: bool = True) -> Path:
    path = _workspace_path(value, must_exist=must_exist)
    if not _path_is_within(path, WORKSPACE / "ratings"):
        raise HTTPException(status_code=400, detail="Measurement logs must remain inside the SAUND.AI ratings directory.")
    return path


def _project_title(record: JobRecord) -> str:
    result = record.result or {}
    plan = result.get("plan") if isinstance(result, dict) else None
    ordered = plan.get("ordered_tracks") if isinstance(plan, dict) else None
    titles: list[str] = []
    if isinstance(ordered, list):
        for item in ordered:
            if isinstance(item, dict) and item.get("title"):
                titles.append(str(item["title"]))
    if not titles:
        for value in record.resource_context.get("track_paths", []):
            titles.append(Path(str(value)).stem)
    if not titles:
        return "SAUND.AI set"
    if len(titles) == 1:
        return titles[0]
    if len(titles) == 2:
        return f"{titles[0]} → {titles[1]}"
    return f"{titles[0]} → {titles[1]} + {len(titles) - 2}"


def _account_project_summary(record: JobRecord, renders_by_plan: dict[str, JobRecord]) -> dict[str, Any]:
    result = record.result if isinstance(record.result, dict) else {}
    plan_path = str(result.get("plan_path")) if result.get("plan_path") else None
    render = renders_by_plan.get(plan_path or "")
    render_result = render.result if render is not None and isinstance(render.result, dict) else {}
    audio_path = result.get("audio_path") or render_result.get("audio_path")
    audio_url = result.get("audio_url") or render_result.get("audio_url")
    if audio_path:
        try:
            candidate = _workspace_path(str(audio_path))
            if candidate.is_file():
                audio_url = _media_url(candidate)
        except HTTPException:
            pass
    return {
        "project_id": record.job_id,
        "job_id": record.job_id,
        "kind": record.kind,
        "status": record.status,
        "title": _project_title(record),
        "created_at_utc": record.created_at_utc,
        "updated_at_utc": record.updated_at_utc,
        "journey": record.resource_context.get("journey"),
        "track_count": len(record.resource_context.get("track_paths", [])),
        "track_paths": list(record.resource_context.get("track_paths", [])),
        "plan_path": plan_path,
        "audio_path": str(audio_path) if audio_path else None,
        "audio_url": str(audio_url) if audio_url else None,
    }


def _auth_response(payload: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
    response = JSONResponse(payload, status_code=status_code)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response


def _set_account_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        ACCOUNT_COOKIE_NAME,
        token,
        max_age=ACCOUNT_SESSION_SECONDS,
        path="/",
        secure=ACCOUNT_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


def _clear_account_cookie(response: Response) -> None:
    response.delete_cookie(
        ACCOUNT_COOKIE_NAME,
        path="/",
        secure=ACCOUNT_COOKIE_SECURE,
        httponly=True,
        samesite="lax",
    )


@app.get("/api/auth/me")
def auth_me(http_request: Request) -> Response:
    snapshot = _auth_snapshot(http_request)
    if snapshot is None:
        response = _auth_response({"authenticated": False, "user": None, "profile": None, "preferences": None})
        if http_request.cookies.get(ACCOUNT_COOKIE_NAME):
            _clear_account_cookie(response)
        return response
    return _auth_response({"authenticated": True, **snapshot})


@app.post("/api/auth/register")
def auth_register(request: AccountRegisterRequest, http_request: Request) -> Response:
    _enforce_auth_rate_limit(http_request, "register")
    if not invite_code_allowed(request.invite_code, mode=HOSTING.signup_mode):
        if HOSTING.signup_mode == "closed":
            raise HTTPException(status_code=403, detail="SAUND.AI registration is currently closed.")
        raise HTTPException(status_code=403, detail="A valid SAUND.AI beta invite code is required.")
    try:
        snapshot = account_store.create_user(
            str(request.email),
            request.password,
            first_name=request.first_name,
            last_name=request.last_name,
            artist_name=request.artist_name,
            country=request.country,
            primary_use=request.primary_use,
            experience_level=request.experience_level,
            preferred_genres=list(request.preferred_genres),
            musical_priorities=list(request.musical_priorities),
            personalization_enabled=request.personalization_enabled,
            improve_saund_ai=request.improve_saund_ai,
            marketing_consent=request.marketing_consent,
        )
    except AccountExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session = account_store.create_session(snapshot["user"]["id"], ttl_seconds=ACCOUNT_SESSION_SECONDS)
    response = _auth_response({"authenticated": True, **snapshot}, status_code=201)
    _set_account_cookie(response, session.token)
    return response


@app.post("/api/auth/login")
def auth_login(request: AccountLoginRequest, http_request: Request) -> Response:
    _enforce_auth_rate_limit(http_request, "login")
    snapshot = account_store.authenticate(str(request.email), request.password)
    if snapshot is None:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    session = account_store.create_session(snapshot["user"]["id"], ttl_seconds=ACCOUNT_SESSION_SECONDS)
    response = _auth_response({"authenticated": True, **snapshot})
    _set_account_cookie(response, session.token)
    return response


@app.get("/api/account/export")
def account_export(
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> Response:
    return _auth_response({"privacy_version": PRIVACY_VERSION, "account": account_store.export_user(_account_user_id(account))})


@app.delete("/api/account")
def account_delete(
    http_request: Request,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> Response:
    listener = _account_listener(account, require_personalization=False)
    purge_listener_profile(listener, WORKSPACE / "ratings" / "listener-profiles.json")
    purge_jsonl_listener(listener, WORKSPACE / "ratings" / "feedback-events.jsonl")
    purge_jsonl_listener(listener, WORKSPACE / "ratings" / "listening-tests.jsonl")
    try:
        CrateMemoryStore(WORKSPACE / "ratings" / "crate-memory.sqlite3").purge_listener(listener)
    except Exception:
        # Account deletion remains available even when optional local memory is
        # absent/corrupt; the DB contains no account credentials or raw audio.
        pass
    user_id = _account_user_id(account)
    if object_store.enabled:
        try:
            object_store.delete_user_objects(user_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Could not remove hosted audio objects; account deletion was not completed.") from exc
    if HOSTING.hosted_mode:
        remove_materialized_tree(_account_audio_root(account))
    account_store.delete_user(user_id)
    response = _auth_response({"deleted": True, "privacy_version": PRIVACY_VERSION})
    _clear_account_cookie(response)
    return response


@app.post("/api/auth/logout")
def auth_logout(http_request: Request) -> Response:
    account_store.delete_session(http_request.cookies.get(ACCOUNT_COOKIE_NAME))
    response = _auth_response({"authenticated": False, "user": None, "profile": None, "preferences": None})
    _clear_account_cookie(response)
    return response


@app.patch("/api/account/preferences")
def account_preferences(
    request: AccountPreferencesRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> Response:
    snapshot = account_store.update_preferences(
        account["user"]["id"],
        personalization_enabled=request.personalization_enabled,
        improve_saund_ai=request.improve_saund_ai,
        marketing_consent=request.marketing_consent,
    )
    return _auth_response({"authenticated": True, **snapshot})


@app.patch("/api/account/profile")
def account_profile(
    request: AccountProfileRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> Response:
    supplied = request.model_fields_set
    snapshot = account_store.update_profile(
        account["user"]["id"],
        first_name=request.first_name,
        last_name=request.last_name,
        artist_name=request.artist_name,
        country=request.country,
        primary_use=request.primary_use,
        experience_level=request.experience_level,
        preferred_genres=list(request.preferred_genres) if request.preferred_genres is not None else None,
        musical_priorities=list(request.musical_priorities) if request.musical_priorities is not None else None,
        artist_name_supplied="artist_name" in supplied,
        primary_use_supplied="primary_use" in supplied,
        experience_level_supplied="experience_level" in supplied,
    )
    return _auth_response({"authenticated": True, **snapshot})


@app.get("/api/account/projects")
def account_projects(
    account: Annotated[dict[str, Any], Depends(_require_account)],
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    user_id = _account_user_id(account)
    primary = manager.list_for_owner(
        user_id,
        kinds={"automatic-set", "set-plan"},
        limit=limit,
    )
    render_jobs = manager.list_for_owner(
        user_id,
        kinds={"set-render"},
        limit=max(100, limit * 4),
    )
    renders_by_plan: dict[str, JobRecord] = {}
    for render in render_jobs:
        source_plan = str(render.resource_context.get("source_plan_path") or "")
        if source_plan and source_plan not in renders_by_plan:
            renders_by_plan[source_plan] = render
    return {
        "projects": [_account_project_summary(record, renders_by_plan) for record in primary],
        "count": len(primary),
    }


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "product": "SAUND.AI",
        "version": __version__,
        "build": build_identity(),
        "workspace": str(WORKSPACE),
        "features": {
            "offline_render": True,
            "account_foundation": True,
            "authenticated_build_gate": True,
            "account_resource_ownership": True,
            "studio_account_gate": True,
            "account_project_history": True,
            "rich_account_profile": True,
            "separate_training_and_marketing_consent": True,
            "unified_decision_engine": True,
            "immutable_execution_plans": True,
            "content_bound_execution_assets": True,
            "executable_transition_ir": True,
            "generic_dsp_execution_engine": True,
            "joint_transition_proposals": True,
            "musical_world_model_6": True,
            "hierarchical_set_director": True,
            "dj_performance_memory": True,
            "multi_critic_rendered_judge": True,
            "critic_guided_surgical_replanning": True,
            "content_addressed_rehearsal_cache": True,
            "release_readiness_gate": True,
            "exact_set_rehearsal_context": True,
            "final_transition_verification": True,
            "atomic_render_publication": True,
            "automatic_delivery": True,
            "automatic_fallback": True,
            "manual_confirmation_in_pro_mode": True,
            "genre_training": True,
            "learned_scoring": True,
            "parallel_analysis": True,
            "parallel_graph_planning": True,
            "pair_evidence_reuse": True,
            "render_measure_quality_gate": True,
            "cross_validated_learning": True,
            "listener_personalization": True,
            "target_energy_arcs": True,
            "automatic_stems_mode": automatic_stem_mode(),
            "stems_available": demucs_available(),
            "supplied_performance_layers": True,
            "automatic_performance_layers": True,
            "layer_intelligence_v1": LayerBrainConfig.from_environment().enabled,
            "dj_executive_observation_v1": executive_status()["enabled"],
            "dj_executive_v2": executive_status()["authoritative"],
            "dj_executive_v3": executive_status()["authoritative"],
            "dj_executive_v4": executive_status()["authoritative"],
            "dj_executive_v5": executive_status()["authoritative"],
            "djmove_intent_v1": executive_status()["djmove_intent_required_for_commit"],
            "adaptive_thinking_v1": executive_status()["adaptive_thinking"]["enabled"],
            "prepared_recovery_v1": executive_status()["prepared_recovery_capable"],
            "incoming_vocal_over_instrumental": True,
            "unified_transition_intelligence": True,
            "stem_ownership_choreography": True,
            "review_needed_replans": True,
            "simple_pro_shared_brain": True,
            "duration_search": True,
            "live_http_streaming": True,
            "local_music_embeddings": True,
            "neural_model_registry": True,
            "neural_beat_downbeat_adapter": True,
            "neural_transition_policy_heads": True,
            "neural_rendered_dj_judge": True,
            "neural_next_track_policy": True,
            "hybrid_next_track_model_1": True,
            "cold_target_intelligence": True,
            "global_taste_ranker": True,
            "unified_ai_planner_1": True,
            "full_set_benchmarks_1": True,
            "consent_gated_personalization": True,
            "privacy_export_delete": True,
            "model_governance_1": True,
            "production_release_gate_1": True,
            "adaptive_rehearsal_1_0_1": True,
            "nonblocking_background_stems_1_0_1": True,
            "resident_embedding_runtime_1_0_1": True,
            "background_learning_refit_1_0_1": True,
            "shadow_authority_gating": True,
            "ab_listening_lab": True,
            "training_lab": TRAINING_LAB_ENABLED,
            "immutable_training_snapshots": True,
            "shadow_preference_training": True,
            "taste_learning_loop_1_1_1": True,
            "runtime_candidate_evidence": True,
            "synthetic_rendered_failures": True,
            "global_taste_shadow_training": True,
            "rendered_dj_judge_shadow_training": True,
            "dj_deck_tempo_handoff_1_2": True,
            "beat_phase_nudge_1_2": True,
            "stem_vocal_exchange_1_2": True,
            "human_dj_move_brain_1_4": True,
            "adaptive_musical_timeline_1_4": True,
            "protected_musical_commitments_1_4": True,
            "explicit_wait_actions_1_4": True,
            "move_aware_track_ordering_1_4": True,
            "host_ready_foundation_1_5": True,
            "large_crate_planning_1_5_1": True,
            "job_wide_progress_1_5_2": True,
            "engineering_integrity_1_5_3": True,
            "canonical_musical_timeline_1_6": True,
            "single_phrase_section_authority_1_6": True,
            "background_track_readiness_1_6_1": BACKGROUND_TRACK_READINESS,
            "modular_feature_store_1_6_1": True,
            "feature_store_reconstruction_1_6_1": True,
            "complete_djmove_generator_1_7": True,
            "lazy_crate_search_1_7_1": True,
            "structural_deck_operations_1_7_2": True,
            "ai_headphones_1_8": True,
            "state_move_taste_learning_1_8_1": True,
            "crate_memory_1_8_2": True,
            "release_benchmarking_1_9": True,
            "benchmark_champion_gate_1_9": True,
            "release_candidate_certification_2_0": True,
            "time_event_aware_set_intelligence_13": True,
            "live_autonomous_hardening_14": True,
            "venue_production_reliability_15": True,
            "docker_authority_passthrough_2_0": True,
            "persisted_schema_preflight_2_0": True,
            "cooperative_job_cancellation_1_5_3": True,
            "job_heartbeat_and_stall_detection_1_5_3": True,
            "single_host_resource_budget_1_5_3": True,
            "managed_postgres_accounts_1_5": account_store.backend == "postgres",
            "direct_object_uploads_1_5": object_store.enabled,
            "hosted_user_library_isolation_1_5": HOSTING.hosted_mode,
            "external_dataset_builder": TRAINING_LAB_ENABLED,
            "weak_label_provenance": True,
            "immutable_bootstrap_corpora": True,
            "transition_ranker": load_transition_ranker(WORKSPACE / "config" / "transition-ranker.json") is not None,
            "live_dependency_available": importlib.util.find_spec("sounddevice") is not None,
        },
    }


def hosting_status() -> dict[str, Any]:
    return HOSTING.public_status(
        database_backend=account_store.backend,
        object_store_enabled=object_store.enabled,
    )


def _safe_tree_stats(path: Path) -> dict[str, int]:
    files = 0
    size = 0
    if not path.exists():
        return {"files": 0, "bytes": 0}
    try:
        for item in path.rglob("*"):
            if not item.is_file():
                continue
            files += 1
            try:
                size += item.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return {"files": files, "bytes": size}


def runtime_status() -> dict[str, Any]:
    return {
        "schema_version": "2.0.0",
        "build": build_identity(),
        "release_candidate": release_candidate_status(WORKSPACE, strict=False),
        "resources": RESOURCE_BUDGET.to_dict(),
        "jobs": {
            "main": manager.status_summary(),
            "stems": stem_manager.status_summary(),
        },
        "cache": {
            "track_analysis": _safe_tree_stats(WORKSPACE / ".djai-cache" / "tracks"),
            "feature_store": FEATURE_STORE.stats(),
            "readiness": _safe_tree_stats(WORKSPACE / ".djai-cache" / "readiness"),
            "rehearsal": _safe_tree_stats(WORKSPACE / ".djai-cache" / "rehearsals"),
            "model_cache_external": True,
        },
        "policies": {
            "job_cancellation": "cooperative-safe-boundary",
            "stuck_detection": "progress-age-with-independent-heartbeat",
            "docker_image_tag": "local-or-explicit-SAUND_IMAGE_TAG",
            "background_track_readiness": BACKGROUND_TRACK_READINESS,
            "feature_invalidation": "per-feature-version",
        },
    }


def phase16_validation_status() -> dict[str, Any]:
    report_path = WORKSPACE / "output" / "phase16-real-music" / "phase16-report.json"
    if not report_path.is_file():
        return {
            "available": False,
            "phase": 16,
            "name": "Real-Music Validation",
            "report_path": str(report_path.relative_to(WORKSPACE)),
            "summary": None,
            "runs": [],
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "phase": 16,
            "name": "Real-Music Validation",
            "report_path": str(report_path.relative_to(WORKSPACE)),
            "summary": None,
            "runs": [],
            "error": str(exc),
        }
    return {
        "available": True,
        **payload,
    }


def readiness() -> Response:
    workspace_ready = os.access(WORKSPACE, os.W_OK)
    database_ready = account_store.ping()
    object_ready = object_store.ready()
    build_ready = bool(build_identity().get("version_consistent", False))
    ready = bool(workspace_ready and database_ready and object_ready and build_ready)
    payload = {
        "status": "ready" if ready else "not_ready",
        "version": __version__,
        "checks": {
            "workspace_writable": workspace_ready,
            "database": database_ready,
            "object_store": object_ready,
            "version_consistent": build_ready,
        },
        "hosting": HOSTING.public_status(
            database_backend=account_store.backend,
            object_store_enabled=object_store.enabled,
        ),
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


def release_status() -> dict[str, Any]:
    """Return operational release state without using HTTP failure status."""
    return release_candidate_status(WORKSPACE, strict=False)


def release_readiness(
    strict: bool = Query(default=False),
    suite: str | None = Query(default=None, min_length=1, max_length=80),
) -> Response:
    """Return operational or strict RC certification readiness.

    Normal service readiness remains `/api/ready`.  Strict mode intentionally
    includes external release evidence (benchmark champion, provenance, model
    governance and media-rights confirmation) and therefore may be blocked on
    a perfectly healthy development installation.
    """
    payload = release_candidate_status(WORKSPACE, strict=strict, benchmark_suite=suite)
    ready = bool(payload.get("release_certifiable")) if strict else bool(payload.get("operational_ready"))
    return JSONResponse(payload, status_code=200 if ready else 503)


def model_status() -> dict[str, Any]:
    registry_path = WORKSPACE / "config" / "model-registry.json"
    payload = registry_status(registry_path)
    payload["governance"] = audit_model_governance(registry_path).to_dict()
    payload["governance_version"] = MODEL_GOVERNANCE_VERSION
    return payload


def _unique_upload_target(root: Path, filename: str) -> Path:
    target = root / filename
    counter = 1
    while target.exists():
        target = root / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
        counter += 1
    return target


def _track_stem_status(path: Path) -> str:
    bundle = resolve_performance_layers(path)
    if bundle is not None and bundle.complete:
        return "ready"
    if find_cached_stems(path):
        return "ready"
    mode = automatic_stem_mode()
    if mode == "off":
        return "disabled"
    if demucs_available():
        return "preparing"
    return "unavailable"


def _stem_required_for_readiness(path: Path) -> bool:
    bundle = resolve_performance_layers(path)
    if bundle is not None and bundle.complete:
        return False
    return automatic_stem_mode() in {"background", "wait"} and demucs_available()


def _track_readiness_payload(path: Path) -> dict[str, Any]:
    stem_status = _track_stem_status(path)
    return FEATURE_STORE.track_readiness(
        path,
        stem_status=stem_status,
        stem_required=_stem_required_for_readiness(path),
    )


def _ensure_track_readiness_job(
    path: Path,
    *,
    owner_user_id: str | None,
    retry_failed: bool = False,
    prepare_stems: bool = True,
) -> str | None:
    if not BACKGROUND_TRACK_READINESS:
        return None
    source = path.expanduser().resolve()
    with _READINESS_LOCK:
        readiness = _track_readiness_payload(source)
        if readiness["dj_ready"]:
            return None
        if readiness.get("state") == "failed" and not retry_failed:
            return str(readiness.get("job_id") or "") or None
        current_job_id = readiness.get("job_id")
        if current_job_id:
            try:
                current = stem_manager.get(str(current_job_id))
            except KeyError:
                current = None
            if current is not None and current.status in {"queued", "running", "cancelling"}:
                return current.job_id

        def readiness_task(context: JobContext, track: Path = source) -> dict[str, Any]:
            FEATURE_STORE.clear_failure(track)
            FEATURE_STORE.set_job(track, context.job_id, state="analyzing")
            try:
                stem_needed = (
                    prepare_stems
                    and _stem_required_for_readiness(track)
                    and not bool(find_cached_stems(track))
                )
                total_steps = len(DJ_READY_FEATURES) + (1 if stem_needed else 0)
                completed_steps = 0
                context.update("track-readiness", 0, total_steps, f"Fingerprinting {track.name}")
                if stem_needed:
                    context.raise_if_cancelled()
                    context.update("track-readiness", 0, total_steps, f"Separating stems for {track.name}")
                    separate_stems(
                        track,
                        model=os.environ.get("DJAI_STEM_MODEL", "htdemucs"),
                        device=os.environ.get("DJAI_STEM_DEVICE", "cpu"),
                    )
                    completed_steps = 1
                    context.update("track-readiness", completed_steps, total_steps, f"{track.name}: stems ready")
                for feature in DJ_READY_FEATURES:
                    FEATURE_STORE.mark_feature(track, feature, "running")
                context.raise_if_cancelled()
                cached = analyze_track_cached(
                    track,
                    cache_dir=WORKSPACE / ".djai-cache" / "tracks",
                    use_cache=True,
                )
                # analyze_track_cached synchronizes all analysis-derived feature
                # projections, including canonical timeline + world model.
                for offset, feature in enumerate(DJ_READY_FEATURES, start=1):
                    context.update(
                        "track-readiness",
                        completed_steps + offset,
                        total_steps,
                        f"{track.name}: {feature.replace('_', ' ')} ready",
                    )
                    context.raise_if_cancelled()
                final = _track_readiness_payload(track)
                FEATURE_STORE.set_job(
                    track,
                    context.job_id,
                    state="ready" if final["dj_ready"] else "partial",
                )
                return {
                    "track_path": _relative(track),
                    "cache_source": cached.cache_source,
                    "readiness": _track_readiness_payload(track),
                }
            except JobCancelled:
                FEATURE_STORE.set_job(track, context.job_id, state="cancelled")
                raise
            except Exception as exc:
                FEATURE_STORE.mark_failed(track, str(exc))
                raise

        job = stem_manager.submit(
            "track-readiness",
            readiness_task,
            owner_user_id=owner_user_id,
            resource_context={"track_path": _relative(source)},
        )
        FEATURE_STORE.set_job(source, job.job_id, state="queued")
        return job.job_id


def _postprocess_uploaded_batch(
    raw_saved: list[dict[str, Any]],
    *,
    upload_owner_user_id: str | None,
) -> dict[str, Any]:
    saved_paths = [Path(item["path_obj"]) for item in raw_saved]
    stem_mode = automatic_stem_mode()
    can_separate = demucs_available() and stem_mode in {"background", "wait"}
    bundles = register_performance_layers(saved_paths)
    bundle_lookup = {
        (Path(bundle.full.path).parent, bundle.key): bundle
        for bundle in bundles
        if bundle.duration_compatible
    }
    visible: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []

    for item in raw_saved:
        target = Path(item["path_obj"])
        key, role = classify_layer_filename(target)
        bundle = bundle_lookup.get((target.parent.resolve(), key))
        if role != "full" and bundle is not None:
            auxiliary.append(
                {
                    "name": target.name,
                    "path": _relative(target),
                    "size": int(item["size"]),
                    "role": role,
                    "bundle_id": bundle.bundle_id,
                }
            )
            continue

        stem_job_id: str | None = None
        bundle_info = bundle_summary(bundle)
        supplied_complete = bool(bundle and bundle.complete)
        if can_separate and not supplied_complete:
            stem_target = target

            def stem_task(context: JobContext, source: Path = stem_target) -> dict[str, Any]:
                context.update("stems", 0, 1, source.name)
                mapping = separate_stems(
                    source,
                    model=os.environ.get("DJAI_STEM_MODEL", "htdemucs"),
                    device=os.environ.get("DJAI_STEM_DEVICE", "cpu"),
                )
                context.update("stems", 1, 1, source.name)
                return {
                    "track_path": _relative(source),
                    "stems": {name: _relative(Path(path)) for name, path in mapping.items()},
                }

            stem_job_id = stem_manager.submit(
                "stem-separation",
                stem_task,
                owner_user_id=upload_owner_user_id,
                resource_context={"track_path": _relative(stem_target)},
            ).job_id

        readiness_job_id = _ensure_track_readiness_job(
            target,
            owner_user_id=upload_owner_user_id,
            # Fresh uploads already have the established stem-separation job
            # above when Demucs is required. Avoid running Demucs twice; the
            # readiness view discovers those stems as soon as that job lands.
            prepare_stems=not bool(stem_job_id),
        )
        stem_status = (
            "supplied_layers"
            if supplied_complete
            else "queued"
            if stem_job_id
            else _track_stem_status(target)
        )
        readiness = FEATURE_STORE.track_readiness(
            target,
            stem_status=stem_status,
            stem_required=_stem_required_for_readiness(target),
        )
        visible.append(
            {
                "name": target.name,
                "path": _relative(target),
                "size": int(item["size"]),
                "media_url": _media_url(target),
                "performance_layers": bundle_info,
                "stem_job_id": stem_job_id,
                "stem_status": stem_status,
                "readiness_job_id": readiness_job_id or readiness.get("job_id"),
                "readiness": readiness,
            }
        )
    return {
        "files": visible,
        "auxiliary_files": auxiliary,
        "automatic_stems": stem_mode,
        "performance_layer_bundles": [bundle_summary(bundle) for bundle in bundles],
    }


@app.post("/api/tracks/upload")
async def upload_tracks(
    http_request: Request,
    files: Annotated[list[UploadFile], File(description="Audio files")],
) -> dict[str, Any]:
    upload_account = _auth_snapshot(http_request)
    if HOSTING.upload_auth_required and upload_account is None:
        upload_account = _require_account(http_request)
    if HOSTING.hosted_mode and object_store.enabled and os.environ.get("SAUND_ALLOW_PROXY_UPLOADS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "direct_upload_required",
                "message": "Hosted SAUND.AI uploads must use the direct object-store upload flow.",
            },
        )
    upload_owner_user_id = _account_user_id(upload_account) if upload_account else None
    upload_root = _account_upload_root(upload_account)
    raw_saved: list[dict[str, Any]] = []

    for upload in files:
        filename = _safe_filename(upload.filename or "track.bin")
        if Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported audio type: {filename}")
        target = _unique_upload_target(upload_root, filename)
        written = 0
        source_digest = hashlib.sha256()
        try:
            with target.open("wb") as handle:
                while True:
                    chunk = upload.file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Upload exceeds the configured size limit: {filename}",
                        )
                    source_digest.update(chunk)
                    handle.write(chunk)
            FEATURE_STORE.register_source_fingerprint(target, source_digest.hexdigest())
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        raw_saved.append({"path_obj": target, "size": written})

    return _postprocess_uploaded_batch(raw_saved, upload_owner_user_id=upload_owner_user_id)


@app.post("/api/tracks/upload-intents")
def create_upload_intents(
    request: HostedUploadIntentRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    if not object_store.enabled:
        raise HTTPException(status_code=404, detail="Direct object uploads are not configured for this deployment.")
    intents: list[dict[str, Any]] = []
    for item in request.files:
        filename = _safe_filename(item.filename)
        if Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported audio type: {filename}")
        if int(item.size) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"Upload exceeds the configured size limit: {filename}")
        intent = object_store.create_upload_intent(
            user_id=_account_user_id(account),
            filename=filename,
            size=int(item.size),
            content_type=item.content_type,
        )
        intents.append({"filename": filename, **intent})
    return {"uploads": intents, "backend": "s3", "max_upload_bytes": MAX_UPLOAD_BYTES}


@app.post("/api/tracks/upload-complete")
def complete_direct_uploads(
    request: HostedUploadCompleteRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    if not object_store.enabled:
        raise HTTPException(status_code=404, detail="Direct object uploads are not configured for this deployment.")
    upload_root = _account_upload_root(account)
    raw_saved: list[dict[str, Any]] = []
    created: list[Path] = []
    try:
        for item in request.files:
            filename = _safe_filename(item.filename)
            if Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported audio type: {filename}")
            if int(item.expected_size) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Upload exceeds the configured size limit: {filename}")
            if not object_store.owns_key(_account_user_id(account), item.object_key):
                raise HTTPException(status_code=404, detail="Uploaded object not found")
            target = _unique_upload_target(upload_root, filename)
            try:
                metadata = object_store.materialize(
                    item.object_key,
                    target,
                    expected_size=int(item.expected_size),
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Could not materialize hosted upload: {filename}") from exc
            created.append(target)
            source_sha256 = str(metadata.get("sha256") or "")
            if source_sha256:
                FEATURE_STORE.register_source_fingerprint(target, source_sha256)
            raw_saved.append({"path_obj": target, "size": int(metadata["size"])})
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return _postprocess_uploaded_batch(raw_saved, upload_owner_user_id=_account_user_id(account))


@app.get("/api/tracks")
def list_tracks(
    http_request: Request,
    folder: str = Query(default="audio"),
    recursive: bool = False,
) -> dict[str, Any]:
    account = _auth_snapshot(http_request)
    if HOSTING.upload_auth_required and account is None:
        account = _require_account(http_request)
    root = _account_audio_root(account) if HOSTING.hosted_mode else _audio_library_path(folder)
    tracks = discover_tracks(root, recursive=recursive)
    records: list[dict[str, Any]] = []
    for path in tracks:
        key, role = classify_layer_filename(path)
        # Only hide an alternate file when there is a real full-master bundle;
        # a standalone file named "Instrumental" remains usable as a normal track.
        if role != "full":
            sibling_full = next(
                (
                    candidate
                    for candidate in tracks
                    if candidate.parent == path.parent
                    and classify_layer_filename(candidate) == (key, "full")
                ),
                None,
            )
            if sibling_full is not None and find_performance_layers(sibling_full) is not None:
                continue
        bundle = resolve_performance_layers(path) if role == "full" else None
        layer_summary = bundle_summary(bundle)
        stem_status = _track_stem_status(path)
        owner_user_id = _account_user_id(account) if account is not None else None
        readiness_job_id = _ensure_track_readiness_job(
            path,
            owner_user_id=owner_user_id,
        )
        readiness = _track_readiness_payload(path)
        records.append(
            {
                "name": path.name,
                "path": _relative(path),
                "size": path.stat().st_size,
                "media_url": _media_url(path),
                "performance_layers": layer_summary,
                "stem_status": stem_status,
                "readiness_job_id": readiness_job_id or readiness.get("job_id"),
                "readiness": readiness,
            }
        )
    return {"folder": _relative(root), "tracks": records}


@app.post("/api/tracks/readiness/prepare", status_code=202)
def prepare_track_readiness(
    account: Annotated[dict[str, Any], Depends(_require_account)],
    track_path: str = Query(...),
) -> dict[str, Any]:
    track = _audio_library_path(track_path, account=account)
    job_id = _ensure_track_readiness_job(
        track,
        owner_user_id=_account_user_id(account),
        retry_failed=True,
    )
    job = None
    if job_id:
        try:
            job = stem_manager.get(job_id).to_dict()
        except KeyError:
            job = None
    return {
        "track_path": _relative(track),
        "job": job,
        "readiness": _track_readiness_payload(track),
    }


@app.post("/api/set/plan", status_code=202)
def create_set_plan(
    request: SetPlanRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    paths = [_audio_library_path(value, account=account) for value in request.track_paths]
    folder = (_account_audio_root(account) if HOSTING.hosted_mode else _audio_library_path(request.folder or "audio")) if not paths else None
    measurement_log = (
        str(_ratings_workspace_path(request.measurement_log, must_exist=False))
        if request.measurement_log
        else None
    )
    listener = _account_listener(account)

    def task(context: JobContext) -> dict[str, Any]:
        context.update_absolute(
            "starting",
            0.02,
            f"Preparing {len(paths) if paths else 'library'} for Studio planning",
            current=0,
            total=len(paths),
        )
        plan_progress = context.ranged_progress_callback(
            0.02,
            0.98,
            stage_ranges={
                "stems": (0.02, 0.08),
                "analysis": (0.08, 0.32),
                "cached": (0.08, 0.32),
                "analyzed": (0.08, 0.32),
                "graph": (0.32, 0.58),
                "order-replan": (0.58, 0.64),
                "measure": (0.64, 0.98),
            },
        )
        if paths:
            if request.resolved_automatic_delivery and automatic_stem_mode() == "wait" and demucs_available():
                # "background" is intentionally non-blocking: upload-time stem
                # jobs may continue independently and planning uses whatever is
                # already cached. Only the explicit "wait" policy synchronizes
                # set creation with Demucs.
                prepare_stems_for_tracks(paths, progress=plan_progress)
            if request.manual:
                plan = plan_manual_paths(
                    paths,
                    use_cache=request.use_cache,
                    tempo_policy=request.tempo_policy,
                    preferred_style=request.preferred_style,
                    journey=request.journey,
                    minimum_body_seconds=request.minimum_body_seconds,
                    minimum_body_fraction=request.minimum_body_fraction,
                    progress=plan_progress,
                    analysis_workers=request.analysis_workers,
                    graph_workers=request.graph_workers,
                    measure_transitions=request.quality_check,
                    measurement_top_k=request.measurement_top_k,
                    measurement_log=measurement_log,
                    automatic_delivery=request.resolved_automatic_delivery,
                    listener=listener,
                    timing_intent=SetTimingIntent(
                    target_duration_seconds=request.target_duration_seconds,
                    peak_fraction=request.peak_fraction,
                    closing_window_seconds=request.closing_window_seconds,
                    ending_energy=request.ending_energy,
                ),
                )
            else:
                analysis_stats = AnalysisBatchStats()
                analyses = analyze_tracks(
                    paths,
                    use_cache=request.use_cache,
                    progress=plan_progress,
                    workers=request.analysis_workers,
                    stats=analysis_stats,
                )
                enrich_analyses_with_cached_stems(analyses)
                enrich_analyses_with_performance_layers(analyses)
                plan = make_set_plan(
                    analyses,
                    source="api-track-list",
                    tempo_policy=request.tempo_policy,
                    preferred_style=request.preferred_style,
                    journey=request.journey,
                    minimum_body_seconds=request.minimum_body_seconds,
                    minimum_body_fraction=request.minimum_body_fraction,
                    progress=plan_progress,
                    graph_workers=request.graph_workers,
                    quality_loop=QualityLoopConfig.from_environment(
                        enabled=request.quality_check,
                        top_k=request.measurement_top_k,
                        log_path=measurement_log,
                    ),
                    automatic_delivery=request.resolved_automatic_delivery,
                    listener=listener,
                    timing_intent=SetTimingIntent(
                    target_duration_seconds=request.target_duration_seconds,
                    peak_fraction=request.peak_fraction,
                    closing_window_seconds=request.closing_window_seconds,
                    ending_energy=request.ending_energy,
                ),
                )
                performance = dict(plan.metadata.get("performance", {}))
                performance["analysis"] = analysis_stats.to_dict()
                plan.metadata["performance"] = performance
        else:
            assert folder is not None
            plan = plan_folder(
                folder,
                use_cache=request.use_cache,
                recursive=request.recursive,
                tempo_policy=request.tempo_policy,
                preferred_style=request.preferred_style,
                journey=request.journey,
                start_track=request.start_track,
                end_track=request.end_track,
                minimum_body_seconds=request.minimum_body_seconds,
                minimum_body_fraction=request.minimum_body_fraction,
                progress=plan_progress,
                analysis_workers=request.analysis_workers,
                graph_workers=request.graph_workers,
                measure_transitions=request.quality_check,
                measurement_top_k=request.measurement_top_k,
                measurement_log=measurement_log,
                automatic_delivery=request.resolved_automatic_delivery,
                listener=listener,
                timing_intent=SetTimingIntent(
                    target_duration_seconds=request.target_duration_seconds,
                    peak_fraction=request.peak_fraction,
                    closing_window_seconds=request.closing_window_seconds,
                    ending_energy=request.ending_energy,
                ),
            )
        context.update_absolute("finalizing", 0.99, "Saving Studio plan", current=1, total=1)
        output = save_set_plan(plan, context.output_dir / "set-plan.json")
        return {
            "plan_path": _relative(output),
            "plan": plan.to_dict(include_analysis=False),
        }

    job = manager.submit(
        "set-plan",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={
            "track_paths": [str(value) for value in request.track_paths],
            "journey": request.journey,
            "manual": request.manual,
            "experience": request.experience,
        },
    )
    return {"job": job.to_dict()}


@app.post("/api/sets/automatic", status_code=202)
def create_automatic_set(
    request: AutomaticSetRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    """Upload/select tracks and receive one finished, self-checked set job."""
    paths = [_audio_library_path(value, account=account) for value in request.track_paths]
    output_name = _safe_filename(request.output_name)
    if not output_name.lower().endswith(".wav"):
        output_name += ".wav"
    listener = _account_listener(account)
    feedback_listener = _account_listener(account, require_personalization=False)
    preferences = dict(account.get("preferences", {}))

    def task(context: JobContext) -> dict[str, Any]:
        job_started = time.perf_counter()
        stage_timings: dict[str, float] = {}
        context.update_absolute(
            "starting",
            0.01,
            f"Preparing {len(paths)} tracks",
            current=0,
            total=len(paths),
        )
        stage_started = job_started
        stem_mode = automatic_stem_mode()
        # Background separation must never block the latency-sensitive Create
        # Set path. Upload already queues Demucs work; here we consume completed
        # caches immediately and fall back to the stereo master for unfinished
        # tracks. Users who explicitly want synchronization can select "wait".
        prepared_stems: dict[str, dict[str, str]] = {
            str(path): find_cached_stems(path) for path in paths
        }
        if stem_mode == "wait" and demucs_available():
            prepared_stems = prepare_stems_for_tracks(
                paths,
                progress=context.ranged_progress_callback(0.01, 0.08),
            )
        else:
            context.update_absolute(
                "stems",
                0.08,
                "Using ready audio layers; background stems stay non-blocking",
                current=len(paths),
                total=len(paths),
            )
        stage_timings["stems_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        analysis_stats = AnalysisBatchStats()
        context.update_absolute(
            "analysis",
            0.08,
            f"Understanding {len(paths)} tracks",
            current=0,
            total=len(paths),
        )
        analyses = analyze_tracks(
            paths,
            use_cache=request.use_cache,
            progress=context.ranged_progress_callback(0.08, 0.32),
            workers=request.analysis_workers,
            stats=analysis_stats,
        )
        stem_count = enrich_analyses_with_cached_stems(analyses)
        layer_count = enrich_analyses_with_performance_layers(analyses)
        stage_timings["analysis_seconds"] = time.perf_counter() - stage_started
        try:
            source_fingerprints = [FEATURE_STORE.cached_source_fingerprint(path) for path in paths]
            crate_fingerprint = (
                combine_crate_fingerprints([value for value in source_fingerprints if value is not None])
                if source_fingerprints and all(value is not None for value in source_fingerprints)
                else None
            )
        except Exception:
            # Benchmark identity is evidence, never a reason to fail Create.
            # Do not hash uncached audio on the latency-sensitive Create path.
            crate_fingerprint = None

        stage_started = time.perf_counter()
        context.update_absolute(
            "graph",
            0.32,
            "Scouting track order and transition opportunities",
            current=0,
            total=max(1, len(paths) * max(0, len(paths) - 1)),
        )
        planning_progress = context.ranged_progress_callback(
            0.32,
            0.80,
            stage_ranges={
                "graph": (0.32, 0.50),
                "order-replan": (0.50, 0.56),
                "measure": (0.56, 0.80),
            },
        )
        plan = make_set_plan(
            analyses,
            source="automatic-api",
            tempo_policy="safe",
            preferred_style="auto",
            journey=request.journey,
            minimum_body_seconds=request.minimum_body_seconds,
            minimum_body_fraction=request.minimum_body_fraction,
            progress=planning_progress,
            graph_workers=request.graph_workers,
            quality_loop=QualityLoopConfig.from_environment(
                enabled=True,
                top_k=request.measurement_top_k,
                log_path=str(WORKSPACE / "ratings" / "automatic-measurements.jsonl"),
            ),
            automatic_delivery=True,
            listener=listener,
            timing_intent=SetTimingIntent(
                target_duration_seconds=request.target_duration_seconds,
                peak_fraction=request.peak_fraction,
                closing_window_seconds=request.closing_window_seconds,
                ending_energy=request.ending_energy,
            ),
        )
        plan.metadata["stem_enriched_tracks"] = stem_count
        plan.metadata["performance_layer_enriched_tracks"] = layer_count
        plan.metadata["automatic_performance_layers"] = {
            "mode": stem_mode,
            "demucs_available": demucs_available(),
            "prepared_tracks": sum(bool(value) for value in prepared_stems.values()),
            "blocking_wait": bool(stem_mode == "wait" and demucs_available()),
            "background_nonblocking": stem_mode == "background",
            "one_master_input": True,
        }
        stage_timings["planning_seconds"] = time.perf_counter() - stage_started
        performance = dict(plan.metadata.get("performance", {}))
        performance["analysis"] = analysis_stats.to_dict()
        performance["latency"] = {**stage_timings}
        plan.metadata["performance"] = performance
        plan_path = save_set_plan(plan, context.output_dir / "set-plan.json")
        context.update_absolute(
            "render",
            0.80,
            f"Rendering {max(1, len(plan.transitions))} transitions into the final set",
            current=0,
            total=max(1, len(plan.transitions) + 1),
        )

        stage_started = time.perf_counter()
        audio_path, report_path = render_set(
            plan,
            context.output_dir / output_name,
            progress=context.ranged_progress_callback(0.80, 0.985),
        )
        stage_timings["render_seconds"] = time.perf_counter() - stage_started
        stage_timings["total_delivery_seconds"] = time.perf_counter() - job_started
        peak_rss = _process_peak_rss_mb()
        if peak_rss is not None:
            stage_timings["process_peak_rss_mb"] = peak_rss
        performance = dict(plan.metadata.get("performance", {}))
        performance["latency"] = dict(stage_timings)
        plan.metadata["performance"] = performance
        plan.metadata["release_benchmark"] = extract_release_benchmark(
            plan,
            suite=f"runtime-{len(paths)}",
            crate_fingerprint=crate_fingerprint,
            stage_timings=stage_timings,
            process_peak_rss_mb=peak_rss,
        ).to_dict()
        # The plan was initially persisted before rendering. Re-save it with
        # final delivery telemetry so later benchmark/diagnostic tools see the
        # exact run that produced the master audio.
        save_set_plan(plan, plan_path)
        context.update_absolute(
            "finalizing",
            0.99,
            "Saving the set and learning from the finished performance",
            current=1,
            total=1,
        )

        append_feedback_event(
            event="render_completed",
            listener=feedback_listener,
            plan_id=plan.plan_id,
            journey=request.journey,
            metadata={"automatic": True, "fallbacks": plan.metadata.get("fallback_transition_count", 0)},
            feedback_path=WORKSPACE / "ratings" / "feedback-events.jsonl",
            profiles_path=WORKSPACE / "ratings" / "listener-profiles.json",
            personalization_enabled=bool(preferences.get("personalization_enabled", False)),
            training_consent=bool(preferences.get("improve_saund_ai", False)),
        )
        def refit_task(_context: JobContext) -> dict[str, Any]:
            try:
                updated_model = maybe_refit_scoring_model(
                    ratings_path=WORKSPACE / "ratings" / "listening-tests.jsonl",
                    measurement_path=WORKSPACE / "ratings" / "automatic-measurements.jsonl",
                    output_path=WORKSPACE / "config" / "scoring-model.json",
                )
                return {"updated_model": _relative(updated_model) if updated_model is not None else None}
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        refit_job = manager.submit(
            "scoring-refit",
            refit_task,
            owner_user_id=_account_user_id(account),
            resource_context={"source_plan_id": plan.plan_id},
        )
        return {
            "plan_path": _relative(plan_path),
            "plan": plan.to_dict(include_analysis=False),
            "audio_path": _relative(audio_path),
            "report_path": _relative(report_path),
            "audio_url": _publish_job_media(audio_path, account=account, job_id=context.job_id),
            "live_stream_url": f"/api/live/stream?plan_path={_relative(plan_path)}",
            "live_manifest_url": f"/api/live/manifest?plan_path={_relative(plan_path)}",
            "performance": stage_timings,
            "learning_refit_job_id": refit_job.job_id,
        }

    job = manager.submit(
        "automatic-set",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={
            "track_paths": [str(value) for value in request.track_paths],
            "journey": request.journey,
            "experience": "simple",
            "automatic_delivery": True,
        },
    )
    return {"job": job.to_dict()}


@app.post("/api/set/render", status_code=202)
def create_render(
    request: RenderRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    plan_path = _require_owned_workspace_path(request.plan_path, account)
    plan = load_set_plan(plan_path)
    if plan.requires_confirmation:
        payload = confirmation_payload(plan.plan_id, plan.order, plan.transitions)
        valid = bool(request.confirmation_token) and verify_confirmation_token(payload, request.confirmation_token or "")
        if not request.confirm or not valid:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This plan contains risky or poor transitions and needs explicit confirmation.",
                    "requires_confirmation": True,
                    "confirmation_token": plan.confirmation_token,
                    "transitions": [edge.to_dict() for edge in plan.transitions if edge.quality != "good"],
                },
            )
        plan.confirmed = True

    output_name = _safe_filename(request.output_name)
    if not output_name.lower().endswith(".wav"):
        output_name += ".wav"

    def task(context: JobContext) -> dict[str, Any]:
        render_started = time.perf_counter()
        context.update_absolute(
            "render",
            0.02,
            f"Rendering {max(1, len(plan.transitions))} transitions",
            current=0,
            total=max(1, len(plan.transitions) + 1),
        )
        output, report = render_set(
            plan,
            context.output_dir / output_name,
            progress=context.ranged_progress_callback(0.02, 0.99),
        )
        render_seconds = time.perf_counter() - render_started
        peak_rss = _process_peak_rss_mb()
        performance = dict(plan.metadata.get("performance", {}))
        latency = dict(performance.get("latency", {}))
        latency["render_seconds"] = render_seconds
        latency["render_job_seconds"] = render_seconds
        if peak_rss is not None:
            latency["process_peak_rss_mb"] = peak_rss
        performance["latency"] = latency
        plan.metadata["performance"] = performance
        plan.metadata["release_benchmark"] = extract_release_benchmark(
            plan,
            suite=f"runtime-{len(plan.order)}",
            stage_timings=latency,
            process_peak_rss_mb=peak_rss,
        ).to_dict()
        save_set_plan(plan, plan_path)
        return {
            "audio_path": _relative(output),
            "report_path": _relative(report),
            "audio_url": _publish_job_media(output, account=account, job_id=context.job_id),
            "performance": latency,
            "release_benchmark": plan.metadata["release_benchmark"],
        }

    job = manager.submit(
        "set-render",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={"source_plan_path": request.plan_path},
    )
    return {"job": job.to_dict()}


@app.post("/api/set/previews", status_code=202)
def create_previews(
    request: PreviewRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    plan_path = _require_owned_workspace_path(request.plan_path, account)
    plan = load_set_plan(plan_path)

    def task(context: JobContext) -> dict[str, Any]:
        context.update_absolute(
            "render-preview",
            0.02,
            f"Preparing {max(1, len(plan.transitions))} transition previews",
            current=0,
            total=max(1, len(plan.transitions)),
        )
        outputs, report = render_set_previews(
            plan,
            context.output_dir / "previews",
            pre_roll_seconds=request.pre_roll_seconds,
            post_roll_seconds=request.post_roll_seconds,
            progress=context.ranged_progress_callback(0.02, 0.99),
        )
        return {
            "preview_paths": [_relative(path) for path in outputs],
            "preview_urls": [_publish_job_media(path, account=account, job_id=context.job_id) for path in outputs],
            "report_path": _relative(report),
        }

    job = manager.submit(
        "set-previews",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={"source_plan_path": request.plan_path},
    )
    return {"job": job.to_dict()}


@app.post("/api/studio/transition-preview", status_code=202)
def create_studio_transition_preview(
    request: StudioTransitionPreviewRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    """Render one already-prepared transition as a Studio rehearsal.

    This endpoint is intentionally non-authoritative: it never edits a set
    plan or installs a different execution program. Selecting an alternative
    in Studio remains a preference until a fresh set-plan job runs through the
    normal Executive / Professional Brain pipeline.
    """
    plan_path = _require_owned_workspace_path(request.plan_path, account)
    plan = load_set_plan(plan_path)
    if request.transition_index >= len(plan.transitions):
        raise HTTPException(status_code=400, detail="Transition index is outside this set plan.")
    edge = plan.transitions[request.transition_index]
    if request.alternative_index is None:
        selected = edge.plan
        variant = "primary"
    else:
        if request.alternative_index >= len(edge.alternatives):
            raise HTTPException(status_code=400, detail="Alternative index is outside this transition.")
        selected = edge.alternatives[request.alternative_index]
        variant = f"alternative-{request.alternative_index + 1}"

    def task(context: JobContext) -> dict[str, Any]:
        context.update_absolute(
            "studio-audition",
            0.05,
            f"Rehearsing {selected.technique.replace('_', ' ')}",
            current=0,
            total=1,
        )
        filename = _safe_filename(
            f"{request.transition_index + 1:02d}-{variant}-{selected.technique}.wav"
        )
        output = render_transition(
            selected,
            context.output_dir / filename,
            pre_roll_seconds=request.pre_roll_seconds,
            post_roll_seconds=request.post_roll_seconds,
        )
        context.update_absolute("studio-audition", 0.99, "Rehearsal ready", current=1, total=1)
        return {
            "preview_path": _relative(output),
            "preview_url": _publish_job_media(output, account=account, job_id=context.job_id),
            "transition_index": request.transition_index,
            "alternative_index": request.alternative_index,
            "variant": variant,
            "execution_signature": selected.execution_signature,
            "technique": selected.technique,
            "intent": selected.intent,
            "score": round(float(selected.score), 6),
        }

    job = manager.submit(
        "studio-transition-preview",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={
            "source_plan_path": request.plan_path,
            "transition_index": request.transition_index,
            "alternative_index": request.alternative_index,
        },
    )
    return {"job": job.to_dict()}


@app.post("/api/stems/separate", status_code=202)
def create_stems(
    request: StemRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    track = _audio_library_path(request.track_path, account=account)

    def task(context: JobContext) -> dict[str, Any]:
        context.update("stems", 0, 1, track.name)
        mapping = separate_stems(
            track,
            model=request.model,
            device=request.device,
            force=request.force,
        )
        context.update("stems", 1, 1, track.name)
        return {"stems": {name: _relative(Path(path)) for name, path in mapping.items()}}

    job = stem_manager.submit(
        "stem-separation",
        task,
        owner_user_id=_account_user_id(account),
        resource_context={"track_path": request.track_path},
    )
    return {"job": job.to_dict()}


@app.post("/api/ratings")
def save_rating(
    request: RatingRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    report = _require_owned_workspace_path(request.report_path, account)
    output = append_listening_rating(
        report,
        alternative_id=request.alternative,
        transition_index=request.transition_index,
        beat_alignment=request.beat,
        phrase_timing=request.phrase,
        bass_cleanliness=request.bass,
        energy_flow=request.energy,
        cue_quality=request.cue,
        overall=request.overall,
        notes=request.notes,
        listener=_account_listener(account, require_personalization=False),
        ratings_path=WORKSPACE / "ratings" / "listening-tests.jsonl",
    )
    return {"ratings_path": _relative(output)}


@app.post("/api/feedback")
def save_feedback(
    request: FeedbackRequest,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    plan_payload: dict[str, Any] | None = None
    plan_id = request.plan_id
    journey = request.journey
    if request.plan_path:
        plan_path = _require_owned_workspace_path(request.plan_path, account)
        plan = load_set_plan(plan_path)
        plan_id = plan_id or plan.plan_id
        journey = journey or str(plan.metadata.get("journey", "balanced"))
        plan_payload = {
            "ordered_tracks": [plan.tracks[index].title for index in plan.order],
            "transition_count": len(plan.transitions),
            "fallback_count": int(plan.metadata.get("fallback_transition_count", 0)),
            "automatic_delivery": plan.automatic_delivery,
        }
    preferences = dict(account.get("preferences", {}))
    output, profile = append_feedback_event(
        event=request.event,
        listener=_account_listener(account, require_personalization=False),
        plan_id=plan_id,
        transition_index=request.transition_index,
        value=request.value,
        journey=journey,
        plan=plan_payload,
        metadata=request.metadata,
        feedback_path=WORKSPACE / "ratings" / "feedback-events.jsonl",
        profiles_path=WORKSPACE / "ratings" / "listener-profiles.json",
        personalization_enabled=bool(preferences.get("personalization_enabled", False)),
        training_consent=bool(preferences.get("improve_saund_ai", False)),
    )
    return {
        "feedback_path": _relative(output),
        "listener_profile": profile or None,
        "personalization_applied": bool(preferences.get("personalization_enabled", False)),
        "training_eligible": bool(preferences.get("improve_saund_ai", False)),
        "privacy_version": PRIVACY_VERSION,
    }


def refit_scoring(request: RefitRequest) -> dict[str, Any]:
    ratings = _workspace_path(request.ratings_path)
    measurements = (
        _workspace_path(request.measurement_path)
        if request.measurement_path and _workspace_path(request.measurement_path, must_exist=False).exists()
        else None
    )
    output = _workspace_path(request.output_path, must_exist=False)
    result = fit_scoring_model(
        ratings,
        output,
        measurement_path=measurements,
    )
    payload = json.loads(result.read_text(encoding="utf-8"))
    return {
        "model_path": _relative(result),
        "sample_count": payload.get("sample_count", 0),
        "train_rmse": payload.get("train_rmse"),
        "cv_rmse": payload.get("cv_rmse"),
        "cv_skill": payload.get("cv_skill"),
        "learned_weight": payload.get("learned_weight"),
    }


def train_genre(request: GenreTrainRequest) -> dict[str, Any]:
    manifest = _workspace_path(request.manifest_path)
    output = _workspace_path(request.output_path, must_exist=False)
    result = train_from_manifest(manifest, output)
    return {"model_path": _relative(result)}



def create_intelligence_listening_session(request: ListeningSessionRequest) -> dict[str, Any]:
    _require_training_lab()
    track_a = _workspace_path(request.track_a)
    track_b = _workspace_path(request.track_b)
    manifest = create_listening_session(
        track_a,
        track_b,
        output_root=WORKSPACE / "output" / "listening-lab",
        candidates=request.candidates,
        cache_dir=WORKSPACE / ".djai-cache" / "tracks",
        use_cache=True,
    )
    items = []
    for item in manifest["candidates"]:
        audio = Path(str(item["audio_path"]))
        items.append({
            "label": item["label"],
            "audio_url": f"/api/media/{_relative(audio)}",
        })
    return {"session_id": manifest["session_id"], "candidates": items, "generation": manifest.get("generation", {})}


def save_intelligence_preference(request: PreferenceVoteRequest) -> dict[str, Any]:
    _require_training_lab()
    manifest_path = _workspace_path(
        f"output/listening-lab/{request.session_id}/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = {str(item["label"]).upper(): item for item in manifest.get("candidates", [])}
    label_a = request.candidate_a.upper()
    label_b = request.candidate_b.upper()
    if label_a not in candidates or label_b not in candidates or label_a == label_b:
        raise HTTPException(status_code=400, detail="Invalid listening-lab candidate labels.")
    output = append_preference(
        candidate_a=candidates[label_a]["plan"],
        candidate_b=candidates[label_b]["plan"],
        choice=request.choice,
        listener=request.listener,
        context={"session_id": request.session_id, "candidate_labels": [label_a, label_b], "reasons": list(request.reasons)},
        path=WORKSPACE / "ratings" / "transition-preferences.jsonl",
    )
    rejection_paths: list[str] = []
    if request.choice == "neither":
        for label in (label_a, label_b):
            rejected = append_feature_rejection(
                candidates[label]["plan"],
                reasons=request.reasons,
                severity=1.0,
                source="taste_lab_neither",
                listener=request.listener,
                path=WORKSPACE / "ratings" / "rejected-transitions.jsonl",
            )
            rejection_paths.append(_relative(rejected))
    return {"preferences_path": _relative(output), "rejection_paths": rejection_paths}


def get_human_review_status() -> dict[str, Any]:
    _require_training_lab()
    return human_preference_status(WORKSPACE)


def create_training_lab_human_review_session(request: HumanReviewSessionRequest) -> dict[str, Any]:
    _require_training_lab()
    manifest = create_human_review_session(
        WORKSPACE,
        evaluator=request.evaluator,
        cohort=request.cohort,
        scenario_id=request.scenario_id,
    )
    items = []
    for item in manifest.get("candidates", []):
        audio = Path(str(item["audio_path"]))
        items.append({
            "label": item["label"],
            "audio_url": f"/api/media/{_relative(audio)}",
        })
    return {
        "session_id": manifest["session_id"],
        "teacher_scenario_id": manifest["teacher_scenario_id"],
        "queue_priority": manifest.get("queue_priority", 0.0),
        "queue_reasons": manifest.get("queue_reasons", []),
        "evaluator_cohort": manifest.get("evaluator_cohort"),
        "candidates": items,
    }


def save_training_lab_human_review_vote(request: HumanReviewVoteRequest) -> dict[str, Any]:
    _require_training_lab()
    output = record_human_review_vote(
        WORKSPACE,
        session_id=request.session_id,
        choice=request.choice,
        reasons=request.reasons,
        confidence=request.confidence,
        experience_years=request.experience_years,
    )
    return {
        "preferences_path": _relative(output),
        "status": human_preference_status(WORKSPACE),
    }


def train_intelligence_ranker(request: RankerTrainRequest) -> dict[str, Any]:
    _require_training_lab()
    preferences = _workspace_path(request.preferences_path, must_exist=False)
    output = _workspace_path(request.output_path, must_exist=False)
    result = train_transition_ranker(preferences, output)
    payload = json.loads(result.read_text(encoding="utf-8"))
    return {
        "model_path": _relative(result),
        "comparison_count": payload.get("comparison_count", 0),
        "cv_accuracy": payload.get("cv_accuracy"),
        "baseline_accuracy": payload.get("baseline_accuracy"),
        "authority": payload.get("authority"),
    }


def intelligence_ranker_status() -> dict[str, Any]:
    _require_training_lab()
    report = benchmark_brain(
        WORKSPACE / "ratings" / "transition-preferences.jsonl",
        WORKSPACE / "config" / "transition-ranker.json",
    )
    return report.to_dict()

def professional_brain_status() -> dict[str, Any]:
    cfg = ProfessionalBrainConfig.from_environment()
    ranker = benchmark_brain(
        WORKSPACE / "ratings" / "transition-preferences.jsonl",
        WORKSPACE / "config" / "transition-ranker.json",
    ).to_dict()
    rejection_path = WORKSPACE / cfg.rejection_path if not Path(cfg.rejection_path).is_absolute() else Path(cfg.rejection_path)
    expert_policy_path = WORKSPACE / "config" / "expert-policy.json"
    try:
        expert_policy = json.loads(expert_policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        expert_policy = {"example_count": 0}
    registry = WORKSPACE / "datasets" / "registry.json"
    production_audit = audit_dataset_registry(registry, lane="production") if registry.exists() else {"ready": False, "blockers": [{"reason": "registry_missing"}]}
    return {
        "version": __version__,
        "professional_brain_enabled": cfg.enabled,
        "brain_config": {
            "analytical_pool": cfg.analytical_pool,
            "finalists": cfg.finalists,
            "retry_rounds": cfg.retry_rounds,
            "diversity_weight": cfg.diversity_weight,
            "minimum_judge_score": cfg.minimum_judge_score,
        },
        "preference_ranker": ranker,
        "bootstrap_runtime": bootstrap_runtime_status(),
        "taste_learning": taste_learning_status(WORKSPACE),
        "deck_performance": deck_runtime_status(),
        "professional_mixing": professional_mixing_status(),
        "professional_mix_contract": professional_mix_contract_status(),
        "structural_operations": structural_operation_status(),
        "dj_move_brain": move_brain_status(),
        "dj_executive": executive_status(),
        "musical_timeline": musical_timeline_status(),
        "large_crate_planning": large_crate_runtime_status(),
        "expert_imitation_examples": int(expert_policy.get("example_count", 0) or 0),
        "rejection_memory_count": len(load_rejections(rejection_path)),
        "crate_memory": crate_memory_status(workspace=WORKSPACE),
        "release_benchmarks": benchmark_runtime_status(WORKSPACE),
        "human_release_gate": human_release_status(WORKSPACE),
        "production_data_ready": bool(production_audit.get("ready")),
        "production_data_blockers": production_audit.get("blockers", []),
        "arena": arena_report(WORKSPACE / "ratings" / "arena.jsonl"),
        "curriculum": training_curriculum(),
    }


def release_benchmark_status() -> dict[str, Any]:
    return benchmark_runtime_status(WORKSPACE)


def get_dataset_builder_sources() -> dict[str, Any]:
    _require_training_lab()
    return source_catalog()


def get_dataset_builder_status() -> dict[str, Any]:
    _require_training_lab()
    return dataset_builder_status(WORKSPACE)


def build_dataset_builder_corpus(
    name: str = Query(default="bootstrap-v1", min_length=1, max_length=80),
    production: bool = Query(default=False),
) -> dict[str, Any]:
    _require_training_lab()
    try:
        corpus = create_bootstrap_corpus(WORKSPACE, name=name, production=production)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    return {"corpus_id": manifest.get("corpus_id", corpus.name), "corpus_path": _relative(corpus), "manifest": manifest}


def train_dataset_builder_shadows(
    corpus_id: str | None = Query(default=None, min_length=1, max_length=160),
) -> dict[str, Any]:
    _require_training_lab()
    try:
        result = train_bootstrap_shadow_models(WORKSPACE, corpus_id=corpus_id)
        clear_bootstrap_runtime_cache()
        result["runtime"] = bootstrap_runtime_status()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_training_lab_status() -> dict[str, Any]:
    _require_training_lab()
    return training_lab_status(WORKSPACE)


def get_training_lab_sessions(limit: int = Query(default=30, ge=1, le=200)) -> dict[str, Any]:
    _require_training_lab()
    return {"sessions": list_training_sessions(WORKSPACE, limit=limit)}


def snapshot_training_lab_dataset(request: TrainingSnapshotRequest) -> dict[str, Any]:
    _require_training_lab()
    try:
        snapshot = create_training_snapshot(WORKSPACE, name=request.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    return {
        "dataset_id": manifest.get("dataset_id", snapshot.name),
        "dataset_path": _relative(snapshot),
        "manifest": manifest,
    }


def train_training_lab_shadow_model(request: TrainingLabTrainRequest) -> dict[str, Any]:
    _require_training_lab()
    snapshot_id = request.snapshot_id
    if snapshot_id is None:
        try:
            snapshot = create_training_snapshot(WORKSPACE, name="taste-learning")
            snapshot_id = snapshot.name
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        return train_taste_learning_heads(WORKSPACE, snapshot_id=snapshot_id, ridge=request.ridge)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def get_taste_learning_status() -> dict[str, Any]:
    _require_training_lab()
    return taste_learning_status(WORKSPACE)


def train_real_taste(request: TrainingLabTrainRequest) -> dict[str, Any]:
    _require_training_lab()
    return train_real_taste_from_snapshot(
        WORKSPACE,
        snapshot_id=request.snapshot_id,
        ridge=request.ridge,
    )


def get_real_taste_status() -> dict[str, Any]:
    _require_training_lab()
    return real_taste_status(WORKSPACE)


def lock_training_lab_human_release(request: HumanReleaseLockRequest) -> dict[str, Any]:
    _require_training_lab()
    output = Path(request.output_path).expanduser() if request.output_path else WORKSPACE / HUMAN_RELEASE_DEFAULT_MANIFEST
    if not output.is_absolute():
        output = WORKSPACE / output
    try:
        locked = lock_human_release_benchmark(
            request.cases_path,
            output,
            incumbent=request.incumbent,
            candidate=request.candidate,
        )
        payload = json.loads(locked.read_text(encoding="utf-8"))
        return {
            "benchmark_path": _relative(locked),
            "benchmark_id": payload.get("benchmark_id"),
            "locked_sha256": payload.get("locked_sha256"),
            "case_count": payload.get("case_count"),
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def create_training_lab_human_release_session(request: HumanReleaseSessionRequest) -> dict[str, Any]:
    _require_training_lab()
    try:
        session = create_human_release_session(
            WORKSPACE,
            evaluator=request.evaluator,
            cohort=request.cohort,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        **session,
        "audio": {
            label: f"/api/media/{_relative(Path(path))}"
            for label, path in dict(session.get("audio", {})).items()
        },
    }


def save_training_lab_human_release_vote(request: HumanReleaseVoteRequest) -> dict[str, Any]:
    _require_training_lab()
    try:
        output = record_human_release_vote(
            WORKSPACE,
            session_id=request.session_id,
            choice=request.choice,
            catastrophic_a=request.catastrophic_a,
            catastrophic_b=request.catastrophic_b,
            reasons=request.reasons,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "votes_path": _relative(output),
        "report": human_release_report(WORKSPACE),
    }


def get_training_lab_human_release_report() -> dict[str, Any]:
    _require_training_lab()
    try:
        return human_release_report(WORKSPACE)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


register_training_routers(
    app,
    {
        "get_dataset_builder_sources": get_dataset_builder_sources,
        "get_dataset_builder_status": get_dataset_builder_status,
        "build_dataset_builder_corpus": build_dataset_builder_corpus,
        "train_dataset_builder_shadows": train_dataset_builder_shadows,
        "get_training_lab_status": get_training_lab_status,
        "get_training_lab_sessions": get_training_lab_sessions,
        "snapshot_training_lab_dataset": snapshot_training_lab_dataset,
        "train_training_lab_shadow_model": train_training_lab_shadow_model,
        "get_taste_learning_status": get_taste_learning_status,
        "train_real_taste": train_real_taste,
        "get_real_taste_status": get_real_taste_status,
        "lock_training_lab_human_release": lock_training_lab_human_release,
        "create_training_lab_human_release_session": create_training_lab_human_release_session,
        "save_training_lab_human_release_vote": save_training_lab_human_release_vote,
        "get_training_lab_human_release_report": get_training_lab_human_release_report,
    },
)


register_status_and_intelligence_routers(
    app,
    {
        "health": health,
        "hosting_status": hosting_status,
        "runtime_status": runtime_status,
        "phase16_validation_status": phase16_validation_status,
        "readiness": readiness,
        "release_status": release_status,
        "release_readiness": release_readiness,
        "model_status": model_status,
        "professional_brain_status": professional_brain_status,
        "release_benchmark_status": release_benchmark_status,
        "refit_scoring": refit_scoring,
        "train_genre": train_genre,
        "create_intelligence_listening_session": create_intelligence_listening_session,
        "save_intelligence_preference": save_intelligence_preference,
        "get_human_review_status": get_human_review_status,
        "create_training_lab_human_review_session": create_training_lab_human_review_session,
        "save_training_lab_human_review_vote": save_training_lab_human_review_vote,
        "train_intelligence_ranker": train_intelligence_ranker,
        "intelligence_ranker_status": intelligence_ranker_status,
    },
)


@app.get("/api/live/manifest")
def get_live_manifest(
    response: Response,
    account: Annotated[dict[str, Any], Depends(_require_account)],
    plan_path: str = Query(...),
) -> dict[str, Any]:
    plan = load_set_plan(_require_owned_workspace_path(plan_path, account))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return live_manifest(plan)


@app.get("/api/live/events")
def live_events(
    account: Annotated[dict[str, Any], Depends(_require_account)],
    plan_path: str = Query(...),
    session_id: str = Query(..., min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
    after: int = -1,
) -> StreamingResponse:
    plan = load_set_plan(_require_owned_workspace_path(plan_path, account))
    ensure_live_telemetry_session(plan.plan_id, session_id)

    def stream():
        sequence = int(after)
        while True:
            event = wait_for_live_telemetry(
                plan.plan_id,
                session_id,
                sequence,
                timeout=12.0,
            )
            if event is None:
                yield ": keep-alive\n\n"
                continue
            sequence = int(event["sequence"])
            yield f"id: {sequence}\ndata: {json.dumps(event)}\n\n"
            if live_telemetry_terminal(event):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/live/stream")
def stream_live_set(
    account: Annotated[dict[str, Any], Depends(_require_account)],
    plan_path: str = Query(...),
    bitrate: str = Query(default="192k", pattern=r"^(96|128|160|192|224|256|320)k$"),
    session_id: str | None = Query(default=None, min_length=8, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$"),
) -> StreamingResponse:
    plan = load_set_plan(_require_owned_workspace_path(plan_path, account))
    if session_id:
        ensure_live_telemetry_session(plan.plan_id, session_id)
    return StreamingResponse(
        iter_mp3_stream(plan, bitrate=bitrate, telemetry_session_id=session_id),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
            "Content-Disposition": f'inline; filename="djai-live-{plan.plan_id[:8]}.mp3"',
        },
    )


def _manager_for_job(job_id: str) -> JobManager:
    for candidate in (manager, stem_manager):
        try:
            candidate.get(job_id)
            return candidate
        except KeyError:
            continue
    raise KeyError(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(
    job_id: str,
    response: Response,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    _, record = _owned_job(job_id, account)
    return record.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    response: Response,
    account: Annotated[dict[str, Any], Depends(_require_account)],
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    job_manager, _ = _owned_job(job_id, account)
    return job_manager.cancel(job_id).to_dict()


@app.get("/api/jobs/{job_id}/events")
def job_events(
    job_id: str,
    account: Annotated[dict[str, Any], Depends(_require_account)],
    after: int = -1,
) -> StreamingResponse:
    job_manager, _ = _owned_job(job_id, account)

    def stream():
        sequence = after
        while True:
            events = job_manager.wait_for_events(job_id, sequence, timeout=12.0)
            if not events:
                yield ": keep-alive\n\n"
            for event in events:
                sequence = int(event["sequence"])
                yield f"id: {sequence}\ndata: {json.dumps(event)}\n\n"
            record = job_manager.get(job_id)
            if record.status in {"completed", "failed", "cancelled"} and (not record.events or sequence >= int(record.events[-1]["sequence"])):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/media/{relative_path:path}")
def media(relative_path: str, http_request: Request) -> Response:
    path = _workspace_path(relative_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")

    # Guest source browsing/upload stays available so users can choose music
    # before login. Generated set artifacts are private to the account whose
    # job produced them. Everything else in the workspace is non-public.
    if _path_is_within(path, WORKSPACE / "audio"):
        if HOSTING.hosted_mode:
            account = _require_account(http_request)
            if not _path_is_within(path, _account_audio_root(account)):
                raise HTTPException(status_code=404, detail="Media file not found")
            hosted_url = object_store.media_url_for_path(path)
            if hosted_url:
                return RedirectResponse(hosted_url, status_code=307, headers={"Cache-Control": "private, no-store"})
        return FileResponse(path, filename=path.name, headers={"Cache-Control": "private, no-store"} if HOSTING.hosted_mode else None)
    if TRAINING_LAB_ENABLED and _path_is_within(path, WORKSPACE / "output" / "listening-lab"):
        return FileResponse(path, filename=path.name)

    resolved_job = _job_for_workspace_path(path)
    if resolved_job is None:
        raise HTTPException(status_code=404, detail="Media file not found")
    account = _require_account(http_request)
    _, record = resolved_job
    if not record.owner_user_id or record.owner_user_id != _account_user_id(account):
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(
        path,
        filename=path.name,
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


@app.exception_handler(ValueError)
def value_error_handler(request: object, exc: ValueError) -> JSONResponse:
    del request
    return JSONResponse(status_code=400, content={"detail": str(exc)})
