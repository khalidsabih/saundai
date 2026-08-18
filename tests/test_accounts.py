from __future__ import annotations

import importlib
import sqlite3
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from djai.accounts import AccountStore


def test_account_store_hashes_passwords_and_revokes_sessions(tmp_path: Path) -> None:
    database = tmp_path / "accounts.sqlite3"
    store = AccountStore(database)

    snapshot = store.create_user(" Test@Example.com ", "correct-horse-battery-staple")
    assert snapshot["user"]["email"] == "test@example.com"
    assert snapshot["preferences"] == {
        "personalization_enabled": False,
        "improve_saund_ai": False,
        "marketing_consent": False,
    }

    with sqlite3.connect(database) as connection:
        stored_hash = connection.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (snapshot["user"]["id"],),
        ).fetchone()[0]
    assert stored_hash.startswith("$argon2id$")
    assert "correct-horse-battery-staple" not in stored_hash

    assert store.authenticate("test@example.com", "wrong-password") is None
    authenticated = store.authenticate("TEST@example.com", "correct-horse-battery-staple")
    assert authenticated is not None
    assert authenticated["user"]["id"] == snapshot["user"]["id"]

    session = store.create_session(snapshot["user"]["id"], ttl_seconds=3600)
    active = store.get_session(session.token)
    assert active is not None
    assert active["user"]["email"] == "test@example.com"

    updated = store.update_preferences(
        snapshot["user"]["id"],
        personalization_enabled=True,
        improve_saund_ai=True,
    )
    assert updated["preferences"] == {
        "personalization_enabled": True,
        "improve_saund_ai": True,
        "marketing_consent": False,
    }

    store.delete_session(session.token)
    assert store.get_session(session.token) is None



def test_account_store_migrates_foundation_2_database_without_losing_user(tmp_path: Path) -> None:
    from argon2 import PasswordHasher

    database = tmp_path / "accounts.sqlite3"
    now = "2026-08-13T20:00:00+00:00"
    password = "legacy-account-password"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                disabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE user_preferences (
                user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                personalization_enabled INTEGER NOT NULL DEFAULT 0,
                improve_saund_ai INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO users(id, email, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("legacy-user", "legacy@example.com", PasswordHasher().hash(password), now, now),
        )
        connection.execute(
            "INSERT INTO user_preferences(user_id, personalization_enabled, improve_saund_ai, updated_at) VALUES (?, 1, 0, ?)",
            ("legacy-user", now),
        )

    store = AccountStore(database)
    snapshot = store.authenticate("legacy@example.com", password)
    assert snapshot is not None
    assert snapshot["user"]["id"] == "legacy-user"
    assert snapshot["profile"] == {
        "first_name": "",
        "last_name": "",
        "artist_name": None,
        "country": "",
        "primary_use": None,
        "experience_level": None,
        "preferred_genres": [],
        "musical_priorities": [],
    }
    assert snapshot["preferences"]["personalization_enabled"] is True
    assert snapshot["preferences"]["marketing_consent"] is False

    updated = store.update_profile(
        "legacy-user",
        first_name="Legacy",
        last_name="Listener",
        country="Germany",
        primary_use="dj",
        primary_use_supplied=True,
        preferred_genres=["House"],
    )
    assert updated["profile"]["first_name"] == "Legacy"
    assert updated["profile"]["primary_use"] == "dj"
    assert updated["profile"]["preferred_genres"] == ["House"]

def test_auth_api_sets_http_only_session_and_build_requires_login(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("SAUND_AUTH_COOKIE_SECURE", "0")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)

    guest = client.get("/api/auth/me")
    assert guest.status_code == 200
    assert guest.json()["authenticated"] is False

    blocked = client.post(
        "/api/sets/automatic",
        json={"track_paths": ["audio/a.wav", "audio/b.wav"]},
    )
    assert blocked.status_code == 401
    assert blocked.json()["detail"]["code"] == "login_required"

    registered = client.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "listener@example.com", "password": "a-long-local-password", "country": "Germany", "primary_use": "personal"},
    )
    assert registered.status_code == 201
    assert registered.json()["authenticated"] is True
    assert registered.json()["profile"]["first_name"] == "Test"
    assert registered.json()["profile"]["country"] == "Germany"
    assert registered.json()["preferences"]["marketing_consent"] is False
    set_cookie = registered.headers["set-cookie"].lower()
    assert "saund_session=" in set_cookie
    assert "httponly" in set_cookie
    assert "samesite=lax" in set_cookie

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "listener@example.com"

    preferences = client.patch(
        "/api/account/preferences",
        json={"personalization_enabled": True, "improve_saund_ai": False},
    )
    assert preferences.status_code == 200
    assert preferences.json()["preferences"]["personalization_enabled"] is True
    assert preferences.json()["preferences"]["improve_saund_ai"] is False

    profile = client.patch(
        "/api/account/profile",
        json={
            "artist_name": "Test Selector",
            "primary_use": "dj",
            "experience_level": "experienced",
            "preferred_genres": ["House", "Techno"],
            "musical_priorities": ["smooth_transitions", "energy_progression"],
        },
    )
    assert profile.status_code == 200
    assert profile.json()["profile"]["artist_name"] == "Test Selector"
    assert profile.json()["profile"]["primary_use"] == "dj"
    assert profile.json()["profile"]["preferred_genres"] == ["House", "Techno"]

    marketing = client.patch("/api/account/preferences", json={"marketing_consent": True})
    assert marketing.status_code == 200
    assert marketing.json()["preferences"]["marketing_consent"] is True

    logout = client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert logout.json()["authenticated"] is False
    assert client.get("/api/auth/me").json()["authenticated"] is False


def test_account_owned_jobs_and_generated_media_are_private(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_STEM_JOB_WORKERS", "1")
    monkeypatch.setenv("SAUND_AUTH_COOKIE_SECURE", "0")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")

    owner = TestClient(api.app)
    other = TestClient(api.app)
    guest = TestClient(api.app)

    owner_registration = owner.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "owner@example.com", "password": "a-long-owner-password", "country": "Germany", "primary_use": "personal"},
    )
    other_registration = other.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "other@example.com", "password": "a-long-other-password", "country": "Germany", "primary_use": "personal"},
    )
    assert owner_registration.status_code == 201
    assert other_registration.status_code == 201
    owner_user_id = owner_registration.json()["user"]["id"]

    job = api.manager.submit(
        "test-private-job",
        lambda context: {"ok": True},
        owner_user_id=owner_user_id,
    )
    artifact = api.JOBS / job.job_id / "private.wav"
    artifact.write_bytes(b"private-audio")
    plan = api.JOBS / job.job_id / "set-plan.json"
    plan.write_text("{}", encoding="utf-8")

    assert owner.get(f"/api/jobs/{job.job_id}").status_code == 200
    assert guest.get(f"/api/jobs/{job.job_id}").status_code == 401
    assert other.get(f"/api/jobs/{job.job_id}").status_code == 404

    media_path = f"output/jobs/{job.job_id}/private.wav"
    assert owner.get(f"/api/media/{media_path}").status_code == 200
    assert guest.get(f"/api/media/{media_path}").status_code == 401
    assert other.get(f"/api/media/{media_path}").status_code == 404

    # Workspace files are never exposed merely because the caller knows a path.
    assert owner.get("/api/media/accounts/accounts.sqlite3").status_code == 404

    source = tmp_path / "audio" / "uploads" / "guest-source.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-audio")
    assert guest.get("/api/media/audio/uploads/guest-source.wav").status_code == 200

    foreign_plan = f"output/jobs/{job.job_id}/set-plan.json"
    assert other.post("/api/set/previews", json={"plan_path": foreign_plan}).status_code == 404
    assert other.post("/api/set/render", json={"plan_path": foreign_plan}).status_code == 404
    assert other.post(
        "/api/studio/transition-preview",
        json={"plan_path": foreign_plan, "transition_index": 0},
    ).status_code == 404
    assert guest.post("/api/set/previews", json={"plan_path": foreign_plan}).status_code == 401
    assert guest.post(
        "/api/studio/transition-preview",
        json={"plan_path": foreign_plan, "transition_index": 0},
    ).status_code == 401


def test_account_projects_are_owner_scoped_and_keep_generation_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_STEM_JOB_WORKERS", "1")
    monkeypatch.setenv("SAUND_AUTH_COOKIE_SECURE", "0")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")

    owner = TestClient(api.app)
    other = TestClient(api.app)
    owner_registration = owner.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "history-owner@example.com", "password": "a-long-history-owner-password", "country": "Germany", "primary_use": "personal"},
    )
    other_registration = other.post(
        "/api/auth/register",
        json={"first_name": "Test", "last_name": "Listener", "email": "history-other@example.com", "password": "a-long-history-other-password", "country": "Germany", "primary_use": "personal"},
    )
    owner_user_id = owner_registration.json()["user"]["id"]
    other_user_id = other_registration.json()["user"]["id"]

    plan_path = "output/jobs/primary/set-plan.json"
    primary = api.manager.submit(
        "set-plan",
        lambda context: {
            "plan_path": plan_path,
            "plan": {
                "ordered_tracks": [
                    {"title": "First Track"},
                    {"title": "Second Track"},
                    {"title": "Third Track"},
                ]
            },
        },
        owner_user_id=owner_user_id,
        resource_context={
            "track_paths": ["audio/first.wav", "audio/second.wav", "audio/third.wav"],
            "journey": "build",
            "experience": "pro",
        },
    )
    render = api.manager.submit(
        "set-render",
        lambda context: {
            "audio_path": "output/jobs/render/final.wav",
            "audio_url": "/api/media/output/jobs/render/final.wav",
        },
        owner_user_id=owner_user_id,
        resource_context={"source_plan_path": plan_path},
    )
    api.manager.submit(
        "automatic-set",
        lambda context: {"plan_path": "output/jobs/other/set-plan.json"},
        owner_user_id=other_user_id,
        resource_context={"track_paths": ["audio/private-a.wav", "audio/private-b.wav"]},
    )

    for job in (primary, render):
        for _ in range(100):
            if api.manager.get(job.job_id).status in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert api.manager.get(job.job_id).status == "completed"

    projects = owner.get("/api/account/projects?limit=10")
    assert projects.status_code == 200
    payload = projects.json()
    assert payload["count"] == 1
    project = payload["projects"][0]
    assert project["project_id"] == primary.job_id
    assert project["title"] == "First Track → Second Track + 1"
    assert project["journey"] == "build"
    assert project["track_count"] == 3
    assert project["track_paths"] == ["audio/first.wav", "audio/second.wav", "audio/third.wav"]
    assert project["audio_path"] == "output/jobs/render/final.wav"

    assert TestClient(api.app).get("/api/account/projects").status_code == 401
    other_projects = other.get("/api/account/projects")
    assert other_projects.status_code == 200
    assert all(item["project_id"] != primary.job_id for item in other_projects.json()["projects"])


def test_job_resource_context_survives_manager_restart(tmp_path: Path) -> None:
    from djai.jobs import JobManager

    root = tmp_path / "jobs"
    manager = JobManager(root, workers=1)
    record = manager.submit(
        "set-plan",
        lambda context: {"ok": True},
        owner_user_id="user-123",
        resource_context={"track_paths": ["audio/a.wav", "audio/b.wav"], "journey": "smooth"},
    )
    for _ in range(100):
        if manager.get(record.job_id).status in {"completed", "failed"}:
            break
        time.sleep(0.01)
    restored = JobManager(root, workers=1).get(record.job_id)
    assert restored.owner_user_id == "user-123"
    assert restored.resource_context == {
        "track_paths": ["audio/a.wav", "audio/b.wav"],
        "journey": "smooth",
    }


def test_account_export_delete_and_personalization_consent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("SAUND_AUTH_COOKIE_SECURE", "0")
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")
    client = TestClient(api.app)
    registered = client.post(
        "/api/auth/register",
        json={
            "first_name": "Privacy",
            "last_name": "Test",
            "email": "privacy@example.com",
            "password": "privacy-password-long",
            "country": "Germany",
            "primary_use": "personal",
        },
    )
    assert registered.status_code == 201

    feedback = client.post("/api/feedback", json={"event": "set_like", "listener": "attacker-selected"})
    assert feedback.status_code == 200
    assert feedback.json()["listener_profile"] is None
    assert feedback.json()["personalization_applied"] is False

    enabled = client.patch(
        "/api/account/preferences",
        json={"personalization_enabled": True, "improve_saund_ai": True},
    )
    assert enabled.status_code == 200
    feedback = client.post("/api/feedback", json={"event": "set_like", "listener": "attacker-selected"})
    assert feedback.json()["listener_profile"]["sample_count"] == 1
    assert feedback.json()["training_eligible"] is True
    assert feedback.json()["listener_profile"]["listener"] != "attacker-selected"

    exported = client.get("/api/account/export")
    assert exported.status_code == 200
    assert exported.json()["account"]["user"]["email"] == "privacy@example.com"
    assert "password" not in str(exported.json()).lower()

    feedback_log = tmp_path / "ratings" / "feedback-events.jsonl"
    assert "acct_" in feedback_log.read_text(encoding="utf-8")
    deleted = client.delete("/api/account")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert "acct_" not in feedback_log.read_text(encoding="utf-8")
    assert client.get("/api/auth/me").json()["authenticated"] is False
