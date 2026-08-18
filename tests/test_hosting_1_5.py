from __future__ import annotations

import importlib
import os
import sys
import uuid
import wave
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from djai.accounts import AccountStore
from djai.hosting import HostingConfig
from djai.object_store import ObjectStore, ObjectStoreConfig


def _hosted_env(monkeypatch, workspace: Path) -> None:
    monkeypatch.setenv("DJAI_WORKSPACE", str(workspace))
    monkeypatch.setenv("SAUND_ENV", "production")
    monkeypatch.setenv("SAUND_HOSTED_MODE", "1")
    monkeypatch.setenv("SAUND_PERSISTENT_WORKSPACE", "1")
    monkeypatch.setenv("SAUND_HOSTED_REQUIRE_POSTGRES", "0")
    monkeypatch.setenv("SAUND_HOSTED_REQUIRE_OBJECT_STORE", "0")
    monkeypatch.setenv("SAUND_AUTH_COOKIE_SECURE", "1")
    monkeypatch.setenv("SAUND_UPLOAD_AUTH_REQUIRED", "1")
    monkeypatch.setenv("SAUND_SIGNUP_MODE", "open")
    monkeypatch.setenv("DJAI_CORS_ORIGINS", "https://app.saund.test")
    monkeypatch.setenv("SAUND_PUBLIC_ORIGIN", "https://app.saund.test")
    monkeypatch.setenv("SAUND_API_ORIGIN", "https://api.saund.test")
    monkeypatch.setenv("DJAI_TRAINING_LAB_ENABLED", "0")
    monkeypatch.setenv("DJAI_CONFIRMATION_SECRET", "hosted-test-secret-that-is-definitely-long-enough")
    monkeypatch.setenv("DJAI_AUTO_STEMS", "off")
    monkeypatch.setenv("DJAI_JOB_WORKERS", "1")
    monkeypatch.setenv("DJAI_STEM_JOB_WORKERS", "1")


def _registration(email: str) -> dict[str, object]:
    return {
        "first_name": "Beta",
        "last_name": "Tester",
        "email": email,
        "password": "correct-horse-battery-staple",
        "country": "Germany",
        "primary_use": "dj",
        "experience_level": "experienced",
        "preferred_genres": ["House"],
        "musical_priorities": ["smooth_transitions"],
    }


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 8000)


def test_hosted_validation_fails_closed_without_release_dependencies(tmp_path: Path) -> None:
    cfg = HostingConfig.from_mapping(
        {
            "SAUND_ENV": "production",
            "SAUND_HOSTED_MODE": "1",
            "SAUND_PUBLIC_ORIGIN": "http://app.example.com",
            "SAUND_API_ORIGIN": "https://api.example.com",
        }
    )
    errors = cfg.validation_errors(
        database_backend="sqlite",
        object_store_enabled=False,
        cookie_secure=False,
        cors_origins=["http://localhost:3000"],
        training_lab_enabled=True,
        confirmation_secret="change-this",
        workspace=tmp_path,
    )
    combined = " | ".join(errors)
    assert "persistent" in combined.lower()
    assert "postgres" in combined.lower()
    assert "object storage" in combined.lower()
    assert "secure" in combined.lower()
    assert "HTTPS" in combined
    assert "Training Lab" in combined


def test_object_store_uses_user_scoped_keys_and_presigned_contract() -> None:
    config = ObjectStoreConfig(
        backend="s3",
        bucket="private",
        endpoint_url="https://example.invalid",
        region="auto",
        access_key_id="key",
        secret_access_key="secret",
        prefix="saund",
        presign_seconds=600,
        delete_after_materialize=False,
    )
    store = ObjectStore(config)

    class FakeClient:
        def generate_presigned_url(self, operation, *, Params, ExpiresIn, HttpMethod):
            assert operation == "put_object"
            assert Params["Bucket"] == "private"
            assert ExpiresIn == 600
            assert HttpMethod == "PUT"
            return "https://upload.example/signed"

    store._client = FakeClient()
    intent = store.create_upload_intent(
        user_id="user-a",
        filename="track.wav",
        size=1234,
        content_type="audio/wav",
    )
    assert intent["upload_url"] == "https://upload.example/signed"
    assert store.owns_key("user-a", intent["object_key"])
    assert not store.owns_key("user-b", intent["object_key"])
    assert "/users/" in intent["object_key"]
    assert "user-a" not in intent["object_key"]


def test_hosted_api_isolates_user_libraries_and_supports_direct_completion(tmp_path: Path, monkeypatch) -> None:
    _hosted_env(monkeypatch, tmp_path)
    sys.modules.pop("djai.api", None)
    api = importlib.import_module("djai.api")

    class FakeStore:
        enabled = True

        def status(self):
            return {"backend": "s3", "enabled": True, "direct_browser_upload": True}

        def ready(self):
            return True

        def owns_key(self, user_id: str, key: str) -> bool:
            return key == f"users/{user_id}/incoming.wav"

        def create_upload_intent(self, *, user_id: str, filename: str, size: int, content_type: str | None = None):
            return {
                "object_key": f"users/{user_id}/incoming.wav",
                "upload_url": "https://object.example/upload",
                "method": "PUT",
                "headers": {"Content-Type": content_type or "audio/wav"},
                "expires_in": 900,
                "expected_size": size,
            }

        def materialize(self, key: str, target: Path, *, expected_size: int | None = None):
            _write_wav(target)
            return {"size": target.stat().st_size, "content_type": "audio/wav", "etag": "fake"}

        def media_url_for_path(self, path: Path):
            return f"https://object.example/{path.name}"

        def delete_user_objects(self, user_id: str):
            return 0

    api.object_store = FakeStore()
    client_a = TestClient(api.app, base_url="https://api.saund.test")
    client_b = TestClient(api.app, base_url="https://api.saund.test")

    assert client_a.get("/api/tracks").status_code == 401
    assert client_a.post("/api/auth/register", json=_registration("a@example.com")).status_code == 201
    assert client_b.post("/api/auth/register", json=_registration("b@example.com")).status_code == 201

    intent = client_a.post(
        "/api/tracks/upload-intents",
        json={"files": [{"filename": "incoming.wav", "size": 16044, "content_type": "audio/wav"}]},
    )
    assert intent.status_code == 200
    object_key = intent.json()["uploads"][0]["object_key"]

    completed = client_a.post(
        "/api/tracks/upload-complete",
        json={"files": [{"filename": "incoming.wav", "object_key": object_key, "expected_size": 16044}]},
    )
    assert completed.status_code == 200
    track = completed.json()["files"][0]
    assert track["media_url"].startswith("https://object.example/")

    listing_a = client_a.get("/api/tracks?recursive=true")
    listing_b = client_b.get("/api/tracks?recursive=true")
    assert [item["name"] for item in listing_a.json()["tracks"]] == ["incoming.wav"]
    assert listing_b.json()["tracks"] == []

    # A second account cannot reach A's materialized source via the API fallback.
    assert client_b.get(f"/api/media/{track['path']}").status_code == 404

    status = client_a.get("/api/hosting/status").json()
    assert status["hosted_mode"] is True
    assert status["deployment_profile"] == "private-beta-single-node"
    assert status["horizontal_worker_scaling"] is False
    assert client_a.get("/api/ready").status_code == 200


@pytest.mark.skipif(not os.environ.get("SAUND_TEST_POSTGRES_URL"), reason="set SAUND_TEST_POSTGRES_URL for managed PostgreSQL integration")
def test_managed_postgres_account_round_trip() -> None:
    store = AccountStore(database_url=os.environ["SAUND_TEST_POSTGRES_URL"])
    assert store.backend == "postgres"
    assert store.ping() is True
    email = f"hosted-integration-{uuid.uuid4().hex}@example.invalid"
    user = store.create_user(email, "correct-horse-battery-staple", first_name="Hosted", last_name="Test")
    try:
        authenticated = store.authenticate(email, "correct-horse-battery-staple")
        assert authenticated is not None
        assert authenticated["id"] == user["id"]
    finally:
        assert store.delete_user(user["id"]) is True
