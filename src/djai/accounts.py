from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


class AccountExistsError(ValueError):
    pass


@dataclass(frozen=True)
class AccountSession:
    token: str
    expires_at: int


class AccountStore:
    """Server-side account/session store with SQLite + PostgreSQL backends.

    Local SAUND keeps using SQLite. Hosted deployments can set
    ``SAUND_DATABASE_URL=postgresql://...`` and use the same account/session
    contract against managed PostgreSQL.

    Passwords are Argon2id hashes. Browser sessions are opaque random values;
    only SHA-256 digests are persisted, so a database copy does not contain
    reusable bearer tokens.
    """

    def __init__(self, path: Path | None = None, *, database_url: str | None = None):
        value = (database_url or "").strip()
        self.database_url = value or None
        self.backend = "postgres" if value.startswith(("postgres://", "postgresql://")) else "sqlite"
        self.path: Path | None = None
        self._psycopg: Any = None
        self._dict_row: Any = None
        if self.backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:  # pragma: no cover - exercised only in hosted image
                raise RuntimeError(
                    "PostgreSQL account storage requires the hosted dependency extra: "
                    "install SAUND.AI with .[hosted]."
                ) from exc
            self._psycopg = psycopg
            self._dict_row = dict_row
        else:
            resolved = Path(path or "accounts/accounts.sqlite3").expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self.path = resolved

        self.password_hasher = PasswordHasher()
        # Used when an email is unknown so failed logins do comparable password
        # verification work and do not trivially reveal account existence by timing.
        self._dummy_hash = self.password_hasher.hash("saund-dummy-password-not-a-user")
        self._initialize()

    def _connect(self):
        if self.backend == "postgres":
            assert self.database_url is not None
            return self._psycopg.connect(self.database_url, row_factory=self._dict_row)
        assert self.path is not None
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _sql(self, query: str) -> str:
        if self.backend == "postgres":
            return query.replace("?", "%s")
        return query

    def _execute(self, connection: Any, query: str, params: tuple[Any, ...] = ()):
        return connection.execute(self._sql(query), params)

    def _initialize(self) -> None:
        if self.backend == "postgres":
            statements = [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0 CHECK(disabled IN (0, 1))
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    first_name TEXT NOT NULL DEFAULT '',
                    last_name TEXT NOT NULL DEFAULT '',
                    artist_name TEXT,
                    country TEXT NOT NULL DEFAULT '',
                    primary_use TEXT,
                    experience_level TEXT,
                    preferred_genres_json TEXT NOT NULL DEFAULT '[]',
                    musical_priorities_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    personalization_enabled INTEGER NOT NULL DEFAULT 0 CHECK(personalization_enabled IN (0, 1)),
                    improve_saund_ai INTEGER NOT NULL DEFAULT 0 CHECK(improve_saund_ai IN (0, 1)),
                    marketing_consent INTEGER NOT NULL DEFAULT 0 CHECK(marketing_consent IN (0, 1)),
                    updated_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at BIGINT NOT NULL,
                    expires_at BIGINT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id)",
                "CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions(expires_at)",
                "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS marketing_consent INTEGER NOT NULL DEFAULT 0 CHECK(marketing_consent IN (0, 1))",
            ]
            with self._connect() as connection:
                for statement in statements:
                    self._execute(connection, statement)
            return

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0 CHECK(disabled IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    first_name TEXT NOT NULL DEFAULT '',
                    last_name TEXT NOT NULL DEFAULT '',
                    artist_name TEXT,
                    country TEXT NOT NULL DEFAULT '',
                    primary_use TEXT,
                    experience_level TEXT,
                    preferred_genres_json TEXT NOT NULL DEFAULT '[]',
                    musical_priorities_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    personalization_enabled INTEGER NOT NULL DEFAULT 0 CHECK(personalization_enabled IN (0, 1)),
                    improve_saund_ai INTEGER NOT NULL DEFAULT 0 CHECK(improve_saund_ai IN (0, 1)),
                    marketing_consent INTEGER NOT NULL DEFAULT 0 CHECK(marketing_consent IN (0, 1)),
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS sessions_expires_at_idx ON sessions(expires_at);
                """
            )
            preference_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(user_preferences)").fetchall()
            }
            if "marketing_consent" not in preference_columns:
                connection.execute(
                    "ALTER TABLE user_preferences ADD COLUMN marketing_consent INTEGER NOT NULL DEFAULT 0 CHECK(marketing_consent IN (0, 1))"
                )

    def ping(self) -> bool:
        try:
            with self._connect() as connection:
                row = self._execute(connection, "SELECT 1 AS ok").fetchone()
            return bool(row and int(row["ok"]) == 1)
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "database_url_configured": bool(self.database_url),
            "sqlite_path": str(self.path) if self.path is not None else None,
            "ready": self.ping(),
        }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().casefold()

    @staticmethod
    def _clean_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(str(value).strip().split())
        return cleaned or None

    @staticmethod
    def _clean_list(values: list[str] | None, *, limit: int = 16) -> list[str]:
        if not values:
            return []
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = " ".join(str(value).strip().split())
            key = cleaned.casefold()
            if not cleaned or key in seen:
                continue
            output.append(cleaned)
            seen.add(key)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _decode_list(value: Any) -> list[str]:
        try:
            decoded = json.loads(str(value or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(decoded, list):
            return []
        return [str(item) for item in decoded if isinstance(item, str)]

    @staticmethod
    def _session_digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def _snapshot(cls, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "user": {
                "id": str(row["id"]),
                "email": str(row["email"]),
                "created_at": str(row["created_at"]),
            },
            "profile": {
                "first_name": str(row["first_name"] or ""),
                "last_name": str(row["last_name"] or ""),
                "artist_name": str(row["artist_name"]) if row["artist_name"] else None,
                "country": str(row["country"] or ""),
                "primary_use": str(row["primary_use"]) if row["primary_use"] else None,
                "experience_level": str(row["experience_level"]) if row["experience_level"] else None,
                "preferred_genres": cls._decode_list(row["preferred_genres_json"]),
                "musical_priorities": cls._decode_list(row["musical_priorities_json"]),
            },
            "preferences": {
                "personalization_enabled": bool(row["personalization_enabled"]),
                "improve_saund_ai": bool(row["improve_saund_ai"]),
                "marketing_consent": bool(row["marketing_consent"]),
            },
        }

    def _select_user(
        self,
        connection: Any,
        *,
        user_id: str | None = None,
        email: str | None = None,
    ) -> Mapping[str, Any] | None:
        if user_id is None and email is None:
            raise ValueError("user_id or email is required")
        column = "u.id" if user_id is not None else "u.email"
        value = user_id if user_id is not None else self._normalize_email(str(email))
        return self._execute(
            connection,
            f"""
            SELECT
                u.id,
                u.email,
                u.password_hash,
                u.created_at,
                u.updated_at,
                u.disabled,
                COALESCE(pr.first_name, '') AS first_name,
                COALESCE(pr.last_name, '') AS last_name,
                pr.artist_name,
                COALESCE(pr.country, '') AS country,
                pr.primary_use,
                pr.experience_level,
                COALESCE(pr.preferred_genres_json, '[]') AS preferred_genres_json,
                COALESCE(pr.musical_priorities_json, '[]') AS musical_priorities_json,
                p.personalization_enabled,
                p.improve_saund_ai,
                COALESCE(p.marketing_consent, 0) AS marketing_consent
            FROM users AS u
            JOIN user_preferences AS p ON p.user_id = u.id
            LEFT JOIN user_profiles AS pr ON pr.user_id = u.id
            WHERE {column} = ?
            """,
            (value,),
        ).fetchone()

    def _is_integrity_error(self, exc: Exception) -> bool:
        if isinstance(exc, sqlite3.IntegrityError):
            return True
        return getattr(exc, "sqlstate", None) == "23505" or exc.__class__.__name__ in {
            "UniqueViolation",
            "IntegrityError",
        }

    def create_user(
        self,
        email: str,
        password: str,
        *,
        first_name: str = "",
        last_name: str = "",
        artist_name: str | None = None,
        country: str = "",
        primary_use: str | None = None,
        experience_level: str | None = None,
        preferred_genres: list[str] | None = None,
        musical_priorities: list[str] | None = None,
        personalization_enabled: bool = False,
        improve_saund_ai: bool = False,
        marketing_consent: bool = False,
    ) -> dict[str, Any]:
        normalized = self._normalize_email(email)
        now = self._now_iso()
        user_id = str(uuid.uuid4())
        password_hash = self.password_hasher.hash(password)
        profile = {
            "first_name": self._clean_text(first_name) or "",
            "last_name": self._clean_text(last_name) or "",
            "artist_name": self._clean_text(artist_name),
            "country": self._clean_text(country) or "",
            "primary_use": self._clean_text(primary_use),
            "experience_level": self._clean_text(experience_level),
            "preferred_genres": self._clean_list(preferred_genres),
            "musical_priorities": self._clean_list(musical_priorities),
        }
        try:
            with self._connect() as connection:
                self._execute(
                    connection,
                    "INSERT INTO users(id, email, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, normalized, password_hash, now, now),
                )
                self._execute(
                    connection,
                    """
                    INSERT INTO user_profiles(
                        user_id, first_name, last_name, artist_name, country,
                        primary_use, experience_level, preferred_genres_json,
                        musical_priorities_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        profile["first_name"],
                        profile["last_name"],
                        profile["artist_name"],
                        profile["country"],
                        profile["primary_use"],
                        profile["experience_level"],
                        json.dumps(profile["preferred_genres"], ensure_ascii=False),
                        json.dumps(profile["musical_priorities"], ensure_ascii=False),
                        now,
                    ),
                )
                self._execute(
                    connection,
                    """
                    INSERT INTO user_preferences(
                        user_id, personalization_enabled, improve_saund_ai,
                        marketing_consent, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        int(personalization_enabled),
                        int(improve_saund_ai),
                        int(marketing_consent),
                        now,
                    ),
                )
                row = self._select_user(connection, user_id=user_id)
        except Exception as exc:
            if self._is_integrity_error(exc):
                raise AccountExistsError("An account with that email already exists.") from exc
            raise
        assert row is not None
        return self._snapshot(row)

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        normalized = self._normalize_email(email)
        with self._connect() as connection:
            row = self._select_user(connection, email=normalized)
            password_hash = str(row["password_hash"]) if row is not None else self._dummy_hash
            try:
                valid = self.password_hasher.verify(password_hash, password)
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                valid = False
            if row is None or not valid or bool(row["disabled"]):
                return None
            if self.password_hasher.check_needs_rehash(password_hash):
                replacement = self.password_hasher.hash(password)
                self._execute(
                    connection,
                    "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                    (replacement, self._now_iso(), str(row["id"])),
                )
            return self._snapshot(row)

    def create_session(self, user_id: str, *, ttl_seconds: int) -> AccountSession:
        now = int(time.time())
        expires_at = now + max(60, int(ttl_seconds))
        token = secrets.token_urlsafe(48)
        digest = self._session_digest(token)
        with self._connect() as connection:
            self._execute(connection, "DELETE FROM sessions WHERE expires_at <= ?", (now,))
            self._execute(
                connection,
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (digest, user_id, now, expires_at),
            )
        return AccountSession(token=token, expires_at=expires_at)

    def get_session(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        now = int(time.time())
        digest = self._session_digest(token)
        with self._connect() as connection:
            session_row = self._execute(
                connection,
                "SELECT user_id, expires_at FROM sessions WHERE token_hash = ? AND expires_at > ?",
                (digest, now),
            ).fetchone()
            if session_row is None:
                return None
            row = self._select_user(connection, user_id=str(session_row["user_id"]))
            if row is None or bool(row["disabled"]):
                return None
            snapshot = self._snapshot(row)
            snapshot["session_expires_at"] = int(session_row["expires_at"])
            return snapshot

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connect() as connection:
            self._execute(
                connection,
                "DELETE FROM sessions WHERE token_hash = ?",
                (self._session_digest(token),),
            )

    def update_preferences(
        self,
        user_id: str,
        *,
        personalization_enabled: bool | None = None,
        improve_saund_ai: bool | None = None,
        marketing_consent: bool | None = None,
    ) -> dict[str, Any]:
        assignments: list[str] = []
        values: list[Any] = []
        if personalization_enabled is not None:
            assignments.append("personalization_enabled = ?")
            values.append(int(personalization_enabled))
        if improve_saund_ai is not None:
            assignments.append("improve_saund_ai = ?")
            values.append(int(improve_saund_ai))
        if marketing_consent is not None:
            assignments.append("marketing_consent = ?")
            values.append(int(marketing_consent))
        if assignments:
            assignments.append("updated_at = ?")
            values.append(self._now_iso())
            values.append(user_id)
            with self._connect() as connection:
                self._execute(
                    connection,
                    f"UPDATE user_preferences SET {', '.join(assignments)} WHERE user_id = ?",
                    tuple(values),
                )
        with self._connect() as connection:
            row = self._select_user(connection, user_id=user_id)
        if row is None:
            raise KeyError(user_id)
        return self._snapshot(row)

    def update_profile(
        self,
        user_id: str,
        *,
        first_name: str | None = None,
        last_name: str | None = None,
        artist_name: str | None = None,
        country: str | None = None,
        primary_use: str | None = None,
        experience_level: str | None = None,
        preferred_genres: list[str] | None = None,
        musical_priorities: list[str] | None = None,
        artist_name_supplied: bool = False,
        primary_use_supplied: bool = False,
        experience_level_supplied: bool = False,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = self._select_user(connection, user_id=user_id)
            if row is None:
                raise KeyError(user_id)
            current = self._snapshot(row)["profile"]
            merged = {
                "first_name": self._clean_text(first_name) if first_name is not None else current["first_name"],
                "last_name": self._clean_text(last_name) if last_name is not None else current["last_name"],
                "artist_name": self._clean_text(artist_name) if artist_name_supplied else current["artist_name"],
                "country": self._clean_text(country) if country is not None else current["country"],
                "primary_use": self._clean_text(primary_use) if primary_use_supplied else current["primary_use"],
                "experience_level": self._clean_text(experience_level) if experience_level_supplied else current["experience_level"],
                "preferred_genres": self._clean_list(preferred_genres) if preferred_genres is not None else current["preferred_genres"],
                "musical_priorities": self._clean_list(musical_priorities) if musical_priorities is not None else current["musical_priorities"],
            }
            self._execute(
                connection,
                """
                INSERT INTO user_profiles(
                    user_id, first_name, last_name, artist_name, country,
                    primary_use, experience_level, preferred_genres_json,
                    musical_priorities_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    artist_name = excluded.artist_name,
                    country = excluded.country,
                    primary_use = excluded.primary_use,
                    experience_level = excluded.experience_level,
                    preferred_genres_json = excluded.preferred_genres_json,
                    musical_priorities_json = excluded.musical_priorities_json,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    merged["first_name"] or "",
                    merged["last_name"] or "",
                    merged["artist_name"],
                    merged["country"] or "",
                    merged["primary_use"],
                    merged["experience_level"],
                    json.dumps(merged["preferred_genres"], ensure_ascii=False),
                    json.dumps(merged["musical_priorities"], ensure_ascii=False),
                    self._now_iso(),
                ),
            )
            updated = self._select_user(connection, user_id=user_id)
        assert updated is not None
        return self._snapshot(updated)

    def export_user(self, user_id: str) -> dict[str, Any]:
        """Return a portable privacy export without password/session secrets."""
        with self._connect() as connection:
            row = self._select_user(connection, user_id=user_id)
        if row is None:
            raise KeyError(user_id)
        snapshot = self._snapshot(row)
        snapshot["export_schema_version"] = "1.0"
        return snapshot

    def delete_user(self, user_id: str) -> bool:
        """Delete credentials, profile, preferences and sessions transactionally."""
        with self._connect() as connection:
            cursor = self._execute(connection, "DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0
