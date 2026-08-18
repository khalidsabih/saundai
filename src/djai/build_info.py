from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from .version import DJAI_VERSION, MUSICAL_TIMELINE_SCHEMA_VERSION
from .feature_store import FEATURE_STORE_SCHEMA_VERSION, FEATURE_VERSIONS


_VERSION_FILES = (
    "VERSION",
    "pyproject.toml",
    "src/djai/version.py",
    "frontend/package.json",
)
_FINGERPRINT_FILES = (
    "VERSION",
    "src/djai/api.py",
    "src/djai/jobs.py",
    "src/djai/music/timeline.py",
    "src/djai/feature_store.py",
    "src/djai/brain/crate_memory.py",
    "src/djai/release_candidate.py",
    "src/djai/cache.py",
    "src/djai/perception/world_model.py",
    "src/djai/parallel.py",
    "src/djai/version.py",
    "frontend/package.json",
    "frontend/components/JobProgress.tsx",
    "frontend/components/TrackLibrary.tsx",
    "compose.yaml",
)


def _workspace_root() -> Path:
    candidate = Path(os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()
    if (candidate / "VERSION").is_file():
        return candidate
    package_root = Path(__file__).resolve().parents[2]
    if (package_root / "VERSION").is_file():
        return package_root
    return candidate


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _versions(root: Path) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "runtime": DJAI_VERSION,
        "version_file": _read_text(root / "VERSION"),
        "python_project": None,
        "frontend": None,
    }
    pyproject = root / "pyproject.toml"
    try:
        values["python_project"] = str(tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"])
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        pass
    package = root / "frontend" / "package.json"
    try:
        values["frontend"] = str(json.loads(package.read_text(encoding="utf-8"))["version"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return values


def _git_identity(root: Path) -> tuple[str | None, bool | None]:
    explicit = os.environ.get("SAUND_BUILD_COMMIT", "").strip()
    if explicit:
        return explicit, None
    if not (root / ".git").exists():
        return None, None
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
        return commit or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    used = 0
    for relative in _FINGERPRINT_FILES:
        path = root / relative
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        used += 1
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()[:16] if used else "unavailable"


@lru_cache(maxsize=1)
def build_identity() -> dict[str, Any]:
    root = _workspace_root()
    versions = _versions(root)
    observed = [value for value in versions.values() if value]
    version_consistent = bool(observed) and len(set(observed)) == 1
    commit, dirty = _git_identity(root)
    return {
        "schema_version": "2.0.0",
        "product": "SAUND.AI",
        "version": DJAI_VERSION,
        "release_channel": "release-candidate" if "rc" in DJAI_VERSION.lower() else "stable",
        "release_candidate": 1 if "rc" in DJAI_VERSION.lower() else None,
        "versions": versions,
        "version_consistent": version_consistent,
        "source_fingerprint": _source_fingerprint(root),
        "commit": commit,
        "source_dirty": dirty,
        "build_time_utc": os.environ.get("SAUND_BUILD_TIME_UTC", "").strip() or None,
        "image_tag": os.environ.get("SAUND_IMAGE_TAG", "").strip() or None,
        "runtime_source": "workspace" if str(Path(__file__).resolve()).startswith(str(root)) else "installed-package",
        "authoritative_version_source": "VERSION",
        "musical_timeline_schema": MUSICAL_TIMELINE_SCHEMA_VERSION,
        "feature_store_schema": FEATURE_STORE_SCHEMA_VERSION,
        "feature_versions": dict(FEATURE_VERSIONS),
        "checked_files": list(_VERSION_FILES),
    }
