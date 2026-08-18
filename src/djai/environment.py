from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DJAIEnvironment:
    workspace: Path
    config_dir: Path
    cache_dir: Path
    output_dir: Path
    ratings_dir: Path

    @classmethod
    def from_environment(cls) -> "DJAIEnvironment":
        workspace = Path(os.environ.get("DJAI_WORKSPACE", Path.cwd())).expanduser().resolve()
        config_dir = Path(os.environ.get("DJAI_CONFIG_DIR", workspace / "config")).expanduser()
        cache_dir = Path(os.environ.get("DJAI_CACHE_DIR", workspace / ".djai-cache")).expanduser()
        output_dir = Path(os.environ.get("DJAI_OUTPUT_DIR", workspace / "output")).expanduser()
        ratings_dir = Path(os.environ.get("DJAI_RATINGS_DIR", workspace / "ratings")).expanduser()
        return cls(
            workspace=workspace,
            config_dir=config_dir.resolve(),
            cache_dir=cache_dir.resolve(),
            output_dir=output_dir.resolve(),
            ratings_dir=ratings_dir.resolve(),
        )

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace / path
        return path.resolve()

    def config_path(self, name: str | Path) -> Path:
        path = Path(name).expanduser()
        if path.is_absolute():
            return path.resolve()
        # A caller may pass "config/foo.json" or only "foo.json".
        parts = path.parts
        if parts and parts[0] == "config":
            path = Path(*parts[1:])
        return (self.config_dir / path).resolve()


def current_environment() -> DJAIEnvironment:
    # Deliberately not cached: tests and local service launchers can set
    # DJAI_WORKSPACE before invoking a command without stale module state.
    return DJAIEnvironment.from_environment()


def workspace_path(value: str | Path) -> Path:
    return current_environment().resolve(value)


def config_path(value: str | Path) -> Path:
    return current_environment().config_path(value)
