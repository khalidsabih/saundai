from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Bump this whenever deterministic DSP semantics change in a way that can
# change rendered samples for an otherwise identical transition program.
RENDER_ENGINE_VERSION = "6.2.0"


@dataclass(frozen=True, slots=True)
class TransitionRenderContext:
    """Exact level/sample-rate context used to rehearse an executable edge.

    Set planning supplies the same fixed gains later used by the full-set
    renderer. Pair-mode may omit the context and keep local gain matching for
    backwards compatibility.
    """

    sample_rate: int
    gain_a: float = 1.0
    gain_b: float | None = None
    target_loudness_dbfs: float = -13.0

    def normalized(self) -> "TransitionRenderContext":
        sample_rate = max(8_000, min(192_000, int(self.sample_rate)))
        gain_a = max(0.0, min(8.0, float(self.gain_a)))
        gain_b = None if self.gain_b is None else max(0.0, min(8.0, float(self.gain_b)))
        return TransitionRenderContext(
            sample_rate=sample_rate,
            gain_a=gain_a,
            gain_b=gain_b,
            target_loudness_dbfs=float(self.target_loudness_dbfs),
        )


def normalization_gain(loudness_dbfs: float, target_dbfs: float = -13.0) -> float:
    """Return the bounded full-set normalization gain for one source track."""

    db = float(np.clip(float(target_dbfs) - float(loudness_dbfs), -6.0, 6.0))
    return float(10.0 ** (db / 20.0))


@lru_cache(maxsize=4096)
def _content_fingerprint_cached(
    path: str,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
    inode: int,
) -> str:
    # Metadata participates only in cache invalidation; the stored value is
    # always a digest of the actual bytes. ctime/inode close the same-size +
    # restored-mtime hole for normal filesystem mutations/replacements.
    del size, mtime_ns, ctime_ns, inode
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_fingerprint(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _content_fingerprint_cached(
        str(resolved),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", int(stat.st_ctime * 1_000_000_000))),
        int(getattr(stat, "st_ino", 0)),
    )


def _layer_paths(bundle: Mapping[str, Any] | None, *, prefix: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for role, value in dict(bundle or {}).items():
        if isinstance(value, Mapping):
            candidate = value.get("path")
            if candidate:
                paths[f"{prefix}:{role}"] = str(Path(str(candidate)).expanduser().resolve())
    return paths


def execution_asset_paths(plan: Any) -> dict[str, str]:
    """Return every concrete audio asset that can alter rendered samples."""

    paths: dict[str, str] = {
        "master_a": str(Path(str(plan.track_a)).expanduser().resolve()),
        "master_b": str(Path(str(plan.track_b)).expanduser().resolve()),
    }
    for role, path in sorted(dict(getattr(plan, "stem_paths_a", {}) or {}).items()):
        if path:
            paths[f"stem_a:{role}"] = str(Path(str(path)).expanduser().resolve())
    for role, path in sorted(dict(getattr(plan, "stem_paths_b", {}) or {}).items()):
        if path:
            paths[f"stem_b:{role}"] = str(Path(str(path)).expanduser().resolve())
    paths.update(_layer_paths(getattr(plan, "performance_layers_a", {}), prefix="layer_a"))
    paths.update(_layer_paths(getattr(plan, "performance_layers_b", {}), prefix="layer_b"))
    return dict(sorted(paths.items()))


def bind_execution_assets(plan: Any) -> None:
    """Snapshot all source paths and their content hashes into a plan."""

    paths = execution_asset_paths(plan)
    fingerprints: dict[str, str] = {}
    for role, path in paths.items():
        try:
            fingerprints[role] = content_fingerprint(path)
        except OSError:
            # Planning-only callers may operate on synthetic TrackAnalysis
            # objects. The sentinel can be signed, but execution validation
            # will reject it until real content is present and the plan is
            # re-sealed.
            fingerprints[role] = "__asset_unavailable__"
    plan.execution_asset_paths = paths
    plan.execution_asset_fingerprints = fingerprints
    plan.renderer_version = RENDER_ENGINE_VERSION


def verify_execution_assets(plan: Any) -> None:
    """Reject a plan if any bound source path or content has changed."""

    expected_paths = dict(getattr(plan, "execution_asset_paths", {}) or {})
    expected_fingerprints = dict(getattr(plan, "execution_asset_fingerprints", {}) or {})
    if not expected_paths or not expected_fingerprints:
        raise ValueError("Transition plan is missing bound execution asset fingerprints; re-plan before rendering.")

    current_paths = execution_asset_paths(plan)
    if current_paths != expected_paths:
        changed_roles = sorted(set(current_paths) | set(expected_paths))
        changed_roles = [role for role in changed_roles if current_paths.get(role) != expected_paths.get(role)]
        raise ValueError(
            "Transition execution assets changed after planning (paths: "
            + ", ".join(changed_roles)
            + "); re-plan before rendering."
        )

    unavailable = [
        role for role, value in expected_fingerprints.items()
        if value == "__asset_unavailable__"
    ]
    if unavailable:
        raise ValueError(
            "Transition execution assets were unavailable when the plan was compiled ("
            + ", ".join(unavailable)
            + "); bind real audio and re-plan before rendering."
        )

    changed: list[str] = []
    for role, path in current_paths.items():
        try:
            actual = content_fingerprint(path)
        except OSError as exc:
            raise ValueError(f"Transition execution asset is unavailable for {role}: {path}") from exc
        if actual != expected_fingerprints.get(role):
            changed.append(role)
    if changed:
        raise ValueError(
            "Transition execution asset content changed after planning ("
            + ", ".join(changed)
            + "); re-plan before rendering."
        )
