from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .analysis import ANALYSIS_VERSION, analyze_track
from .feature_store import (
    feature_store_for_analysis_cache,
    load_analysis_from_features,
    persist_analysis_features,
)
from .models import TrackAnalysis
from .parallel import (
    prepare_process_environment,
    process_context,
    resolve_worker_selection,
    worker_initializer,
)
from .serialization import track_analysis_from_dict
from .version import CACHE_SCHEMA_VERSION

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aif", ".aiff", ".ogg"}
ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(slots=True)
class CachedAnalysis:
    analysis: TrackAnalysis
    cache_hit: bool
    fingerprint: str
    cache_path: Path
    cache_source: str = "analysis"


@dataclass(slots=True)
class AnalysisBatchStats:
    total_tracks: int = 0
    cache_hits: int = 0
    analyzed_tracks: int = 0
    feature_store_hits: int = 0
    workers: int = 1
    parallel: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_tracks": self.total_tracks,
            "cache_hits": self.cache_hits,
            "analyzed_tracks": self.analyzed_tracks,
            "feature_store_hits": self.feature_store_hits,
            "workers": self.workers,
            "parallel": self.parallel,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


def discover_tracks(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Track folder not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a folder, got: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    tracks = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    from .layers import collapse_performance_layer_paths

    primary = collapse_performance_layer_paths(tracks, validate=True)
    return sorted(primary, key=lambda path: path.name.casefold())


def fingerprint_file(
    path: str | Path,
    *,
    analysis_version: str | None = None,
    cache_schema_version: str | None = None,
) -> str:
    """Fingerprint both the audio bytes and the analysis contract.

    Changing the analysis algorithm or cache schema automatically changes the
    cache filename, so stale analyses can never be silently reused.
    """

    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    analysis_version = ANALYSIS_VERSION if analysis_version is None else analysis_version
    cache_schema_version = CACHE_SCHEMA_VERSION if cache_schema_version is None else cache_schema_version
    digest = hashlib.sha256()
    digest.update(cache_schema_version.encode("utf-8"))
    digest.update(analysis_version.encode("utf-8"))
    digest.update(str(resolved).encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_cached_analysis(
    cache_path: Path,
    *,
    resolved: Path,
    fingerprint: str,
) -> TrackAnalysis | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if payload.get("analysis_version") != ANALYSIS_VERSION:
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        if payload.get("path") != str(resolved):
            return None
        analysis = track_analysis_from_dict(payload["analysis"])
        if analysis.path != str(resolved):
            return None
        if analysis.analysis_version != ANALYSIS_VERSION:
            return None
        return analysis
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _sync_feature_store(analysis: TrackAnalysis, *, cache_dir: str | Path) -> None:
    # Feature storage is an additive 1.6.1 capability. A damaged feature-store
    # artifact must never make the proven legacy analysis path unusable.
    try:
        store = feature_store_for_analysis_cache(cache_dir)
        persist_analysis_features(analysis, store=store)
    except (OSError, KeyError, TypeError, ValueError):
        return


def analyze_track_cached(
    path: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
) -> CachedAnalysis:
    resolved = Path(path).expanduser().resolve()
    fingerprint = fingerprint_file(resolved)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{fingerprint}.json"

    if use_cache and cache_path.exists():
        analysis = _load_cached_analysis(
            cache_path, resolved=resolved, fingerprint=fingerprint
        )
        if analysis is not None:
            # Transparently migrate legacy warm caches into the modular store.
            _sync_feature_store(analysis, cache_dir=cache_root)
            return CachedAnalysis(analysis, True, fingerprint, cache_path, "legacy")

    if use_cache:
        # 1.6.1 can recover a complete current TrackAnalysis from independently
        # versioned projections even when the monolithic cache is absent.
        try:
            store = feature_store_for_analysis_cache(cache_root)
            analysis = load_analysis_from_features(
                resolved,
                store=store,
                expected_analysis_version=ANALYSIS_VERSION,
            )
        except (OSError, KeyError, TypeError, ValueError):
            analysis = None
        if analysis is not None:
            payload: dict[str, object] = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "analysis_version": ANALYSIS_VERSION,
                "fingerprint": fingerprint,
                "path": str(resolved),
                "analysis": analysis.to_dict(include_beats=True),
            }
            _atomic_json_write(cache_path, payload)
            return CachedAnalysis(analysis, True, fingerprint, cache_path, "feature_store")

    analysis = analyze_track(resolved)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": fingerprint,
        "path": str(resolved),
        "analysis": analysis.to_dict(include_beats=True),
    }
    _atomic_json_write(cache_path, payload)
    _sync_feature_store(analysis, cache_dir=cache_root)
    return CachedAnalysis(analysis, False, fingerprint, cache_path, "analysis")


def _analyze_track_worker(
    path: str,
    cache_dir: str,
    use_cache: bool,
) -> CachedAnalysis:
    return analyze_track_cached(path, cache_dir=cache_dir, use_cache=use_cache)


def clear_analysis_cache(cache_dir: str | Path = ".djai-cache/tracks") -> int:
    root = Path(cache_dir).expanduser().resolve()
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob("*.json"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _update_batch_stats(
    stats: AnalysisBatchStats | None,
    *,
    results: list[CachedAnalysis],
    workers: int,
    parallel: bool,
    elapsed_seconds: float,
) -> None:
    if stats is None:
        return
    stats.total_tracks = len(results)
    stats.cache_hits = sum(1 for item in results if item.cache_hit)
    stats.analyzed_tracks = len(results) - stats.cache_hits
    stats.feature_store_hits = sum(1 for item in results if item.cache_source == "feature_store")
    stats.workers = workers
    stats.parallel = parallel
    stats.elapsed_seconds = elapsed_seconds


def analyze_tracks(
    paths: Iterable[str | Path],
    *,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    progress: ProgressCallback | None = None,
    workers: int | None = None,
    stats: AnalysisBatchStats | None = None,
) -> list[TrackAnalysis]:
    """Analyze independent tracks serially or with a deterministic process pool."""

    started = time.perf_counter()
    path_list = [Path(path).expanduser().resolve() for path in paths]
    total = len(path_list)
    if total == 0:
        _update_batch_stats(
            stats,
            results=[],
            workers=1,
            parallel=False,
            elapsed_seconds=time.perf_counter() - started,
        )
        return []

    selection = resolve_worker_selection("analysis", total, requested=workers)
    cache_root = str(Path(cache_dir).expanduser().resolve())

    # Analyze each distinct resolved path once while preserving the caller's
    # order (and any deliberate repeated paths) in the final result.
    positions: dict[str, list[int]] = {}
    for index, path in enumerate(path_list):
        positions.setdefault(str(path), []).append(index)
    unique_paths = [str(path) for path in dict.fromkeys(str(path) for path in path_list)]

    result_slots: list[CachedAnalysis | None] = [None] * total
    completed = 0

    if not selection.parallel or len(unique_paths) <= 1:
        for unique_path in unique_paths:
            path = Path(unique_path)
            if progress:
                progress("analysis", completed, total, path.name)
            cached = analyze_track_cached(path, cache_dir=cache_root, use_cache=use_cache)
            for position in positions[unique_path]:
                result_slots[position] = cached
                completed += 1
            if progress:
                state = "cached" if cached.cache_hit else "analyzed"
                progress(state, completed, total, path.name)
        actual_workers = 1
        parallel = False
    else:
        prepare_process_environment()
        actual_workers = min(selection.workers, len(unique_paths))
        parallel = actual_workers > 1
        if progress:
            progress("analysis", 0, total, f"{actual_workers} analysis workers")
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            mp_context=process_context(),
            initializer=worker_initializer,
        ) as executor:
            futures = {
                executor.submit(_analyze_track_worker, path, cache_root, use_cache): path
                for path in unique_paths
            }
            try:
                for future in as_completed(futures):
                    unique_path = futures[future]
                    cached = future.result()
                    for position in positions[unique_path]:
                        result_slots[position] = cached
                        completed += 1
                    if progress:
                        state = "cached" if cached.cache_hit else "analyzed"
                        progress(state, completed, total, Path(unique_path).name)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    results = [item for item in result_slots if item is not None]
    if len(results) != total:
        raise RuntimeError("DJAI analysis batch completed with missing results.")
    elapsed = time.perf_counter() - started
    _update_batch_stats(
        stats,
        results=results,
        workers=actual_workers,
        parallel=parallel,
        elapsed_seconds=elapsed,
    )
    return [item.analysis for item in results]


def analyze_folder(
    folder: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/tracks",
    use_cache: bool = True,
    recursive: bool = False,
    progress: ProgressCallback | None = None,
    workers: int | None = None,
    stats: AnalysisBatchStats | None = None,
) -> list[TrackAnalysis]:
    tracks = discover_tracks(folder, recursive=recursive)
    if len(tracks) < 2:
        raise ValueError("DJAI needs at least two supported audio files for a set.")
    return analyze_tracks(
        tracks,
        cache_dir=cache_dir,
        use_cache=use_cache,
        progress=progress,
        workers=workers,
        stats=stats,
    )
