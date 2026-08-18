from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import TrackAnalysis
from .serialization import track_analysis_from_dict
from .version import MUSICAL_TIMELINE_SCHEMA_VERSION
from .perception.world_model import WORLD_MODEL_VERSION, build_track_world_model
from .music.timeline import build_musical_timeline

FEATURE_STORE_SCHEMA_VERSION = "1.0"
SOURCE_FINGERPRINT_SCHEMA_VERSION = "1.0"

# Feature contracts are intentionally independent. A timeline change must not
# invalidate expensive embeddings, and an embedding-model change must not
# destroy rhythm/structure work. Bump only the contract that actually changes.
FEATURE_VERSIONS: dict[str, str] = {
    "descriptors": "descriptors/v1",
    "rhythm": "rhythm/v1",
    "structure": "structure/v1",
    "embeddings": "embeddings/v1",
    "cues": "cues/v1",
    "timeline": f"timeline/{MUSICAL_TIMELINE_SCHEMA_VERSION}",
    "world_model": f"world-model/{WORLD_MODEL_VERSION}",
    "stems": "stems/v2",
}

ANALYSIS_FEATURES = ("descriptors", "rhythm", "structure", "embeddings", "cues")
DJ_READY_FEATURES = (*ANALYSIS_FEATURES, "timeline", "world_model")

_DESCRIPTOR_FIELDS = (
    "duration_seconds",
    "sample_rate",
    "key",
    "loudness_dbfs",
    "energy",
    "brightness",
    "percussive_ratio",
    "genre",
    "vocal_probability",
    "bass_activity",
    "spectral_rolloff",
    "onset_density",
    "waveform_peaks",
    "analysis_version",
    "perception_confidence",
)
_RHYTHM_FIELDS = (
    "bpm",
    "bpm_confidence",
    "beats_seconds",
    "downbeats_seconds",
    "beat_grid_confidence",
    "beat_interval_cv",
    "downbeat_confidence",
    "time_signature",
    "time_signature_confidence",
    "tempo_curve",
    "variable_tempo",
)
_STRUCTURE_FIELDS = (
    "structure_confidence",
    "bar_features",
)
_EMBEDDING_FIELDS = (
    "music_embedding",
    "section_embeddings",
    "embedding_model",
    "embedding_backend",
    "embedding_fingerprint",
    "neural_evidence",
    "neural_evidence_fingerprint",
)
_CUE_FIELDS = (
    "entry_candidates",
    "exit_candidates",
    "cue_model",
    "cue_model_fingerprint",
    "cue_model_predictions",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_component(value: str) -> str:
    return value.replace("..", "_").replace("/", "__").replace("\\", "__")


@dataclass(frozen=True, slots=True)
class FeatureArtifact:
    feature: str
    version: str
    fingerprint: str
    path: Path
    created_at_utc: str
    payload: dict[str, Any]


class FeatureStore:
    """Versioned, source-fingerprint keyed feature storage for one SAUND workspace.

    The store deliberately sits alongside the legacy TrackAnalysis cache. 1.6.1
    can reconstruct a TrackAnalysis from modular projections when all current
    contracts are available, while the legacy cache remains a safe fallback.
    """

    def __init__(self, root: str | Path = ".djai-cache/features") -> None:
        self.root = Path(root).expanduser().resolve()
        self.source_index_root = self.root.parent / "source-index"
        self.readiness_root = self.root.parent / "readiness"

    def _source_index_path(self, resolved: Path) -> Path:
        path_key = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        return self.source_index_root / f"{path_key}.json"

    def register_source_fingerprint(self, path: str | Path, fingerprint: str) -> str:
        resolved = Path(path).expanduser().resolve()
        stat = resolved.stat()
        normalized = str(fingerprint).strip().lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("Source fingerprint must be a SHA-256 hex digest.")
        _atomic_json_write(
            self._source_index_path(resolved),
            {
                "schema_version": SOURCE_FINGERPRINT_SCHEMA_VERSION,
                "path": str(resolved),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "fingerprint": normalized,
                "updated_at_utc": _utc_now(),
            },
        )
        return normalized

    def cached_source_fingerprint(self, path: str | Path) -> str | None:
        """Return a valid cached source digest without reading the audio body."""
        resolved = Path(path).expanduser().resolve()
        try:
            stat = resolved.stat()
            cached = json.loads(self._source_index_path(resolved).read_text(encoding="utf-8"))
            fingerprint = str(cached.get("fingerprint") or "").strip().lower()
            if (
                cached.get("schema_version") == SOURCE_FINGERPRINT_SCHEMA_VERSION
                and int(cached.get("size", -1)) == int(stat.st_size)
                and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
                and len(fingerprint) == 64
                and all(char in "0123456789abcdef" for char in fingerprint)
            ):
                return fingerprint
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return None

    def source_fingerprint(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        cached = self.cached_source_fingerprint(resolved)
        if cached is not None:
            return cached

        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
        return self.register_source_fingerprint(resolved, fingerprint)

    def feature_path(self, fingerprint: str, feature: str, version: str | None = None) -> Path:
        resolved_version = version or FEATURE_VERSIONS[feature]
        return (
            self.root
            / fingerprint
            / _safe_component(feature)
            / f"{_safe_component(resolved_version)}.json"
        )

    def put(
        self,
        path: str | Path,
        feature: str,
        payload: dict[str, Any],
        *,
        version: str | None = None,
        fingerprint: str | None = None,
    ) -> FeatureArtifact:
        resolved = Path(path).expanduser().resolve()
        if feature not in FEATURE_VERSIONS and version is None:
            raise KeyError(f"Unknown feature contract: {feature}")
        version = version or FEATURE_VERSIONS[feature]
        fingerprint = fingerprint or self.source_fingerprint(resolved)
        created = _utc_now()
        target = self.feature_path(fingerprint, feature, version)
        _atomic_json_write(
            target,
            {
                "schema_version": FEATURE_STORE_SCHEMA_VERSION,
                "feature": feature,
                "version": version,
                "fingerprint": fingerprint,
                "source_path": str(resolved),
                "created_at_utc": created,
                "payload": payload,
            },
        )
        return FeatureArtifact(feature, version, fingerprint, target, created, payload)

    def get(
        self,
        path: str | Path,
        feature: str,
        *,
        version: str | None = None,
        fingerprint: str | None = None,
    ) -> FeatureArtifact | None:
        resolved = Path(path).expanduser().resolve()
        if feature not in FEATURE_VERSIONS and version is None:
            return None
        version = version or FEATURE_VERSIONS[feature]
        fingerprint = fingerprint or self.source_fingerprint(resolved)
        target = self.feature_path(fingerprint, feature, version)
        try:
            wrapper = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if (
            wrapper.get("schema_version") != FEATURE_STORE_SCHEMA_VERSION
            or wrapper.get("feature") != feature
            or wrapper.get("version") != version
            or wrapper.get("fingerprint") != fingerprint
            or not isinstance(wrapper.get("payload"), dict)
        ):
            return None
        return FeatureArtifact(
            feature=feature,
            version=version,
            fingerprint=fingerprint,
            path=target,
            created_at_utc=str(wrapper.get("created_at_utc", "")),
            payload=dict(wrapper["payload"]),
        )

    def has(
        self,
        path: str | Path,
        feature: str,
        *,
        version: str | None = None,
        fingerprint: str | None = None,
    ) -> bool:
        return self.get(path, feature, version=version, fingerprint=fingerprint) is not None

    def delete_feature_version(
        self,
        path: str | Path,
        feature: str,
        version: str,
    ) -> bool:
        fingerprint = self.source_fingerprint(path)
        target = self.feature_path(fingerprint, feature, version)
        if not target.exists():
            return False
        target.unlink()
        return True

    def readiness_path(self, fingerprint: str) -> Path:
        return self.readiness_root / f"{fingerprint}.json"

    def _read_readiness(self, fingerprint: str) -> dict[str, Any]:
        path = self.readiness_path(fingerprint)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _write_readiness(self, fingerprint: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["schema_version"] = FEATURE_STORE_SCHEMA_VERSION
        payload["fingerprint"] = fingerprint
        payload["updated_at_utc"] = _utc_now()
        _atomic_json_write(self.readiness_path(fingerprint), payload)

    def set_job(self, path: str | Path, job_id: str | None, *, state: str = "queued") -> None:
        resolved = Path(path).expanduser().resolve()
        fingerprint = self.source_fingerprint(resolved)
        payload = self._read_readiness(fingerprint)
        payload.update({"source_path": str(resolved), "job_id": job_id, "state": state})
        payload.setdefault("features", {})
        self._write_readiness(fingerprint, payload)

    def mark_feature(
        self,
        path: str | Path,
        feature: str,
        status: str,
        *,
        version: str | None = None,
        message: str | None = None,
        fingerprint: str | None = None,
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        fingerprint = fingerprint or self.source_fingerprint(resolved)
        payload = self._read_readiness(fingerprint)
        features = dict(payload.get("features", {}))
        record = dict(features.get(feature, {}))
        record.update(
            {
                "status": status,
                "version": version or FEATURE_VERSIONS.get(feature, "unknown"),
                "updated_at_utc": _utc_now(),
            }
        )
        if message:
            record["message"] = message
        elif "message" in record and status == "ready":
            record.pop("message", None)
        features[feature] = record
        payload.update({"source_path": str(resolved), "features": features})
        if status in {"running", "queued"}:
            payload["state"] = "analyzing" if status == "running" else "queued"
        self._write_readiness(fingerprint, payload)

    def mark_failed(self, path: str | Path, message: str) -> None:
        resolved = Path(path).expanduser().resolve()
        fingerprint = self.source_fingerprint(resolved)
        payload = self._read_readiness(fingerprint)
        payload.update({"source_path": str(resolved), "state": "failed", "error": message})
        self._write_readiness(fingerprint, payload)

    def clear_failure(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        fingerprint = self.source_fingerprint(resolved)
        payload = self._read_readiness(fingerprint)
        payload.pop("error", None)
        if payload.get("state") == "failed":
            payload["state"] = "pending"
        self._write_readiness(fingerprint, payload)

    def track_readiness(
        self,
        path: str | Path,
        *,
        stem_status: str = "unknown",
        stem_required: bool = False,
    ) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        fingerprint = self.source_fingerprint(resolved)
        stored = self._read_readiness(fingerprint)
        stored_features = dict(stored.get("features", {}))
        features: dict[str, dict[str, Any]] = {}
        ready_count = 0
        required = list(DJ_READY_FEATURES)

        for feature in DJ_READY_FEATURES:
            version = FEATURE_VERSIONS[feature]
            available = self.has(resolved, feature, version=version, fingerprint=fingerprint)
            previous = dict(stored_features.get(feature, {}))
            status = "ready" if available else str(previous.get("status", "pending"))
            if status == "ready" and not available:
                status = "stale"
            if available:
                ready_count += 1
            features[feature] = {
                "status": status,
                "version": version,
                "available": available,
                **({"message": previous["message"]} if previous.get("message") else {}),
            }

        stem_ready = stem_status in {"ready", "supplied_layers"}
        stem_skipped = stem_status in {"disabled", "unavailable", "unknown"} and not stem_required
        stem_feature_status = (
            "ready"
            if stem_ready
            else "skipped"
            if stem_skipped
            else "running"
            if stem_status in {"queued", "preparing", "running"}
            else "pending"
        )
        features["stems"] = {
            "status": stem_feature_status,
            "version": FEATURE_VERSIONS["stems"],
            "available": stem_ready,
        }
        denominator = len(required) + (1 if stem_required else 0)
        numerator = ready_count + (1 if stem_required and stem_ready else 0)
        core_ready = ready_count == len(required)
        dj_ready = core_ready and (stem_ready if stem_required else True)

        observed_states = {item["status"] for item in features.values()}
        if dj_ready:
            state = "ready"
            message = "DJ ready"
        elif stored.get("state") == "failed" and not core_ready:
            state = "failed"
            message = str(stored.get("error") or "Track preparation failed")
        elif observed_states & {"running"} or stored.get("state") in {"queued", "analyzing", "running"}:
            state = "analyzing"
            next_feature = next(
                (name for name in required if features[name]["status"] != "ready"),
                "stems" if stem_required and not stem_ready else "features",
            )
            message = f"Preparing {next_feature.replace('_', ' ')}…"
        elif ready_count:
            state = "partial"
            message = "Track intelligence partially prepared"
        else:
            state = "pending"
            message = "Waiting for track preparation"

        return {
            "schema_version": FEATURE_STORE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "state": state,
            "dj_ready": dj_ready,
            "progress": round(numerator / max(1, denominator), 4),
            "message": message,
            "job_id": stored.get("job_id"),
            "features": features,
            "updated_at_utc": stored.get("updated_at_utc"),
        }

    def stats(self) -> dict[str, Any]:
        artifacts = 0
        bytes_used = 0
        tracks: set[str] = set()
        if self.root.exists():
            for path in self.root.rglob("*.json"):
                if not path.is_file():
                    continue
                artifacts += 1
                try:
                    bytes_used += path.stat().st_size
                except OSError:
                    pass
                try:
                    relative = path.relative_to(self.root)
                    if relative.parts:
                        tracks.add(relative.parts[0])
                except ValueError:
                    pass
        return {
            "schema_version": FEATURE_STORE_SCHEMA_VERSION,
            "track_fingerprints": len(tracks),
            "artifacts": artifacts,
            "bytes": bytes_used,
            "feature_versions": dict(FEATURE_VERSIONS),
        }


def _projection(data: dict[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: data[field] for field in fields if field in data}


def persist_analysis_features(
    analysis: TrackAnalysis,
    *,
    store: FeatureStore,
    path: str | Path | None = None,
) -> list[str]:
    """Persist independent feature projections from one complete analysis.

    The current analyzer can still calculate several of these in one pass. The
    storage contracts are split now so later releases can recompute one feature
    family without invalidating every expensive artifact.
    """
    source = Path(path or analysis.path).expanduser().resolve()
    fingerprint = store.source_fingerprint(source)
    data = analysis.to_dict(include_beats=True)
    groups = {
        "descriptors": _projection(data, _DESCRIPTOR_FIELDS),
        "rhythm": _projection(data, _RHYTHM_FIELDS),
        "structure": _projection(data, _STRUCTURE_FIELDS),
        "embeddings": _projection(data, _EMBEDDING_FIELDS),
        "cues": _projection(data, _CUE_FIELDS),
    }
    written: list[str] = []
    for feature, payload in groups.items():
        store.put(source, feature, payload, fingerprint=fingerprint)
        written.append(feature)

    timeline = build_musical_timeline(analysis)
    store.put(source, "timeline", timeline.to_dict(), fingerprint=fingerprint)
    written.append("timeline")

    world_model = build_track_world_model(analysis)
    store.put(source, "world_model", world_model.to_dict(), fingerprint=fingerprint)
    written.append("world_model")
    return written


def load_analysis_from_features(
    path: str | Path,
    *,
    store: FeatureStore,
    expected_analysis_version: str | None = None,
) -> TrackAnalysis | None:
    """Reassemble TrackAnalysis from current modular contracts when possible."""
    if expected_analysis_version is None:
        from .analysis import ANALYSIS_VERSION as expected_analysis_version  # Lazy import avoids cache/analysis cycles.

    resolved = Path(path).expanduser().resolve()
    fingerprint = store.source_fingerprint(resolved)
    artifacts = {
        feature: store.get(resolved, feature, fingerprint=fingerprint)
        for feature in ANALYSIS_FEATURES
    }
    if any(artifact is None for artifact in artifacts.values()):
        return None

    payload: dict[str, Any] = {"path": str(resolved)}
    for feature in ANALYSIS_FEATURES:
        artifact = artifacts[feature]
        assert artifact is not None
        payload.update(artifact.payload)
    # Preserve the existing contract guard: if the analyzer contract itself is
    # bumped, old projections do not silently masquerade as current analysis.
    if payload.get("analysis_version") != expected_analysis_version:
        return None
    try:
        restored = track_analysis_from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None
    restored.path = str(resolved)
    return restored


def feature_store_for_analysis_cache(cache_dir: str | Path) -> FeatureStore:
    cache_root = Path(cache_dir).expanduser().resolve()
    # Standard layout: .djai-cache/tracks -> .djai-cache/features. Custom test
    # caches keep their feature store beside the requested analysis directory.
    parent = cache_root.parent
    return FeatureStore(parent / "features")
