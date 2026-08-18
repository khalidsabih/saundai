from __future__ import annotations

"""Runtime adapters for research bootstrap priors.

The bootstrap corpora intentionally live in a feature space that is separate
from SAUND.AI's deterministic execution/safety contract.  This module bridges
that gap in a bounded way:

* cue and duration priors may reorder *valid* candidates in local/advisory mode;
* sequence priors may contribute to the hybrid next-track score when identity
  evidence is available;
* production automatically falls back to shadow unless an explicit production
  approval flag is present;
* no bootstrap prior can bypass render measurement or hard audio failures.

The adapters are deliberately small and auditable.  They consume the exact
JSON artifacts emitted by ``dataset-train-bootstrap`` and do not require the
raw external datasets at runtime.
"""

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..environment import current_environment
from ..models import TrackAnalysis, TransitionCandidate

BOOTSTRAP_RUNTIME_VERSION = "1.1"
_UUID_RE = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")


def _truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return float(np.clip(value, lo, hi))


@dataclass(frozen=True, slots=True)
class BootstrapRuntimeConfig:
    requested_mode: str
    mode: str
    cue_weight: float
    duration_weight: float
    sequence_weight: float
    production_approved: bool
    production_guarded: bool

    @classmethod
    def from_environment(cls) -> "BootstrapRuntimeConfig":
        production = os.environ.get("SAUND_ENV", "development").strip().lower() == "production"
        default_mode = "shadow" if production else "advisory"
        requested = os.environ.get("DJAI_BOOTSTRAP_RUNTIME_MODE", default_mode).strip().lower()
        if requested not in {"off", "shadow", "advisory"}:
            requested = default_mode
        approved = _truthy("SAUND_BOOTSTRAP_PRODUCTION_APPROVED", False)
        guarded = bool(production and requested == "advisory" and not approved)
        mode = "shadow" if guarded else requested
        return cls(
            requested_mode=requested,
            mode=mode,
            cue_weight=_float_env("DJAI_BOOTSTRAP_CUE_WEIGHT", 0.18, 0.0, 0.30),
            duration_weight=_float_env("DJAI_BOOTSTRAP_DURATION_WEIGHT", 0.34, 0.0, 0.55),
            sequence_weight=_float_env("DJAI_BOOTSTRAP_SEQUENCE_WEIGHT", 0.18, 0.0, 0.30),
            production_approved=approved,
            production_guarded=guarded,
        )


@dataclass(frozen=True, slots=True)
class BootstrapPrediction:
    model_id: str
    available: bool
    score: float = 0.5
    confidence: float = 0.0
    raw_score: float = 0.0
    mode: str = "shadow"
    effective_weight: float = 0.0
    corpus_id: str = ""
    source: str = ""
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("score", "confidence", "raw_score", "effective_weight"):
            payload[key] = round(float(payload[key]), 6)
        return payload


@dataclass(frozen=True, slots=True)
class TrackIdentity:
    recording_id: str = ""
    artist: str = ""
    artist_key: str = ""
    confidence: float = 0.0
    source: str = "unresolved"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BootstrapBundle:
    root: str
    corpus_id: str
    cue: dict[str, Any] | None = None
    duration: dict[str, Any] | None = None
    adjacency: dict[str, Any] | None = None
    generalizer: dict[str, Any] | None = None


def _stable_artist_key(value: str) -> str:
    text = " ".join(str(value or "").casefold().split())
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    return f"artist-{digest}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _model_root() -> Path | None:
    override = os.environ.get("DJAI_BOOTSTRAP_MODEL_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = current_environment().workspace / path
        return path.resolve() if path.exists() else None

    root = current_environment().config_dir / "bootstrap-models"
    if not root.exists():
        return None
    candidates: list[tuple[float, str, Path]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        report = _read_json(directory / "training-report.json") or {}
        trained = str(report.get("trained_at_utc") or "")
        try:
            mtime = (directory / "training-report.json").stat().st_mtime
        except OSError:
            mtime = directory.stat().st_mtime
        candidates.append((mtime, trained, directory))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1], item[2].name))[2].resolve()


@lru_cache(maxsize=8)
def _load_bundle_cached(root_text: str) -> BootstrapBundle:
    root = Path(root_text)
    cue = _read_json(root / "cue_point_prior.json")
    duration = _read_json(root / "transition_duration_structure.json")
    adjacency = _read_json(root / "next_track_adjacency_prior.json")
    generalizer = _read_json(root / "next_track_sequence_generalizer.json")
    corpus_id = ""
    for payload in (cue, duration, adjacency, generalizer):
        if payload and payload.get("corpus_id"):
            corpus_id = str(payload["corpus_id"])
            break
    return BootstrapBundle(
        root=str(root),
        corpus_id=corpus_id or root.name,
        cue=cue,
        duration=duration,
        adjacency=adjacency,
        generalizer=generalizer,
    )


def load_bootstrap_bundle() -> BootstrapBundle | None:
    root = _model_root()
    return _load_bundle_cached(str(root)) if root is not None else None


def clear_bootstrap_runtime_cache() -> None:
    _load_bundle_cached.cache_clear()
    _resolve_identity_cached.cache_clear()
    connections = getattr(_SEQUENCE_CONNECTIONS, "connections", None)
    if isinstance(connections, dict):
        for item in connections.values():
            connection = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            try:
                connection.close()
            except (AttributeError, sqlite3.Error):
                pass
        connections.clear()


def _metric_confidence(payload: Mapping[str, Any], *, default: float) -> float:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        test = metrics.get("test")
        if isinstance(test, Mapping):
            primary = test.get("top1_accuracy", test.get("accuracy"))
            try:
                value = float(primary)
            except (TypeError, ValueError):
                value = default
            if math.isfinite(value):
                return float(np.clip(value, 0.0, 1.0))
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, Mapping):
        metric_rows = evaluation.get("metrics")
        if isinstance(metric_rows, Mapping) and isinstance(metric_rows.get("test"), Mapping):
            model = metric_rows["test"].get("model")
            if isinstance(model, Mapping):
                try:
                    value = float(model.get("hit_at_10", default))
                except (TypeError, ValueError):
                    value = default
                if math.isfinite(value):
                    return float(np.clip(value, 0.0, 1.0))
    return float(np.clip(default, 0.0, 1.0))


def _linear_score(payload: Mapping[str, Any], features: Mapping[str, float]) -> float | None:
    names = payload.get("feature_names")
    weights = payload.get("weights")
    means = payload.get("feature_mean")
    scales = payload.get("feature_scale")
    if not isinstance(names, list) or not isinstance(weights, list) or len(names) != len(weights):
        return None
    means = means if isinstance(means, Mapping) else {}
    scales = scales if isinstance(scales, Mapping) else {}
    total = float(payload.get("bias", 0.0) or 0.0)
    try:
        for name, weight in zip(names, weights, strict=True):
            key = str(name)
            value = float(features.get(key, 0.0))
            mean = float(means.get(key, 0.0) or 0.0)
            scale = abs(float(scales.get(key, 1.0) or 1.0))
            if scale < 1e-9:
                scale = 1.0
            total += float(weight) * ((value - mean) / scale)
    except (TypeError, ValueError, OverflowError):
        return None
    return total if math.isfinite(total) else None


def _rank_scores(raw: Sequence[float]) -> list[float]:
    if not raw:
        return []
    if len(raw) == 1:
        return [1.0]
    order = sorted(range(len(raw)), key=lambda index: (raw[index], -index))
    ranks = [0.0] * len(raw)
    denominator = float(len(raw) - 1)
    for position, index in enumerate(order):
        ranks[index] = position / denominator
    return ranks


def _cue_features(track: TrackAnalysis, candidate: TransitionCandidate) -> dict[str, float]:
    bpm = max(0.0, float(track.bpm))
    candidate_time = max(0.0, float(candidate.time_seconds))
    grid_offset = float(track.beats_seconds[0]) if track.beats_seconds else 0.0
    if bpm > 0.0:
        beat_position = ((candidate_time - grid_offset) * bpm / 60.0) + 1.0
        bar_position = beat_position / 4.0
        bar_phase = ((beat_position - 1.0) % 4.0) / 4.0
        phase_8 = (bar_position % 8.0) / 8.0
        phase_16 = (bar_position % 16.0) / 16.0
    else:
        beat_position = bar_position = bar_phase = phase_8 = phase_16 = 0.0
    tau = 2.0 * math.pi
    return {
        "candidate_time_seconds": candidate_time,
        "candidate_log_time": math.log1p(candidate_time),
        "bpm": bpm,
        "candidate_beat_position": beat_position,
        "candidate_bar_position": bar_position,
        "candidate_bar_phase": bar_phase,
        "candidate_bar_phase_sin": math.sin(tau * bar_phase),
        "candidate_bar_phase_cos": math.cos(tau * bar_phase),
        "candidate_8bar_phase": phase_8,
        "candidate_8bar_phase_sin": math.sin(tau * phase_8),
        "candidate_8bar_phase_cos": math.cos(tau * phase_8),
        "candidate_16bar_phase": phase_16,
        "candidate_16bar_phase_sin": math.sin(tau * phase_16),
        "candidate_16bar_phase_cos": math.cos(tau * phase_16),
    }


def _candidate_key(candidate: TransitionCandidate) -> tuple[str, int, float, str]:
    return (
        str(candidate.kind),
        int(candidate.bar_index),
        round(float(candidate.time_seconds), 6),
        str(candidate.section_label),
    )


def cue_candidate_priors(
    track: TrackAnalysis,
    candidates: Sequence[TransitionCandidate],
) -> dict[tuple[str, int, float, str], BootstrapPrediction]:
    cfg = BootstrapRuntimeConfig.from_environment()
    bundle = load_bootstrap_bundle()
    payload = bundle.cue if bundle else None
    if cfg.mode == "off" or not payload:
        return {}
    feature_spaces = {str(value) for value in payload.get("feature_spaces", [])}
    if "external-cue-structural-v3-ranking" not in feature_spaces:
        return {}
    # The external cue model was trained on 4/4 beat-grid geometry.  Do not
    # pretend its phase features transfer to other meters.
    if int(track.time_signature or 4) != 4 or float(track.bpm) <= 0.0:
        return {}

    raw: list[float] = []
    valid: list[TransitionCandidate] = []
    for candidate in candidates:
        score = _linear_score(payload, _cue_features(track, candidate))
        if score is None:
            continue
        valid.append(candidate)
        raw.append(score)
    normalized = _rank_scores(raw)
    if not valid:
        return {}

    heldout = _metric_confidence(payload, default=0.5)
    runtime_evidence = float(np.clip(
        0.45 * float(track.beat_grid_confidence)
        + 0.25 * float(track.structure_confidence)
        + 0.30 * float(track.bpm_confidence),
        0.0,
        1.0,
    ))
    confidence = float(np.clip(math.sqrt(max(0.0, heldout * runtime_evidence)), 0.0, 1.0))
    weight = cfg.cue_weight * confidence if cfg.mode == "advisory" else 0.0
    corpus_id = str(payload.get("corpus_id") or (bundle.corpus_id if bundle else ""))
    return {
        _candidate_key(candidate): BootstrapPrediction(
            model_id="cue_point_prior",
            available=True,
            score=normalized[index],
            confidence=confidence,
            raw_score=raw[index],
            mode=cfg.mode,
            effective_weight=weight,
            corpus_id=corpus_id,
            source="bootstrap_external_cue_structural_v3",
            details={
                "heldout_top1": round(heldout, 6),
                "runtime_evidence": round(runtime_evidence, 6),
                "candidate_count": len(valid),
            },
        )
        for index, candidate in enumerate(valid)
    }


def cue_pair_prediction(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    *,
    exit_priors: Mapping[tuple[str, int, float, str], BootstrapPrediction] | None = None,
    entry_priors: Mapping[tuple[str, int, float, str], BootstrapPrediction] | None = None,
) -> BootstrapPrediction:
    exits = exit_priors if exit_priors is not None else cue_candidate_priors(track_a, track_a.exit_candidates)
    entries = entry_priors if entry_priors is not None else cue_candidate_priors(track_b, track_b.entry_candidates)
    left = exits.get(_candidate_key(exit_candidate))
    right = entries.get(_candidate_key(entry_candidate))
    cfg = BootstrapRuntimeConfig.from_environment()
    if left is None or right is None:
        return BootstrapPrediction(
            model_id="cue_point_prior",
            available=False,
            mode=cfg.mode,
            reason="cue prior unavailable for one or both candidates",
        )
    confidence = min(left.confidence, right.confidence)
    weight = cfg.cue_weight * confidence if cfg.mode == "advisory" else 0.0
    return BootstrapPrediction(
        model_id="cue_point_prior",
        available=True,
        score=float(np.clip(0.5 * left.score + 0.5 * right.score, 0.0, 1.0)),
        confidence=confidence,
        raw_score=0.5 * left.raw_score + 0.5 * right.raw_score,
        mode=cfg.mode,
        effective_weight=weight,
        corpus_id=left.corpus_id or right.corpus_id,
        source="bootstrap_pairwise_cue_prior",
        details={
            "exit_score": round(left.score, 6),
            "entry_score": round(right.score, 6),
            "exit_raw": round(left.raw_score, 6),
            "entry_raw": round(right.raw_score, 6),
        },
    )


def _duration_features(seconds: float) -> dict[str, float]:
    duration = max(float(seconds), 1e-6)
    return {
        "duration_seconds": duration,
        "duration_log_seconds": math.log(duration),
        "duration_under_4s": 1.0 if duration < 4.0 else 0.0,
        "duration_over_32s": 1.0 if duration > 32.0 else 0.0,
    }


def rank_duration_values(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    durations: Sequence[int],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    original = tuple(int(value) for value in durations)
    cfg = BootstrapRuntimeConfig.from_environment()
    bundle = load_bootstrap_bundle()
    payload = bundle.duration if bundle else None
    if cfg.mode == "off" or not payload or not original:
        return original, {
            "model_id": "transition_duration_structure",
            "available": False,
            "mode": cfg.mode,
            "reason": "duration bootstrap prior unavailable",
        }
    feature_spaces = {str(value) for value in payload.get("feature_spaces", [])}
    if "external-transition-duration-v4-ranking" not in feature_spaces:
        return original, {
            "model_id": "transition_duration_structure",
            "available": False,
            "mode": cfg.mode,
            "reason": "duration feature space mismatch",
        }

    pair_bpm = math.sqrt(max(float(track_a.bpm), 1e-6) * max(float(track_b.bpm), 1e-6))
    seconds = [float(value) * 4.0 * 60.0 / pair_bpm for value in original]
    raw = [_linear_score(payload, _duration_features(value)) for value in seconds]
    if any(value is None for value in raw):
        return original, {
            "model_id": "transition_duration_structure",
            "available": False,
            "mode": cfg.mode,
            "reason": "duration artifact could not be evaluated",
        }
    raw_values = [float(value) for value in raw if value is not None]
    learned = _rank_scores(raw_values)
    base = [1.0 - (index / max(1, len(original) - 1)) for index in range(len(original))]
    heldout = _metric_confidence(payload, default=0.5)
    runtime_evidence = float(np.clip(
        0.5 * min(float(track_a.beat_grid_confidence), float(track_b.beat_grid_confidence))
        + 0.5 * min(float(track_a.structure_confidence), float(track_b.structure_confidence)),
        0.0,
        1.0,
    ))
    confidence = float(np.clip(math.sqrt(max(0.0, heldout * runtime_evidence)), 0.0, 1.0))
    effective = cfg.duration_weight * confidence if cfg.mode == "advisory" else 0.0
    combined = [
        (1.0 - effective) * base[index] + effective * learned[index]
        for index in range(len(original))
    ]
    order = sorted(range(len(original)), key=lambda index: (combined[index], -index), reverse=True)
    ordered = tuple(original[index] for index in order) if cfg.mode == "advisory" else original
    return ordered, {
        "model_id": "transition_duration_structure",
        "available": True,
        "mode": cfg.mode,
        "corpus_id": str(payload.get("corpus_id") or (bundle.corpus_id if bundle else "")),
        "confidence": round(confidence, 6),
        "effective_weight": round(effective, 6),
        "heldout_top1": round(heldout, 6),
        "changed_order": ordered != original,
        "original": list(original),
        "ordered": list(ordered),
        "candidates": [
            {
                "bars": original[index],
                "seconds": round(seconds[index], 6),
                "raw_score": round(raw_values[index], 6),
                "learned_rank_score": round(learned[index], 6),
                "combined_score": round(combined[index], 6),
            }
            for index in range(len(original))
        ],
    }


def _sidecar_candidates(path: Path) -> Iterable[Path]:
    yield path.with_name(path.name + ".saund.json")
    yield path.with_suffix(".saund.json")


@lru_cache(maxsize=4096)
def _resolve_identity_cached(path_text: str) -> TrackIdentity:
    path = Path(path_text)
    for sidecar in _sidecar_candidates(path):
        payload = _read_json(sidecar)
        if not payload:
            continue
        recording = str(
            payload.get("recording_mbid")
            or payload.get("musicbrainz_recording_id")
            or payload.get("recording_id")
            or payload.get("mbid")
            or ""
        ).strip()
        artist = str(payload.get("artist") or payload.get("artist_name") or "").strip()
        if recording or artist:
            return TrackIdentity(
                recording_id=recording,
                artist=artist,
                artist_key=_stable_artist_key(artist),
                confidence=1.0 if recording and artist else 0.86,
                source=f"sidecar:{sidecar.name}",
            )

    stem = path.stem
    recording_match = _UUID_RE.search(stem)
    recording = recording_match.group(0).lower() if recording_match else ""
    artist = ""
    for separator in (" - ", " – ", " — "):
        if separator in stem:
            left = stem.split(separator, 1)[0].strip()
            left = re.sub(r"^\d{1,3}\s*[-._]\s*", "", left).strip()
            if left:
                artist = left
                break
    if recording or artist:
        confidence = 0.78 if recording and artist else 0.62 if recording else 0.44
        return TrackIdentity(
            recording_id=recording,
            artist=artist,
            artist_key=_stable_artist_key(artist),
            confidence=confidence,
            source="filename",
        )
    return TrackIdentity()


def resolve_track_identity(track: TrackAnalysis) -> TrackIdentity:
    return _resolve_identity_cached(str(Path(track.path).expanduser().resolve()))


_SEQUENCE_CONNECTIONS = threading.local()


def _runtime_sequence_index(bundle: BootstrapBundle) -> Path | None:
    root = Path(bundle.root)
    candidates: list[Path] = []
    if bundle.adjacency and isinstance(bundle.adjacency.get("runtime_index"), Mapping):
        value = str(bundle.adjacency["runtime_index"].get("path") or "").strip()
        if value:
            candidates.append(root / value)
    candidates.append(root / "sequence-runtime.sqlite")
    return next((path for path in candidates if path.exists() and path.is_file()), None)


def _sequence_connection(path: Path) -> sqlite3.Connection:
    connections = getattr(_SEQUENCE_CONNECTIONS, "connections", None)
    if not isinstance(connections, dict):
        connections = {}
        _SEQUENCE_CONNECTIONS.connections = connections
    resolved = path.resolve()
    stat = resolved.stat()
    signature = (int(stat.st_mtime_ns), int(stat.st_size))
    key = str(resolved)
    existing = connections.get(key)
    if isinstance(existing, tuple) and len(existing) == 2 and existing[0] == signature:
        return existing[1]
    if existing is not None:
        old_connection = existing[1] if isinstance(existing, tuple) and len(existing) == 2 else existing
        try:
            old_connection.close()
        except (AttributeError, sqlite3.Error):
            pass
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, check_same_thread=False)
    connection.execute("PRAGMA query_only=ON")
    connections[key] = (signature, connection)
    return connection


def _runtime_index_evidence(
    bundle: BootstrapBundle,
    source: TrackIdentity,
    target: TrackIdentity,
) -> list[tuple[float, float, dict[str, Any]]]:
    path = _runtime_sequence_index(bundle)
    if path is None:
        return []
    try:
        connection = _sequence_connection(path)
    except sqlite3.Error:
        return []
    evidence: list[tuple[float, float, dict[str, Any]]] = []
    try:
        if source.recording_id and target.recording_id:
            row = connection.execute(
                "SELECT count,outgoing_count FROM adjacency WHERE source_track=? AND target_track=?",
                (source.recording_id, target.recording_id),
            ).fetchone()
            if row:
                probability = int(row[0]) / max(1, int(row[1]))
                score = float(np.clip(probability / 0.20, 0.0, 1.0))
                evidence.append((score, 0.86, {
                    "kind": "runtime_exact_adjacency",
                    "count": int(row[0]),
                    "conditional_probability": round(probability, 8),
                }))
        if source.artist_key and target.recording_id:
            row = connection.execute(
                "SELECT count,outgoing_count FROM artist_track WHERE source_artist_key=? AND target_track=?",
                (source.artist_key, target.recording_id),
            ).fetchone()
            if row:
                probability = int(row[0]) / max(1, int(row[1]))
                score = float(np.clip(probability / 0.12, 0.0, 1.0))
                evidence.append((score, 0.64, {
                    "kind": "runtime_source_artist_to_track",
                    "count": int(row[0]),
                    "conditional_probability": round(probability, 8),
                }))
        if source.artist_key and target.artist_key:
            row = connection.execute(
                "SELECT count,outgoing_count FROM artist_pair WHERE source_artist_key=? AND target_artist_key=?",
                (source.artist_key, target.artist_key),
            ).fetchone()
            if row:
                probability = int(row[0]) / max(1, int(row[1]))
                score = float(np.clip(probability / 0.12, 0.0, 1.0))
                evidence.append((score, 0.56, {
                    "kind": "runtime_source_artist_to_target_artist",
                    "count": int(row[0]),
                    "conditional_probability": round(probability, 8),
                }))
    except sqlite3.Error:
        return []
    return evidence


def _adjacency_lookup(payload: Mapping[str, Any], source: str, target: str) -> tuple[float, dict[str, Any]] | None:
    if not source or not target:
        return None
    rows = payload.get("top_edges")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("from_track_id")) == source and str(row.get("to_track_id")) == target:
            probability = float(row.get("conditional_probability", 0.0) or 0.0)
            score = float(np.clip(probability / 0.25, 0.0, 1.0))
            return score, {"kind": "exact_adjacency", "count": int(row.get("count", 0) or 0), "probability": probability}
    return None


def _artist_target_lookup(payload: Mapping[str, Any], source_artist_key: str, target: str) -> tuple[float, dict[str, Any]] | None:
    if not source_artist_key or not target:
        return None
    rows = payload.get("top_artist_targets")
    if not isinstance(rows, list):
        return None
    best_count = 0
    matches: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("source_artist_key")) != source_artist_key:
            continue
        best_count = max(best_count, int(row.get("count", 0) or 0))
        if str(row.get("to_track_id")) == target:
            matches.append(row)
    if not matches:
        return None
    row = matches[0]
    count = int(row.get("count", 0) or 0)
    rank = int(row.get("artist_rank", 999999) or 999999)
    count_score = count / max(1, best_count)
    rank_score = 1.0 / (1.0 + math.log2(max(1, rank)))
    score = float(np.clip(0.58 * count_score + 0.42 * rank_score, 0.0, 1.0))
    return score, {"kind": "source_artist_to_track", "count": count, "artist_rank": rank}


def _artist_pair_lookup(payload: Mapping[str, Any], source_artist_key: str, target_artist_key: str) -> tuple[float, dict[str, Any]] | None:
    if not source_artist_key or not target_artist_key:
        return None
    cold = payload.get("cold_target_intelligence")
    if not isinstance(cold, Mapping):
        return None
    rows = cold.get("top_source_artist_target_artists")
    if not isinstance(rows, list):
        return None
    best_count = 0
    match: Mapping[str, Any] | None = None
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("source_artist_key")) != source_artist_key:
            continue
        best_count = max(best_count, int(row.get("count", 0) or 0))
        if str(row.get("target_artist_key")) == target_artist_key:
            match = row
    if match is None:
        return None
    count = int(match.get("count", 0) or 0)
    rank = int(match.get("artist_rank", 999999) or 999999)
    count_score = count / max(1, best_count)
    rank_score = 1.0 / (1.0 + math.log2(max(1, rank)))
    score = float(np.clip(0.58 * count_score + 0.42 * rank_score, 0.0, 1.0))
    return score, {"kind": "source_artist_to_target_artist", "count": count, "artist_rank": rank}


def sequence_pair_prediction(track_a: TrackAnalysis, track_b: TrackAnalysis) -> BootstrapPrediction:
    cfg = BootstrapRuntimeConfig.from_environment()
    bundle = load_bootstrap_bundle()
    if cfg.mode == "off" or bundle is None:
        return BootstrapPrediction(
            model_id="bootstrap_next_track",
            available=False,
            mode=cfg.mode,
            reason="bootstrap model bundle unavailable",
        )
    source = resolve_track_identity(track_a)
    target = resolve_track_identity(track_b)
    evidence: list[tuple[float, float, dict[str, Any]]] = _runtime_index_evidence(bundle, source, target)

    # Older 1.0.x shadow runs exported only top-N JSON summaries.  Keep those
    # as a fallback; a 1.1 retrain writes the full train-only SQLite index.
    if bundle.adjacency and not any(item[2].get("kind") == "runtime_exact_adjacency" for item in evidence):
        exact = _adjacency_lookup(bundle.adjacency, source.recording_id, target.recording_id)
        if exact:
            heldout = _metric_confidence(bundle.adjacency, default=0.10)
            evidence.append((exact[0], min(0.95, 0.50 + heldout), exact[1]))

    if bundle.generalizer:
        artist_track = None if any(item[2].get("kind") == "runtime_source_artist_to_track" for item in evidence) else _artist_target_lookup(bundle.generalizer, source.artist_key, target.recording_id)
        if artist_track:
            heldout = _metric_confidence(bundle.generalizer, default=0.07)
            evidence.append((artist_track[0], min(0.72, 0.34 + 2.5 * heldout), artist_track[1]))
        artist_pair = None if any(item[2].get("kind") == "runtime_source_artist_to_target_artist" for item in evidence) else _artist_pair_lookup(bundle.generalizer, source.artist_key, target.artist_key)
        if artist_pair:
            evidence.append((artist_pair[0], 0.48, artist_pair[1]))

    if not evidence:
        return BootstrapPrediction(
            model_id="bootstrap_next_track",
            available=False,
            mode=cfg.mode,
            corpus_id=bundle.corpus_id,
            reason="no runtime identity match in bootstrap sequence index/summaries",
            details={"source_identity": source.to_dict(), "target_identity": target.to_dict()},
        )

    denominator = sum(conf for _, conf, _ in evidence)
    score = sum(value * conf for value, conf, _ in evidence) / max(1e-9, denominator)
    identity_confidence = min(source.confidence, target.confidence)
    confidence = float(np.clip((denominator / len(evidence)) * identity_confidence, 0.0, 1.0))
    weight = cfg.sequence_weight * confidence if cfg.mode == "advisory" else 0.0
    return BootstrapPrediction(
        model_id="bootstrap_next_track",
        available=True,
        score=score,
        confidence=confidence,
        raw_score=score,
        mode=cfg.mode,
        effective_weight=weight,
        corpus_id=bundle.corpus_id,
        source="bootstrap_listenbrainz_sequence",
        details={
            "source_identity": source.to_dict(),
            "target_identity": target.to_dict(),
            "evidence": [details for _, _, details in evidence],
        },
    )


def bootstrap_runtime_status() -> dict[str, Any]:
    cfg = BootstrapRuntimeConfig.from_environment()
    bundle = load_bootstrap_bundle()
    models: dict[str, Any] = {}
    if bundle:
        for name, payload in (
            ("cue_point_prior", bundle.cue),
            ("transition_duration_structure", bundle.duration),
            ("next_track_adjacency_prior", bundle.adjacency),
            ("next_track_sequence_generalizer", bundle.generalizer),
        ):
            models[name] = {
                "available": payload is not None,
                "schema_version": str(payload.get("schema_version", "")) if payload else "",
                "trained_at_utc": str(payload.get("trained_at_utc", "")) if payload else "",
                "research_shadow": bool(payload.get("research_shadow", True)) if payload else True,
                "promotable_to_live": bool(payload.get("promotable_to_live", False)) if payload else False,
                "heldout_confidence": round(_metric_confidence(payload, default=0.0), 6) if payload else 0.0,
            }
    return {
        "schema_version": BOOTSTRAP_RUNTIME_VERSION,
        "mode": cfg.mode,
        "requested_mode": cfg.requested_mode,
        "production_guarded": cfg.production_guarded,
        "production_approved": cfg.production_approved,
        "weights": {
            "cue": cfg.cue_weight,
            "duration": cfg.duration_weight,
            "sequence": cfg.sequence_weight,
        },
        "model_root": bundle.root if bundle else None,
        "corpus_id": bundle.corpus_id if bundle else None,
        "models": models,
        "policy": {
            "hard_audio_safety_authoritative": True,
            "bootstrap_can_bypass_render_gate": False,
            "production_research_default": "shadow",
            "local_default": "advisory",
        },
    }
