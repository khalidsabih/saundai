from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import librosa
import numpy as np
from scipy.signal import correlate, correlation_lags

from .audio import load_audio, rms

LayerKind = Literal["full", "instrumental", "vocals"]
LAYER_REGISTRY_VERSION = "1.0"
DEFAULT_LAYER_REGISTRY = Path(".djai-cache/performance-layers.json")

_LAYER_PATTERNS: tuple[tuple[LayerKind, re.Pattern[str]], ...] = (
    (
        "vocals",
        re.compile(
            r"(?:[\s._-]+|\s*[\[(])(?:acapella|a\s*cappella|accapella|vocals?|vox)(?:\s*[\])])?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "instrumental",
        re.compile(
            r"(?:[\s._-]+|\s*[\[(])(?:instrumental|instrumentals|inst)(?:\s*[\])])?\s*$",
            re.IGNORECASE,
        ),
    ),
)



@dataclass(slots=True)
class LayerSource:
    path: str
    offset_seconds: float = 0.0
    alignment_confidence: float = 0.0
    supplied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceLayers:
    """User-supplied performance material aligned to a full master timeline."""

    bundle_id: str
    key: str
    full: LayerSource
    instrumental: LayerSource | None = None
    vocals: LayerSource | None = None
    duration_compatible: bool = True
    verified: bool = False
    warnings: list[str] = field(default_factory=list)
    source_quality: str = "supplied_original"
    quality_confidence: float = 1.0

    @property
    def has_instrumental(self) -> bool:
        return self.instrumental is not None and Path(self.instrumental.path).exists()

    @property
    def has_vocals(self) -> bool:
        return self.vocals is not None and Path(self.vocals.path).exists()

    @property
    def complete(self) -> bool:
        return self.has_instrumental and self.has_vocals

    @property
    def fingerprint(self) -> str:
        payload = {
            "bundle_id": self.bundle_id,
            "key": self.key,
            "full": self.full.to_dict(),
            "instrumental": self.instrumental.to_dict() if self.instrumental else None,
            "vocals": self.vocals.to_dict() if self.vocals else None,
            "duration_compatible": bool(self.duration_compatible),
            "verified": bool(self.verified),
            "source_quality": self.source_quality,
            "quality_confidence": round(float(self.quality_confidence), 4),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "key": self.key,
            "full": self.full.to_dict(),
            "instrumental": self.instrumental.to_dict() if self.instrumental else None,
            "vocals": self.vocals.to_dict() if self.vocals else None,
            "duration_compatible": bool(self.duration_compatible),
            "verified": bool(self.verified),
            "warnings": list(self.warnings),
            "source_quality": self.source_quality,
            "quality_confidence": round(float(self.quality_confidence), 4),
            "complete": self.complete,
            "fingerprint": self.fingerprint,
        }


def _registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    explicit = os.environ.get("DJAI_LAYER_REGISTRY")
    if explicit:
        return Path(explicit).expanduser().resolve()
    workspace = os.environ.get("DJAI_WORKSPACE")
    if workspace:
        return (Path(workspace).expanduser().resolve() / ".djai-cache" / "performance-layers.json")
    return DEFAULT_LAYER_REGISTRY.expanduser().resolve()


def _normalized_key(text: str) -> str:
    value = re.sub(r"[\[\](){}]", " ", text)
    value = re.sub(r"[^A-Za-z0-9]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", value)


def classify_layer_filename(path: str | Path) -> tuple[str, LayerKind]:
    """Return a stable bundle key and the musical role implied by a filename."""
    stem = Path(path).stem
    kind: LayerKind = "full"
    cleaned = stem
    for candidate_kind, pattern in _LAYER_PATTERNS:
        if pattern.search(stem):
            kind = candidate_kind
            cleaned = pattern.sub(" ", stem)
            break
    cleaned = re.sub(r"\b(?:clean|dirty|explicit|radio|extended|original\s+mix)\b", " ", cleaned, flags=re.IGNORECASE)
    key = _normalized_key(cleaned) or _normalized_key(stem)
    return key, kind


def _bundle_id(parent: Path, key: str) -> str:
    payload = f"{parent.resolve()}::{key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _load_registry(path: str | Path | None = None) -> dict[str, Any]:
    source = _registry_path(path)
    if not source.exists():
        return {"version": LAYER_REGISTRY_VERSION, "bundles": {}}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": LAYER_REGISTRY_VERSION, "bundles": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("bundles"), dict):
        return {"version": LAYER_REGISTRY_VERSION, "bundles": {}}
    payload["version"] = LAYER_REGISTRY_VERSION
    return payload


def _save_registry(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    target = _registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _duration_seconds(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        audio, sr = load_audio(path, sample_rate=8_000, mono=True)
        return float(audio.shape[-1] / max(sr, 1))


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return array
    array = array - float(np.mean(array))
    scale = float(np.std(array))
    return array / max(scale, 1e-8)


def _alignment_envelope(path: Path, *, sample_rate: int = 8_000, duration: float = 90.0) -> tuple[np.ndarray, float]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration, dtype=np.float32)
    if y.size < 512:
        return np.zeros(0, dtype=np.float64), 0.0
    hop = 256
    # RMS captures phrase shape for acapellas while onset strength remains useful
    # for instrumentals. Their average is robust across separately mastered files.
    rms_env = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop, center=True)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    length = min(rms_env.size, onset.size)
    if length <= 4:
        return np.zeros(0, dtype=np.float64), 0.0
    envelope = 0.55 * _standardize(rms_env[:length]) + 0.45 * _standardize(onset[:length])
    return envelope, hop / sr


def estimate_alignment_offset(
    full_path: str | Path,
    layer_path: str | Path,
    *,
    max_offset_seconds: float = 4.0,
) -> tuple[float, float]:
    """Estimate source-time offset of a supplied layer relative to the full mix.

    Positive means the layer has extra leading material/silence, so a full-track
    cue at t should read the layer near t + offset.
    """
    full_env, seconds_per_frame = _alignment_envelope(Path(full_path).expanduser().resolve())
    layer_env, layer_seconds_per_frame = _alignment_envelope(Path(layer_path).expanduser().resolve())
    if not full_env.size or not layer_env.size or seconds_per_frame <= 0.0:
        return 0.0, 0.0
    seconds_per_frame = max(seconds_per_frame, layer_seconds_per_frame)
    length = min(full_env.size, layer_env.size)
    full_env = full_env[:length]
    layer_env = layer_env[:length]
    values = correlate(layer_env, full_env, mode="full", method="fft")
    lags = correlation_lags(layer_env.size, full_env.size, mode="full")
    limit = max(1, int(round(max_offset_seconds / seconds_per_frame)))
    mask = np.abs(lags) <= limit
    if not np.any(mask):
        return 0.0, 0.0
    local_values = values[mask]
    local_lags = lags[mask]
    best_index = int(np.argmax(local_values))
    lag = int(local_lags[best_index])
    denominator = math.sqrt(float(np.sum(full_env * full_env) * np.sum(layer_env * layer_env))) + 1e-9
    confidence = max(0.0, min(1.0, float(local_values[best_index]) / denominator))
    return round(lag * seconds_per_frame, 6), round(confidence, 4)


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    full_raw = record.get("full")
    if not full_raw:
        record.update({"duration_compatible": False, "verified": False, "warnings": ["No full master is registered yet."]})
        return record
    full = Path(str(full_raw)).expanduser().resolve()
    warnings: list[str] = []
    try:
        full_duration = _duration_seconds(full)
    except Exception as exc:
        record.update({"duration_compatible": False, "verified": False, "warnings": [f"Could not inspect full master: {exc}"]})
        return record

    duration_compatible = True
    alignment: dict[str, dict[str, float]] = {}
    for role in ("instrumental", "vocals"):
        raw = record.get(role)
        if not raw:
            continue
        path = Path(str(raw)).expanduser().resolve()
        try:
            duration = _duration_seconds(path)
            tolerance = max(2.5, full_duration * 0.025)
            if abs(duration - full_duration) > tolerance:
                duration_compatible = False
                warnings.append(
                    f"{role} duration differs from the full master by {abs(duration - full_duration):.2f}s."
                )
            offset, confidence = estimate_alignment_offset(full, path)
            # Weak evidence is not allowed to invent a timing shift. Official
            # alternate masters are commonly sample-aligned, so zero is safer.
            if confidence < 0.10:
                offset = 0.0
                warnings.append(f"{role} alignment evidence is weak; assuming shared zero point.")
            alignment[role] = {"offset_seconds": offset, "confidence": confidence}
        except Exception as exc:
            alignment[role] = {"offset_seconds": 0.0, "confidence": 0.0}
            warnings.append(f"Could not validate {role}: {exc}")

    record["alignment"] = alignment
    record["duration_compatible"] = duration_compatible
    record["verified"] = bool(duration_compatible and record.get("instrumental") and record.get("vocals"))
    record["warnings"] = warnings
    return record



def register_performance_bundle(
    full: str | Path,
    *,
    instrumental: str | Path | None = None,
    vocals: str | Path | None = None,
    registry_path: str | Path | None = None,
    validate: bool = True,
) -> PerformanceLayers:
    """Explicitly register alternate masters when filenames are non-standard."""
    full_path = Path(full).expanduser().resolve()
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    key, _ = classify_layer_filename(full_path)
    bundle_id = _bundle_id(full_path.parent, key)
    payload = _load_registry(registry_path)
    bundles = payload.setdefault("bundles", {})
    record = dict(bundles.get(bundle_id, {}))
    record.update(
        {
            "bundle_id": bundle_id,
            "key": key,
            "parent": str(full_path.parent),
            "source_quality": "supplied_original",
            "full": str(full_path),
        }
    )
    for role, raw in (("instrumental", instrumental), ("vocals", vocals)):
        if raw is None:
            continue
        source = Path(raw).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        record[role] = str(source)
    if validate:
        record = _validate_record(record)
    bundles[bundle_id] = record
    _save_registry(payload, registry_path)
    bundle = _record_to_bundle(record)
    if bundle is None:  # pragma: no cover - full is guaranteed above
        raise RuntimeError("Could not create performance-layer bundle.")
    return bundle


def register_performance_layers(
    paths: Iterable[str | Path],
    *,
    registry_path: str | Path | None = None,
    validate: bool = True,
) -> list[PerformanceLayers]:
    """Register a batch of user-supplied full/instrumental/vocal files.

    Files can arrive over multiple uploads. The registry merges records by
    directory + normalized song name.
    """
    payload = _load_registry(registry_path)
    bundles = payload.setdefault("bundles", {})
    touched: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            continue
        key, kind = classify_layer_filename(path)
        bundle_id = _bundle_id(path.parent, key)
        record = dict(bundles.get(bundle_id, {}))
        record.update({"bundle_id": bundle_id, "key": key, "parent": str(path.parent), "source_quality": "supplied_original"})
        record[kind] = str(path)
        bundles[bundle_id] = record
        touched.add(bundle_id)

    results: list[PerformanceLayers] = []
    for bundle_id in touched:
        record = dict(bundles[bundle_id])
        if validate and record.get("full") and (record.get("instrumental") or record.get("vocals")):
            record = _validate_record(record)
            bundles[bundle_id] = record
        bundle = _record_to_bundle(record)
        if bundle is not None:
            results.append(bundle)
    _save_registry(payload, registry_path)
    return results


def _record_to_bundle(record: dict[str, Any]) -> PerformanceLayers | None:
    full_raw = record.get("full")
    if not full_raw:
        return None
    alignment = dict(record.get("alignment", {}))

    def source(role: str) -> LayerSource | None:
        raw = record.get(role)
        if not raw:
            return None
        info = dict(alignment.get(role, {}))
        return LayerSource(
            path=str(Path(str(raw)).expanduser().resolve()),
            offset_seconds=float(info.get("offset_seconds", 0.0)),
            alignment_confidence=float(info.get("confidence", 0.0)),
        )

    return PerformanceLayers(
        bundle_id=str(record.get("bundle_id", "")),
        key=str(record.get("key", "")),
        full=LayerSource(path=str(Path(str(full_raw)).expanduser().resolve()), alignment_confidence=1.0),
        instrumental=source("instrumental"),
        vocals=source("vocals"),
        duration_compatible=bool(record.get("duration_compatible", True)),
        verified=bool(record.get("verified", False)),
        warnings=[str(value) for value in record.get("warnings", [])],
        source_quality=str(record.get("source_quality", "supplied_original")),
        quality_confidence=float(record.get("quality_confidence", 1.0)),
    )


def find_performance_layers(
    track: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> PerformanceLayers | None:
    resolved = Path(track).expanduser().resolve()
    payload = _load_registry(registry_path)
    for record in payload.get("bundles", {}).values():
        if not isinstance(record, dict) or not record.get("full"):
            continue
        try:
            full = Path(str(record["full"])).expanduser().resolve()
        except Exception:
            continue
        if full == resolved:
            bundle = _record_to_bundle(record)
            if bundle is not None and bundle.duration_compatible:
                return bundle
    # Discovery fallback lets CLI/folder workflows use conventionally named
    # files without requiring the web upload endpoint to have registered them.
    key, kind = classify_layer_filename(resolved)
    if kind != "full":
        return None
    siblings = [path for path in resolved.parent.iterdir() if path.is_file()]
    matching = [path for path in siblings if classify_layer_filename(path)[0] == key]
    if len(matching) > 1:
        register_performance_layers(matching, registry_path=registry_path, validate=True)
        return find_performance_layers(resolved, registry_path=registry_path)
    return None



def collapse_performance_layer_paths(
    paths: Iterable[str | Path],
    *,
    registry_path: str | Path | None = None,
    validate: bool = True,
) -> list[Path]:
    """Collapse conventionally named alternate masters into one full track path."""
    resolved = [Path(path).expanduser().resolve() for path in paths]
    groups: dict[tuple[Path, str], list[tuple[Path, LayerKind]]] = {}
    for path in resolved:
        key, role = classify_layer_filename(path)
        groups.setdefault((path.parent, key), []).append((path, role))

    hidden: set[Path] = set()
    for (_, _), members in groups.items():
        full = next((path for path, role in members if role == "full"), None)
        alternates = [path for path, role in members if role != "full"]
        if full is None or not alternates:
            continue
        registered = register_performance_layers(
            [full, *alternates],
            registry_path=registry_path,
            validate=validate,
        )
        bundle = next(
            (item for item in registered if Path(item.full.path).resolve() == full.resolve()),
            None,
        )
        # Only collapse alternates when validation says they are plausibly the
        # same performance timeline. A mismatched remix/edit must remain a
        # separate library track rather than being silently hidden.
        if bundle is not None and bundle.duration_compatible:
            hidden.update(alternates)
    return [path for path in resolved if path not in hidden]



def performance_layers_from_cached_stems(track: str | Path) -> PerformanceLayers | None:
    """Expose cached separator output through the same performance-layer contract.

    Supplied originals remain authoritative. This adapter lets one uploaded full
    master automatically gain full/instrumental/vocal performance material.
    """
    from .stems import ensure_cached_performance_material, find_cached_stems, load_stem_manifest

    resolved = Path(track).expanduser().resolve()
    if not resolved.exists():
        return None
    mapping = find_cached_stems(resolved)
    if not mapping:
        return None
    manifest = load_stem_manifest(resolved)
    instrumental = manifest.get("instrumental")
    quality = dict(manifest.get("quality", {}))
    if not instrumental or not Path(str(instrumental)).exists():
        material = ensure_cached_performance_material(resolved, mapping)
        instrumental = material.get("instrumental")
        quality = dict(material.get("quality", {}))
    if not instrumental or not Path(str(instrumental)).exists():
        return None
    confidence = max(0.0, min(1.0, float(quality.get("quality_confidence", 0.70))))
    warnings = [str(value) for value in quality.get("warnings", [])]
    key, _ = classify_layer_filename(resolved)
    return PerformanceLayers(
        bundle_id=f"ai-{_bundle_id(resolved.parent, key)}",
        key=key,
        full=LayerSource(path=str(resolved), alignment_confidence=1.0, supplied=True),
        instrumental=LayerSource(
            path=str(Path(str(instrumental)).resolve()),
            offset_seconds=0.0,
            alignment_confidence=confidence,
            supplied=False,
        ),
        vocals=LayerSource(
            path=str(Path(mapping["vocals"]).resolve()),
            offset_seconds=0.0,
            alignment_confidence=confidence,
            supplied=False,
        ),
        duration_compatible=bool(quality.get("duration_compatible", True)),
        verified=confidence >= 0.45,
        warnings=warnings,
        source_quality="ai_separated",
        quality_confidence=confidence,
    )


def resolve_performance_layers(
    track: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> PerformanceLayers | None:
    """Return the best available full/instrumental/vocal bundle for a track.

    Per-role priority is supplied original first, then AI-separated material.
    A partial supplied bundle may therefore be completed by cached separation.
    """
    supplied = find_performance_layers(track, registry_path=registry_path)
    generated = performance_layers_from_cached_stems(track)
    if supplied is None:
        return generated
    if generated is None or supplied.complete:
        return supplied

    instrumental = supplied.instrumental if supplied.has_instrumental else generated.instrumental
    vocals = supplied.vocals if supplied.has_vocals else generated.vocals
    generated_used = (not supplied.has_instrumental and instrumental is not None) or (not supplied.has_vocals and vocals is not None)
    confidence = 1.0 if not generated_used else generated.quality_confidence
    warnings = list(dict.fromkeys([*supplied.warnings, *generated.warnings]))
    return PerformanceLayers(
        bundle_id=supplied.bundle_id,
        key=supplied.key,
        full=supplied.full,
        instrumental=instrumental,
        vocals=vocals,
        duration_compatible=supplied.duration_compatible and generated.duration_compatible,
        verified=supplied.duration_compatible and confidence >= 0.45,
        warnings=warnings,
        source_quality="hybrid_supplied_ai",
        quality_confidence=confidence,
    )


def enrich_analysis_with_performance_layers(
    analysis: "Any",
    bundle: PerformanceLayers | None = None,
    *,
    sample_rate: int = 22_050,
    window_bars: int = 8,
) -> "Any":
    """Use the best vocal layer as authoritative local vocal evidence."""
    active = bundle or resolve_performance_layers(analysis.path)
    if active is None or not active.has_vocals:
        return analysis
    vocal_spec = active.vocals
    assert vocal_spec is not None
    vocals, _ = load_audio(vocal_spec.path, sample_rate=sample_rate, mono=True)
    values = np.asarray(vocals, dtype=np.float32).reshape(-1)
    if not values.size:
        return analysis
    global_rms = rms(values)
    if global_rms <= 1e-8:
        return analysis

    # Estimate how often the supplied vocal source is materially active. This
    # is a presence probability, not a loudness decomposition of the master.
    hop = max(128, int(sample_rate * 0.05))
    frame = max(hop * 4, int(sample_rate * 0.20))
    envelope = librosa.feature.rms(y=values, frame_length=frame, hop_length=hop, center=True)[0]
    activity_fraction = float(np.mean(envelope >= max(global_rms * 0.18, 1e-6))) if envelope.size else 0.0
    analysis.vocal_probability = round(max(0.0, min(1.0, activity_fraction)), 4)

    bar_seconds = max(3, analysis.time_signature) * 60.0 / max(analysis.bpm, 1.0)
    for candidate in [*analysis.entry_candidates, *analysis.exit_candidates]:
        usable_bars = max(1, min(window_bars, int(candidate.available_bars or 1)))
        start_seconds = max(0.0, candidate.time_seconds + vocal_spec.offset_seconds)
        start = min(values.size, max(0, int(round(start_seconds * sample_rate))))
        end = min(values.size, start + max(1, int(round(usable_bars * bar_seconds * sample_rate))))
        local = rms(values[start:end])
        activity = max(0.0, min(1.0, local / max(global_rms * 1.20, 1e-8)))
        candidate.vocal_probability = round(max(0.0001, activity), 4)
        candidate.stem_vocal_activity = round(max(candidate.stem_vocal_activity, activity), 4)

    analysis.performance_layer_analysis = {
        "available": 1.0,
        "complete": 1.0 if active.complete else 0.0,
        "verified": 1.0 if active.verified else 0.0,
        "vocal_activity_fraction": round(activity_fraction, 4),
        "alignment_confidence": round(vocal_spec.alignment_confidence, 4),
        "quality_confidence": round(float(active.quality_confidence), 4),
        "source_quality": active.source_quality,
        "bundle_fingerprint": active.fingerprint,
    }
    return analysis


def enrich_analyses_with_performance_layers(analyses: Iterable["Any"]) -> int:
    enriched = 0
    for analysis in analyses:
        bundle = resolve_performance_layers(analysis.path)
        if bundle is None:
            continue
        before = bool(getattr(analysis, "performance_layer_analysis", {}))
        enrich_analysis_with_performance_layers(analysis, bundle)
        if not before and getattr(analysis, "performance_layer_analysis", {}):
            enriched += 1
    return enriched


def bundle_summary(bundle: PerformanceLayers | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "bundle_id": bundle.bundle_id,
        "complete": bundle.complete,
        "verified": bundle.verified,
        "instrumental": bundle.has_instrumental,
        "vocals": bundle.has_vocals,
        "source_quality": bundle.source_quality,
        "quality_confidence": round(float(bundle.quality_confidence), 4),
        "warnings": list(bundle.warnings),
    }


def layer_source_time(spec: dict[str, Any], role: str, full_time: float) -> float:
    value = spec.get(role)
    if not isinstance(value, dict):
        return float(full_time)
    return max(0.0, float(full_time) + float(value.get("offset_seconds", 0.0)))


def _plan_layer_payload(bundle: PerformanceLayers) -> dict[str, Any]:
    return {
        "bundle_id": bundle.bundle_id,
        "fingerprint": bundle.fingerprint,
        "source_quality": bundle.source_quality,
        "quality_confidence": round(float(bundle.quality_confidence), 4),
        "verified": bundle.verified,
        "full": bundle.full.to_dict(),
        "instrumental": bundle.instrumental.to_dict() if bundle.instrumental else None,
        "vocals": bundle.vocals.to_dict() if bundle.vocals else None,
    }


def augment_layer_transition_plans(
    track_a: "Any",
    track_b: "Any",
    plans: Sequence["Any"],
) -> list["Any"]:
    """Add executable layer alternatives from supplied or AI-separated material."""
    import copy

    from .decision import execution_signature, seal_execution_plan
    from .brain.layer_intelligence import (
        LayerBrainConfig,
        assess_incoming_vocal_over_instrumental,
        incoming_vocal_over_instrumental_choreography,
    )
    from .models import TechniqueAssessment
    from .performance.program import attach_performance_program

    bundle_a = resolve_performance_layers(track_a.path)
    bundle_b = resolve_performance_layers(track_b.path)
    if bundle_a is None and bundle_b is None:
        return list(plans)

    output = list(plans)
    seen = {plan.execution_signature for plan in output}
    layer_cfg = LayerBrainConfig.from_environment()
    bases = [plan for plan in plans if plan.beatmatched and plan.transition_mode != "emergency_cut"][:6]
    for base in bases:
        # Layer Intelligence V1: clear A's vocal by switching to its
        # instrumental, introduce B's vocal as the new lead, transfer the
        # instrumental bed from A to B, then reveal B's full master.  This is
        # deliberately a complete ownership choreography rather than a vocal
        # effect pasted on top of a normal stereo crossfade.
        if (
            layer_cfg.enabled
            and bundle_a
            and bundle_a.complete
            and bundle_a.verified
            and bundle_b
            and bundle_b.complete
            and bundle_b.verified
        ):
            safety = assess_incoming_vocal_over_instrumental(
                track_a,
                track_b,
                base,
                quality_a=bundle_a.quality_confidence,
                quality_b=bundle_b.quality_confidence,
                config=layer_cfg,
            )
            if safety.feasible:
                plan = copy.deepcopy(base)
                plan.technique = "layer_incoming_vocal_handoff"
                plan.intent = "vocal_handoff"
                plan.style = "safe"
                plan.alternative_id = "layer:incoming_vocal_handoff"
                source_confidence = min(bundle_a.quality_confidence, bundle_b.quality_confidence)
                # Reward genuine collision avoidance and source quality while
                # leaving the rendered judge as the deciding authority.
                advantage = 0.030 + 0.050 * safety.vocal_collision_safety + 0.035 * safety.score
                utility = min(
                    0.995,
                    max(float(base.metadata.get("selection_utility", base.score)), float(base.score))
                    + advantage * source_confidence,
                )
                plan.score = round(min(0.99, max(base.score, utility)), 4)
                plan.analytical_score = plan.score
                plan.technique_score = round(min(0.99, 0.76 + 0.18 * safety.score), 4)
                plan.performance_layers_a = _plan_layer_payload(bundle_a)
                plan.performance_layers_b = _plan_layer_payload(bundle_b)
                choreography = incoming_vocal_over_instrumental_choreography()
                plan.rationale = [
                    *base.rationale,
                    "Introduce the incoming vocal over the outgoing instrumental, then transfer instrumental ownership to the incoming track before revealing its full master.",
                    "The full-master transition remains available as a fallback if rendered layer safety does not pass.",
                ]
                plan.technique_assessments = [
                    *base.technique_assessments,
                    TechniqueAssessment(
                        technique="layer_incoming_vocal_handoff",  # type: ignore[arg-type]
                        intent="vocal_handoff",
                        feasible=True,
                        score=plan.technique_score,
                        transition_bars=plan.transition_bars,
                        beatmatched=True,
                        tempo_strategy=plan.tempo_strategy,
                        transition_mode=plan.transition_mode,
                        reasons=list(safety.reasons),
                    ),
                ]
                source_quality = (
                    bundle_a.source_quality
                    if bundle_a.source_quality == bundle_b.source_quality
                    else f"{bundle_a.source_quality}->{bundle_b.source_quality}"
                )
                plan.metadata = {
                    **plan.metadata,
                    "layer_mix": "incoming_vocal_over_outgoing_instrumental",
                    "layer_intelligence_version": "1.0",
                    "layer_bundle_a": bundle_a.fingerprint,
                    "layer_bundle_b": bundle_b.fingerprint,
                    "layer_source_quality": source_quality,
                    "layer_quality_confidence": round(source_confidence, 4),
                    "layer_safety": safety.to_dict(),
                    "layer_choreography": choreography,
                    "layer_fallback": "full_master_transition",
                    "ai_headphones_required_for_authority": True,
                }
                attach_performance_program(plan)
                seal_execution_plan(plan)
                if plan.execution_signature not in seen:
                    output.append(plan)
                    seen.add(plan.execution_signature)

        # Carry an outgoing vocal over the incoming clean instrumental. This is
        # most useful when stereo-vs-stereo analysis predicts vocal collision.
        if bundle_a and bundle_a.has_vocals and bundle_b and bundle_b.has_instrumental:
            if base.compatibility.harmonic >= 0.44 and base.compatibility.beat_grid >= 0.40:
                plan = copy.deepcopy(base)
                plan.technique = "layer_vocal_bridge"
                plan.intent = "vocal_handoff"
                plan.style = "creative"
                plan.alternative_id = "creative"
                safety_need = 1.0 - max(0.0, min(1.0, base.compatibility.vocal_overlap))
                source_confidence = min(bundle_a.quality_confidence, bundle_b.quality_confidence)
                plan.technique_score = round(min(0.98, (0.82 + 0.10 * safety_need) * (0.82 + 0.18 * source_confidence)), 4)
                plan.score = round(min(0.99, max(base.score, 0.72) + (0.035 + 0.035 * safety_need) * source_confidence), 4)
                plan.analytical_score = plan.score
                plan.performance_layers_a = _plan_layer_payload(bundle_a)
                plan.performance_layers_b = _plan_layer_payload(bundle_b)
                plan.rationale = [
                    *base.rationale,
                    f"{bundle_a.source_quality} outgoing vocals are carried over the {bundle_b.source_quality} incoming instrumental before the full incoming master is revealed.",
                ]
                plan.technique_assessments = [
                    *base.technique_assessments,
                    TechniqueAssessment(
                        technique="layer_vocal_bridge",  # type: ignore[arg-type]
                        intent="vocal_handoff",
                        feasible=True,
                        score=plan.technique_score,
                        transition_bars=plan.transition_bars,
                        beatmatched=True,
                        tempo_strategy=plan.tempo_strategy,
                        transition_mode=plan.transition_mode,
                        reasons=["Vocal and instrumental performance layers are available on the full-master timeline."],
                    ),
                ]
                source_quality = (
                    bundle_a.source_quality
                    if bundle_a.source_quality == bundle_b.source_quality
                    else f"{bundle_a.source_quality}->{bundle_b.source_quality}"
                )
                plan.metadata = {
                    **plan.metadata,
                    "layer_mix": "vocal_bridge",
                    "layer_bundle_a": bundle_a.fingerprint,
                    "layer_bundle_b": bundle_b.fingerprint,
                    "layer_source_quality": source_quality,
                    "layer_quality_confidence": round(source_confidence, 4),
                }
                attach_performance_program(plan)
                seal_execution_plan(plan)
                if plan.execution_signature not in seen:
                    output.append(plan)
                    seen.add(plan.execution_signature)

        # Remove competing vocals during the overlap, then reveal B's full mix.
        if bundle_b and bundle_b.has_instrumental:
            plan = copy.deepcopy(base)
            plan.technique = "layer_instrumental_handoff"
            plan.intent = "groove_handoff" if base.intent != "energy_release" else base.intent
            plan.style = "safe"
            plan.alternative_id = "safe"
            safety_need = 1.0 - max(0.0, min(1.0, base.compatibility.vocal_overlap))
            completeness = 1.0 if bundle_a and bundle_a.has_instrumental else 0.55
            source_confidence = bundle_b.quality_confidence
            if bundle_a and bundle_a.has_instrumental:
                source_confidence = min(source_confidence, bundle_a.quality_confidence)
            plan.technique_score = round(min(0.96, (0.78 + 0.10 * safety_need + 0.06 * completeness) * (0.84 + 0.16 * source_confidence)), 4)
            plan.score = round(min(0.98, max(base.score, 0.70) + (0.025 + 0.04 * safety_need) * source_confidence), 4)
            plan.analytical_score = plan.score
            if bundle_a:
                plan.performance_layers_a = _plan_layer_payload(bundle_a)
            plan.performance_layers_b = _plan_layer_payload(bundle_b)
            plan.rationale = [
                *base.rationale,
                f"{bundle_b.source_quality} instrumental material suppresses vocal competition during the handoff, then the incoming full master is revealed.",
            ]
            plan.technique_assessments = [
                *base.technique_assessments,
                TechniqueAssessment(
                    technique="layer_instrumental_handoff",  # type: ignore[arg-type]
                    intent=plan.intent,
                    feasible=True,
                    score=plan.technique_score,
                    transition_bars=plan.transition_bars,
                    beatmatched=True,
                    tempo_strategy=plan.tempo_strategy,
                    transition_mode=plan.transition_mode,
                    reasons=["An incoming instrumental performance layer is available on the full-master timeline."],
                ),
            ]
            plan.metadata = {
                **plan.metadata,
                "layer_mix": "instrumental_handoff",
                "layer_bundle_a": bundle_a.fingerprint if bundle_a else None,
                "layer_bundle_b": bundle_b.fingerprint,
                "layer_source_quality": bundle_b.source_quality,
                "layer_quality_confidence": round(source_confidence, 4),
            }
            attach_performance_program(plan)
            seal_execution_plan(plan)
            if plan.execution_signature not in seen:
                output.append(plan)
                seen.add(plan.execution_signature)

    return output
