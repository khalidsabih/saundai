from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import numpy as np
    from .models import TrackAnalysis, TransitionPlan

from .cache import fingerprint_file


STEM_NAMES = ("vocals", "drums", "bass", "other")
STEM_MANIFEST_VERSION = "2.0"


def demucs_available() -> bool:
    return importlib.util.find_spec("demucs") is not None or shutil.which("demucs") is not None


def stem_cache_directory(
    track: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
) -> Path:
    # Keep the v1 directory key so existing expensive Demucs outputs remain
    # reusable. 4.2 upgrades the manifest/instrumental in-place.
    return Path(cache_dir).expanduser().resolve() / fingerprint_file(
        track, analysis_version="stems-v1", cache_schema_version="stems-v1"
    )


def find_cached_stems(
    track: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
) -> dict[str, str]:
    root = stem_cache_directory(track, cache_dir=cache_dir)
    mapping = {name: str(root / f"{name}.wav") for name in STEM_NAMES}
    return mapping if all(Path(path).exists() for path in mapping.values()) else {}


def stem_manifest_path(
    track: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
) -> Path:
    return stem_cache_directory(track, cache_dir=cache_dir) / "manifest.json"


def load_stem_manifest(
    track: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
) -> dict[str, Any]:
    path = stem_manifest_path(track, cache_dir=cache_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@contextmanager
def _stem_lock(destination: Path, *, timeout_seconds: float = 60 * 60) -> Iterable[None]:
    """Serialize separation of the same track across API/background workers."""
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".separation.lock"
    deadline = time.monotonic() + max(10.0, timeout_seconds)
    while True:
        try:
            descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii", errors="ignore"))
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
                if age > timeout_seconds:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for stem separation lock: {lock}")
            time.sleep(0.25)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _rms(values: "np.ndarray") -> float:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array)) + 1e-12)) if array.size else 0.0


def _peak(values: "np.ndarray") -> float:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_cached_instrumental(
    track: str | Path,
    stems: dict[str, str] | None = None,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
    force: bool = False,
) -> str:
    """Create instrumental = drums + bass + other on the Demucs timeline."""
    import numpy as np
    import soundfile as sf

    mapping = stems or find_cached_stems(track, cache_dir=cache_dir)
    required = ("drums", "bass", "other")
    if not all(mapping.get(name) and Path(mapping[name]).exists() for name in required):
        raise RuntimeError("Cannot construct an instrumental without drums, bass and other stems.")
    destination = stem_cache_directory(track, cache_dir=cache_dir) / "instrumental.wav"
    if destination.exists() and not force:
        return str(destination)

    arrays: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    for name in required:
        values, sr = sf.read(mapping[name], dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = int(sr)
            channels = int(values.shape[1])
        if int(sr) != sample_rate or values.shape[1] != channels:
            raise RuntimeError("Stem files do not share a common sample rate/channel layout.")
        arrays.append(np.asarray(values, dtype=np.float32))
    length = min(value.shape[0] for value in arrays)
    instrumental = np.sum(np.stack([value[:length] for value in arrays], axis=0), axis=0, dtype=np.float32)
    # Float WAV preserves the separator's level relationship; the transition
    # renderer performs local level matching before use.
    sf.write(destination, instrumental, sample_rate, subtype="FLOAT")
    return str(destination)


def assess_stem_quality(
    track: str | Path,
    stems: dict[str, str] | None = None,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
    sample_rate: int = 16_000,
    duration_seconds: float = 75.0,
) -> dict[str, Any]:
    """Return conservative *proxies* for separator integrity, not ground truth SDR.

    We can verify timeline/duration/reconstruction behavior without pretending
    to know whether every isolated source is perceptually artifact-free.
    """
    import numpy as np

    from .audio import load_audio

    resolved = Path(track).expanduser().resolve()
    mapping = stems or find_cached_stems(resolved, cache_dir=cache_dir)
    if not mapping:
        return {
            "quality_confidence": 0.0,
            "quality_tier": "missing",
            "reconstruction_similarity": 0.0,
            "residual_rms_ratio": 1.0,
            "duration_compatible": False,
            "warnings": ["No complete stem cache exists."],
        }

    master, _ = load_audio(resolved, sample_rate=sample_rate, mono=True)
    maximum = min(master.shape[-1], int(sample_rate * max(10.0, duration_seconds)))
    master = np.asarray(master[..., :maximum], dtype=np.float32).reshape(-1)
    loaded: dict[str, np.ndarray] = {}
    for name in STEM_NAMES:
        values, _ = load_audio(mapping[name], sample_rate=sample_rate, mono=True)
        loaded[name] = np.asarray(values, dtype=np.float32).reshape(-1)
    minimum = min([master.size, *[value.size for value in loaded.values()]])
    if minimum < sample_rate:
        return {
            "quality_confidence": 0.25,
            "quality_tier": "low",
            "reconstruction_similarity": 0.0,
            "residual_rms_ratio": 1.0,
            "duration_compatible": False,
            "warnings": ["Stem material is too short for integrity validation."],
        }
    master = master[:minimum]
    sources = {name: value[:minimum] for name, value in loaded.items()}
    reconstruction = np.sum(np.stack(list(sources.values()), axis=0), axis=0, dtype=np.float32)
    master_rms = _rms(master)
    residual_rms_ratio = _rms(master - reconstruction) / max(master_rms, 1e-8)
    if float(np.std(master)) <= 1e-9 or float(np.std(reconstruction)) <= 1e-9:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(master, reconstruction)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
    correlation_score = _clamp((correlation + 1.0) / 2.0)
    residual_score = _clamp(1.0 - residual_rms_ratio / 0.35)
    reconstruction_score = _clamp(0.68 * correlation_score + 0.32 * residual_score)

    stem_rms = {name: _rms(values) for name, values in sources.items()}
    silent = [name for name, value in stem_rms.items() if value < max(1e-7, master_rms * 0.0015)]
    stem_peaks = {name: _peak(values) for name, values in sources.items()}
    clipping = max(stem_peaks.values(), default=0.0) > 1.25

    # Full Demucs outputs are generated on one timeline. File-length mismatch
    # beyond 100 ms is a strong signal of corruption/copy failure.
    lengths = [Path(mapping[name]).stat().st_size for name in STEM_NAMES if Path(mapping[name]).exists()]
    size_spread = (max(lengths) - min(lengths)) / max(max(lengths), 1) if lengths else 1.0
    duration_compatible = size_spread < 0.08

    integrity = _clamp(
        0.68 * reconstruction_score
        + 0.17 * (1.0 if duration_compatible else 0.25)
        + 0.10 * (0.0 if clipping else 1.0)
        + 0.05 * (1.0 - min(1.0, len(silent) / 3.0))
    )
    warnings: list[str] = []
    if reconstruction_score < 0.62:
        warnings.append("Stem sum does not reconstruct the master closely; use aggressive layer choreography cautiously.")
    if clipping:
        warnings.append("One or more separated stems exceed the expected floating-point peak range.")
    if len(silent) >= 2:
        warnings.append("Multiple stems are nearly silent; source separation may be unreliable for this recording.")
    if not duration_compatible:
        warnings.append("Stem file sizes differ unexpectedly; timeline integrity is uncertain.")
    tier = "high" if integrity >= 0.78 else "usable" if integrity >= 0.56 else "low"
    return {
        "quality_confidence": round(integrity, 4),
        "quality_tier": tier,
        "reconstruction_similarity": round(reconstruction_score, 4),
        "residual_rms_ratio": round(float(residual_rms_ratio), 4),
        "duration_compatible": bool(duration_compatible),
        "silent_stems": silent,
        "stem_rms": {name: round(value, 7) for name, value in stem_rms.items()},
        "stem_peaks": {name: round(value, 6) for name, value in stem_peaks.items()},
        "warnings": warnings,
    }


def ensure_cached_performance_material(
    track: str | Path,
    stems: dict[str, str] | None = None,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
) -> dict[str, Any]:
    mapping = stems or find_cached_stems(track, cache_dir=cache_dir)
    if not mapping:
        return {}
    instrumental = build_cached_instrumental(track, mapping, cache_dir=cache_dir)
    quality = assess_stem_quality(track, mapping, cache_dir=cache_dir)
    return {
        "full": str(Path(track).expanduser().resolve()),
        "instrumental": instrumental,
        "vocals": mapping["vocals"],
        "stems": dict(mapping),
        "quality": quality,
        "source_quality": "ai_separated",
    }


def _write_manifest(
    track: Path,
    mapping: dict[str, str],
    *,
    model: str,
    device: str,
    cache_dir: str | Path,
) -> None:
    destination = stem_cache_directory(track, cache_dir=cache_dir)
    material = ensure_cached_performance_material(track, mapping, cache_dir=cache_dir)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "version": STEM_MANIFEST_VERSION,
                "track": str(track),
                "model": model,
                "device": device,
                "stems": mapping,
                "instrumental": material.get("instrumental"),
                "quality": material.get("quality", {}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _sync_stem_feature_store(
    track: Path,
    mapping: dict[str, str],
    *,
    cache_dir: str | Path,
) -> None:
    try:
        from .feature_store import FEATURE_VERSIONS, FeatureStore

        root = Path(cache_dir).expanduser().resolve().parent / "features"
        FeatureStore(root).put(
            track,
            "stems",
            {
                "paths": {name: str(Path(value).resolve()) for name, value in mapping.items()},
                "complete": bool(mapping),
                "source": "demucs-cache",
                "manifest_version": STEM_MANIFEST_VERSION,
            },
            version=FEATURE_VERSIONS["stems"],
        )
    except (OSError, KeyError, TypeError, ValueError):
        # Demucs output remains authoritative if diagnostics cannot be written.
        return


def separate_stems(
    track: str | Path,
    *,
    cache_dir: str | Path = ".djai-cache/stems",
    model: str = "htdemucs",
    device: str = "cpu",
    force: bool = False,
) -> dict[str, str]:
    resolved = Path(track).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Track not found: {resolved}")
    cached = find_cached_stems(resolved, cache_dir=cache_dir)
    if cached and not force:
        # Upgrade an older v1 cache by deriving the instrumental/quality
        # manifest without re-running the expensive separator.
        manifest = load_stem_manifest(resolved, cache_dir=cache_dir)
        if manifest.get("version") != STEM_MANIFEST_VERSION or not manifest.get("instrumental"):
            _write_manifest(resolved, cached, model=str(manifest.get("model", model)), device=str(manifest.get("device", device)), cache_dir=cache_dir)
        _sync_stem_feature_store(resolved, cached, cache_dir=cache_dir)
        return cached
    if not demucs_available():
        raise RuntimeError(
            "Demucs is not installed. Use the stems Docker target or "
            "install DJAI with the 'stems' extra."
        )

    destination = stem_cache_directory(resolved, cache_dir=cache_dir)
    with _stem_lock(destination):
        cached = find_cached_stems(resolved, cache_dir=cache_dir)
        if cached and not force:
            manifest = load_stem_manifest(resolved, cache_dir=cache_dir)
            if manifest.get("version") != STEM_MANIFEST_VERSION or not manifest.get("instrumental"):
                _write_manifest(resolved, cached, model=str(manifest.get("model", model)), device=str(manifest.get("device", device)), cache_dir=cache_dir)
            _sync_stem_feature_store(resolved, cached, cache_dir=cache_dir)
            return cached
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="djai-demucs-") as temporary:
            command = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                model,
                "-d",
                device,
                "-o",
                temporary,
                str(resolved),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Demucs failed:\n" + (completed.stderr or completed.stdout)[-4000:]
                )
            candidates = list(Path(temporary).glob(f"{model}/**/{STEM_NAMES[0]}.wav"))
            if not candidates:
                raise RuntimeError("Demucs completed but no stem files were found.")
            source_root = candidates[0].parent
            mapping: dict[str, str] = {}
            for name in STEM_NAMES:
                source = source_root / f"{name}.wav"
                target = destination / f"{name}.wav"
                if not source.exists():
                    raise RuntimeError(f"Missing Demucs stem: {source}")
                shutil.copy2(source, target)
                mapping[name] = str(target)
            _write_manifest(resolved, mapping, model=model, device=device, cache_dir=cache_dir)
            _sync_stem_feature_store(resolved, mapping, cache_dir=cache_dir)
            return mapping


def automatic_stem_mode() -> str:
    """Return off, background, or wait according to environment policy."""
    value = os.environ.get("DJAI_AUTO_STEMS", "background").strip().lower()
    return value if value in {"off", "background", "wait"} else "background"


def prepare_stems_for_tracks(
    tracks: Iterable[str | Path],
    *,
    model: str | None = None,
    device: str | None = None,
    progress=None,
) -> dict[str, dict[str, str]]:
    """Synchronize stem preparation only for the explicit ``wait`` policy.

    ``background`` is latency-sensitive: upload-time separation may continue in
    the stem job pool while set planning consumes any completed cache and falls
    back to the stereo master for unfinished tracks. Calling this helper can
    therefore never turn ``background`` into an accidental blocking wait.
    """
    mode = automatic_stem_mode()
    paths = [Path(track).expanduser().resolve() for track in tracks]
    if mode != "wait" or not demucs_available():
        return {str(path): find_cached_stems(path) for path in paths}
    result: dict[str, dict[str, str]] = {}
    for index, path in enumerate(paths):
        # Clean supplied originals outrank AI separation. Import locally to
        # avoid a module-level stems <-> layers cycle.
        try:
            from .layers import find_performance_layers
            supplied = find_performance_layers(path)
        except Exception:
            supplied = None
        if supplied is not None and supplied.complete and supplied.duration_compatible:
            result[str(path)] = {}
            if progress:
                progress("stems", index + 1, len(paths), f"Using supplied performance layers for {path.name}")
            continue
        if progress:
            progress("stems", index, len(paths), f"Preparing performance layers for {path.name}")
        try:
            mapping = separate_stems(
                path,
                model=model or os.environ.get("DJAI_STEM_MODEL", "htdemucs"),
                device=device or os.environ.get("DJAI_STEM_DEVICE", "cpu"),
            )
        except Exception as exc:
            # Source separation is an enhancement, never a reason to make a
            # valid stereo master unplayable. Keep automatic delivery alive
            # and let the planner use the original master.
            mapping = find_cached_stems(path)
            if progress:
                progress(
                    "stems",
                    index + 1,
                    len(paths),
                    f"Layer preparation unavailable for {path.name}; continuing stereo ({type(exc).__name__})",
                )
        result[str(path)] = mapping
    if progress:
        progress("stems", len(paths), len(paths), "Performance layers ready")
    return result


def enrich_analysis_with_stems(
    analysis: "TrackAnalysis",
    stems: dict[str, str] | None = None,
    *,
    sample_rate: int = 22_050,
    window_bars: int = 8,
) -> "TrackAnalysis":
    """Replace heuristic vocal/bass cue evidence with cached stem evidence."""
    import numpy as np

    from .audio import load_audio

    mapping = stems or find_cached_stems(analysis.path)
    if not mapping:
        return analysis
    required = {"vocals", "bass", "drums", "other"}
    if not required.issubset(mapping):
        return analysis

    loaded: dict[str, np.ndarray] = {}
    for name in required:
        audio, _ = load_audio(mapping[name], sample_rate=sample_rate, mono=True)
        loaded[name] = np.asarray(audio, dtype=np.float32).reshape(-1)
    minimum = min((values.size for values in loaded.values()), default=0)
    if minimum <= 0:
        return analysis
    for name in loaded:
        loaded[name] = loaded[name][:minimum]

    global_rms = {name: _rms(values) for name, values in loaded.items()}
    total_global = sum(global_rms.values()) + 1e-9
    analysis.vocal_probability = round(global_rms["vocals"] / total_global, 4)
    analysis.bass_activity = round(global_rms["bass"] / total_global, 4)

    bar_seconds = max(3, analysis.time_signature) * 60.0 / max(analysis.bpm, 1.0)
    for candidate in [*analysis.entry_candidates, *analysis.exit_candidates]:
        usable_bars = max(1, min(window_bars, int(candidate.available_bars or 1)))
        start = max(0, min(minimum - 1, int(round(candidate.time_seconds * sample_rate))))
        end = min(minimum, start + max(1, int(round(usable_bars * bar_seconds * sample_rate))))
        local_rms = {name: _rms(values[start:end]) for name, values in loaded.items()}
        total = sum(local_rms.values()) + 1e-9
        vocal = max(0.0, min(1.0, local_rms["vocals"] / total * 1.55))
        bass = max(0.0, min(1.0, local_rms["bass"] / total * 1.55))
        candidate.stem_vocal_activity = round(vocal, 4)
        candidate.stem_bass_activity = round(bass, 4)
        candidate.vocal_probability = round(max(0.0001, vocal), 4)
        candidate.bass_activity = round(max(0.0001, bass), 4)

    manifest = load_stem_manifest(analysis.path)
    quality = dict(manifest.get("quality", {}))
    analysis.stem_analysis = {
        "available": 1.0,
        "vocal_activity": analysis.vocal_probability,
        "bass_activity": analysis.bass_activity,
        "sample_rate": float(sample_rate),
        "quality_confidence": float(quality.get("quality_confidence", 0.72)),
        "reconstruction_similarity": float(quality.get("reconstruction_similarity", 0.0)),
    }
    return analysis


def enrich_analyses_with_cached_stems(
    analyses: list["TrackAnalysis"],
    *,
    sample_rate: int = 22_050,
) -> int:
    enriched = 0
    for analysis in analyses:
        mapping = find_cached_stems(analysis.path)
        if mapping:
            enrich_analysis_with_stems(analysis, mapping, sample_rate=sample_rate)
            enriched += 1
    return enriched


def measure_stem_transition_safety(
    plan: "TransitionPlan",
    *,
    sample_rate: int = 22_050,
) -> dict[str, float] | None:
    """Measure vocal and bass overlap from isolated stems when available."""
    import numpy as np

    from .audio import load_audio
    from .renderer import _render_segment

    required = ("vocals", "bass")
    if not all(plan.stem_paths_a.get(name) and plan.stem_paths_b.get(name) for name in required):
        return None
    target_length = max(1, int(round(plan.transition_seconds * sample_rate)))
    safety: dict[str, float] = {}
    for name in required:
        source_a, _ = load_audio(plan.stem_paths_a[name], sample_rate=sample_rate, mono=False)
        source_b, _ = load_audio(plan.stem_paths_b[name], sample_rate=sample_rate, mono=False)
        rendered_a = _render_segment(
            source_a,
            sample_rate,
            plan.exit_time_a,
            plan.source_end_a,
            rate=plan.render_rate_a,
            gain=1.0,
            target_length=target_length,
        )
        rendered_b = _render_segment(
            source_b,
            sample_rate,
            plan.entry_time_b,
            plan.source_end_b,
            rate=plan.render_rate_b,
            gain=1.0,
            target_length=target_length,
        ) if plan.source_duration_b > 0 else np.zeros_like(rendered_a)
        local_a = _rms(rendered_a)
        local_b = _rms(rendered_b)
        global_a = _rms(source_a)
        global_b = _rms(source_b)
        activity_a = min(1.0, local_a / max(global_a * 1.35, 1e-8))
        activity_b = min(1.0, local_b / max(global_b * 1.35, 1e-8))
        overlap_risk = float(np.sqrt(max(0.0, activity_a * activity_b)))
        safety[name] = max(0.0, min(1.0, 1.0 - overlap_risk))
    return {
        "vocal_cleanliness": round(safety["vocals"], 4),
        "stem_bass_cleanliness": round(safety["bass"], 4),
    }
