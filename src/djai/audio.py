from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


AudioArray = np.ndarray


def load_audio(
    path: str | Path,
    *,
    sample_rate: int | None = 44_100,
    mono: bool = False,
) -> tuple[AudioArray, int]:
    """Load audio as float32, channel-first when mono=False."""
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        y, sr = librosa.load(
            audio_path,
            sr=sample_rate,
            mono=mono,
            dtype=np.float32,
        )
    except Exception as exc:  # pragma: no cover - depends on local codecs
        raise RuntimeError(
            f"Could not decode '{audio_path.name}'. Install FFmpeg and libsndfile, "
            "or convert the track to WAV/FLAC."
        ) from exc

    y = np.asarray(y, dtype=np.float32)
    if mono:
        if y.ndim != 1:
            y = librosa.to_mono(y)
        return y, int(sr)

    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        # Keep the MVP renderer stereo. Downmix extra channels predictably.
        mono_mix = librosa.to_mono(y)
        y = np.vstack([mono_mix, mono_mix])
    elif y.shape[0] == 1:
        y = np.vstack([y[0], y[0]])

    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def write_audio(path: str | Path, y: AudioArray, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim == 1:
        data = audio
    elif audio.ndim == 2:
        data = audio.T
    else:
        raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")

    sf.write(output_path, data, sample_rate, subtype="PCM_24")
    return output_path


def slice_audio(
    y: AudioArray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> AudioArray:
    start = max(0, int(round(start_seconds * sample_rate)))
    length = max(0, int(round(duration_seconds * sample_rate)))
    end = min(y.shape[-1], start + length)
    segment = y[..., start:end]
    return pad_to_length(segment, length)


def pad_to_length(y: AudioArray, length: int) -> AudioArray:
    if length <= 0:
        return y[..., :0]
    current = y.shape[-1]
    if current >= length:
        return y[..., :length]
    pad_width = [(0, 0)] * y.ndim
    pad_width[-1] = (0, length - current)
    return np.pad(y, pad_width, mode="constant")


def rms(y: AudioArray) -> float:
    values = np.asarray(y, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values) + 1e-12))


def rms_dbfs(y: AudioArray) -> float:
    return float(20.0 * np.log10(max(rms(y), 1e-8)))


def peak_normalize(y: AudioArray, peak: float = 0.98) -> AudioArray:
    maximum = float(np.max(np.abs(y))) if y.size else 0.0
    if maximum <= 1e-9:
        return np.asarray(y, dtype=np.float32)
    gain = min(1.0, peak / maximum)
    return np.asarray(y * gain, dtype=np.float32)
