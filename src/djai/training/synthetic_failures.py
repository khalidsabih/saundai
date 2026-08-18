from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


@dataclass(frozen=True, slots=True)
class SyntheticFailure:
    kind: str
    severity: float
    technical_label: str
    expected_bad: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def corrupt_audio(audio: np.ndarray, sample_rate: int, kind: str, severity: float) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).copy()
    if values.ndim == 1:
        values = values[:, None]
    severity = float(np.clip(severity, 0.0, 1.0))
    if kind == "clipping":
        gain = 1.5 + severity * 6.0
        values = np.clip(values * gain, -1.0, 1.0)
    elif kind == "silence_gap":
        width = max(1, int(sample_rate * (0.03 + 0.45 * severity)))
        center = values.shape[0] // 2
        values[max(0, center - width // 2) : min(values.shape[0], center + width // 2)] = 0.0
    elif kind == "gain_jump":
        center = values.shape[0] // 2
        values[center:] *= 0.25 + 1.8 * severity
    elif kind == "phase_inversion":
        if values.shape[1] == 1:
            values = np.repeat(values, 2, axis=1)
        values[:, 1] *= -1.0
    elif kind == "transient_double":
        delay = max(1, int(sample_rate * (0.008 + 0.070 * severity)))
        echo = np.zeros_like(values)
        echo[delay:] = values[:-delay]
        values = np.clip(values + (0.25 + 0.65 * severity) * echo, -1.0, 1.0)
    elif kind == "bass_overload":
        sos = butter(4, 180.0, btype="lowpass", fs=sample_rate, output="sos")
        bass = sosfilt(sos, values, axis=0).astype(np.float32)
        values = np.clip(values + (0.8 + 2.2 * severity) * bass, -1.0, 1.0)
    elif kind == "timing_shift":
        # Delay the transition material without circular wrap. This creates a
        # controlled boundary/beat-offset failure rather than merely changing
        # gain or spectrum.
        delay = max(1, int(sample_rate * (0.025 + 0.155 * severity)))
        shifted = np.zeros_like(values)
        if delay < values.shape[0]:
            shifted[delay:] = values[:-delay]
        values = shifted
    else:
        raise ValueError(f"Unknown synthetic failure kind: {kind}")
    return values.astype(np.float32)


def generate_synthetic_failure_set(
    source: str | Path,
    output_dir: str | Path,
    *,
    severities: tuple[float, ...] = (0.3, 0.6, 0.9),
) -> Path:
    source_path = Path(source).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(source_path, always_2d=True, dtype="float32")
    kinds = ("clipping", "silence_gap", "gain_jump", "phase_inversion", "transient_double", "bass_overload", "timing_shift")
    records: list[dict[str, object]] = []
    for kind in kinds:
        for severity in severities:
            damaged = corrupt_audio(audio, sr, kind, severity)
            name = f"{source_path.stem}__{kind}__{severity:.2f}.wav"
            destination = root / name
            sf.write(destination, damaged, sr, subtype="PCM_24")
            label = SyntheticFailure(kind=kind, severity=severity, technical_label=kind)
            records.append({"source": str(source_path), "audio": str(destination), **label.to_dict()})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    return manifest
