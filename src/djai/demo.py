from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import peak_normalize, write_audio


SAMPLE_RATE = 44_100


def _tone(frequency: float, length: int, sample_rate: int, phase: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time + phase)


def _add_event(
    destination: np.ndarray,
    start: int,
    signal: np.ndarray,
    gain: float = 1.0,
) -> None:
    if start >= destination.size:
        return
    end = min(destination.size, start + signal.size)
    destination[start:end] += gain * signal[: end - start]


def _kick(sample_rate: int) -> np.ndarray:
    length = int(0.42 * sample_rate)
    time = np.arange(length, dtype=np.float64) / sample_rate
    frequency = 48.0 + 95.0 * np.exp(-time * 28.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    envelope = np.exp(-time * 12.0)
    click = np.exp(-time * 90.0) * np.random.default_rng(4).normal(0.0, 0.18, length)
    return (np.sin(phase) * envelope + click).astype(np.float32)


def _hat(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    length = int(0.09 * sample_rate)
    time = np.arange(length, dtype=np.float64) / sample_rate
    envelope = np.exp(-time * 52.0)
    noise = rng.normal(0.0, 1.0, length)
    # A first difference removes much of the low-frequency noise energy.
    bright = np.concatenate([[0.0], np.diff(noise)])
    return (bright * envelope * 0.18).astype(np.float32)


def _bass_note(frequency: float, seconds: float, sample_rate: int) -> np.ndarray:
    length = int(seconds * sample_rate)
    time = np.arange(length, dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, time / 0.012) * np.exp(-time * 2.2)
    signal = np.sin(2.0 * np.pi * frequency * time)
    signal += 0.22 * np.sin(2.0 * np.pi * frequency * 2.0 * time)
    return (signal * envelope * 0.28).astype(np.float32)


def _pad_chord(frequencies: tuple[float, ...], seconds: float, sample_rate: int) -> np.ndarray:
    length = int(seconds * sample_rate)
    time = np.arange(length, dtype=np.float64) / sample_rate
    attack = np.clip(time / 0.35, 0.0, 1.0)
    release = np.clip((seconds - time) / 0.45, 0.0, 1.0)
    envelope = attack * release
    signal = np.zeros(length, dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        signal += np.sin(2.0 * np.pi * frequency * time + index * 0.37)
    signal /= max(1, len(frequencies))
    return (signal * envelope * 0.18).astype(np.float32)


def _arrangement_level(bar: int, total_bars: int, variant: str) -> tuple[float, float, float]:
    if variant == "a":
        if bar < 4:
            return 0.75, 0.0, 0.0
        if bar < 14:
            return 1.0, 1.0, 0.75
        if bar < 18:
            return 0.45, 0.20, 0.90
        if bar < total_bars - 2:
            return 0.90, 0.85, 0.62
        return 0.70, 0.0, 0.0
    if bar < 4:
        return 0.70, 0.0, 0.0
    if bar < 8:
        return 0.90, 0.45, 0.25
    if bar < 18:
        return 1.0, 1.0, 0.82
    if bar < total_bars - 2:
        return 0.82, 0.55, 0.50
    return 0.65, 0.0, 0.0


def generate_demo_track(
    output_path: str | Path,
    *,
    bpm: float,
    root_hz: float,
    minor_third_hz: float,
    fifth_hz: float,
    variant: str,
    bars: int = 24,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    rng = np.random.default_rng(19 if variant == "a" else 31)
    beat_seconds = 60.0 / bpm
    bar_seconds = beat_seconds * 4.0
    duration = bars * bar_seconds
    length = int(np.ceil(duration * sample_rate))
    mono = np.zeros(length, dtype=np.float32)

    kick = _kick(sample_rate)
    hat = _hat(sample_rate, rng)
    bass = _bass_note(root_hz, beat_seconds * 0.88, sample_rate)
    chord = _pad_chord(
        (root_hz * 4.0, minor_third_hz * 4.0, fifth_hz * 4.0),
        bar_seconds * 0.96,
        sample_rate,
    )

    for bar in range(bars):
        drums_level, bass_level, pad_level = _arrangement_level(bar, bars, variant)
        bar_start_seconds = bar * bar_seconds
        bar_start = int(round(bar_start_seconds * sample_rate))

        if pad_level > 0.0:
            _add_event(mono, bar_start, chord, pad_level)

        for beat in range(4):
            beat_start = int(round((bar_start_seconds + beat * beat_seconds) * sample_rate))
            _add_event(mono, beat_start, kick, drums_level)
            if bass_level > 0.0:
                _add_event(mono, beat_start, bass, bass_level)

            eighth_start = int(round((bar_start_seconds + (beat + 0.5) * beat_seconds) * sample_rate))
            _add_event(mono, eighth_start, hat, drums_level)

    # Gentle stereo width without introducing timing ambiguity in the beat grid.
    delay = int(0.006 * sample_rate)
    right = np.pad(mono[:-delay], (delay, 0)) if delay else mono.copy()
    stereo = np.vstack([mono, right * 0.96]).astype(np.float32)
    stereo = peak_normalize(stereo, peak=0.86)
    return write_audio(output_path, stereo, sample_rate)


def generate_demo_pair(output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    track_a = generate_demo_track(
        directory / "demo_a_126bpm_Aminor.wav",
        bpm=126.0,
        root_hz=55.0,
        minor_third_hz=65.406,
        fifth_hz=82.407,
        variant="a",
    )
    track_b = generate_demo_track(
        directory / "demo_b_128bpm_Eminor.wav",
        bpm=128.0,
        root_hz=41.203,
        minor_third_hz=48.999,
        fifth_hz=61.735,
        variant="b",
    )
    return track_a, track_b


def generate_demo_set(
    output_dir: str | Path,
    *,
    bars: int = 32,
) -> list[Path]:
    """Create a small copyright-free 4/4 set for end-to-end verification."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("01_opening_122bpm_Aminor.wav", 122.0, 55.0, 65.406, 82.407, "a"),
        ("02_groove_124bpm_Eminor.wav", 124.0, 41.203, 48.999, 61.735, "b"),
        ("03_drive_126bpm_Bminor.wav", 126.0, 61.735, 73.416, 92.499, "a"),
        ("04_peak_128bpm_Fsharpminor.wav", 128.0, 46.249, 55.0, 69.296, "b"),
        ("05_close_125bpm_Dminor.wav", 125.0, 36.708, 43.654, 55.0, "a"),
    ]
    outputs: list[Path] = []
    for filename, bpm, root, third, fifth, variant in specifications:
        outputs.append(
            generate_demo_track(
                directory / filename,
                bpm=bpm,
                root_hz=root,
                minor_third_hz=third,
                fifth_hz=fifth,
                variant=variant,
                bars=bars,
            )
        )
    return outputs
