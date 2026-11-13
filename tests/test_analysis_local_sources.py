from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from djai.analysis import (
    ANALYSIS_VERSION,
    HOP_LENGTH,
    _build_source_frame_features,
    _source_window_scores,
    analyze_track,
)
from djai.audio import write_audio


def _two_section_signal(sample_rate: int = 22_050, seconds: float = 24.0) -> np.ndarray:
    times = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    signal = np.zeros_like(times, dtype=np.float32)

    first = times < seconds / 2.0
    modulation = 0.50 + 0.50 * np.sin(2.0 * np.pi * 4.4 * times[first]) ** 2
    signal[first] = (
        0.24
        * modulation
        * (
            np.sin(2.0 * np.pi * 330.0 * times[first])
            + 0.65 * np.sin(2.0 * np.pi * 550.0 * times[first])
            + 0.30 * np.sin(2.0 * np.pi * 770.0 * times[first])
        )
    ).astype(np.float32)

    second = ~first
    signal[second] = (
        0.42 * np.sin(2.0 * np.pi * 72.0 * times[second])
    ).astype(np.float32)
    for beat in np.arange(seconds / 2.0, seconds, 0.5):
        start = int(beat * sample_rate)
        length = int(0.12 * sample_rate)
        end = min(signal.size, start + length)
        envelope = np.exp(-np.arange(end - start) / (0.028 * sample_rate))
        phase = np.arange(end - start) / sample_rate
        signal[start:end] += (
            0.55 * envelope * np.sin(2.0 * np.pi * 58.0 * phase)
        ).astype(np.float32)

    return np.clip(signal, -0.98, 0.98)


def test_local_source_windows_distinguish_midrange_and_bass() -> None:
    sample_rate = 22_050
    seconds = 24.0
    signal = _two_section_signal(sample_rate, seconds)
    harmonic, percussive = librosa.effects.hpss(signal)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        bins_per_octave=36,
    )
    features = _build_source_frame_features(
        y=signal,
        sr=sample_rate,
        harmonic=harmonic,
        percussive=percussive,
        chroma=chroma,
    )

    midpoint = int(
        librosa.time_to_frames(
            seconds / 2.0,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
        )
    )
    first_vocal, first_bass = _source_window_scores(features, 0, midpoint)
    second_vocal, second_bass = _source_window_scores(
        features,
        midpoint,
        features.frame_times.size,
    )

    assert first_vocal > second_vocal + 0.05
    assert second_bass > first_bass + 0.50


def test_analysis_populates_candidate_local_context(tmp_path: Path) -> None:
    sample_rate = 22_050
    seconds = 48.0
    signal = np.tile(_two_section_signal(sample_rate, seconds / 2.0), 2)

    # Add a stable 120 BPM click grid so candidate generation is deterministic.
    click_length = int(0.025 * sample_rate)
    click = np.exp(-np.arange(click_length) / (0.004 * sample_rate)).astype(np.float32)
    for beat_time in np.arange(0.0, seconds, 0.5):
        start = int(beat_time * sample_rate)
        end = min(signal.size, start + click_length)
        signal[start:end] += 0.15 * click[: end - start]

    path = write_audio(
        tmp_path / "local-context.wav",
        np.vstack([signal, signal]),
        sample_rate,
    )
    analysis = analyze_track(path)
    candidates = [*analysis.entry_candidates, *analysis.exit_candidates]

    assert analysis.analysis_version == ANALYSIS_VERSION
    assert candidates
    assert all(candidate.local_key is not None for candidate in candidates)
    assert all(len(candidate.local_chroma) == 12 for candidate in candidates)
    assert len({round(candidate.bass_activity, 2) for candidate in candidates}) > 1
    assert len({round(candidate.vocal_probability, 2) for candidate in candidates}) > 1
