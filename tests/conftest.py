from __future__ import annotations

from pathlib import Path

import pytest

from djai.models import (
    GenreEstimate,
    KeyEstimate,
    TrackAnalysis,
    TransitionCandidate,
)


def key(camelot: str = "8A", confidence: float = 0.9) -> KeyEstimate:
    mapping = {
        "8A": ("A", 9, "minor"),
        "9A": ("E", 4, "minor"),
        "10A": ("B", 11, "minor"),
        "7A": ("D", 2, "minor"),
    }
    tonic, pitch, mode = mapping[camelot]
    return KeyEstimate(tonic=tonic, pitch_class=pitch, mode=mode, camelot=camelot, confidence=confidence)  # type: ignore[arg-type]


def candidate(
    kind: str,
    time_seconds: float,
    bar_index: int,
    section: str,
    *,
    bpm: float,
    local_key: KeyEstimate | None = None,
    energy: float = 0.55,
    score: float = 0.82,
    vocal: float = 0.12,
    bass: float = 0.30,
) -> TransitionCandidate:
    return TransitionCandidate(
        kind=kind,  # type: ignore[arg-type]
        time_seconds=time_seconds,
        bar_index=bar_index,
        score=score,
        local_energy=energy,
        stability=0.88,
        available_bars=32,
        boundary_strength=0.86,
        phrase_strength=0.84,
        beat_grid_confidence=0.91,
        local_bpm=bpm,
        section_label=section,  # type: ignore[arg-type]
        energy_before=max(0.0, energy - 0.05),
        energy_after=energy,
        rhythm_change=0.35,
        harmonic_change=0.08,
        vocal_probability=vocal,
        bass_activity=bass,
        local_key=local_key,
        local_chroma=[1.0 if index == (local_key.pitch_class if local_key else 0) else 0.05 for index in range(12)],
        local_harmonic_stability=0.86 if local_key else 0.0,
    )


def analysis(
    path: Path,
    *,
    bpm: float,
    camelot: str = "8A",
    energy: float = 0.55,
    exit_section: str = "outro",
    entry_section: str = "intro",
) -> TrackAnalysis:
    local = key(camelot)
    return TrackAnalysis(
        path=str(path.resolve()),
        duration_seconds=300.0,
        sample_rate=22050,
        bpm=bpm,
        bpm_confidence=0.92,
        beats_seconds=[index * 60.0 / bpm for index in range(600)],
        downbeats_seconds=[index * 4.0 * 60.0 / bpm for index in range(150)],
        key=local,
        loudness_dbfs=-12.0,
        energy=energy,
        brightness=0.28,
        percussive_ratio=0.62,
        beat_grid_confidence=0.91,
        beat_interval_cv=0.01,
        downbeat_confidence=0.82,
        structure_confidence=0.84,
        entry_candidates=[
            candidate("entry", 8.0, 4, entry_section, bpm=bpm, local_key=local, energy=energy - 0.05),
            candidate("entry", 64.0, 32, "drop", bpm=bpm, local_key=local, energy=min(1.0, energy + 0.16)),
        ],
        exit_candidates=[
            candidate("exit", 210.0, 105, exit_section, bpm=bpm, local_key=local, energy=energy - 0.04),
            candidate("exit", 150.0, 75, "breakdown", bpm=bpm, local_key=local, energy=max(0.1, energy - 0.18)),
        ],
        genre=GenreEstimate(label="house", confidence=0.86, probabilities={"house": 0.86}),
        time_signature=4,
        time_signature_confidence=0.9,
        variable_tempo=False,
        vocal_probability=0.18,
        bass_activity=0.45,
        spectral_rolloff=0.42,
        onset_density=2.0,
        waveform_peaks=[0.2, 0.4, 0.8, 0.5],
        analysis_version="1.3.1",
    )


@pytest.fixture
def track_pair(tmp_path: Path) -> tuple[TrackAnalysis, TrackAnalysis]:
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    a_path.write_bytes(b"a")
    b_path.write_bytes(b"b")
    return (
        analysis(a_path, bpm=124.0, camelot="8A", energy=0.56),
        analysis(b_path, bpm=126.0, camelot="9A", energy=0.62),
    )
