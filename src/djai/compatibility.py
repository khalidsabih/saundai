from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .learning import learned_prediction

from .models import (
    CompatibilityBreakdown,
    KeyEstimate,
    SectionLabel,
    TrackAnalysis,
    TransitionCandidate,
)


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def match_bpm(bpm_a: float, bpm_b: float) -> tuple[float, float, float]:
    """Return matched B BPM, relative difference, and B time-stretch rate."""
    if bpm_a <= 0.0 or bpm_b <= 0.0:
        return bpm_b, 1.0, 1.0

    options = [bpm_b * factor for factor in (0.5, 1.0, 2.0)]
    matched = min(options, key=lambda value: abs(math.log(max(value, 1e-9) / bpm_a)))
    difference = abs(matched - bpm_a) / max(bpm_a, 1e-9)
    rate = bpm_a / bpm_b
    if matched != bpm_b:
        rate = bpm_a / matched
    return float(matched), float(difference), float(rate)


def tempo_compatibility(bpm_a: float, bpm_b: float) -> tuple[float, float]:
    _, difference, _ = match_bpm(bpm_a, bpm_b)
    score = math.exp(-((difference / 0.07) ** 2))
    return _clamp(score), difference


def _parse_camelot(value: str) -> tuple[int, str]:
    return int(value[:-1]), value[-1].upper()


def harmonic_compatibility(key_a: KeyEstimate, key_b: KeyEstimate) -> float:
    number_a, letter_a = _parse_camelot(key_a.camelot)
    number_b, letter_b = _parse_camelot(key_b.camelot)
    wheel_distance = min((number_a - number_b) % 12, (number_b - number_a) % 12)

    if number_a == number_b and letter_a == letter_b:
        base = 1.0
    elif number_a == number_b and letter_a != letter_b:
        base = 0.96
    elif wheel_distance == 1 and letter_a == letter_b:
        base = 0.92
    elif wheel_distance == 1:
        base = 0.80
    elif wheel_distance == 2 and letter_a == letter_b:
        base = 0.74
    else:
        semitone_distance = min(
            (key_a.pitch_class - key_b.pitch_class) % 12,
            (key_b.pitch_class - key_a.pitch_class) % 12,
        )
        if semitone_distance in (0, 5):
            base = 0.78
        elif semitone_distance in (2, 3, 4, 7):
            base = 0.62
        elif semitone_distance == 6:
            base = 0.28
        else:
            base = 0.45

    confidence = math.sqrt(max(0.0, key_a.confidence * key_b.confidence))
    return _clamp(confidence * base + (1.0 - confidence) * 0.64)




def _normalized_chroma(values: list[float]) -> np.ndarray | None:
    """Return a valid normalized 12-bin chroma profile, or None."""
    profile = np.asarray(values, dtype=np.float64).reshape(-1)
    if profile.size != 12:
        return None
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    profile = np.maximum(profile, 0.0)
    total = float(np.sum(profile))
    if total <= 1e-10:
        return None
    return profile / total


def _chroma_similarity(values_a: list[float], values_b: list[float]) -> float | None:
    """Compare local pitch-class content using the Bhattacharyya coefficient."""
    profile_a = _normalized_chroma(values_a)
    profile_b = _normalized_chroma(values_b)
    if profile_a is None or profile_b is None:
        return None
    return _clamp(float(np.sum(np.sqrt(profile_a * profile_b))))


def _effective_local_key_confidence(candidate: TransitionCandidate) -> float:
    """Combine key-estimator confidence with local harmonic stability."""
    if candidate.local_key is None:
        return 0.0
    key_confidence = _clamp(candidate.local_key.confidence)
    stability = _clamp(candidate.local_harmonic_stability)
    return _clamp(math.sqrt(max(0.0, key_confidence * stability)))


def _local_harmonic_evidence(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
) -> tuple[float, float, float, float | None]:
    """Return final harmonic score plus local diagnostics.

    Returns:
        final_score:
            Confidence-weighted blend used by DJAI.
        local_score:
            Compatibility of the selected local regions before confidence
            blending. Zero means local analysis is unavailable.
        evidence_confidence:
            Reliability of the local comparison.
        chroma_similarity:
            Direct local pitch-class overlap, when both profiles are present.
    """
    global_score = harmonic_compatibility(track_a.key, track_b.key)

    local_key_a = exit_candidate.local_key
    local_key_b = entry_candidate.local_key
    has_local_a = local_key_a is not None
    has_local_b = local_key_b is not None

    # Older cached analyses have no local harmonic fields. Preserve the prior
    # conservative global-key behavior until those tracks are reanalyzed.
    if not has_local_a and not has_local_b:
        change_a = _clamp(exit_candidate.harmonic_change)
        change_b = _clamp(entry_candidate.harmonic_change)
        stability = math.sqrt(max(0.0, (1.0 - change_a) * (1.0 - change_b)))
        neutral = 0.56
        confidence = 0.35 + 0.65 * stability
        final = confidence * global_score + (1.0 - confidence) * neutral
        return _clamp(final), 0.0, 0.0, None

    # If only one candidate was reanalyzed, compare its local key against the
    # other track's global key, but deliberately reduce local authority.
    key_a = local_key_a if local_key_a is not None else track_a.key
    key_b = local_key_b if local_key_b is not None else track_b.key
    local_key_score = harmonic_compatibility(key_a, key_b)

    chroma_similarity = _chroma_similarity(
        exit_candidate.local_chroma,
        entry_candidate.local_chroma,
    )
    if chroma_similarity is None:
        local_score = local_key_score
    else:
        # Key/Camelot relationship is primary; direct pitch-class overlap catches
        # selected regions whose actual chord content does not overlap cleanly.
        local_score = _clamp(
            0.74 * local_key_score
            + 0.26 * chroma_similarity
        )

    confidence_a = (
        _effective_local_key_confidence(exit_candidate)
        if has_local_a
        else _clamp(track_a.key.confidence) * 0.40
    )
    confidence_b = (
        _effective_local_key_confidence(entry_candidate)
        if has_local_b
        else _clamp(track_b.key.confidence) * 0.40
    )
    evidence_confidence = _clamp(
        math.sqrt(max(0.0, confidence_a * confidence_b))
    )

    coverage = (int(has_local_a) + int(has_local_b)) / 2.0
    local_authority = _clamp(
        evidence_confidence * (0.58 + 0.32 * coverage)
    )
    final_score = (
        local_authority * local_score
        + (1.0 - local_authority) * global_score
    )
    return (
        _clamp(final_score),
        _clamp(local_score),
        evidence_confidence,
        chroma_similarity,
    )


def local_harmonic_compatibility(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
) -> float:
    """Return harmonic compatibility for the selected local cue regions."""
    final_score, _, _, _ = _local_harmonic_evidence(
        track_a,
        track_b,
        exit_candidate,
        entry_candidate,
    )
    return final_score


def genre_compatibility(track_a: TrackAnalysis, track_b: TrackAnalysis) -> float:
    probs_a = track_a.genre.probabilities
    probs_b = track_b.genre.probabilities
    if probs_a and probs_b:
        labels = set(probs_a) | set(probs_b)
        overlap = sum(
            math.sqrt(max(0.0, probs_a.get(label, 0.0)) * max(0.0, probs_b.get(label, 0.0)))
            for label in labels
        )
        return _clamp(overlap)
    if track_a.genre.label == "unknown" or track_b.genre.label == "unknown":
        return 0.62
    if track_a.genre.label == track_b.genre.label:
        return 0.96
    families = [
        {"house", "tech_house", "melodic_techno", "techno", "trance"},
        {"hip_hop", "pop"},
    ]
    for family in families:
        if track_a.genre.label in family and track_b.genre.label in family:
            return 0.78
    return 0.46


def _source_conflict_scores(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
) -> tuple[float, float]:
    """Return vocal-overlap and bass-cleanliness scores for the selected cues.

    These are compatibility scores: 1.0 means clean/safe, 0.0 means strong
    collision risk. Risk depends on both sources being active at once, not on
    either track merely containing vocals or bass somewhere globally.
    """
    vocal_a = _clamp(exit_candidate.vocal_probability or track_a.vocal_probability)
    vocal_b = _clamp(entry_candidate.vocal_probability or track_b.vocal_probability)
    bass_a = _clamp(exit_candidate.bass_activity or track_a.bass_activity)
    bass_b = _clamp(entry_candidate.bass_activity or track_b.bass_activity)

    energy_a = _clamp(exit_candidate.local_energy)
    energy_b = _clamp(entry_candidate.local_energy)
    overlap_energy = math.sqrt(max(0.0, energy_a * energy_b))

    # Product/geometric interaction is more appropriate than min(): two very
    # vocal sections are substantially worse than one vocal section plus an
    # instrumental region.
    vocal_interaction = math.sqrt(max(0.0, vocal_a * vocal_b))
    bass_interaction = math.sqrt(max(0.0, bass_a * bass_b))

    # Harmonic motion makes overlapping vocals/melodic phrases less predictable.
    harmonic_motion = _clamp(
        0.5 * exit_candidate.harmonic_change
        + 0.5 * entry_candidate.harmonic_change
    )

    vocal_risk = vocal_interaction * (
        0.52 + 0.30 * overlap_energy + 0.18 * harmonic_motion
    )

    # Bass collision is especially problematic when both sections are energetic
    # and rhythmically active.
    rhythm_activity = _clamp(
        0.5 * exit_candidate.rhythm_change
        + 0.5 * entry_candidate.rhythm_change
    )
    bass_risk = bass_interaction * (
        0.50 + 0.34 * overlap_energy + 0.16 * rhythm_activity
    )

    return _clamp(1.0 - vocal_risk), _clamp(1.0 - bass_risk)


def section_compatibility(exit_label: SectionLabel, entry_label: SectionLabel) -> float:
    """Context-neutral section compatibility.

    The old table made outro->intro the universal optimum, which caused the
    automatic set compiler to behave like a playlist player with a crossfade.
    This table now rewards phrase-role compatibility while leaving the specific
    artistic choice (smooth handoff, lift, drop switch, etc.) to the intent
    policy in decision.py.
    """
    matrix: dict[SectionLabel, dict[SectionLabel, float]] = {
        "outro": {
            "intro": 0.74,
            "groove": 0.90,
            "drop": 0.78,
            "breakdown": 0.62,
            "build": 0.78,
            "outro": 0.35,
            "unknown": 0.60,
        },
        "breakdown": {
            "drop": 0.98,
            "intro": 0.72,
            "groove": 0.88,
            "build": 0.82,
            "breakdown": 0.58,
            "outro": 0.30,
            "unknown": 0.64,
        },
        "build": {
            "drop": 0.96,
            "groove": 0.84,
            "intro": 0.66,
            "build": 0.62,
            "breakdown": 0.55,
            "outro": 0.35,
            "unknown": 0.62,
        },
        "groove": {
            "intro": 0.76,
            "groove": 0.96,
            "drop": 0.88,
            "breakdown": 0.72,
            "build": 0.86,
            "outro": 0.40,
            "unknown": 0.66,
        },
        "drop": {
            "intro": 0.60,
            "groove": 0.76,
            "breakdown": 0.68,
            "drop": 0.62,
            "build": 0.64,
            "outro": 0.32,
            "unknown": 0.56,
        },
        "intro": {
            "intro": 0.56,
            "groove": 0.72,
            "drop": 0.66,
            "build": 0.62,
            "breakdown": 0.50,
            "outro": 0.28,
            "unknown": 0.52,
        },
        "unknown": {
            "intro": 0.58,
            "groove": 0.72,
            "drop": 0.68,
            "build": 0.64,
            "breakdown": 0.56,
            "outro": 0.38,
            "unknown": 0.55,
        },
    }
    return matrix[exit_label][entry_label]


def score_transition(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    exit_candidate: TransitionCandidate,
    entry_candidate: TransitionCandidate,
    *,
    scoring_model: Mapping[str, Any] | None | object = ...,
) -> CompatibilityBreakdown:
    global_tempo, global_tempo_difference = tempo_compatibility(track_a.bpm, track_b.bpm)
    local_bpm_a = exit_candidate.local_bpm or track_a.bpm
    local_bpm_b = entry_candidate.local_bpm or track_b.bpm
    local_tempo, local_tempo_difference = tempo_compatibility(local_bpm_a, local_bpm_b)
    tempo = _clamp(0.68 * global_tempo + 0.32 * local_tempo)
    (
        harmonic,
        local_harmonic,
        local_harmonic_confidence,
        local_chroma_similarity,
    ) = _local_harmonic_evidence(
        track_a,
        track_b,
        exit_candidate,
        entry_candidate,
    )

    energy_delta = entry_candidate.local_energy - exit_candidate.local_energy
    # A slight upward movement is generally useful, but the wide curve also
    # permits deliberate resets and downward transitions.
    energy = math.exp(-(((energy_delta - 0.04) / 0.36) ** 2))
    energy = _clamp(energy)

    texture_distance = (
        0.55 * abs(track_a.percussive_ratio - track_b.percussive_ratio)
        + 0.45 * abs(track_a.brightness - track_b.brightness)
    )
    texture = _clamp(math.exp(-2.2 * texture_distance))

    phrase_a = exit_candidate.phrase_strength or exit_candidate.score
    phrase_b = entry_candidate.phrase_strength or entry_candidate.score
    exit_boundary = exit_candidate.boundary_strength or exit_candidate.score
    entry_boundary = entry_candidate.boundary_strength or entry_candidate.score
    phrase_pair = math.sqrt(max(0.0, phrase_a * phrase_b))
    phrase_weakest = min(phrase_a, phrase_b)
    phrase = _clamp(
        0.46 * phrase_pair
        + 0.34 * phrase_weakest
        + 0.20 * min(exit_boundary, entry_boundary)
    )
    boundary = _clamp(
        math.sqrt(max(0.0, exit_boundary) * max(0.0, entry_boundary))
    )
    track_grid_a = track_a.beat_grid_confidence or track_a.bpm_confidence or 0.55
    track_grid_b = track_b.beat_grid_confidence or track_b.bpm_confidence or 0.55
    exit_grid = exit_candidate.beat_grid_confidence or track_grid_a
    entry_grid = entry_candidate.beat_grid_confidence or track_grid_b
    beat_grid = _clamp(
        (
            max(0.0, track_grid_a)
            * max(0.0, track_grid_b)
            * max(0.0, exit_grid)
            * max(0.0, entry_grid)
        )
        ** 0.25
    )
    section_fit = section_compatibility(
        exit_candidate.section_label,
        entry_candidate.section_label,
    )
    genre = genre_compatibility(track_a, track_b)
    vocal_overlap, bass_conflict = _source_conflict_scores(
        track_a, track_b, exit_candidate, entry_candidate
    )
    weakest_candidate = min(exit_candidate.score, entry_candidate.score)
    sections = _clamp(
        0.18 * exit_candidate.score
        + 0.18 * entry_candidate.score
        + 0.20 * weakest_candidate
        + 0.17 * boundary
        + 0.17 * section_fit
        + 0.10 * min(exit_candidate.stability, entry_candidate.stability)
    )

    overall = (
        0.225 * tempo
        + 0.120 * harmonic
        + 0.080 * energy
        + 0.035 * texture
        + 0.125 * beat_grid
        + 0.110 * phrase
        + 0.145 * sections
        + 0.040 * genre
        + 0.060 * vocal_overlap
        + 0.060 * bass_conflict
    )

    hand_tuned_overall = _clamp(overall)
    learned_features = {
        "tempo": tempo,
        "harmonic": harmonic,
        "energy": energy,
        "texture": texture,
        "beat_grid": beat_grid,
        "phrase": phrase,
        "sections": sections,
        "genre": genre,
        "vocal_overlap": vocal_overlap,
        "bass_conflict": bass_conflict,
        "boundary": boundary,
        "section_fit": section_fit,
    }
    resolved_model = None if scoring_model is ... else scoring_model
    if scoring_model is ...:
        from .learning import load_scoring_model

        resolved_model = load_scoring_model()
    learned, learned_weight = learned_prediction(
        learned_features,
        model=resolved_model,
        genre_pair=(track_a.genre.label, track_b.genre.label),
    )
    learned_adjustment = 0.0
    if learned is not None and learned_weight > 0.0:
        overall = (1.0 - learned_weight) * hand_tuned_overall + learned_weight * learned
        learned_adjustment = overall - hand_tuned_overall

    # A near-zero tempo score cannot be hidden by matching key and loudness.
    # Such pairs may still be usable through a deliberate cut, but not as an
    # automatically approved beatmatched transition.
    effective_difference = max(global_tempo_difference, local_tempo_difference)
    if effective_difference > 0.10:
        overall = min(overall, 0.45)
    elif effective_difference > 0.06:
        overall = min(overall, 0.64)

    # Weak beat grids make long transparent blends unreliable even when global
    # BPM and key happen to look compatible.
    if beat_grid < 0.32:
        overall = min(overall, 0.50)

    # When both local regions are analyzed confidently, a severe local
    # harmonic clash must not be hidden by the tracks' global Camelot keys.
    if local_harmonic_confidence >= 0.58 and local_harmonic > 0.0:
        if local_harmonic < 0.28:
            overall = min(overall, 0.48)
        elif local_harmonic < 0.40:
            overall = min(overall, 0.58)

    # Two strongly active vocal or bass sections should not be presented as a
    # high-confidence transparent overlap even if BPM/key happen to match.
    if vocal_overlap < 0.30:
        overall = min(overall, 0.58)
    if bass_conflict < 0.28:
        overall = min(overall, 0.56)
    if vocal_overlap < 0.30 and bass_conflict < 0.30:
        overall = min(overall, 0.50)

    return CompatibilityBreakdown(
        tempo=round(_clamp(tempo), 4),
        harmonic=round(_clamp(harmonic), 4),
        energy=round(_clamp(energy), 4),
        texture=round(_clamp(texture), 4),
        phrase=round(_clamp(phrase), 4),
        sections=round(_clamp(sections), 4),
        overall=round(_clamp(overall), 4),
        tempo_difference_percent=round(global_tempo_difference * 100.0, 3),
        energy_delta=round(float(energy_delta), 4),
        beat_grid=round(_clamp(beat_grid), 4),
        boundary=round(_clamp(boundary), 4),
        section_fit=round(_clamp(section_fit), 4),
        local_tempo_difference_percent=round(local_tempo_difference * 100.0, 3),
        genre=round(_clamp(genre), 4),
        vocal_overlap=round(_clamp(vocal_overlap), 4),
        bass_conflict=round(_clamp(bass_conflict), 4),
        learned_adjustment=round(float(learned_adjustment), 4),
        local_harmonic=round(_clamp(local_harmonic), 4),
        local_key_a=(
            exit_candidate.local_key.camelot
            if exit_candidate.local_key is not None
            else None
        ),
        local_key_b=(
            entry_candidate.local_key.camelot
            if entry_candidate.local_key is not None
            else None
        ),
        local_key_confidence_a=round(
            _clamp(exit_candidate.local_key.confidence)
            if exit_candidate.local_key is not None
            else 0.0,
            4,
        ),
        local_key_confidence_b=round(
            _clamp(entry_candidate.local_key.confidence)
            if entry_candidate.local_key is not None
            else 0.0,
            4,
        ),
    )