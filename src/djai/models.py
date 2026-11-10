from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .version import DECISION_VERSION, PLANNER_VERSION

CandidateKind = Literal["entry", "exit"]
SectionLabel = Literal[
    "intro",
    "groove",
    "build",
    "breakdown",
    "drop",
    "outro",
    "unknown",
]
TransitionTechnique = Literal[
    "eq_bass_swap",
    "equal_power_blend",
    "filter_build",
    "echo_out_drop_in",
    "loop_roll",
    "quick_cut",
    "backspin",
    "double_drop",
    "stem_vocal_overlay",
    "stem_choreography",
    "layer_vocal_bridge",
    "layer_incoming_vocal_handoff",
    "layer_instrumental_handoff",
]
TransitionStyle = Literal["safe", "creative", "emergency"]
TransitionIntent = Literal[
    "smooth_continuation",
    "groove_handoff",
    "energy_lift",
    "energy_release",
    "breakdown_to_drop",
    "vocal_handoff",
    "dramatic_switch",
    "emergency_escape",
]
TempoPolicy = Literal["safe", "stretch", "cut"]
TempoStrategy = Literal["native_blend", "constant_stretch", "deck_handoff", "native_cut"]
TransitionMode = Literal["beatmatched_blend", "emergency_cut"]
QualityClass = Literal["good", "risky", "poor"]


@dataclass(slots=True)
class KeyEstimate:
    tonic: str
    pitch_class: int
    mode: Literal["major", "minor"]
    camelot: str
    confidence: float

    @property
    def label(self) -> str:
        suffix = "maj" if self.mode == "major" else "min"
        return f"{self.tonic} {suffix}"


@dataclass(slots=True)
class GenreEstimate:
    label: str = "unknown"
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str = "centroid-v1"


@dataclass(slots=True)
class TempoPoint:
    time_seconds: float
    bpm: float
    confidence: float


@dataclass(slots=True)
class TransitionCandidate:
    kind: CandidateKind
    time_seconds: float
    bar_index: int
    score: float
    local_energy: float
    stability: float
    available_bars: int
    boundary_strength: float = 0.0
    phrase_strength: float = 0.0
    beat_grid_confidence: float = 0.0
    local_bpm: float = 0.0
    section_label: SectionLabel = "unknown"
    energy_before: float = 0.0
    energy_after: float = 0.0
    rhythm_change: float = 0.0
    harmonic_change: float = 0.0
    vocal_probability: float = 0.0
    bass_activity: float = 0.0
    local_key: KeyEstimate | None = None
    local_chroma: list[float] = field(default_factory=list)
    local_harmonic_stability: float = 0.0
    stem_vocal_activity: float = 0.0
    stem_bass_activity: float = 0.0




@dataclass(slots=True)
class BarFeature:
    """Bar-synchronous musical state used by the professional DJ brain.

    Values are normalized to 0..1 unless otherwise noted.  Keeping this
    representation in TrackAnalysis lets the policy reason about musical
    trajectories without re-decoding the source file during every search.
    """

    bar_index: int
    start_seconds: float
    end_seconds: float
    section_label: SectionLabel = "unknown"
    energy: float = 0.0
    rhythm: float = 0.0
    brightness: float = 0.0
    percussive_activity: float = 0.0
    vocal_probability: float = 0.0
    bass_activity: float = 0.0
    harmonic_tension: float = 0.0
    boundary_strength: float = 0.0
    phrase_strength: float = 0.0
    beat_grid_confidence: float = 0.0
    local_bpm: float = 0.0
    chroma: list[float] = field(default_factory=list)

@dataclass(slots=True)
class TrackAnalysis:
    path: str
    duration_seconds: float
    sample_rate: int
    bpm: float
    bpm_confidence: float
    beats_seconds: list[float]
    downbeats_seconds: list[float]
    key: KeyEstimate
    loudness_dbfs: float
    energy: float
    brightness: float
    percussive_ratio: float
    integrated_lufs: float = -80.0
    true_peak_dbtp: float = -80.0
    beat_grid_confidence: float = 0.0
    beat_interval_cv: float = 1.0
    downbeat_confidence: float = 0.0
    structure_confidence: float = 0.0
    entry_candidates: list[TransitionCandidate] = field(default_factory=list)
    exit_candidates: list[TransitionCandidate] = field(default_factory=list)
    genre: GenreEstimate = field(default_factory=GenreEstimate)
    time_signature: int = 4
    time_signature_confidence: float = 0.0
    tempo_curve: list[TempoPoint] = field(default_factory=list)
    variable_tempo: bool = False
    vocal_probability: float = 0.0
    bass_activity: float = 0.0
    spectral_rolloff: float = 0.0
    onset_density: float = 0.0
    waveform_peaks: list[float] = field(default_factory=list)
    analysis_version: str = "1.2.0"
    stem_analysis: dict[str, float] = field(default_factory=dict)
    performance_layer_analysis: dict[str, Any] = field(default_factory=dict)
    music_embedding: list[float] = field(default_factory=list)
    section_embeddings: dict[str, list[float]] = field(default_factory=dict)
    embedding_model: str = "none"
    embedding_backend: str = "none"
    embedding_fingerprint: str = ""
    # DJAI 5.0 neural evidence is data; registry authority decides whether a
    # model may influence planning.
    neural_evidence: dict[str, Any] = field(default_factory=dict)
    neural_evidence_fingerprint: str = ""
    cue_model: str = "none"
    cue_model_fingerprint: str = ""
    cue_model_predictions: list[dict[str, float]] = field(default_factory=list)
    bar_features: list[BarFeature] = field(default_factory=list)
    perception_confidence: float = 0.0

    @property
    def title(self) -> str:
        from pathlib import Path

        return Path(self.path).stem

    def to_dict(self, *, include_beats: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if not include_beats:
            data["beats_seconds"] = data["beats_seconds"][:16]
            data["downbeats_seconds"] = data["downbeats_seconds"][:16]
            data["beat_count"] = len(self.beats_seconds)
            data["downbeat_count"] = len(self.downbeats_seconds)
        return data


@dataclass(slots=True)
class CompatibilityBreakdown:
    tempo: float
    harmonic: float
    energy: float
    texture: float
    phrase: float
    sections: float
    overall: float
    tempo_difference_percent: float
    energy_delta: float
    beat_grid: float = 0.0
    boundary: float = 0.0
    section_fit: float = 0.0
    local_tempo_difference_percent: float = 0.0
    genre: float = 0.0
    vocal_overlap: float = 0.0
    bass_conflict: float = 0.0
    learned_adjustment: float = 0.0
    local_harmonic: float = 0.0
    local_key_a: str | None = None
    local_key_b: str | None = None
    local_key_confidence_a: float = 0.0
    local_key_confidence_b: float = 0.0


@dataclass(slots=True)
class TransitionMeasurement:
    version: str
    sample_rate: int
    duration_seconds: float
    loudness_continuity: float = 0.0
    spectral_continuity: float = 0.0
    bass_cleanliness: float = 0.0
    phase_safety: float = 0.0
    artifact_safety: float = 0.0
    silence_safety: float = 0.0
    vocal_cleanliness: float = 0.0
    stem_bass_cleanliness: float = 0.0
    stem_evidence: bool = False
    overall: float = 0.0
    peak: float = 0.0
    clipping_fraction: float = 0.0
    phase_correlation: float = 1.0
    loudness_jump_db: float = 0.0
    loudness_jump_lu: float = 0.0
    transition_lufs: float = -80.0
    true_peak_dbtp: float = -80.0
    true_peak_headroom_db: float = 0.0
    low_frequency_buildup_db: float = 0.0
    silence_fraction: float = 0.0
    boundary_jump: float = 0.0
    rhythm_coherence: float = 0.0
    transient_clarity: float = 0.0
    groove_continuity: float = 0.0
    harmonic_continuity: float = 0.0
    energy_intent_fit: float = 0.0
    drop_payoff: float = 0.5
    musical_overall: float = 0.0
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TechniqueAssessment:
    technique: TransitionTechnique
    intent: TransitionIntent
    feasible: bool
    score: float
    transition_bars: int
    beatmatched: bool
    tempo_strategy: TempoStrategy
    transition_mode: TransitionMode
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TransitionPlan:
    track_a: str
    track_b: str
    exit_time_a: float
    entry_time_b: float
    transition_bars: int
    transition_seconds: float
    target_bpm: float
    time_stretch_rate_b: float
    time_stretch_applied: bool
    technique: TransitionTechnique
    score: float
    compatibility: CompatibilityBreakdown
    tempo_policy: TempoPolicy = "safe"
    tempo_strategy: TempoStrategy = "native_blend"
    transition_mode: TransitionMode = "beatmatched_blend"
    beatmatched: bool = True
    source_bpm_a: float = 0.0
    source_bpm_b: float = 0.0
    rendered_bpm_b: float = 0.0
    warning: str | None = None
    forced: bool = False
    style: TransitionStyle = "safe"
    alternative_id: str = "safe"
    rationale: list[str] = field(default_factory=list)
    exit_section: SectionLabel = "unknown"
    entry_section: SectionLabel = "unknown"
    cue_confidence: float = 0.0
    requires_confirmation: bool = False
    quality: QualityClass = "risky"
    genre_pair: tuple[str, str] = ("unknown", "unknown")
    stem_paths_a: dict[str, str] = field(default_factory=dict)
    stem_paths_b: dict[str, str] = field(default_factory=dict)
    # Supplied full/instrumental/vocal material. This is intentionally
    # separate from true four-stem paths so the renderer never pretends an
    # instrumental is equivalent to drums+bass+other stems.
    performance_layers_a: dict[str, Any] = field(default_factory=dict)
    performance_layers_b: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Unified decision/execution contract.
    intent: TransitionIntent = "smooth_continuation"
    intent_score: float = 0.0
    technique_score: float = 0.0
    technique_assessments: list[TechniqueAssessment] = field(default_factory=list)
    effective_bpm_a: float = 0.0
    effective_bpm_b: float = 0.0
    render_rate_a: float = 1.0
    render_rate_b: float = 1.0
    # Deck-performance state. render_rate_b is the incoming rate during the
    # overlap; after the outgoing deck yields, B can glide back to native BPM.
    phase_nudge_seconds_b: float = 0.0
    tempo_release_seconds_b: float = 0.0
    tempo_release_source_duration_b: float = 0.0
    tempo_release_start_rate_b: float = 1.0
    tempo_release_end_rate_b: float = 1.0
    post_transition_rate_b: float = 1.0
    post_transition_bpm_b: float = 0.0
    source_duration_a: float = 0.0
    source_duration_b: float = 0.0
    compiled_execution: bool = False
    execution_signature: str = ""
    # Content-addressed execution contract. Every master/stem/layer that can
    # alter rendered samples is bound when the plan is compiled.
    execution_asset_paths: dict[str, str] = field(default_factory=dict)
    execution_asset_fingerprints: dict[str, str] = field(default_factory=dict)
    renderer_version: str = ""

    # Render-and-measure quality gate. These fields affect selection and
    # reporting, not the immutable audio-execution signature.
    analytical_score: float = 0.0
    measured_score: float = 0.0
    measurement: TransitionMeasurement | None = None
    measurement_passed: bool = False
    measurement_rank: int = 0
    quality_checked: bool = False

    # Automatic-delivery and personalization metadata. These fields do not
    # alter the immutable audio execution contract.
    fallback_level: int = 0
    fallback_reason: str | None = None
    auto_approved: bool = False
    listener_adjustment: float = 0.0

    # Professional DJ Brain immutable performance-program contract.
    performance_program: dict[str, Any] = field(default_factory=dict)
    performance_program_fingerprint: str = ""

    decision_version: str = DECISION_VERSION
    planner_version: str = PLANNER_VERSION

    @property
    def source_end_a(self) -> float:
        return self.exit_time_a + max(0.0, self.source_duration_a)

    @property
    def source_start_b(self) -> float:
        return max(0.0, self.entry_time_b + self.phase_nudge_seconds_b)

    @property
    def source_end_b(self) -> float:
        return self.source_start_b + max(0.0, self.source_duration_b)

    @property
    def post_transition_source_end_b(self) -> float:
        return self.source_end_b + max(0.0, self.tempo_release_source_duration_b)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
