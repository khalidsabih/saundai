from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

JourneyName = Literal["balanced", "smooth", "build", "warmup", "peak", "sunset"]
AccountPrimaryUse = Literal["personal", "dj", "producer", "creator", "venue_event", "other"]
AccountExperienceLevel = Literal["starting", "intermediate", "experienced", "professional"]
AccountGenre = Literal[
    "House",
    "Techno",
    "Hip-Hop",
    "Drum & Bass",
    "Pop",
    "Afro",
    "Trance",
    "R&B",
    "Other",
]
AccountMusicalPriority = Literal[
    "smooth_transitions",
    "energy_progression",
    "creative_transitions",
    "harmonic_mixing",
    "club_ready_sound",
    "discovery",
]


class AccountRegisterRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=12, max_length=256)
    artist_name: str | None = Field(default=None, max_length=120)
    country: str = Field(min_length=2, max_length=80)
    primary_use: AccountPrimaryUse
    experience_level: AccountExperienceLevel | None = None
    preferred_genres: list[AccountGenre] = Field(default_factory=list, max_length=9)
    musical_priorities: list[AccountMusicalPriority] = Field(default_factory=list, max_length=6)
    personalization_enabled: bool = False
    improve_saund_ai: bool = False
    marketing_consent: bool = False
    invite_code: str | None = Field(default=None, max_length=128)


class AccountLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class AccountPreferencesRequest(BaseModel):
    personalization_enabled: bool | None = None
    improve_saund_ai: bool | None = None
    marketing_consent: bool | None = None


class AccountProfileRequest(BaseModel):
    first_name: str | None = Field(default=None, min_length=1, max_length=80)
    last_name: str | None = Field(default=None, min_length=1, max_length=80)
    artist_name: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, min_length=2, max_length=80)
    primary_use: AccountPrimaryUse | None = None
    experience_level: AccountExperienceLevel | None = None
    preferred_genres: list[AccountGenre] | None = Field(default=None, max_length=9)
    musical_priorities: list[AccountMusicalPriority] | None = Field(default=None, max_length=6)



class HostedUploadFile(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    content_type: str | None = Field(default=None, max_length=160)


class HostedUploadIntentRequest(BaseModel):
    files: list[HostedUploadFile] = Field(min_length=1, max_length=64)


class HostedUploadCompleteFile(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    object_key: str = Field(min_length=1, max_length=1024)
    expected_size: int = Field(gt=0)


class HostedUploadCompleteRequest(BaseModel):
    files: list[HostedUploadCompleteFile] = Field(min_length=1, max_length=64)


class SetPlanRequest(BaseModel):
    folder: str | None = "audio"
    track_paths: list[str] = Field(default_factory=list)
    manual: bool = False
    experience: Literal["simple", "pro"] = "simple"
    automatic_delivery: bool | None = None
    listener: str = "local"
    tempo_policy: Literal["safe", "stretch", "cut"] = "safe"
    preferred_style: Literal["auto", "safe", "creative", "emergency"] = "auto"
    journey: JourneyName = "balanced"
    target_duration_seconds: float | None = Field(default=None, ge=300.0, le=43200.0)
    peak_fraction: float = Field(default=0.65, ge=0.15, le=0.95)
    closing_window_seconds: float = Field(default=600.0, ge=30.0, le=7200.0)
    ending_energy: float = Field(default=0.45, ge=0.0, le=1.0)
    start_track: str | None = None
    end_track: str | None = None
    minimum_body_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    minimum_body_fraction: float = Field(default=0.35, ge=0.10, le=0.90)
    use_cache: bool = True
    recursive: bool = False
    analysis_workers: int | None = Field(default=None, ge=0, le=64)
    graph_workers: int | None = Field(default=None, ge=0, le=64)
    quality_check: bool = True
    measurement_top_k: int = Field(default=4, ge=1, le=24)
    measurement_log: str | None = "ratings/automatic-measurements.jsonl"

    @property
    def resolved_automatic_delivery(self) -> bool:
        if self.automatic_delivery is not None:
            return self.automatic_delivery
        return self.experience == "simple"


class AutomaticSetRequest(BaseModel):
    track_paths: list[str] = Field(min_length=2)
    journey: JourneyName = "balanced"
    target_duration_seconds: float | None = Field(default=None, ge=300.0, le=43200.0)
    peak_fraction: float = Field(default=0.65, ge=0.15, le=0.95)
    closing_window_seconds: float = Field(default=600.0, ge=30.0, le=7200.0)
    ending_energy: float = Field(default=0.45, ge=0.0, le=1.0)
    listener: str = "local"
    output_name: str = "djai-set.wav"
    minimum_body_seconds: float = Field(default=30.0, ge=5.0, le=300.0)
    minimum_body_fraction: float = Field(default=0.35, ge=0.10, le=0.90)
    use_cache: bool = True
    measurement_top_k: int = Field(default=4, ge=1, le=24)
    analysis_workers: int | None = Field(default=None, ge=0, le=64)
    graph_workers: int | None = Field(default=None, ge=0, le=64)


class RenderRequest(BaseModel):
    plan_path: str
    output_name: str = "full-set.wav"
    confirm: bool = False
    confirmation_token: str | None = None


class PreviewRequest(BaseModel):
    plan_path: str
    pre_roll_seconds: float = Field(default=8.0, ge=0.0, le=30.0)
    post_roll_seconds: float = Field(default=8.0, ge=0.0, le=30.0)


class StudioTransitionPreviewRequest(BaseModel):
    """Render a rehearsal-only primary or alternative transition.

    Studio previews never mutate the authoritative set plan. They expose an
    already-prepared executable transition so a human can compare it before
    asking the Executive to create a new plan.
    """

    plan_path: str
    transition_index: int = Field(ge=0)
    alternative_index: int | None = Field(default=None, ge=0, le=24)
    pre_roll_seconds: float = Field(default=8.0, ge=0.0, le=30.0)
    post_roll_seconds: float = Field(default=8.0, ge=0.0, le=30.0)


class RatingRequest(BaseModel):
    report_path: str
    alternative: Literal["safe", "creative", "emergency"] | None = None
    transition_index: int | None = Field(default=None, ge=0)
    beat: int = Field(ge=1, le=5)
    phrase: int = Field(ge=1, le=5)
    bass: int = Field(ge=1, le=5)
    energy: int = Field(ge=1, le=5)
    cue: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    notes: str = ""
    listener: str = "web"


class FeedbackRequest(BaseModel):
    event: Literal[
        "set_like",
        "set_dislike",
        "transition_like",
        "transition_dislike",
        "alternative_selected",
        "replanned_after_preview",
        "render_completed",
        "downloaded",
        "skip",
        "replay",
        "playback_completed",
    ]
    listener: str = "local"
    plan_id: str | None = None
    plan_path: str | None = None
    transition_index: int | None = Field(default=None, ge=0)
    value: float | None = Field(default=None, ge=-1.0, le=1.0)
    journey: JourneyName | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RefitRequest(BaseModel):
    ratings_path: str = "ratings/listening-tests.jsonl"
    measurement_path: str | None = "ratings/automatic-measurements.jsonl"
    output_path: str = "config/scoring-model.json"


class GenreTrainRequest(BaseModel):
    manifest_path: str
    output_path: str = "config/genre-model.json"


class StemRequest(BaseModel):
    track_path: str
    model: str = "htdemucs"
    device: str = "cpu"
    force: bool = False

class ListeningSessionRequest(BaseModel):
    track_a: str
    track_b: str
    candidates: int = Field(default=4, ge=2, le=8)


class PreferenceVoteRequest(BaseModel):
    session_id: str
    candidate_a: str
    candidate_b: str
    choice: Literal["a", "b", "tie", "neither"]
    listener: str = "local"
    reasons: list[str] = Field(default_factory=list, max_length=12)


class HumanReviewSessionRequest(BaseModel):
    evaluator: str = Field(min_length=1, max_length=160)
    cohort: Literal["general_listener", "experienced_dj", "professional_dj"]
    scenario_id: str | None = Field(default=None, max_length=160)


class HumanReviewVoteRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=160)
    choice: Literal["a", "b", "tie", "neither"]
    reasons: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    experience_years: float | None = Field(default=None, ge=0.0, le=80.0)


class RankerTrainRequest(BaseModel):
    preferences_path: str = "ratings/transition-preferences.jsonl"
    output_path: str = "config/transition-ranker.json"


class TrainingSnapshotRequest(BaseModel):
    name: str | None = Field(default=None, max_length=64)


class TrainingLabTrainRequest(BaseModel):
    snapshot_id: str | None = Field(default=None, max_length=96)
    ridge: float = Field(default=0.18, ge=0.001, le=10.0)


class HumanReleaseLockRequest(BaseModel):
    cases_path: str = Field(min_length=1, max_length=1024)
    incumbent: str = Field(min_length=1, max_length=160)
    candidate: str = Field(min_length=1, max_length=160)
    output_path: str | None = Field(default=None, max_length=1024)


class HumanReleaseSessionRequest(BaseModel):
    evaluator: str = Field(min_length=1, max_length=160)
    cohort: Literal["general_listener", "experienced_dj", "professional_dj"]


class HumanReleaseVoteRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=160)
    choice: Literal["a", "b", "tie", "neither"]
    catastrophic_a: bool = False
    catastrophic_b: bool = False
    reasons: list[str] = Field(default_factory=list, max_length=12)
