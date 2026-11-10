from __future__ import annotations

from dataclasses import asdict, dataclass

from ..models import TrackAnalysis, TransitionPlan

STYLE_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class StyleProfile:
    name: str
    preferred_techniques: tuple[str, ...]
    discouraged_techniques: tuple[str, ...]
    preferred_bars: tuple[int, ...]
    long_blend_bias: float
    reset_tolerance: float
    vocal_overlap_tolerance: float
    bass_handoff_priority: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("preferred_techniques", "discouraged_techniques", "preferred_bars", "reasons"):
            payload[key] = list(payload[key])
        payload["version"] = STYLE_POLICY_VERSION
        return payload


_PROFILES: dict[str, StyleProfile] = {
    "house": StyleProfile(
        "house",
        ("eq_bass_swap", "equal_power_blend", "stem_choreography", "filter_build"),
        ("backspin", "quick_cut"),
        (16, 32, 8, 4),
        0.90, 0.18, 0.35, 0.95,
        ("Preserve phrase-locked groove and make bass ownership explicit across longer blends.",),
    ),
    "techno": StyleProfile(
        "techno",
        ("stem_choreography", "eq_bass_swap", "filter_build", "loop_roll", "equal_power_blend"),
        ("backspin",),
        (16, 32, 8, 4),
        0.95, 0.25, 0.42, 0.92,
        ("Favor sustained layering, tension continuity, and controlled low-frequency ownership.",),
    ),
    "drum_and_bass": StyleProfile(
        "drum_and_bass",
        ("double_drop", "stem_choreography", "eq_bass_swap", "filter_build", "quick_cut"),
        ("equal_power_blend",),
        (4, 8, 2, 16),
        0.35, 0.58, 0.34, 0.98,
        ("Prefer fast phrase decisions, drop alignment, and decisive bass handoffs.",),
    ),
    "hip_hop_open_format": StyleProfile(
        "hip_hop_open_format",
        ("quick_cut", "echo_out_drop_in", "backspin", "stem_vocal_overlay", "layer_vocal_bridge"),
        ("equal_power_blend",),
        (1, 2, 4, 8),
        0.05, 0.92, 0.18, 0.65,
        ("Cuts and resets are legitimate musical moves; protect lead vocals and strong entry events.",),
    ),
    "pop": StyleProfile(
        "pop",
        ("layer_vocal_bridge", "stem_vocal_overlay", "echo_out_drop_in", "filter_build", "quick_cut"),
        ("double_drop", "backspin"),
        (4, 8, 2, 16),
        0.20, 0.62, 0.10, 0.58,
        ("Protect hooks and lead vocals; prefer shorter, clearly structured handoffs.",),
    ),
    "trance_progressive": StyleProfile(
        "trance_progressive",
        ("eq_bass_swap", "filter_build", "equal_power_blend", "stem_choreography"),
        ("quick_cut", "backspin"),
        (16, 32, 8, 4),
        1.00, 0.12, 0.32, 0.90,
        ("Preserve long harmonic and energy development across phrase-aligned blends.",),
    ),
    "afro_organic": StyleProfile(
        "afro_organic",
        ("eq_bass_swap", "equal_power_blend", "stem_choreography", "layer_instrumental_handoff"),
        ("backspin",),
        (16, 8, 32, 4),
        0.82, 0.22, 0.28, 0.92,
        ("Protect percussion and groove continuity while keeping bass ownership clean.",),
    ),
    "general": StyleProfile(
        "general",
        ("eq_bass_swap", "filter_build", "stem_choreography", "equal_power_blend"),
        (),
        (8, 16, 4, 2),
        0.50, 0.45, 0.30, 0.75,
        ("Use a neutral professional policy when style evidence is weak or mixed.",),
    ),
}

_HOUSE = {"house", "deep_house", "tech_house", "progressive_house", "afro_house"}
_TECHNO = {"techno", "melodic_techno", "minimal_techno"}
_DNB = {"drum_and_bass", "dnb", "jungle"}
_HIPHOP = {"hip_hop", "hiphop", "rap", "rnb", "open_format"}
_POP = {"pop", "dance_pop"}
_TRANCE = {"trance", "progressive_trance", "uplifting_trance", "progressive"}
_AFRO = {"afro_house", "afrobeat", "afrobeats", "organic_house"}


def _family(label: str) -> str:
    value = str(label or "").strip().lower()
    if value in _DNB:
        return "drum_and_bass"
    if value in _HIPHOP:
        return "hip_hop_open_format"
    if value in _POP:
        return "pop"
    if value in _TRANCE:
        return "trance_progressive"
    if value in _TECHNO:
        return "techno"
    if value in _AFRO:
        return "afro_organic"
    if value in _HOUSE:
        return "house"
    return "general"


def style_profile(track_a: TrackAnalysis, track_b: TrackAnalysis) -> StyleProfile:
    left = _family(track_a.genre.label)
    right = _family(track_b.genre.label)
    if left == right:
        return _PROFILES[left]
    if {left, right} <= {"hip_hop_open_format", "pop"}:
        return _PROFILES["hip_hop_open_format"]
    if {left, right} <= {"house", "afro_organic"}:
        return _PROFILES["afro_organic"]
    if {left, right} <= {"house", "techno", "trance_progressive"}:
        # Mixed electronic pairs retain blend culture but use neutral limits.
        return _PROFILES["general"]
    return _PROFILES["general"]


def style_plan_adjustment(
    plan: TransitionPlan,
    profile: StyleProfile,
) -> tuple[float, list[str]]:
    """Return a bounded style preference. Never changes feasibility."""
    adjustment = 0.0
    reasons: list[str] = []
    technique = str(plan.technique)

    if technique in profile.preferred_techniques:
        rank = profile.preferred_techniques.index(technique)
        adjustment += max(0.015, 0.055 - 0.010 * rank)
        reasons.append(f"{technique} is idiomatic for {profile.name}.")
    if technique in profile.discouraged_techniques:
        adjustment -= 0.050
        reasons.append(f"{technique} is less idiomatic for {profile.name}.")

    if plan.transition_bars in profile.preferred_bars:
        rank = profile.preferred_bars.index(plan.transition_bars)
        adjustment += max(0.008, 0.032 - 0.006 * rank)

    if plan.transition_bars >= 16:
        adjustment += 0.035 * (2.0 * profile.long_blend_bias - 1.0)
    elif plan.transition_bars <= 2:
        adjustment += 0.035 * (2.0 * profile.reset_tolerance - 1.0)

    reset = technique in {"quick_cut", "backspin", "echo_out_drop_in"} or plan.transition_mode == "emergency_cut"
    if reset:
        adjustment += 0.045 * (2.0 * profile.reset_tolerance - 1.0)

    vocal_fit = float(plan.compatibility.vocal_overlap)
    if plan.transition_bars >= 8 and vocal_fit < profile.vocal_overlap_tolerance:
        adjustment -= 0.055
        reasons.append("Style profile discourages this much lead-vocal coexistence.")

    bass_fit = float(plan.compatibility.bass_conflict)
    if plan.transition_bars >= 8 and bass_fit < 0.50:
        if technique in {"eq_bass_swap", "stem_choreography", "layer_instrumental_handoff"}:
            adjustment += 0.025 * profile.bass_handoff_priority
        else:
            adjustment -= 0.035 * profile.bass_handoff_priority

    return max(-0.10, min(0.10, adjustment)), reasons
