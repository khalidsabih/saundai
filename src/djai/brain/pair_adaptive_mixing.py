from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..decision import seal_execution_plan
from ..models import TrackAnalysis, TransitionPlan
from ..performance.program import attach_performance_program

PAIR_ADAPTIVE_MIXING_VERSION = "1.0"

VARIANT_NAMES = ("conservative", "balanced", "assertive")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _local_vocals(plan: TransitionPlan, track_a: TrackAnalysis, track_b: TrackAnalysis) -> tuple[float, float]:
    outgoing = float(plan.metadata.get("exit_vocal_probability", track_a.vocal_probability) or 0.0)
    incoming = float(plan.metadata.get("entry_vocal_probability", track_b.vocal_probability) or 0.0)
    return _clamp(outgoing), _clamp(incoming)


@dataclass(frozen=True, slots=True)
class PairAutomation:
    version: str
    variant: str
    bass_out_start: float
    bass_out_end: float
    bass_in_start: float
    bass_in_end: float
    vocal_out_start: float
    vocal_out_end: float
    vocal_in_start: float
    vocal_in_end: float
    drums_in_start: float
    drums_in_end: float
    bed_out_start: float
    bed_out_end: float
    bed_in_start: float
    bed_in_end: float
    low_overlap_gap: float
    vocal_overlap_gap: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def design_pair_automation(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    variant: str = "balanced",
) -> PairAutomation:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown pair-adaptive variant: {variant}")

    professional = dict(plan.metadata.get("professional_mix", {}) or {})
    vocal_conflict = _clamp(float(professional.get("vocal_conflict", 1.0 - plan.compatibility.vocal_overlap) or 0.0))
    bass_conflict = _clamp(float(professional.get("bass_conflict", 1.0 - plan.compatibility.bass_conflict) or 0.0))
    melodic_conflict = _clamp(float(professional.get("melodic_conflict", 1.0 - plan.compatibility.harmonic) or 0.0))
    rhythm = _clamp(float(professional.get("rhythm_compatibility", 0.58 * plan.compatibility.beat_grid + 0.42 * plan.compatibility.phrase) or 0.0))
    phrase = _clamp(float(professional.get("phrase_compatibility", plan.compatibility.phrase) or 0.0))
    outgoing_vocal, incoming_vocal = _local_vocals(plan, track_a, track_b)

    # Variant pressure changes how early B is introduced and how much safety
    # margin is left around ownership transfers. The pair evidence remains the
    # dominant factor.
    pressure = {"conservative": -0.08, "balanced": 0.0, "assertive": 0.07}[variant]

    bass_center = _clamp(
        0.54
        + 0.10 * bass_conflict
        + 0.05 * melodic_conflict
        - 0.05 * rhythm
        + pressure,
        0.38,
        0.72,
    )
    bass_gap = _clamp(0.035 + 0.16 * bass_conflict + 0.04 * melodic_conflict - 0.025 * rhythm, 0.025, 0.20)
    bass_fade = _clamp(0.12 + 0.04 * (1.0 - rhythm), 0.08, 0.18)
    bass_out_end = _clamp(bass_center - bass_gap / 2.0, 0.25, 0.78)
    bass_out_start = _clamp(bass_out_end - bass_fade, 0.12, bass_out_end - 0.03)
    bass_in_start = _clamp(bass_center + bass_gap / 2.0, bass_out_end + 0.02, 0.86)
    bass_in_end = _clamp(bass_in_start + bass_fade, bass_in_start + 0.04, 0.96)

    vocal_risk = _clamp(0.55 * vocal_conflict + 0.25 * min(outgoing_vocal, incoming_vocal) + 0.20 * melodic_conflict)
    vocal_center = _clamp(0.60 + 0.12 * vocal_risk + 0.04 * melodic_conflict + pressure, 0.42, 0.82)
    vocal_gap = _clamp(0.08 + 0.24 * vocal_risk, 0.06, 0.30)
    vocal_fade = _clamp(0.15 + 0.05 * vocal_risk, 0.10, 0.23)
    vocal_out_end = _clamp(vocal_center - vocal_gap / 2.0, 0.25, 0.82)
    vocal_out_start = _clamp(vocal_out_end - vocal_fade, 0.10, vocal_out_end - 0.04)
    vocal_in_start = _clamp(vocal_center + vocal_gap / 2.0, vocal_out_end + 0.03, 0.92)
    vocal_in_end = _clamp(vocal_in_start + vocal_fade, vocal_in_start + 0.04, 0.995)

    drums_in_start = _clamp(0.03 + 0.16 * (1.0 - rhythm) + 0.05 * (1.0 - phrase) + pressure * 0.45, 0.02, 0.30)
    drums_in_end = _clamp(drums_in_start + 0.28 + 0.14 * (1.0 - rhythm), drums_in_start + 0.16, 0.62)

    bed_conflict = _clamp(0.65 * melodic_conflict + 0.35 * vocal_risk)
    bed_center = _clamp(0.52 + 0.12 * bed_conflict + pressure * 0.65, 0.38, 0.72)
    bed_overlap = _clamp(0.22 - 0.13 * bed_conflict + 0.07 * phrase, 0.07, 0.30)
    bed_out_end = _clamp(bed_center + bed_overlap / 2.0, 0.48, 0.90)
    bed_out_start = _clamp(bed_out_end - 0.34, 0.10, bed_out_end - 0.08)
    bed_in_start = _clamp(bed_center - bed_overlap / 2.0, 0.05, 0.75)
    bed_in_end = _clamp(bed_in_start + 0.36, bed_in_start + 0.10, 0.94)

    reasons = [
        f"bass conflict {bass_conflict:.0%} sets low-frequency ownership gap",
        f"vocal conflict {vocal_conflict:.0%} sets lead-vocal separation",
        f"rhythm compatibility {rhythm:.0%} sets incoming drum reveal",
        f"melodic conflict {melodic_conflict:.0%} sets musical-bed overlap",
    ]
    if variant == "conservative":
        reasons.append("conservative variant gives outgoing material more time before incoming ownership.")
    elif variant == "assertive":
        reasons.append("assertive variant establishes the incoming record earlier while preserving ownership safety.")
    else:
        reasons.append("balanced variant follows the pair-derived ownership centers.")

    return PairAutomation(
        version=PAIR_ADAPTIVE_MIXING_VERSION,
        variant=variant,
        bass_out_start=round(bass_out_start, 6),
        bass_out_end=round(bass_out_end, 6),
        bass_in_start=round(bass_in_start, 6),
        bass_in_end=round(bass_in_end, 6),
        vocal_out_start=round(vocal_out_start, 6),
        vocal_out_end=round(vocal_out_end, 6),
        vocal_in_start=round(vocal_in_start, 6),
        vocal_in_end=round(vocal_in_end, 6),
        drums_in_start=round(drums_in_start, 6),
        drums_in_end=round(drums_in_end, 6),
        bed_out_start=round(bed_out_start, 6),
        bed_out_end=round(bed_out_end, 6),
        bed_in_start=round(bed_in_start, 6),
        bed_in_end=round(bed_in_end, 6),
        low_overlap_gap=round(max(0.0, bass_in_start - bass_out_end), 6),
        vocal_overlap_gap=round(max(0.0, vocal_in_start - vocal_out_end), 6),
        reasons=tuple(reasons),
    )


def pair_adaptive_variants(
    track_a: TrackAnalysis,
    track_b: TrackAnalysis,
    plan: TransitionPlan,
    *,
    maximum_variants: int = 3,
) -> list[TransitionPlan]:
    """Compile a tiny set of pair-specific automation variants for rehearsal.

    This runs only at the commit/rehearsal boundary. It does not expand the
    analytical transition graph.
    """
    professional = dict(plan.metadata.get("professional_mix", {}) or {})
    family = str(professional.get("family") or "legacy")
    structural_reset = bool(professional.get("structural_reset", False))
    if structural_reset or plan.transition_mode == "emergency_cut":
        return [copy.deepcopy(plan)]

    count = max(1, min(len(VARIANT_NAMES), int(maximum_variants)))
    names = VARIANT_NAMES[:count]
    output: list[TransitionPlan] = []
    for variant in names:
        candidate = copy.deepcopy(plan)
        automation = design_pair_automation(track_a, track_b, candidate, variant=variant)
        candidate.metadata = {
            **candidate.metadata,
            "pair_adaptive_mix": {
                **automation.to_dict(),
                "family": family,
                "generated_at_commit_boundary": True,
                "graph_search_expanded": False,
                "renderer_may_not_reinterpret": True,
            },
        }
        candidate.alternative_id = (
            f"{candidate.alternative_id or candidate.technique}-pair-{variant}"
        )
        candidate.rationale = [*candidate.rationale, *automation.reasons]
        attach_performance_program(candidate)
        seal_execution_plan(candidate)
        output.append(candidate)
    return output


def pair_adaptive_status() -> dict[str, Any]:
    return {
        "version": PAIR_ADAPTIVE_MIXING_VERSION,
        "commit_boundary_only": True,
        "graph_search_expanded": False,
        "variants": list(VARIANT_NAMES),
        "controls": [
            "bass_ownership_timing",
            "vocal_ownership_timing",
            "incoming_drums",
            "musical_bed_overlap",
            "master_low_eq_handoff",
        ],
        "hard_safety_authoritative": True,
        "renderer_contract": "immutable_program_ir",
    }
