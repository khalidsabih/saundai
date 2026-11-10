from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import TransitionPlan

PROFESSIONAL_MIX_CONTRACT_VERSION = "1.0.0"

RESET_TECHNIQUES = {"echo_out_drop_in", "quick_cut", "backspin"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _curve_value(points: list[dict[str, Any]], position: float) -> float:
    if not points:
        return 0.0
    ordered = sorted(points, key=lambda point: float(point.get("position", 0.0)))
    x = _clamp(position)
    if x <= float(ordered[0].get("position", 0.0)):
        return float(ordered[0].get("value", 0.0))
    for left, right in zip(ordered, ordered[1:]):
        lx = float(left.get("position", 0.0))
        rx = float(right.get("position", 1.0))
        if lx <= x <= rx:
            if rx <= lx:
                return float(right.get("value", 0.0))
            t = (x - lx) / (rx - lx)
            return float(left.get("value", 0.0)) + t * (
                float(right.get("value", 0.0)) - float(left.get("value", 0.0))
            )
    return float(ordered[-1].get("value", 0.0))


def _master_activity(program: dict[str, Any], deck: str, position: float) -> float:
    curves = dict(program.get("curves", {}) or {})
    master = curves.get(f"deck_{deck}_gain", [])
    if master:
        value = _curve_value(list(master), position)
    else:
        value = 1.0
    low = curves.get(f"deck_{deck}_low_gain", [])
    high = curves.get(f"deck_{deck}_high_gain", [])
    if low or high:
        low_value = _curve_value(list(low), position) if low else 1.0
        high_value = _curve_value(list(high), position) if high else 1.0
        value *= 0.45 * low_value + 0.55 * high_value
    return _clamp(value)


def _role_activity(program: dict[str, Any], deck: str, role: str, position: float) -> float:
    source_mode = str(program.get("source_mode", "master"))
    if source_mode == "stems":
        group = dict(program.get(f"stem_curves_{deck}", {}) or {})
        return _clamp(_curve_value(list(group.get(role, []) or []), position))
    if source_mode == "layers":
        group = dict(program.get(f"layer_curves_{deck}", {}) or {})
        if role == "vocals":
            return _clamp(_curve_value(list(group.get("vocals", []) or []), position))
        if role in {"bass", "drums", "melody"}:
            instrumental = list(group.get("instrumental", []) or [])
            master = list(group.get("master", []) or [])
            return _clamp(max(_curve_value(instrumental, position), _curve_value(master, position)))
    return _master_activity(program, deck, position)


def _fraction(predicate) -> float:
    positions = [index / 40.0 for index in range(41)]
    return sum(1 for position in positions if predicate(position)) / len(positions)


def _first_position(predicate) -> float | None:
    for index in range(101):
        position = index / 100.0
        if predicate(position):
            return position
    return None


def _last_position(predicate) -> float | None:
    for index in range(100, -1, -1):
        position = index / 100.0
        if predicate(position):
            return position
    return None


@dataclass(slots=True)
class ProfessionalMixContract:
    version: str
    applicable: bool
    passed: bool
    quality: float
    family: str
    source_mode: str
    continuous_mix_required: bool
    structural_reset: bool
    reset_justified: bool
    incoming_alignment: str
    dual_deck_overlap_fraction: float = 0.0
    incoming_established_position: float | None = None
    outgoing_release_position: float | None = None
    dual_bass_fraction: float = 0.0
    dual_vocal_fraction: float = 0.0
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_professional_mix_contract(plan: TransitionPlan) -> ProfessionalMixContract:
    policy = dict(plan.metadata.get("professional_mix", {}) or {})
    program = dict(plan.performance_program or {})
    if not policy or not program:
        return ProfessionalMixContract(
            version=PROFESSIONAL_MIX_CONTRACT_VERSION,
            applicable=False,
            passed=True,
            quality=0.68,
            family="legacy_unannotated",
            source_mode=str(program.get("source_mode", "unknown")),
            continuous_mix_required=False,
            structural_reset=bool(plan.technique in RESET_TECHNIQUES or plan.transition_mode == "emergency_cut"),
            reset_justified=False,
            incoming_alignment=str(dict(program.get("annotations", {}) or {}).get("incoming_alignment", "unknown")),
            warnings=["Legacy plan has no professional mixing contract annotation."],
        )

    family = str(policy.get("family", "unknown"))
    continuous_required = bool(policy.get("continuous_mix_required", False))
    structural_reset = bool(policy.get("structural_reset", False))
    reset_justified = bool(policy.get("reset_justified", False))
    source_mode = str(program.get("source_mode", "master"))
    annotations = dict(program.get("annotations", {}) or {})
    incoming_alignment = str(annotations.get("incoming_alignment", "full"))
    failures: list[str] = []
    warnings: list[str] = []

    positions = [index / 40.0 for index in range(41)]
    activity_a = [_master_activity(program, "a", position) for position in positions]
    activity_b = [_master_activity(program, "b", position) for position in positions]
    dual_overlap = sum(1 for a, b in zip(activity_a, activity_b) if a >= 0.20 and b >= 0.20) / len(positions)
    incoming_established = _first_position(lambda position: _master_activity(program, "b", position) >= 0.42)
    outgoing_release = _last_position(lambda position: _master_activity(program, "a", position) >= 0.42)

    dual_bass = _fraction(
        lambda position: _role_activity(program, "a", "bass", position) >= 0.62
        and _role_activity(program, "b", "bass", position) >= 0.62
    )
    dual_vocal = _fraction(
        lambda position: _role_activity(program, "a", "vocals", position) >= 0.58
        and _role_activity(program, "b", "vocals", position) >= 0.58
    )

    if structural_reset and not reset_justified:
        failures.append("Reset-style transition is not explicitly justified by the compatibility-driven mix policy.")

    if continuous_required:
        if plan.transition_mode == "emergency_cut":
            failures.append("A continuous professional mix cannot execute through emergency-cut mode.")
        if not plan.beatmatched:
            failures.append("A continuous professional mix must keep a coherent rhythmic relationship.")
        if incoming_alignment == "tail":
            failures.append("Incoming audio starts late inside the transition; zero-filled runway is forbidden for continuous mixing.")
        rendered_b = plan.source_duration_b / max(plan.render_rate_b, 1e-9)
        if incoming_alignment != "loop_extend" and rendered_b + 1e-5 < 0.90 * plan.transition_seconds:
            failures.append("Incoming source material is too short to establish the next deck across the committed transition.")
        if dual_overlap < 0.22:
            failures.append("The two decks do not coexist long enough to count as a continuous DJ handoff.")
        if incoming_established is None:
            failures.append("The incoming deck never becomes musically established during the transition.")
        if outgoing_release is None:
            failures.append("The outgoing deck has no stable ownership window before release.")
        if incoming_established is not None and outgoing_release is not None and incoming_established > outgoing_release + 0.08:
            failures.append("The outgoing deck surrenders ownership before the incoming deck is established.")

        if source_mode in {"stems", "layers"}:
            if dual_bass > 0.22:
                failures.append("Both bass layers own the low end for too much of the transition.")
            elif dual_bass > 0.12:
                warnings.append("Bass ownership overlaps longer than the preferred professional window.")
            if dual_vocal > 0.20:
                failures.append("Competing lead-vocal ownership overlaps for too much of the transition.")
            elif dual_vocal > 0.10:
                warnings.append("Lead-vocal ownership briefly overlaps; rendered rehearsal must confirm it is intentional.")
        else:
            vocal_conflict = _clamp(float(policy.get("vocal_conflict", 0.0) or 0.0))
            bass_conflict = _clamp(float(policy.get("bass_conflict", 0.0) or 0.0))
            melodic_conflict = _clamp(float(policy.get("melodic_conflict", 0.0) or 0.0))
            low_ownership = any(
                key in dict(program.get("curves", {}) or {})
                for key in ("deck_a_low_gain", "deck_b_low_gain")
            )
            long_master_overlap = plan.transition_bars >= 8 and not structural_reset
            if long_master_overlap and plan.compatibility.vocal_overlap < 0.24:
                failures.append(
                    "Long full-master overlap has severe lead-vocal collision risk; use separated sources, a shorter handoff, or replan."
                )
            elif plan.compatibility.vocal_overlap < 0.48:
                warnings.append("Master-only transition has elevated vocal collision risk; prefer stems/layers when available.")

            if long_master_overlap and plan.compatibility.bass_conflict < 0.24 and not low_ownership:
                failures.append(
                    "Long full-master overlap has severe simultaneous bass ownership without explicit low-frequency handoff."
                )
            elif plan.compatibility.bass_conflict < 0.48 and not low_ownership:
                warnings.append("Master-only transition has elevated bass collision risk without explicit low-frequency ownership automation.")

            harmonic_fit = max(
                float(plan.compatibility.harmonic),
                float(plan.compatibility.local_harmonic),
            )
            if long_master_overlap and harmonic_fit < 0.20:
                failures.append(
                    "Long full-master overlap has severe harmonic incompatibility for a transparent blend."
                )

            if (
                long_master_overlap
                and plan.compatibility.phrase < 0.22
                and plan.compatibility.boundary < 0.30
            ):
                failures.append(
                    "Long continuous overlap is not anchored to a credible phrase or structural boundary."
                )

            if family == "structural_bridge":
                if max(vocal_conflict, melodic_conflict) >= 0.68:
                    failures.append(
                        "This structural bridge has too much lead-vocal/melodic conflict for a full-master overlap; use separated sources or replan the pair."
                    )
                if bass_conflict >= 0.62 and not low_ownership:
                    failures.append(
                        "This structural bridge needs explicit low-frequency ownership before a full-master overlap can be accepted."
                    )

    score = 1.0
    score -= min(0.32, 0.32 * max(0.0, 0.34 - dual_overlap) / 0.34)
    score -= min(0.22, 0.55 * dual_bass)
    score -= min(0.22, 0.55 * dual_vocal)
    if incoming_alignment == "loop_extend":
        score -= 0.03
    if structural_reset:
        score = min(score, 0.78 if reset_justified else 0.35)
    score -= 0.18 * len(failures)
    score -= 0.025 * len(warnings)
    score = _clamp(score)

    return ProfessionalMixContract(
        version=PROFESSIONAL_MIX_CONTRACT_VERSION,
        applicable=True,
        passed=not failures,
        quality=round(score, 4),
        family=family,
        source_mode=source_mode,
        continuous_mix_required=continuous_required,
        structural_reset=structural_reset,
        reset_justified=reset_justified,
        incoming_alignment=incoming_alignment,
        dual_deck_overlap_fraction=round(dual_overlap, 4),
        incoming_established_position=(None if incoming_established is None else round(incoming_established, 4)),
        outgoing_release_position=(None if outgoing_release is None else round(outgoing_release, 4)),
        dual_bass_fraction=round(dual_bass, 4),
        dual_vocal_fraction=round(dual_vocal, 4),
        hard_failures=failures,
        warnings=warnings,
    )


def professional_mix_contract_status() -> dict[str, Any]:
    return {
        "version": PROFESSIONAL_MIX_CONTRACT_VERSION,
        "continuous_mix_requires": [
            "rhythmic_relationship",
            "no_zero_filled_incoming_runway",
            "meaningful_dual_deck_overlap",
            "incoming_established_before_outgoing_release",
            "single_bass_owner_when_separated_sources_exist",
            "controlled_lead_vocal_ownership_when_separated_sources_exist",
            "no_severe_master_vocal_or_bass_collision_in_long_overlaps",
            "credible_phrase_or_boundary_alignment_for_long_overlaps",
            "no_severe_harmonic_conflict_in_long_transparent_blends",
        ],
        "structural_resets_require_explicit_justification": True,
        "combined_audio_quality_alone_is_insufficient": True,
    }
