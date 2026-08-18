from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import TransitionPlan


def _record_id_values(execution_signature: str, *, accepted: bool, rank: int, measured_score: float) -> str:
    payload = f"{execution_signature}|{accepted}|{rank}|{float(measured_score):.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _record_id(plan: TransitionPlan, *, accepted: bool, rank: int) -> str:
    return _record_id_values(
        plan.execution_signature,
        accepted=accepted,
        rank=rank,
        measured_score=float(plan.measured_score or 0.0),
    )


def measurement_record(
    plan: TransitionPlan,
    *,
    accepted: bool,
    rank: int,
    candidate_count: int,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measurement = plan.measurement
    return {
        "schema": "djai-transition-measurement-v1",
        "record_id": _record_id(plan, accepted=accepted, rank=rank),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "rank": rank,
        "candidate_count": candidate_count,
        "track_a": plan.track_a,
        "track_b": plan.track_b,
        "execution_signature": plan.execution_signature,
        "style": plan.style,
        "intent": plan.intent,
        "technique": plan.technique,
        "transition_bars": plan.transition_bars,
        "genre_pair": list(plan.genre_pair),
        "fallback_level": plan.fallback_level,
        "auto_approved": plan.auto_approved,
        "exit_time_a": plan.exit_time_a,
        "entry_time_b": plan.entry_time_b,
        "analytical_score": plan.analytical_score,
        "measured_score": plan.measured_score,
        "final_score": plan.score,
        "naturalness_score": plan.metadata.get("transition_naturalness_score"),
        "naturalness": plan.metadata.get("transition_naturalness"),
        "measurement_passed": plan.measurement_passed,
        "compatibility": {
            "tempo": plan.compatibility.tempo,
            "harmonic": plan.compatibility.harmonic,
            "energy": plan.compatibility.energy,
            "beat_grid": plan.compatibility.beat_grid,
            "phrase": plan.compatibility.phrase,
            "sections": plan.compatibility.sections,
            "vocal_safety": plan.compatibility.vocal_overlap,
            "bass_safety": plan.compatibility.bass_conflict,
        },
        "measurement": measurement.to_dict() if measurement is not None else None,
        "context": dict(context or {}),
    }


def append_measurement_records(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def measurement_log_offset(path: str | Path | None) -> int:
    """Return the byte offset where a new planning run will begin appending."""
    if not path:
        return 0
    destination = Path(path).expanduser().resolve()
    try:
        return int(destination.stat().st_size)
    except OSError:
        return 0


def finalize_measurement_run(
    path: str | Path | None,
    *,
    start_offset: int,
    accepted_signatures: Iterable[str],
    accepted_plans: Iterable[TransitionPlan] | None = None,
) -> None:
    """Make telemetry acceptance match the *final* set after global replanning.

    A transition can be locally accepted during an early route attempt and then
    disappear because a later junction causes the Set Brain to re-order the
    crate.  Those rehearsals remain useful training evidence, but only execution
    signatures that survive into the final SetPlan may retain ``accepted=true``.
    Existing records from earlier runs are left byte-for-byte untouched.
    """
    if not path:
        return
    destination = Path(path).expanduser().resolve()
    try:
        raw = destination.read_bytes()
    except OSError:
        return
    offset=max(0,min(int(start_offset),len(raw)))
    prefix=raw[:offset]
    suffix=raw[offset:]
    accepted_plan_map = {
        plan.execution_signature: plan
        for plan in (accepted_plans or ())
        if plan.execution_signature and plan.measurement is not None
    }
    accepted_set={str(value) for value in accepted_signatures} | set(accepted_plan_map)
    seen_signatures: set[str] = set()
    rewritten: list[bytes] = []
    changed=False
    for raw_line in suffix.splitlines(keepends=False):
        if not raw_line.strip():
            continue
        try:
            record=json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError,json.JSONDecodeError):
            rewritten.append(raw_line + b"\n")
            continue
        if not isinstance(record,dict) or record.get("schema") != "djai-transition-measurement-v1":
            rewritten.append(raw_line + b"\n")
            continue
        signature=str(record.get("execution_signature", ""))
        if signature:
            seen_signatures.add(signature)
        final_accept=bool(signature and signature in accepted_set)
        prior_accept=bool(record.get("accepted"))
        if prior_accept != final_accept:
            changed=True
        record["accepted"]=final_accept
        try:
            rank=int(record.get("rank",0) or 0)
            measured=float(record.get("measured_score",0.0) or 0.0)
            record["record_id"]=_record_id_values(signature,accepted=final_accept,rank=rank,measured_score=measured)
        except (TypeError,ValueError):
            pass
        context=record.get("context") if isinstance(record.get("context"),dict) else {}
        context={**context,"set_plan_final_acceptance":final_accept}
        if prior_accept and not final_accept:
            context["superseded_by_set_replan"]=True
        record["context"]=context
        rewritten.append((json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8"))

    # A final automatic-continuity replacement can be rendered and measured
    # *after* the ordinary rehearsal logger ran (for example, a hard-cut
    # candidate replaced by the deterministic filtered-overlap contract).
    # Materialize that final measured plan into the telemetry log so the log
    # describes the set that was actually delivered, not merely the candidates
    # that existed before the final product-policy replacement.
    for signature, plan in accepted_plan_map.items():
        if signature in seen_signatures:
            continue
        record = measurement_record(
            plan,
            accepted=True,
            rank=1,
            candidate_count=1,
            context={
                "mode": "set_plan_final_materialization",
                "set_plan_final_acceptance": True,
                "materialized_final_plan_record": True,
            },
        )
        rewritten.append((json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8"))
        changed = True

    if not suffix and not rewritten:
        return
    if suffix and not changed and all(b"set_plan_final_acceptance" in line for line in rewritten):
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+".finalize.tmp")
    try:
        temporary.write_bytes(prefix+b"".join(rewritten))
        os.replace(temporary,destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
