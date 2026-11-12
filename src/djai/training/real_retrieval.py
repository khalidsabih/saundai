from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ..brain.set_state import build_set_state
from ..intelligence.neural_policy import transition_feature_dict
from ..serialization import track_analysis_from_dict, transition_plan_from_dict
from .pairwise_ranker import train_mlp_pairwise_next_track_ranker

REAL_RETRIEVAL_DATA_VERSION = "1.1"


def _split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, Counter[str]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    splits: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            splits[str(payload.get("split") or "unknown")] += 1
    return count, splits


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def import_aotm_sequences(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Normalize AotM ordered playlists/bigrams without inventing audio features.

    Supports the public AOTM-2011 playlist JSON(.gz) and the ordered-bigram
    CSV(.gz) form. The output is human-sequencing supervision only; it is not
    granted runtime authority until IDs are resolved to a feature catalog.
    """
    source_path = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise ValueError(f"AotM source not found: {source_path}")

    rows: list[dict[str, Any]] = []
    if source_path.name.endswith(".csv") or source_path.name.endswith(".csv.gz"):
        with _open_text(source_path) as handle:
            reader = csv.reader(handle)
            for index, values in enumerate(reader):
                if len(values) < 2:
                    continue
                left, right = str(values[0]).strip(), str(values[1]).strip()
                if not left or not right or left == right:
                    continue
                group = f"aotm-bigram:{index}"
                rows.append({
                    "schema_version": REAL_RETRIEVAL_DATA_VERSION,
                    "source": "aotm_ordered_bigrams",
                    "group": group,
                    "split": _split(group),
                    "source_track_id": left,
                    "target_track_id": right,
                    "category": None,
                    "position": None,
                    "sequence_length": 2,
                    "human_ordered_positive": True,
                    "weight": 1.0,
                })
    else:
        with _open_text(source_path) as handle:
            playlists = json.load(handle)
        if not isinstance(playlists, list):
            raise ValueError("AotM playlist JSON must contain a list.")
        for playlist_index, playlist in enumerate(playlists):
            if not isinstance(playlist, Mapping):
                continue
            mix_id = str(playlist.get("mix_id") or playlist_index)
            category = str(playlist.get("category") or "") or None
            segments = playlist.get("filtered_lists")
            if not isinstance(segments, list):
                continue
            for segment_index, segment in enumerate(segments):
                if not isinstance(segment, list):
                    continue
                ids = [str(item).strip() for item in segment if str(item).strip()]
                for position, (left, right) in enumerate(zip(ids, ids[1:])):
                    if left == right:
                        continue
                    group = f"aotm:{mix_id}:{segment_index}:{position}"
                    rows.append({
                        "schema_version": REAL_RETRIEVAL_DATA_VERSION,
                        "source": "aotm2011",
                        "group": group,
                        "split": _split(f"aotm-playlist:{mix_id}"),
                        "source_track_id": left,
                        "target_track_id": right,
                        "category": category,
                        "position": position,
                        "sequence_length": len(ids),
                        "human_ordered_positive": True,
                        "weight": 1.0,
                    })

    count, splits = _write_jsonl(destination, rows)
    return {
        "schema_version": REAL_RETRIEVAL_DATA_VERSION,
        "source": str(source_path),
        "output": str(destination),
        "rows": count,
        "splits": dict(sorted(splits.items())),
        "policy": {
            "human_sequence_signal": True,
            "technical_safety_signal": False,
            "runtime_feature_resolved": False,
            "production_authority": False,
        },
    }


def _load_historical_set_plan_evidence(path: str | Path) -> Any:
    """Load immutable planning evidence without execution-version validation.

    Historical Phase16 plans are evidence, not executable programs. Normal
    renderer/playback loading remains strict through set_planner.load_set_plan.
    """
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    tracks_payload = payload.get("tracks")
    transitions_payload = payload.get("transitions")
    if not isinstance(tracks_payload, list) or not isinstance(transitions_payload, list):
        raise ValueError(f"Historical set plan is missing tracks/transitions: {source}")

    tracks = []
    for item in tracks_payload:
        if not isinstance(item, Mapping) or not isinstance(item.get("analysis"), Mapping):
            raise ValueError(f"Historical set plan lacks embedded track analysis: {source}")
        tracks.append(track_analysis_from_dict(dict(item["analysis"])))

    transitions = []
    for item in transitions_payload:
        if not isinstance(item, Mapping) or not isinstance(item.get("plan"), Mapping):
            continue
        transition = transition_plan_from_dict(dict(item["plan"]))
        transitions.append(
            SimpleNamespace(
                from_index=int(item.get("from_index", -1)),
                to_index=int(item.get("to_index", -1)),
                plan=transition,
            )
        )

    order = [int(value) for value in payload.get("order", [])]
    if not order or len(order) != len(tracks):
        raise ValueError(f"Historical set plan has invalid order: {source}")
    return SimpleNamespace(
        plan_id=str(payload.get("plan_id") or source.stem),
        tracks=tracks,
        order=order,
        transitions=transitions,
        metadata=dict(payload.get("metadata") or {}),
    )


def _accepted_target(plan: Any) -> tuple[float, str]:
    measured = float(getattr(plan, "measured_score", None) or getattr(plan, "score", 0.5) or 0.5)
    target = max(0.0, min(1.0, measured))
    fallback_level = int(getattr(plan, "fallback_level", 0) or 0)
    metadata = dict(getattr(plan, "metadata", {}) or {})
    reason = str(metadata.get("fallback_reason") or metadata.get("reason") or "").lower()
    rescue_kind = str(metadata.get("rescue_kind") or "").lower()
    true_emergency = (
        "all alternate track routes" in reason
        or "no evidence matrix remained" in reason
        or "final escape" in reason
    )
    if true_emergency:
        return min(target, 0.20), "accepted_emergency_escape"
    if fallback_level > 0:
        if rescue_kind == "structural_reset":
            return min(target, 0.55), "accepted_structural_reset"
        return min(target, 0.40), "accepted_fallback"
    return target, "accepted_professional"


def _crate_fingerprint(plan: Any) -> str:
    route_meta = dict((plan.metadata or {}).get("complete_move_planner", {}) or {})
    training = route_meta.get("training_evidence")
    if isinstance(training, Mapping) and str(training.get("crate_fingerprint") or ""):
        return str(training["crate_fingerprint"])
    payload = sorted(
        (
            {
                "path": str(track.path),
                "bpm": round(float(track.bpm), 4),
                "duration": round(float(track.duration_seconds), 3),
            }
            for track in plan.tracks
        ),
        key=lambda item: (item["path"], item["bpm"], item["duration"]),
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _compiled_edge_targets(plan: Any) -> dict[tuple[int, int], tuple[float, str]]:
    return {
        (int(edge.from_index), int(edge.to_index)): _accepted_target(edge.plan)
        for edge in plan.transitions
    }


def _state_from_training_payload(payload: Mapping[str, Any] | None) -> Any:
    state = payload if isinstance(payload, Mapping) else {}
    return SimpleNamespace(
        energy=float(state.get("energy", 0.5) or 0.5),
        tension=float(state.get("tension", 0.5) or 0.5),
        vocal_density=float(state.get("vocal_density", 0.5) or 0.5),
    )


def _dense_phase16_rows(plan: Any, training: Mapping[str, Any], journey: str) -> list[dict[str, Any]]:
    """Mine every serious route-search target preserved by the planner."""
    crate = _crate_fingerprint(plan)
    compiled_targets = _compiled_edge_targets(plan)
    final_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in training.get("final_adjacent_pairs", [])
        if isinstance(pair, list) and len(pair) == 2
    }
    blocked_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in training.get("blocked_pairs", [])
        if isinstance(pair, list) and len(pair) == 2
    }

    rows: list[dict[str, Any]] = []
    attempts = training.get("attempts", [])
    for attempt in attempts if isinstance(attempts, list) else []:
        if not isinstance(attempt, Mapping):
            continue
        attempt_index = int(attempt.get("attempt_index", 0) or 0)
        attempt_completed = bool(attempt.get("completed", False))
        attempt_order = [
            int(value) for value in attempt.get("order", [])
            if isinstance(value, (int, float))
        ]
        future_pairs = set(zip(attempt_order, attempt_order[1:]))
        decisions = attempt.get("decisions", [])
        for decision in decisions if isinstance(decisions, list) else []:
            if not isinstance(decision, Mapping):
                continue
            current = int(decision.get("current_index", -1))
            decision_index = int(decision.get("decision_index", 0) or 0)
            winner_index = int(decision.get("winner_index", -1))
            if not 0 <= current < len(plan.tracks):
                continue
            group = (
                f"phase16-dense:{crate}:{journey}:"
                f"{plan.plan_id}:{attempt_index}:{decision_index}"
            )
            state = _state_from_training_payload(
                decision.get("state") if isinstance(decision.get("state"), Mapping) else None
            )

            candidate_rows: list[dict[str, Any]] = []
            seen_targets: set[int] = set()
            candidates = decision.get("candidates", [])
            for candidate in candidates if isinstance(candidates, list) else []:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_index = int(candidate.get("next_index", -1))
                if (
                    candidate_index in seen_targets
                    or not 0 <= candidate_index < len(plan.tracks)
                    or candidate_index == current
                ):
                    continue
                seen_targets.add(candidate_index)
                pair = (current, candidate_index)
                blocked = bool(candidate.get("blocked_by_executive", False)) or pair in blocked_pairs
                final_adjacent = bool(candidate.get("survived_final_adjacency", False)) or pair in final_pairs
                won_attempt = bool(candidate.get("won_attempt", False)) or candidate_index == winner_index
                selection_score = max(0.0, min(1.0, float(candidate.get("selection_score", 0.5) or 0.5)))
                optionality = max(0.0, min(1.0, float(candidate.get("future_optionality", 0.5) or 0.5)))

                if blocked:
                    target, label, weight = 0.0, "professional_rejection", 1.0
                    future_survival = 0.0
                elif final_adjacent:
                    target, compiled_label = compiled_targets.get(pair, (0.85, "final_route"))
                    label, weight = compiled_label, 1.0
                    future_survival = 1.0
                elif won_attempt and not attempt_completed:
                    # The planner locally preferred this edge, but the route
                    # eventually stranded. This is the key new dead-end signal.
                    target, label, weight = min(0.22, 0.10 + 0.12 * optionality), "route_dead_end", 0.85
                    future_survival = 0.0
                elif won_attempt and attempt_completed:
                    target, label, weight = min(0.72, 0.48 + 0.20 * optionality), "viable_route_alternative", 0.55
                    future_survival = 0.75
                elif pair in future_pairs:
                    target, label, weight = min(0.62, 0.38 + 0.18 * optionality), "future_route_edge", 0.45
                    future_survival = 0.65
                else:
                    # Serious retrieved/generated alternative. Preserve the
                    # planner score as a weak prior, but keep it below proven
                    # final-route evidence.
                    target = min(0.48, 0.12 + 0.30 * selection_score + 0.06 * optionality)
                    label, weight = "serious_route_alternative", 0.30
                    future_survival = 0.35

                features = transition_feature_dict(
                    plan.tracks[current],
                    plan.tracks[candidate_index],
                    set_state=state,
                )
                candidate_rows.append({
                    "schema_version": REAL_RETRIEVAL_DATA_VERSION,
                    "id": f"{group}:{candidate_index}",
                    "group": group,
                    "split": "train",
                    "split_unit": crate,
                    "task": "next_track",
                    "source": "phase16_dense_route_evidence",
                    "journey": journey,
                    "features": features,
                    "target": round(float(target), 6),
                    "weight": round(float(weight), 6),
                    "label": label,
                    "current_index": current,
                    "candidate_index": candidate_index,
                    "attempt_index": attempt_index,
                    "attempt_completed": attempt_completed,
                    "won_attempt": won_attempt,
                    "blocked_by_executive": blocked,
                    "survived_final_adjacency": final_adjacent,
                    "future_survival": round(float(future_survival), 6),
                })

            if len(candidate_rows) >= 2:
                rows.extend(candidate_rows)
    return rows


def _legacy_phase16_rows(plan: Any, journey: str) -> list[dict[str, Any]]:
    """Backward-compatible sparse mining for plans created before evidence v1."""
    metadata = dict(plan.metadata or {})
    replan_meta = dict(metadata.get("unified_transition_intelligence", {}) or {})
    blocked_pairs = {
        (int(pair[0]), int(pair[1]))
        for pair in replan_meta.get("blocked_edges", [])
        if isinstance(pair, list) and len(pair) == 2
    }
    route_meta = dict(metadata.get("complete_move_planner", {}) or {})
    latest_route = route_meta.get("latest_route") if isinstance(route_meta.get("latest_route"), Mapping) else {}
    decisions = latest_route.get("decisions", []) if isinstance(latest_route, Mapping) else []
    crate = _crate_fingerprint(plan)

    candidate_map: dict[int, set[int]] = {}
    for decision in decisions if isinstance(decisions, list) else []:
        if not isinstance(decision, Mapping):
            continue
        current = int(decision.get("current_index", -1))
        for candidate in decision.get("candidates", []) if isinstance(decision.get("candidates"), list) else []:
            if isinstance(candidate, Mapping) and candidate.get("next_index") is not None:
                candidate_map.setdefault(current, set()).add(int(candidate["next_index"]))

    rows: list[dict[str, Any]] = []
    prefix_indices: list[int] = []
    prefix_plans: list[Any] = []
    for position, edge in enumerate(plan.transitions):
        current = int(edge.from_index)
        chosen = int(edge.to_index)
        if not prefix_indices:
            prefix_indices = list(plan.order[: position + 1])
        state = build_set_state([plan.tracks[index] for index in prefix_indices], prefix_plans)
        group = f"phase16-legacy:{crate}:{journey}:{plan.plan_id}:{position}"
        accepted_target, accepted_label = _accepted_target(edge.plan)
        rows.append({
            "schema_version": REAL_RETRIEVAL_DATA_VERSION,
            "id": f"{group}:accepted:{chosen}",
            "group": group,
            "split": "train",
            "split_unit": crate,
            "task": "next_track",
            "source": "phase16_real_rehearsal_legacy",
            "journey": journey,
            "features": transition_feature_dict(plan.tracks[current], plan.tracks[chosen], set_state=state),
            "target": round(accepted_target, 6),
            "weight": 1.0,
            "label": accepted_label,
            "current_index": current,
            "candidate_index": chosen,
        })
        alternatives = set(candidate_map.get(current, set()))
        alternatives.update(to_index for from_index, to_index in blocked_pairs if from_index == current)
        alternatives.discard(chosen)
        for candidate_index in sorted(alternatives):
            if not 0 <= candidate_index < len(plan.tracks):
                continue
            rejected = (current, candidate_index) in blocked_pairs
            rows.append({
                "schema_version": REAL_RETRIEVAL_DATA_VERSION,
                "id": f"{group}:{'rejected' if rejected else 'alternative'}:{candidate_index}",
                "group": group,
                "split": "train",
                "split_unit": crate,
                "task": "next_track",
                "source": "phase16_real_rehearsal_legacy",
                "journey": journey,
                "features": transition_feature_dict(plan.tracks[current], plan.tracks[candidate_index], set_state=state),
                "target": 0.0 if rejected else min(0.45, max(0.05, accepted_target - 0.35)),
                "weight": 1.0 if rejected else 0.30,
                "label": "professional_rejection" if rejected else "route_alternative",
                "current_index": current,
                "candidate_index": candidate_index,
            })
        prefix_indices = list(plan.order[: position + 2])
        prefix_plans.append(edge.plan)

    counts = Counter(str(row["group"]) for row in rows)
    return [row for row in rows if counts[str(row["group"])] >= 2]


def phase16_retrieval_rows(plan_path: str | Path) -> list[dict[str, Any]]:
    """Convert one real set plan into grouped next-track preference rows."""
    plan = _load_historical_set_plan_evidence(plan_path)
    metadata = dict(plan.metadata or {})
    journey = str(metadata.get("journey") or "balanced")
    route_meta = dict(metadata.get("complete_move_planner", {}) or {})
    training = route_meta.get("training_evidence")
    if isinstance(training, Mapping) and isinstance(training.get("attempts"), list):
        dense = _dense_phase16_rows(plan, training, journey)
        if dense:
            return dense
    return _legacy_phase16_rows(plan, journey)


def _assign_crate_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Prevent one music crate from leaking across train/validation/test."""
    crates = sorted({str(row.get("split_unit") or "") for row in rows if row.get("split_unit")})
    if not crates:
        return {}
    if len(crates) == 1:
        assignment = {crates[0]: "train"}
    elif len(crates) == 2:
        assignment = {crates[0]: "train", crates[1]: "validation"}
    else:
        ranked = sorted(
            crates,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
        )
        assignment = {crate: "train" for crate in ranked}
        assignment[ranked[-2]] = "validation"
        assignment[ranked[-1]] = "test"
    for row in rows:
        unit = str(row.get("split_unit") or "")
        if unit in assignment:
            row["split"] = assignment[unit]
    return assignment


def build_mixed_retrieval_snapshot(
    workspace: str | Path,
    *,
    synthetic_next_track: str | Path,
    phase16_plans: Iterable[str | Path],
    aotm_source: str | Path | None = None,
    name: str = "next-track-real-v1",
) -> dict[str, Any]:
    """Build a shadow-only retriever dataset from synthetic + real rehearsal data.

    AotM is normalized alongside the trainable rows but deliberately remains a
    separate human-sequence corpus until MSD/audio feature resolution is added.
    """
    root = Path(workspace).expanduser().resolve()
    synthetic = Path(synthetic_next_track).expanduser().resolve()
    if not synthetic.is_file():
        raise ValueError(f"Synthetic next-track dataset not found: {synthetic}")

    plan_paths = [Path(path).expanduser().resolve() for path in phase16_plans]
    phase_rows: list[dict[str, Any]] = []
    for plan_path in plan_paths:
        phase_rows.extend(phase16_retrieval_rows(plan_path))
    crate_splits = _assign_crate_splits(phase_rows)

    identity = hashlib.sha256(
        json.dumps({
            "version": REAL_RETRIEVAL_DATA_VERSION,
            "synthetic": str(synthetic),
            "plans": [str(path) for path in plan_paths],
            "aotm": str(aotm_source or ""),
            "name": name,
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    destination = root / "datasets" / "real-retrieval" / f"{name}-{identity}"
    destination.mkdir(parents=True, exist_ok=True)

    phase_path = destination / "phase16-next-track.jsonl"
    phase_count, phase_splits = _write_jsonl(phase_path, phase_rows)

    # Keep corpora side-by-side rather than concatenating incompatible
    # feature namespaces. Research-synthetic rows use state_*/candidate_* keys,
    # while Phase16 rows use the exact live transition_feature_dict contract.
    # A future explicit feature bridge may align them; until then, only the
    # runtime-aligned Phase16 rows are eligible to train the live retriever.
    synthetic_copy = destination / "synthetic-next-track-prior.jsonl"
    synthetic_count = 0
    with synthetic_copy.open("w", encoding="utf-8") as out:
        with synthetic.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    synthetic_count += 1

    runtime_path = destination / "next-track-runtime.jsonl"
    _write_jsonl(runtime_path, phase_rows)

    aotm_report = None
    if aotm_source is not None:
        aotm_report = import_aotm_sequences(aotm_source, destination / "aotm-sequences.jsonl")

    manifest = {
        "schema_version": REAL_RETRIEVAL_DATA_VERSION,
        "dataset_id": destination.name,
        "policy": {
            "research_only": True,
            "shadow_training_only": True,
            "production_authority": False,
            "hard_safety_authoritative": True,
            "aotm_requires_feature_resolution_before_runtime_training": True,
            "crate_level_split_isolation": True,
            "dense_route_attempt_evidence": True,
        },
        "sources": {
            "synthetic_next_track": str(synthetic),
            "synthetic_rows": synthetic_count,
            "phase16_plans": [str(path) for path in plan_paths],
            "phase16_rows": phase_count,
            "phase16_splits": dict(sorted(phase_splits.items())),
            "phase16_groups": len({str(row.get("group")) for row in phase_rows}),
            "phase16_crates": len(crate_splits),
            "crate_split_assignment": dict(sorted(crate_splits.items())),
            "dense_evidence_rows": sum(
                1 for row in phase_rows if row.get("source") == "phase16_dense_route_evidence"
            ),
            "legacy_evidence_rows": sum(
                1 for row in phase_rows if row.get("source") == "phase16_real_rehearsal_legacy"
            ),
            "aotm": aotm_report,
        },
        "outputs": {
            "runtime_next_track": runtime_path.name,
            "phase16_next_track": phase_path.name,
            "synthetic_prior": synthetic_copy.name,
            "aotm_sequences": "aotm-sequences.jsonl" if aotm_report else None,
        },
        "feature_contract": {
            "runtime_trainable": "transition_feature_dict",
            "synthetic_prior_namespace": "state_* + candidate_*",
            "aotm_namespace": "MSD track ids + playlist order",
            "cross_source_feature_bridge_ready": False,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest = root / "datasets" / "real-retrieval" / "LATEST"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(destination.name + "\n", encoding="utf-8")
    return {"dataset_path": str(destination), "manifest": manifest}


def train_mixed_retrieval_shadow(
    workspace: str | Path,
    *,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    dataset_root = root / "datasets" / "real-retrieval"
    if dataset_id is None:
        latest = dataset_root / "LATEST"
        if not latest.is_file():
            raise ValueError("No real-retrieval snapshot is available.")
        dataset_id = latest.read_text(encoding="utf-8").strip()
    snapshot = dataset_root / str(dataset_id)
    dataset = snapshot / "next-track-runtime.jsonl"
    if not dataset.is_file():
        raise ValueError(f"Runtime-aligned retriever dataset not found: {dataset}")
    model_root = root / "config" / "real-retrieval-models"
    model_root.mkdir(parents=True, exist_ok=True)
    output = model_root / f"next-track-mlp-{dataset_id}.json"
    trained = train_mlp_pairwise_next_track_ranker(
        dataset,
        output,
        hidden_sizes=(64, 32),
        epochs=120,
        learning_rate=0.0035,
        l2=0.0015,
        seed=1337,
    )
    payload = json.loads(trained.read_text(encoding="utf-8"))
    payload.update({
        "model_id": f"next-track-real-retrieval-{dataset_id}",
        "learning_head": "next_track_policy",
        "training_dataset_id": str(dataset_id),
        "training_data_class": "phase16_real_rehearsal_runtime_aligned",
        "synthetic_prior_corpus_present": (snapshot / "synthetic-next-track-prior.jsonl").is_file(),
        "synthetic_runtime_feature_bridge_resolved": False,
        "aotm_sequence_corpus_present": (snapshot / "aotm-sequences.jsonl").is_file(),
        "aotm_runtime_features_resolved": False,
        "shadow_only": True,
        "authority": 0.0,
        "production_authority": False,
        "safety_contract": (
            "This model may rank/filter candidates in research shadow evaluation only; "
            "professional rehearsal, deterministic execution validation, and hard safety remain authoritative."
        ),
    })
    trained.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (model_root / "NEXT_TRACK_REAL_RETRIEVAL_LATEST").write_text(trained.name + "\n", encoding="utf-8")
    return {
        "schema_version": REAL_RETRIEVAL_DATA_VERSION,
        "dataset_id": str(dataset_id),
        "model_path": str(trained.relative_to(root)),
        "shadow_only": True,
        "authority": 0.0,
        "metrics": payload.get("metrics", {}),
        "aotm_runtime_features_resolved": False,
        "synthetic_runtime_feature_bridge_resolved": False,
    }
