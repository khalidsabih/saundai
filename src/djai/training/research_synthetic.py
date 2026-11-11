from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .neural_heads import train_neural_head
from .pairwise_ranker import train_pairwise_next_track_ranker

RESEARCH_SYNTHETIC_SCHEMA_VERSION = "1.0"
RESEARCH_SYNTHETIC_IMPORT_VERSION = "1.0"

_EXPECTED_FILES = {
    "transition_priors.jsonl": "transition_prior",
    "switch_points.jsonl": "switch_point_detection",
    "functional_structure.jsonl": "functional_segment",
    "time_aware_set.jsonl": "time_aware_next_track",
    "hard_negatives.jsonl": "hard_negative",
}
_FORBIDDEN_KEYS = {
    "audio",
    "audio_path",
    "raw_audio",
    "raw_audio_path",
    "local_path",
    "source_path",
    "track_path",
}


@dataclass(frozen=True, slots=True)
class ResearchSyntheticValidation:
    source_dir: str
    total_examples: int
    counts: dict[str, int]
    source_ids: tuple[str, ...]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
            "source_dir": self.source_dir,
            "total_examples": self.total_examples,
            "counts": dict(self.counts),
            "source_ids": list(self.source_ids),
            "manifest_sha256": self.manifest_sha256,
            "valid": True,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name} line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path.name} line {line_number} must be a JSON object.")
            yield line_number, payload


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                return str(key)
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_forbidden_key(item)
            if found is not None:
                return found
    return None


def validate_research_synthetic_dataset(source_dir: str | Path) -> ResearchSyntheticValidation:
    root = Path(source_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Research synthetic manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid research synthetic manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Research synthetic manifest must contain a JSON object.")
    if str(manifest.get("schema_version") or "") != RESEARCH_SYNTHETIC_SCHEMA_VERSION:
        raise ValueError("Unsupported research synthetic schema version.")

    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Research synthetic manifest is missing policy metadata.")
    if bool(policy.get("raw_audio_included")):
        raise ValueError("Research synthetic corpus must not bundle raw audio.")
    if bool(policy.get("human_preference")):
        raise ValueError("Research synthetic corpus cannot claim human preference labels.")
    if bool(policy.get("production_authority")):
        raise ValueError("Research synthetic corpus cannot carry production authority.")

    source_ids = {
        str(item.get("source_id") or "")
        for item in manifest.get("sources", [])
        if isinstance(item, Mapping)
    }
    source_ids.discard("")
    if not source_ids:
        raise ValueError("Research synthetic manifest must declare at least one provenance source.")

    expected_counts = manifest.get("counts")
    if not isinstance(expected_counts, Mapping):
        raise ValueError("Research synthetic manifest is missing per-task counts.")

    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for filename, expected_task in _EXPECTED_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required research synthetic file not found: {path}")
        for line_number, row in _iter_jsonl(path):
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"{filename} line {line_number} is missing example_id.")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate research synthetic example_id: {example_id}")
            seen_ids.add(example_id)
            if str(row.get("schema_version") or "") != RESEARCH_SYNTHETIC_SCHEMA_VERSION:
                raise ValueError(f"{filename} line {line_number} has an unsupported schema version.")
            if str(row.get("task") or "") != expected_task:
                raise ValueError(
                    f"{filename} line {line_number} expected task {expected_task!r}; "
                    f"found {row.get('task')!r}."
                )
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError(f"{filename} line {line_number} is missing provenance.")
            if bool(provenance.get("human_preference")) or bool(provenance.get("production_authority")):
                raise ValueError(f"{filename} line {line_number} improperly claims human/production authority.")
            if str(provenance.get("confidence_tier") or "") not in {"A", "B", "C"}:
                raise ValueError(f"{filename} line {line_number} has invalid confidence_tier.")
            row_sources = {str(item) for item in provenance.get("source_ids", []) if str(item)}
            if not row_sources or not row_sources.issubset(source_ids):
                raise ValueError(f"{filename} line {line_number} references undeclared provenance sources.")
            forbidden = _contains_forbidden_key(row)
            if forbidden is not None:
                raise ValueError(f"{filename} line {line_number} contains forbidden raw/local field: {forbidden}")
            if not isinstance(row.get("context"), Mapping):
                raise ValueError(f"{filename} line {line_number} is missing context.")
            if not isinstance(row.get("candidates"), list) or not row["candidates"]:
                raise ValueError(f"{filename} line {line_number} requires at least one candidate.")
            if not isinstance(row.get("label"), Mapping):
                raise ValueError(f"{filename} line {line_number} is missing label.")
            counts[expected_task] += 1

        manifest_key = filename.removesuffix(".jsonl")
        declared = int(expected_counts.get(manifest_key, -1))
        if declared != counts[expected_task]:
            raise ValueError(
                f"Manifest count mismatch for {filename}: declared {declared}, "
                f"observed {counts[expected_task]}."
            )

    declared_total = int(manifest.get("total_examples", -1))
    observed_total = sum(counts.values())
    if declared_total != observed_total:
        raise ValueError(
            f"Manifest total_examples mismatch: declared {declared_total}, observed {observed_total}."
        )
    return ResearchSyntheticValidation(
        source_dir=str(root),
        total_examples=observed_total,
        counts=dict(sorted(counts.items())),
        source_ids=tuple(sorted(source_ids)),
        manifest_sha256=_sha256(manifest_path),
    )


def _feature_value(features: dict[str, float], name: str, value: Any) -> None:
    if isinstance(value, bool):
        features[name] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        features[name] = float(value)


def _flatten_features(prefix: str, value: Any, features: dict[str, float]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(item, str):
                features[f"{name}__{item}"] = 1.0
            elif isinstance(item, (bool, int, float)):
                _feature_value(features, name, item)
    elif isinstance(value, (bool, int, float)):
        _feature_value(features, prefix, value)


def _base_training_row(row: Mapping[str, Any], *, suffix: str = "") -> dict[str, Any]:
    example_id = str(row["example_id"])
    provenance = dict(row["provenance"])
    confidence = float(dict(row["label"]).get("confidence", 0.5) or 0.5)
    return {
        "id": example_id + suffix,
        "group": example_id,
        "split": _stable_split(example_id),
        "rights_cleared": False,
        "research_only": True,
        "human_preference": False,
        "production_authority": False,
        "label_source": str(provenance.get("label_source") or "research_synthetic_weak_supervision"),
        "confidence_tier": str(provenance.get("confidence_tier") or "C"),
        "source_ids": list(provenance.get("source_ids") or []),
        "weight": max(0.05, min(1.0, confidence)),
    }


def _transition_prior_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _base_training_row(row)
    features: dict[str, float] = {}
    _flatten_features("context", row["context"], features)
    _flatten_features("candidate", row["candidates"][0], features)
    label = str(dict(row["label"]).get("technical_prior") or "")
    target = {"reject": 0.0, "plausible": 0.65, "preferred_prior": 1.0}.get(label)
    if target is None:
        raise ValueError(f"Unknown transition prior label: {label}")
    output.update(task="transition_prior", features=features, target=target, label=label)
    return output


def _switch_point_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _base_training_row(row)
    features: dict[str, float] = {}
    _flatten_features("context", row["context"], features)
    output.update(
        task="switch_point_detection",
        features=features,
        target=1.0 if bool(dict(row["label"]).get("is_switch_point")) else 0.0,
        label="switch_point" if bool(dict(row["label"]).get("is_switch_point")) else "not_switch_point",
    )
    return output


def _functional_segment_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _base_training_row(row)
    features: dict[str, float] = {}
    _flatten_features("context", row["context"], features)
    output.update(
        task="functional_segment",
        features=features,
        label=str(dict(row["label"]).get("segment") or ""),
    )
    return output


def _time_aware_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = dict(row["context"])
    label = dict(row["label"])
    ranking = [str(item) for item in label.get("ranking", [])]
    rank_index = {candidate_id: index for index, candidate_id in enumerate(ranking)}
    candidates = [dict(item) for item in row["candidates"] if isinstance(item, Mapping)]
    if not ranking or set(rank_index) != {str(item.get("candidate_id") or "") for item in candidates}:
        raise ValueError(f"Time-aware example {row['example_id']} has an invalid ranking.")

    output: list[dict[str, Any]] = []
    denominator = max(1, len(ranking) - 1)
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        base = _base_training_row(row, suffix=f":{candidate_id}")
        features: dict[str, float] = {}
        _flatten_features("state", context, features)
        _flatten_features("candidate", candidate, features)
        rank = rank_index[candidate_id]
        target = 1.0 - (rank / denominator)
        base.update(
            task="next_track",
            features=features,
            target=round(target, 6),
            label="preferred" if rank == 0 else "alternative",
            candidate_id=candidate_id,
            rank=rank,
            candidate_count=len(candidates),
        )
        output.append(base)
    return output


def _hard_negative_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = _base_training_row(row)
    features: dict[str, float] = {}
    _flatten_features("context", row["context"], features)
    _flatten_features("candidate", row["candidates"][0], features)
    output.update(
        task="technical_rejection",
        features=features,
        target=0.0,
        label="reject",
    )
    return output


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


def create_research_synthetic_snapshot(
    workspace: str | Path,
    source_dir: str | Path,
    *,
    name: str = "research-synthetic-v1",
) -> Path:
    root = Path(workspace).expanduser().resolve()
    source = Path(source_dir).expanduser().resolve()
    validation = validate_research_synthetic_dataset(source)

    identity_payload = {
        "import_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
        "manifest_sha256": validation.manifest_sha256,
        "name": str(name),
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    dataset_id = f"{str(name).strip().replace(' ', '-')}-{digest}"
    destination = root / "datasets" / "research-synthetic" / dataset_id
    if destination.exists():
        manifest = destination / "manifest.json"
        if manifest.is_file():
            return destination
        raise ValueError(f"Research synthetic snapshot path already exists but is incomplete: {destination}")

    destination.mkdir(parents=True, exist_ok=False)
    raw_root = destination / "source"
    raw_root.mkdir()

    for filename in ["manifest.json", *_EXPECTED_FILES.keys()]:
        shutil.copy2(source / filename, raw_root / filename)

    transition_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    next_track_rows: list[dict[str, Any]] = []
    hard_negative_rows: list[dict[str, Any]] = []

    for _, row in _iter_jsonl(source / "transition_priors.jsonl"):
        transition_rows.append(_transition_prior_row(row))
    for _, row in _iter_jsonl(source / "switch_points.jsonl"):
        switch_rows.append(_switch_point_row(row))
    for _, row in _iter_jsonl(source / "functional_structure.jsonl"):
        structure_rows.append(_functional_segment_row(row))
    for _, row in _iter_jsonl(source / "time_aware_set.jsonl"):
        next_track_rows.extend(_time_aware_rows(row))
    for _, row in _iter_jsonl(source / "hard_negatives.jsonl"):
        hard_negative_rows.append(_hard_negative_row(row))

    outputs = {
        "transition_prior": (destination / "transition-prior.jsonl", transition_rows),
        "switch_point_detection": (destination / "switch-points.jsonl", switch_rows),
        "functional_segment": (destination / "functional-segments.jsonl", structure_rows),
        "next_track": (destination / "next-track.jsonl", next_track_rows),
        "technical_rejection": (destination / "hard-negatives.jsonl", hard_negative_rows),
    }

    output_manifest: dict[str, Any] = {
        "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
        "dataset_id": dataset_id,
        "source_manifest_sha256": validation.manifest_sha256,
        "source_total_examples": validation.total_examples,
        "source_counts": validation.counts,
        "source_ids": list(validation.source_ids),
        "split_policy": {
            "method": "sha256_group_hash",
            "group": "source example_id",
            "train_percent": 80,
            "validation_percent": 10,
            "test_percent": 10,
            "candidate_group_isolation": True,
        },
        "policy": {
            "research_only": True,
            "human_preference": False,
            "production_authority": False,
            "rights_cleared": False,
            "shadow_training_only": True,
            "human_finetuning_required_for_taste": True,
            "hard_safety_authoritative": True,
        },
        "outputs": {},
    }

    for task, (path, rows) in outputs.items():
        count, split_counts = _write_jsonl(path, rows)
        output_manifest["outputs"][task] = {
            "path": path.name,
            "rows": count,
            "splits": dict(sorted(split_counts.items())),
            "sha256": _sha256(path),
        }

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "datasets" / "research-synthetic" / "LATEST").write_text(
        dataset_id + "\n",
        encoding="utf-8",
    )
    return destination


def research_synthetic_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    dataset_root = root / "datasets" / "research-synthetic"
    latest_path = dataset_root / "LATEST"
    if not latest_path.is_file():
        return {
            "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
            "available": False,
            "reason": "no research synthetic snapshot",
        }
    dataset_id = latest_path.read_text(encoding="utf-8").strip()
    manifest_path = dataset_root / dataset_id / "manifest.json"
    if not manifest_path.is_file():
        return {
            "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
            "available": False,
            "dataset_id": dataset_id,
            "reason": "latest research synthetic snapshot manifest is missing",
        }
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
        "available": True,
        "dataset_id": dataset_id,
        "dataset_path": str(manifest_path.parent),
        "manifest": payload,
    }


def train_research_synthetic_shadow(
    workspace: str | Path,
    *,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    dataset_root = root / "datasets" / "research-synthetic"
    if dataset_id is None:
        latest = dataset_root / "LATEST"
        if not latest.is_file():
            raise ValueError("No imported research synthetic snapshot is available.")
        dataset_id = latest.read_text(encoding="utf-8").strip()
    snapshot = dataset_root / str(dataset_id)
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Research synthetic snapshot not found: {dataset_id}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("policy") if isinstance(manifest.get("policy"), Mapping) else {}
    if not bool(policy.get("shadow_training_only", False)):
        raise ValueError("Research synthetic snapshot is not marked shadow-training-only.")
    if bool(policy.get("production_authority", False)):
        raise ValueError("Research synthetic snapshot improperly carries production authority.")

    next_track_dataset = snapshot / "next-track.jsonl"
    if not next_track_dataset.is_file():
        raise ValueError("Research synthetic snapshot is missing next-track training rows.")

    model_root = root / "config" / "research-synthetic-models"
    model_root.mkdir(parents=True, exist_ok=True)

    baseline_output = model_root / f"next-track-baseline-{dataset_id}.json"
    baseline_trained = train_neural_head(
        "next_track",
        next_track_dataset,
        baseline_output,
        require_rights_cleared=False,
        iterations=900,
        learning_rate=0.025,
        l2=0.003,
    )
    baseline_payload = json.loads(baseline_trained.read_text(encoding="utf-8"))
    baseline_payload.update(
        {
            "model_id": f"next-track-research-synthetic-baseline-{dataset_id}",
            "learning_head": "next_track_policy",
            "training_dataset_id": str(dataset_id),
            "training_data_class": "research_synthetic_weak_supervision",
            "trainer": "scalar_logistic_baseline",
            "shadow_only": True,
            "authority": 0.0,
            "promoted_to_live": False,
            "human_preference": False,
            "production_authority": False,
            "rights_cleared_only": False,
            "safety_contract": (
                "Research synthetic supervision may shape a shadow next-track prior only; "
                "hard feasibility, rendered validation, Executive authority, and human release "
                "gates remain authoritative."
            ),
        }
    )
    baseline_trained.write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ranker_output = model_root / f"next-track-pairwise-{dataset_id}.json"
    ranker_trained = train_pairwise_next_track_ranker(
        next_track_dataset,
        ranker_output,
        iterations=700,
        learning_rate=0.04,
        l2=0.004,
    )
    ranker_payload = json.loads(ranker_trained.read_text(encoding="utf-8"))
    ranker_payload.update(
        {
            "model_id": f"next-track-research-synthetic-pairwise-{dataset_id}",
            "learning_head": "next_track_policy",
            "training_dataset_id": str(dataset_id),
            "training_data_class": "research_synthetic_weak_supervision",
            "trainer": "weighted_pairwise_logistic_ranknet",
            "shadow_only": True,
            "authority": 0.0,
            "promoted_to_live": False,
            "human_preference": False,
            "production_authority": False,
            "rights_cleared_only": False,
            "safety_contract": (
                "Research synthetic supervision may shape a shadow next-track prior only; "
                "hard feasibility, rendered validation, Executive authority, and human release "
                "gates remain authoritative."
            ),
        }
    )
    ranker_trained.write_text(
        json.dumps(ranker_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    baseline_metrics = dict(baseline_payload.get("metrics", {}))
    baseline_validation = dict(baseline_metrics.get("validation_ranking", {}))
    baseline_test = dict(baseline_metrics.get("test_ranking", {}))
    ranker_metrics = dict(ranker_payload.get("metrics", {}))
    ranker_validation = dict(ranker_metrics.get("validation_ranking", {}))
    ranker_test = dict(ranker_metrics.get("test_ranking", {}))

    def delta(metric: str) -> dict[str, float | None]:
        baseline_value = baseline_test.get(metric)
        ranker_value = ranker_test.get(metric)
        difference = None
        if isinstance(baseline_value, (int, float)) and isinstance(ranker_value, (int, float)):
            difference = round(float(ranker_value) - float(baseline_value), 6)
        return {
            "baseline": baseline_value,
            "pairwise": ranker_value,
            "delta": difference,
        }

    comparison = {
        "validation": {
            "baseline": baseline_validation,
            "pairwise": ranker_validation,
        },
        "test": {
            "top1_accuracy": delta("top1_accuracy"),
            "top2_recall": delta("top2_recall"),
            "pairwise_accuracy": delta("pairwise_accuracy"),
            "spearman": delta("spearman"),
            "ndcg": delta("ndcg"),
            "mrr": delta("mrr"),
        },
    }

    (model_root / "NEXT_TRACK_RESEARCH_BASELINE_LATEST").write_text(
        baseline_trained.name + "\n",
        encoding="utf-8",
    )
    (model_root / "NEXT_TRACK_RESEARCH_LATEST").write_text(
        ranker_trained.name + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": RESEARCH_SYNTHETIC_IMPORT_VERSION,
        "dataset_id": str(dataset_id),
        "task": "next_track",
        "shadow_only": True,
        "authority": 0.0,
        "baseline_model_path": str(baseline_trained.relative_to(root)),
        "model_path": str(ranker_trained.relative_to(root)),
        "baseline_metrics": baseline_metrics,
        "metrics": ranker_metrics,
        "comparison": comparison,
    }
