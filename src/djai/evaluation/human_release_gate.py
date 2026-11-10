from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from ..training.human_preference import evaluator_id

HUMAN_RELEASE_GATE_VERSION = "1.0"
HumanReleaseChoice = Literal["incumbent", "candidate", "tie", "neither"]
EvaluatorCohort = Literal["general_listener", "experienced_dj", "professional_dj"]

DEFAULT_MANIFEST = "benchmarks/human-release/locked-benchmark.json"
DEFAULT_VOTES = "ratings/human-release-gate.jsonl"
DEFAULT_SESSIONS = "output/human-release-gate"


@dataclass(frozen=True, slots=True)
class HumanReleaseThresholds:
    minimum_locked_cases: int = 150
    minimum_covered_cases: int = 150
    minimum_decisive_votes: int = 150
    minimum_unique_evaluators: int = 3
    minimum_professional_dj_votes: int = 20
    target_candidate_win_rate: float = 0.65
    lower_confidence_must_exceed: float = 0.50
    maximum_candidate_catastrophic_rate: float = 0.01
    maximum_new_catastrophic_cases: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "version": str(payload.get("version") or HUMAN_RELEASE_GATE_VERSION),
        "incumbent": str(payload.get("incumbent") or ""),
        "candidate": str(payload.get("candidate") or ""),
        "cases": [
            {
                "case_id": str(item.get("case_id") or ""),
                "category": str(item.get("category") or "unknown"),
                "style": str(item.get("style") or "unknown"),
                "incumbent_sha256": str(item.get("incumbent_sha256") or ""),
                "candidate_sha256": str(item.get("candidate_sha256") or ""),
                "incumbent_production_safe": bool(item.get("incumbent_production_safe", True)),
                "candidate_production_safe": bool(item.get("candidate_production_safe", True)),
                "incumbent_hard_failure_count": int(item.get("incumbent_hard_failure_count", 0) or 0),
                "candidate_hard_failure_count": int(item.get("candidate_hard_failure_count", 0) or 0),
            }
            for item in payload.get("cases", [])
            if isinstance(item, Mapping)
        ],
    }


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    canonical = _canonical_manifest_payload(payload)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def lock_human_release_benchmark(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    incumbent: str,
    candidate: str,
) -> Path:
    """Lock a local blind benchmark by hashing every rendered A/B asset."""
    source = Path(cases_path).expanduser().resolve()
    if not source.exists():
        raise ValueError("Human release benchmark cases file does not exist.")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _read_jsonl(source):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("Every human release benchmark case requires case_id.")
        if case_id in seen:
            raise ValueError(f"Duplicate human release benchmark case_id: {case_id}")
        seen.add(case_id)
        incumbent_audio = Path(str(row.get("incumbent_audio") or "")).expanduser().resolve()
        candidate_audio = Path(str(row.get("candidate_audio") or "")).expanduser().resolve()
        if not incumbent_audio.exists() or not candidate_audio.exists():
            raise ValueError(f"Benchmark case {case_id} is missing incumbent or candidate audio.")
        cases.append(
            {
                "case_id": case_id,
                "category": str(row.get("category") or "unknown"),
                "style": str(row.get("style") or "unknown"),
                "incumbent_audio": str(incumbent_audio),
                "candidate_audio": str(candidate_audio),
                "incumbent_sha256": _sha256_file(incumbent_audio),
                "candidate_sha256": _sha256_file(candidate_audio),
                "incumbent_production_safe": bool(row.get("incumbent_production_safe", True)),
                "candidate_production_safe": bool(row.get("candidate_production_safe", True)),
                "incumbent_hard_failure_count": int(row.get("incumbent_hard_failure_count", 0) or 0),
                "candidate_hard_failure_count": int(row.get("candidate_hard_failure_count", 0) or 0),
            }
        )
    if not cases:
        raise ValueError("Human release benchmark requires at least one case.")

    payload: dict[str, Any] = {
        "version": HUMAN_RELEASE_GATE_VERSION,
        "incumbent": str(incumbent),
        "candidate": str(candidate),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked": True,
        "local_only": True,
        "training_export_allowed": False,
        "case_count": len(cases),
        "cases": cases,
    }
    digest = _manifest_digest(payload)
    payload["benchmark_id"] = f"human-release-{digest[:20]}"
    payload["locked_sha256"] = digest

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if str(existing.get("locked_sha256") or "") != digest:
            raise ValueError("Locked human release benchmark already exists with different contents.")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_locked_benchmark(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ValueError("Locked human release benchmark does not exist.")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not bool(payload.get("locked")):
        raise ValueError("Human release benchmark is not locked.")
    expected = str(payload.get("locked_sha256") or "")
    actual = _manifest_digest(payload)
    if not expected or expected != actual:
        raise ValueError("Human release benchmark manifest hash mismatch.")
    for item in payload.get("cases", []):
        if not isinstance(item, Mapping):
            raise ValueError("Invalid benchmark case.")
        for side in ("incumbent", "candidate"):
            audio = Path(str(item.get(f"{side}_audio") or "")).expanduser().resolve()
            if not audio.exists():
                raise ValueError(f"Locked benchmark audio is missing for {item.get('case_id')}.")
            if _sha256_file(audio) != str(item.get(f"{side}_sha256") or ""):
                raise ValueError(f"Locked benchmark audio hash mismatch for {item.get('case_id')} ({side}).")
    return payload


def _session_path(workspace: Path, session_id: str) -> Path:
    return workspace / DEFAULT_SESSIONS / session_id / "manifest.json"


def create_human_release_session(
    workspace: str | Path,
    *,
    evaluator: str,
    cohort: EvaluatorCohort,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    resolved_benchmark_path = Path(benchmark_path or (root / DEFAULT_MANIFEST)).expanduser().resolve()
    benchmark = load_locked_benchmark(resolved_benchmark_path)
    reviewer = evaluator_id(evaluator, cohort)
    votes = _read_jsonl(root / DEFAULT_VOTES)
    reviewed = {
        str(row.get("case_id") or "")
        for row in votes
        if str(row.get("benchmark_id") or "") == str(benchmark["benchmark_id"])
        and str(row.get("evaluator_id") or "") == reviewer
    }
    available = [
        item for item in benchmark.get("cases", [])
        if isinstance(item, Mapping) and str(item.get("case_id") or "") not in reviewed
    ]
    if not available:
        raise ValueError("No unreviewed locked benchmark case is available for this evaluator.")
    available.sort(key=lambda item: str(item.get("case_id") or ""))
    case = available[0]

    order = ["incumbent", "candidate"]
    secrets.SystemRandom().shuffle(order)
    labels = {"A": order[0], "B": order[1]}
    session_id = f"human-release-session-{secrets.token_hex(6)}"
    session_root = root / DEFAULT_SESSIONS / session_id
    session_root.mkdir(parents=True, exist_ok=False)
    session = {
        "version": HUMAN_RELEASE_GATE_VERSION,
        "session_id": session_id,
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_path": str(resolved_benchmark_path),
        "locked_sha256": benchmark["locked_sha256"],
        "case_id": case["case_id"],
        "evaluator_id": reviewer,
        "evaluator_cohort": cohort,
        "blind": True,
        "labels": labels,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audio": {
            "A": str(case[f"{labels['A']}_audio"]),
            "B": str(case[f"{labels['B']}_audio"]),
        },
    }
    (session_root / "manifest.json").write_text(
        json.dumps(session, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "session_id": session_id,
        "benchmark_id": benchmark["benchmark_id"],
        "case_id": case["case_id"],
        "cohort": cohort,
        "blind": True,
        "audio": dict(session["audio"]),
    }


def record_human_release_vote(
    workspace: str | Path,
    *,
    session_id: str,
    choice: Literal["a", "b", "tie", "neither"],
    catastrophic_a: bool = False,
    catastrophic_b: bool = False,
    reasons: Iterable[str] = (),
) -> Path:
    root = Path(workspace).expanduser().resolve()
    session_path = _session_path(root, session_id)
    if not session_path.exists():
        raise ValueError("Human release session does not exist.")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    benchmark_path = Path(str(session.get("benchmark_path") or (root / DEFAULT_MANIFEST))).expanduser().resolve()
    benchmark = load_locked_benchmark(benchmark_path)
    if session.get("benchmark_id") != benchmark.get("benchmark_id"):
        raise ValueError("Human release session belongs to a different benchmark.")
    labels = session.get("labels") if isinstance(session.get("labels"), Mapping) else {}
    if labels.get("A") not in {"incumbent", "candidate"} or labels.get("B") not in {"incumbent", "candidate"}:
        raise ValueError("Human release session label mapping is invalid.")

    if choice == "a":
        resolved: HumanReleaseChoice = str(labels["A"])  # type: ignore[assignment]
    elif choice == "b":
        resolved = str(labels["B"])  # type: ignore[assignment]
    else:
        resolved = choice

    catastrophe = {
        str(labels["A"]): bool(catastrophic_a),
        str(labels["B"]): bool(catastrophic_b),
    }
    case_id = str(session.get("case_id") or "")
    case = next(
        (item for item in benchmark.get("cases", []) if isinstance(item, Mapping) and str(item.get("case_id") or "") == case_id),
        None,
    )
    if case is None:
        raise ValueError("Human release session case is not part of the locked benchmark.")

    row = {
        "version": HUMAN_RELEASE_GATE_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "locked_sha256": benchmark["locked_sha256"],
        "case_id": case_id,
        "incumbent": benchmark["incumbent"],
        "candidate": benchmark["candidate"],
        "choice": resolved,
        "evaluator_id": session["evaluator_id"],
        "evaluator_cohort": session["evaluator_cohort"],
        "session_id": session_id,
        "incumbent_catastrophic": bool(catastrophe.get("incumbent", False)),
        "candidate_catastrophic": bool(catastrophe.get("candidate", False)),
        "incumbent_technical_failure": (
            not bool(case.get("incumbent_production_safe", True))
            or int(case.get("incumbent_hard_failure_count", 0) or 0) > 0
        ),
        "candidate_technical_failure": (
            not bool(case.get("candidate_production_safe", True))
            or int(case.get("candidate_hard_failure_count", 0) or 0) > 0
        ),
        "reasons": list(dict.fromkeys(str(item).strip().lower() for item in reasons if str(item).strip())),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = root / DEFAULT_VOTES
    existing_votes = _read_jsonl(destination)
    if any(
        str(item.get("session_id") or "") == session_id
        or (
            str(item.get("benchmark_id") or "") == str(benchmark["benchmark_id"])
            and str(item.get("case_id") or "") == case_id
            and str(item.get("evaluator_id") or "") == str(session.get("evaluator_id") or "")
        )
        for item in existing_votes
    ):
        raise ValueError("This human release comparison has already been submitted by this evaluator.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return destination


def _wilson(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def human_release_report(
    workspace: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    thresholds: HumanReleaseThresholds | None = None,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    cfg = thresholds or HumanReleaseThresholds()
    benchmark = load_locked_benchmark(benchmark_path or (root / DEFAULT_MANIFEST))
    votes = [
        row for row in _read_jsonl(root / DEFAULT_VOTES)
        if str(row.get("benchmark_id") or "") == str(benchmark["benchmark_id"])
        and str(row.get("locked_sha256") or "") == str(benchmark["locked_sha256"])
    ]

    candidate_wins = sum(row.get("choice") == "candidate" for row in votes)
    incumbent_wins = sum(row.get("choice") == "incumbent" for row in votes)
    ties = sum(row.get("choice") == "tie" for row in votes)
    neither = sum(row.get("choice") == "neither" for row in votes)
    decisive = candidate_wins + incumbent_wins
    effective_total = decisive + ties
    effective_candidate = candidate_wins + 0.5 * ties
    win_rate = effective_candidate / effective_total if effective_total else 0.0
    lower, upper = _wilson(effective_candidate, effective_total)

    case_votes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in votes:
        case_votes[str(row.get("case_id") or "")].append(row)
    covered_cases = len([case_id for case_id in case_votes if case_id])
    candidate_catastrophic_cases = 0
    incumbent_catastrophic_cases = 0
    new_catastrophic_cases = 0
    for rows in case_votes.values():
        candidate_bad = any(
            bool(row.get("candidate_catastrophic")) or bool(row.get("candidate_technical_failure"))
            for row in rows
        )
        incumbent_bad = any(
            bool(row.get("incumbent_catastrophic")) or bool(row.get("incumbent_technical_failure"))
            for row in rows
        )
        candidate_catastrophic_cases += int(candidate_bad)
        incumbent_catastrophic_cases += int(incumbent_bad)
        new_catastrophic_cases += int(candidate_bad and not incumbent_bad)
    candidate_catastrophic_rate = (
        candidate_catastrophic_cases / covered_cases if covered_cases else 0.0
    )
    evaluators = {str(row.get("evaluator_id") or "") for row in votes if str(row.get("evaluator_id") or "")}
    cohorts = Counter(str(row.get("evaluator_cohort") or "unknown") for row in votes)
    reasons = Counter(
        str(reason)
        for row in votes
        for reason in (row.get("reasons") or [])
        if str(reason)
    )

    blockers: list[str] = []
    locked_cases = int(benchmark.get("case_count", len(benchmark.get("cases", []))) or 0)
    if locked_cases < cfg.minimum_locked_cases:
        blockers.append(f"locked benchmark has {locked_cases} cases; need {cfg.minimum_locked_cases}")
    if covered_cases < cfg.minimum_covered_cases:
        blockers.append(f"only {covered_cases} unique cases have votes; need {cfg.minimum_covered_cases}")
    if decisive < cfg.minimum_decisive_votes:
        blockers.append(f"only {decisive} decisive votes; need {cfg.minimum_decisive_votes}")
    if len(evaluators) < cfg.minimum_unique_evaluators:
        blockers.append(f"only {len(evaluators)} unique evaluators; need {cfg.minimum_unique_evaluators}")
    professional_votes = int(cohorts.get("professional_dj", 0))
    if professional_votes < cfg.minimum_professional_dj_votes:
        blockers.append(
            f"only {professional_votes} professional-DJ votes; need {cfg.minimum_professional_dj_votes}"
        )
    if win_rate < cfg.target_candidate_win_rate:
        blockers.append(
            f"candidate effective win rate {win_rate:.1%} is below {cfg.target_candidate_win_rate:.1%}"
        )
    if lower <= cfg.lower_confidence_must_exceed:
        blockers.append(
            f"candidate 95% lower confidence bound {lower:.1%} does not exceed {cfg.lower_confidence_must_exceed:.1%}"
        )
    if candidate_catastrophic_rate > cfg.maximum_candidate_catastrophic_rate:
        blockers.append(
            f"candidate catastrophic mix rate {candidate_catastrophic_rate:.2%} exceeds {cfg.maximum_candidate_catastrophic_rate:.2%}"
        )
    if new_catastrophic_cases > cfg.maximum_new_catastrophic_cases:
        blockers.append(
            f"candidate introduces {new_catastrophic_cases} new catastrophic cases; allowed {cfg.maximum_new_catastrophic_cases}"
        )

    return {
        "release_gate_version": HUMAN_RELEASE_GATE_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "locked_sha256": benchmark["locked_sha256"],
        "benchmark_locked": True,
        "incumbent": benchmark["incumbent"],
        "candidate": benchmark["candidate"],
        "locked_case_count": locked_cases,
        "covered_case_count": covered_cases,
        "votes": len(votes),
        "decisive": decisive,
        "incumbent_wins": incumbent_wins,
        "candidate_wins": candidate_wins,
        "ties": ties,
        "neither": neither,
        "candidate_effective_win_rate": round(win_rate, 6),
        "confidence_interval_95": [round(lower, 6), round(upper, 6)],
        "candidate_catastrophic_case_count": candidate_catastrophic_cases,
        "incumbent_catastrophic_case_count": incumbent_catastrophic_cases,
        "new_catastrophic_case_count": new_catastrophic_cases,
        "candidate_catastrophic_mix_rate": round(candidate_catastrophic_rate, 6),
        "unique_evaluators": len(evaluators),
        "cohorts": dict(cohorts),
        "top_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(12)
        ],
        "thresholds": cfg.to_dict(),
        "blockers": blockers,
        "deployment_ready": not blockers,
        "training_data_allowed": False,
        "policy": {
            "locked_benchmark_never_training_data": True,
            "blind_identity_required": True,
            "technical_and_human_catastrophes_count": True,
            "candidate_must_beat_incumbent": True,
            "promotion_requires_deployment_ready": True,
        },
    }



def human_release_status(workspace: str | Path) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    manifest = root / DEFAULT_MANIFEST
    if not manifest.exists():
        return {
            "release_gate_version": HUMAN_RELEASE_GATE_VERSION,
            "available": False,
            "benchmark_locked": False,
            "deployment_ready": False,
            "reason": "no locked human release benchmark",
        }
    try:
        report = human_release_report(root)
    except ValueError as exc:
        return {
            "release_gate_version": HUMAN_RELEASE_GATE_VERSION,
            "available": False,
            "benchmark_locked": False,
            "deployment_ready": False,
            "reason": str(exc),
        }
    return {
        "available": True,
        **report,
    }
