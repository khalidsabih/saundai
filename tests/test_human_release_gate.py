from __future__ import annotations

import json
from pathlib import Path

import pytest

from djai.evaluation.human_release_gate import (
    HumanReleaseThresholds,
    create_human_release_session,
    human_release_report,
    load_locked_benchmark,
    lock_human_release_benchmark,
    record_human_release_vote,
)


def _locked_benchmark(tmp_path: Path, case_count: int = 4) -> Path:
    rows = []
    for index in range(case_count):
        incumbent = tmp_path / f"incumbent-{index}.wav"
        candidate = tmp_path / f"candidate-{index}.wav"
        incumbent.write_bytes(f"incumbent-{index}".encode())
        candidate.write_bytes(f"candidate-{index}".encode())
        rows.append(
            {
                "case_id": f"case-{index:03d}",
                "category": "hard-transition",
                "style": "house" if index % 2 == 0 else "open-format",
                "incumbent_audio": str(incumbent),
                "candidate_audio": str(candidate),
                "incumbent_production_safe": True,
                "candidate_production_safe": True,
                "incumbent_hard_failure_count": 0,
                "candidate_hard_failure_count": 0,
            }
        )
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    destination = tmp_path / "benchmarks/human-release/locked-benchmark.json"
    lock_human_release_benchmark(
        cases,
        destination,
        incumbent="main-c8e7a65",
        candidate="taste-candidate",
    )
    return destination


def _vote_candidate_for_all_cases(
    workspace: Path,
    benchmark: Path,
    *,
    evaluator: str,
    cohort: str,
    case_count: int,
    catastrophic_case: int | None = None,
) -> None:
    for index in range(case_count):
        session = create_human_release_session(
            workspace,
            evaluator=evaluator,
            cohort=cohort,  # type: ignore[arg-type]
            benchmark_path=benchmark,
        )
        manifest = json.loads(
            (
                workspace
                / "output/human-release-gate"
                / session["session_id"]
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        choice = "a" if manifest["labels"]["A"] == "candidate" else "b"
        candidate_is_a = manifest["labels"]["A"] == "candidate"
        record_human_release_vote(
            workspace,
            session_id=session["session_id"],
            choice=choice,
            catastrophic_a=(catastrophic_case == index and candidate_is_a),
            catastrophic_b=(catastrophic_case == index and not candidate_is_a),
            reasons=["phrasing", "naturalness"],
        )


def test_locked_benchmark_hashes_audio_and_detects_tampering(tmp_path: Path):
    benchmark_path = _locked_benchmark(tmp_path, case_count=2)
    benchmark = load_locked_benchmark(benchmark_path)

    assert benchmark["locked"] is True
    assert benchmark["training_export_allowed"] is False
    assert benchmark["case_count"] == 2
    assert benchmark["benchmark_id"].startswith("human-release-")

    candidate = Path(benchmark["cases"][0]["candidate_audio"])
    candidate.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="audio hash mismatch"):
        load_locked_benchmark(benchmark_path)


def test_release_session_is_blind_and_duplicate_submission_is_blocked(tmp_path: Path):
    benchmark_path = _locked_benchmark(tmp_path, case_count=1)
    session = create_human_release_session(
        tmp_path,
        evaluator="reviewer-one",
        cohort="professional_dj",
        benchmark_path=benchmark_path,
    )

    assert session["blind"] is True
    assert set(session["audio"]) == {"A", "B"}
    assert "candidate" not in session["audio"]
    manifest = json.loads(
        (
            tmp_path
            / "output/human-release-gate"
            / session["session_id"]
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    choice = "a" if manifest["labels"]["A"] == "candidate" else "b"

    record_human_release_vote(
        tmp_path,
        session_id=session["session_id"],
        choice=choice,
    )
    with pytest.raises(ValueError, match="already been submitted"):
        record_human_release_vote(
            tmp_path,
            session_id=session["session_id"],
            choice=choice,
        )


def test_release_gate_passes_only_with_locked_coverage_human_win_and_no_catastrophe(tmp_path: Path):
    benchmark_path = _locked_benchmark(tmp_path, case_count=4)
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="pro-a",
        cohort="professional_dj",
        case_count=4,
    )
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="pro-b",
        cohort="professional_dj",
        case_count=4,
    )
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="experienced-a",
        cohort="experienced_dj",
        case_count=4,
    )

    report = human_release_report(
        tmp_path,
        benchmark_path=benchmark_path,
        thresholds=HumanReleaseThresholds(
            minimum_locked_cases=4,
            minimum_covered_cases=4,
            minimum_decisive_votes=12,
            minimum_unique_evaluators=3,
            minimum_professional_dj_votes=8,
            target_candidate_win_rate=0.65,
            lower_confidence_must_exceed=0.50,
            maximum_candidate_catastrophic_rate=0.01,
            maximum_new_catastrophic_cases=0,
        ),
    )

    assert report["deployment_ready"] is True
    assert report["candidate_effective_win_rate"] == pytest.approx(1.0)
    assert report["covered_case_count"] == 4
    assert report["unique_evaluators"] == 3
    assert report["candidate_catastrophic_mix_rate"] == 0.0
    assert report["blockers"] == []


def test_release_gate_blocks_new_catastrophic_mix_even_when_candidate_wins(tmp_path: Path):
    benchmark_path = _locked_benchmark(tmp_path, case_count=4)
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="pro-a",
        cohort="professional_dj",
        case_count=4,
        catastrophic_case=0,
    )
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="pro-b",
        cohort="professional_dj",
        case_count=4,
    )
    _vote_candidate_for_all_cases(
        tmp_path,
        benchmark_path,
        evaluator="experienced-a",
        cohort="experienced_dj",
        case_count=4,
    )

    report = human_release_report(
        tmp_path,
        benchmark_path=benchmark_path,
        thresholds=HumanReleaseThresholds(
            minimum_locked_cases=4,
            minimum_covered_cases=4,
            minimum_decisive_votes=12,
            minimum_unique_evaluators=3,
            minimum_professional_dj_votes=8,
            target_candidate_win_rate=0.65,
            maximum_candidate_catastrophic_rate=0.01,
            maximum_new_catastrophic_cases=0,
        ),
    )

    assert report["candidate_effective_win_rate"] == pytest.approx(1.0)
    assert report["deployment_ready"] is False
    assert report["new_catastrophic_case_count"] == 1
    assert any("catastrophic" in blocker for blocker in report["blockers"])


def test_default_release_gate_requires_real_benchmark_scale(tmp_path: Path):
    benchmark_path = _locked_benchmark(tmp_path, case_count=4)
    report = human_release_report(tmp_path, benchmark_path=benchmark_path)

    assert report["deployment_ready"] is False
    assert any("need 150" in blocker for blocker in report["blockers"])
