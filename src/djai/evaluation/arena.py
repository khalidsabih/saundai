from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

ArenaChoice = Literal["champion", "challenger", "tie", "neither"]


@dataclass(slots=True)
class ArenaVote:
    champion: str
    challenger: str
    choice: ArenaChoice
    evaluator: str
    context_id: str
    technical_regression: bool = False
    notes: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def append_arena_vote(vote: ArenaVote, path: str | Path = "ratings/arena.jsonl") -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(vote.to_dict(), sort_keys=True) + "\n")
    return destination


def _wilson(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def arena_report(
    path: str | Path = "ratings/arena.jsonl",
    *,
    champion: str | None = None,
    challenger: str | None = None,
    target_win_rate: float = 0.60,
    minimum_decisive: int = 50,
) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    votes: list[dict[str, object]] = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if champion and item.get("champion") != champion:
                continue
            if challenger and item.get("challenger") != challenger:
                continue
            votes.append(item)
    champion_wins = sum(item.get("choice") == "champion" for item in votes)
    challenger_wins = sum(item.get("choice") == "challenger" for item in votes)
    ties = sum(item.get("choice") == "tie" for item in votes)
    neither = sum(item.get("choice") == "neither" for item in votes)
    decisive = champion_wins + challenger_wins
    effective_challenger = challenger_wins + 0.5 * ties
    effective_total = decisive + ties
    rate = effective_challenger / effective_total if effective_total else 0.0
    lower, upper = _wilson(effective_challenger, effective_total)
    regressions = sum(bool(item.get("technical_regression")) for item in votes)
    deploy = decisive >= minimum_decisive and rate >= target_win_rate and lower > 0.50 and regressions == 0
    return {
        "path": str(source),
        "champion": champion,
        "challenger": challenger,
        "votes": len(votes),
        "decisive": decisive,
        "champion_wins": champion_wins,
        "challenger_wins": challenger_wins,
        "ties": ties,
        "neither": neither,
        "challenger_effective_win_rate": round(rate, 6),
        "confidence_interval_95": [round(lower, 6), round(upper, 6)],
        "technical_regressions": regressions,
        "gate": {
            "minimum_decisive": minimum_decisive,
            "target_win_rate": target_win_rate,
            "lower_confidence_must_exceed": 0.50,
            "technical_regressions_allowed": 0,
        },
        "deployment_ready": deploy,
    }
