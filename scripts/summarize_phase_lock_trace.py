#!/usr/bin/env python3
"""Summarize rejected phase-lock configurations without changing planning.

Usage:
    python scripts/summarize_phase_lock_trace.py output/phase6/phase-rejections.jsonl

Counts unique exact candidate configurations separately from repeated
evaluations. Residual=999ms is an *unknown/insufficient evidence sentinel*,
never a measured phase error.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return round(ordered[max(0, index)], 3)


def summarize(path: Path) -> dict[str, object]:
    raw = 0
    malformed = 0
    unique: dict[tuple[object, ...], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not an object")
            except (json.JSONDecodeError, ValueError):
                malformed += 1
                continue
            # The diagnostic key is specific to an evaluated cue, rate and
            # overlap: one pair may contain many genuinely different ideas.
            key = tuple(record.get(field) for field in (
                "track_a", "track_b", "exit_time_a_seconds",
                "entry_time_b_seconds", "transition_seconds",
                "render_rate_a", "render_rate_b",
                "phase_nudge_seconds_b",
            ))
            unique.setdefault(key, record)

    reasons = Counter(str(item.get("reason", "unknown")) for item in unique.values())
    per_pair = defaultdict(lambda: {"unique": 0, "measured": 0, "insufficient": 0})
    residuals: list[float] = []
    for row in unique.values():
        key = f"{row.get('track_a', '?')} -> {row.get('track_b', '?')}"
        group = per_pair[key]
        group["unique"] += 1
        value = row.get("max_residual_ms")
        reason = str(row.get("reason", ""))
        # Missing evidence deliberately uses a 999ms sentinel. It is not
        # legitimate measured drift and must never inflate a percentile.
        if (
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value < 999.0
            and "insufficient" not in reason.lower()
            and "too weak" not in reason.lower()
        ):
            group["measured"] += 1
            residuals.append(float(value))
        else:
            group["insufficient"] += 1
    return {
        "raw_rejection_events": raw,
        "malformed_lines": malformed,
        "unique_candidate_configurations": len(unique),
        "repeated_evaluations": max(0, raw - malformed - len(unique)),
        "reasons_unique": dict(reasons.most_common()),
        "measured_residual_count": len(residuals),
        "measured_residual_median_ms": round(median(residuals), 3) if residuals else None,
        "measured_residual_p90_ms": _percentile(residuals, .90),
        "most_rejected_pairs": [
            {"pair": pair, **counts}
            for pair, counts in sorted(
                per_pair.items(),
                key=lambda item: (-item[1]["unique"], item[0]),
            )[:12]
        ],
        "notes": [
            "A rejection event is not necessarily a distinct cue configuration.",
            "999ms is an unknown-residual sentinel, not observed phase drift.",
            "Only rejected candidates are logged; no recall or acceptance rate can be inferred.",
            "This diagnostic does not change hard beat or audio safety thresholds.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
