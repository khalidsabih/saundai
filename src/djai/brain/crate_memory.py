from __future__ import annotations

"""SAUND.AI 1.8.2 persistent record and pair memory.

Crate Memory is deliberately smaller in authority than the Complete DJMove
planner and AI Headphones.  It remembers record-specific and directional-pair
experience so repeated use of a crate can improve cue, tempo, stem and failure
priors without turning yesterday's preference into a hard rule.

The store is SQLite rather than another append-only JSONL evidence stream.  Raw
audio is never stored.  Runtime planning consumes only bounded aggregate hints;
protected payoffs, feasibility, private rendered rehearsal and deterministic DSP
safety remain authoritative.
"""

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models import TrackAnalysis
from .dj_move import DJMove

CRATE_MEMORY_VERSION = "1.8.2"
CRATE_MEMORY_SCHEMA_VERSION = 1
DEFAULT_CRATE_MEMORY_PATH = "ratings/crate-memory.sqlite3"
DEFAULT_MAX_ADJUSTMENT = 0.065

_OBSERVED_ANALYSIS_IDS: set[str] = set()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _json_dict(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): float(item) for key, item in value.items() if isinstance(item, (int, float))}
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): float(item) for key, item in payload.items() if isinstance(item, (int, float))}


def _json_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [_safe_int(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [_safe_int(item) for item in payload]


def _inc_counter(payload: str | Mapping[str, float] | None, key: str, amount: float) -> str:
    counter = _json_dict(payload)
    normalized = str(key or "unknown").strip().lower() or "unknown"
    counter[normalized] = round(float(counter.get(normalized, 0.0)) + max(0.0, float(amount)), 6)
    return json.dumps(counter, sort_keys=True, separators=(",", ":"))



def _memory_enabled() -> bool:
    return os.environ.get("DJAI_CRATE_MEMORY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _maximum_adjustment() -> float:
    raw = os.environ.get("DJAI_CRATE_MEMORY_MAX_ADJUSTMENT", str(DEFAULT_MAX_ADJUSTMENT))
    try:
        return max(0.0, min(0.12, float(raw)))
    except ValueError:
        return DEFAULT_MAX_ADJUSTMENT

def _workspace_root(workspace: str | Path | None = None) -> Path:
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    raw = os.environ.get("DJAI_WORKSPACE", "").strip()
    return Path(raw or ".").expanduser().resolve()


def crate_memory_path(
    path: str | Path | None = None,
    *,
    workspace: str | Path | None = None,
) -> Path:
    if path is not None:
        target = Path(path).expanduser()
        return target.resolve() if target.is_absolute() else (_workspace_root(workspace) / target).resolve()
    configured = os.environ.get("DJAI_CRATE_MEMORY_PATH", DEFAULT_CRATE_MEMORY_PATH).strip() or DEFAULT_CRATE_MEMORY_PATH
    target = Path(configured).expanduser()
    return target.resolve() if target.is_absolute() else (_workspace_root(workspace) / target).resolve()


def track_memory_id(track: TrackAnalysis | str | Path | Mapping[str, Any]) -> str:
    if isinstance(track, TrackAnalysis):
        path = track.path
    elif isinstance(track, Mapping):
        path = str(track.get("path") or track.get("track") or track.get("track_a") or track.get("track_b") or "")
    else:
        path = str(track)
    normalized = str(Path(path).expanduser()).replace("\\", "/").casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _track_label(track: TrackAnalysis | str | Path | Mapping[str, Any]) -> str:
    if isinstance(track, TrackAnalysis):
        return track.title
    if isinstance(track, Mapping):
        raw = str(track.get("path") or track.get("track") or track.get("track_a") or track.get("track_b") or "unknown")
    else:
        raw = str(track)
    return Path(raw).stem or "unknown"


def _move_payload(move: DJMove | Mapping[str, Any]) -> dict[str, Any]:
    return move.to_dict() if isinstance(move, DJMove) else dict(move)


def _candidate_move(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate.get("dj_move"), Mapping):
        return dict(candidate["dj_move"])
    return dict(candidate)


def _deck_operation_types(move: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in move.get("structural_operations", []) if isinstance(move.get("structural_operations"), (list, tuple)) else []:
        values.add(str(item).upper())
    for item in move.get("deck_operations", []) if isinstance(move.get("deck_operations"), (list, tuple)) else []:
        if isinstance(item, Mapping):
            values.add(str(item.get("operation") or item.get("kind") or item.get("type") or "").upper())
    return values


def _loop_start_bar(move: Mapping[str, Any]) -> int | None:
    for item in move.get("deck_operations", []) if isinstance(move.get("deck_operations"), (list, tuple)) else []:
        if not isinstance(item, Mapping):
            continue
        operation = str(item.get("operation") or item.get("kind") or item.get("type") or "").upper()
        if operation != "LOOP_OUTGOING":
            continue
        for key in ("start_bar", "source_bar", "bar"):
            if key in item:
                return max(0, _safe_int(item.get(key)))
    if "LOOP_OUTGOING" in _deck_operation_types(move):
        return max(0, _safe_int(move.get("exit_bar")))
    return None


def _tempo_ratio(move: Mapping[str, Any]) -> float | None:
    native = _safe_float(move.get("incoming_native_bpm"), 0.0)
    matched = _safe_float(move.get("matched_bpm"), 0.0)
    if native <= 1.0 or matched <= 1.0:
        return None
    return matched / native


def _preference_db_for_path(preferences_path: str | Path) -> Path | None:
    source = Path(preferences_path).expanduser().resolve()
    if source.name != "transition-preferences.jsonl":
        return None
    return source.with_name("crate-memory.sqlite3")


@dataclass(frozen=True, slots=True)
class CrateMemoryHint:
    available: bool
    authority: float
    score: float
    adjustment: float
    observations: float
    track_observations: float
    pair_observations: float
    track_id: str
    next_track_id: str
    learned_exit_bar: float | None = None
    learned_entry_bar: float | None = None
    learned_transition_bars: float | None = None
    learned_tempo_percent: float | None = None
    learned_loop_bar: float | None = None
    preferred_techniques: tuple[str, ...] = ()
    preferred_stem_strategies: tuple[str, ...] = ()
    known_failure_reasons: tuple[str, ...] = ()
    vocal_end_bar: int | None = None
    structural_loop_bars: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CRATE_MEMORY_VERSION,
            "available": self.available,
            "authority": round(self.authority, 6),
            "score": round(self.score, 6),
            "adjustment": round(self.adjustment, 6),
            "observations": round(self.observations, 4),
            "track_observations": round(self.track_observations, 4),
            "pair_observations": round(self.pair_observations, 4),
            "track_id": self.track_id,
            "next_track_id": self.next_track_id,
            "learned_exit_bar": None if self.learned_exit_bar is None else round(self.learned_exit_bar, 3),
            "learned_entry_bar": None if self.learned_entry_bar is None else round(self.learned_entry_bar, 3),
            "learned_transition_bars": None if self.learned_transition_bars is None else round(self.learned_transition_bars, 3),
            "learned_tempo_percent": None if self.learned_tempo_percent is None else round(self.learned_tempo_percent, 3),
            "learned_loop_bar": None if self.learned_loop_bar is None else round(self.learned_loop_bar, 3),
            "preferred_techniques": list(self.preferred_techniques),
            "preferred_stem_strategies": list(self.preferred_stem_strategies),
            "known_failure_reasons": list(self.known_failure_reasons),
            "vocal_end_bar": self.vocal_end_bar,
            "structural_loop_bars": list(self.structural_loop_bars),
            "hard_safety_authoritative": True,
        }


class CrateMemoryStore:
    def __init__(self, path: str | Path | None = None, *, workspace: str | Path | None = None) -> None:
        self.path = crate_memory_path(path, workspace=workspace)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS track_memory (
                track_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                observations REAL NOT NULL DEFAULT 0,
                positive REAL NOT NULL DEFAULT 0,
                negative REAL NOT NULL DEFAULT 0,
                exit_bar_sum REAL NOT NULL DEFAULT 0,
                exit_bar_weight REAL NOT NULL DEFAULT 0,
                loop_bar_sum REAL NOT NULL DEFAULT 0,
                loop_bar_weight REAL NOT NULL DEFAULT 0,
                tempo_ratio_sum REAL NOT NULL DEFAULT 0,
                tempo_ratio_sq_sum REAL NOT NULL DEFAULT 0,
                tempo_ratio_weight REAL NOT NULL DEFAULT 0,
                vocal_end_bar INTEGER,
                structural_loop_bars_json TEXT NOT NULL DEFAULT '[]',
                updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pair_memory (
                track_a_id TEXT NOT NULL,
                track_b_id TEXT NOT NULL,
                label_a TEXT NOT NULL,
                label_b TEXT NOT NULL,
                observations REAL NOT NULL DEFAULT 0,
                positive REAL NOT NULL DEFAULT 0,
                negative REAL NOT NULL DEFAULT 0,
                exit_bar_sum REAL NOT NULL DEFAULT 0,
                exit_bar_weight REAL NOT NULL DEFAULT 0,
                entry_bar_sum REAL NOT NULL DEFAULT 0,
                entry_bar_weight REAL NOT NULL DEFAULT 0,
                transition_bars_sum REAL NOT NULL DEFAULT 0,
                transition_bars_weight REAL NOT NULL DEFAULT 0,
                technique_positive_json TEXT NOT NULL DEFAULT '{}',
                technique_negative_json TEXT NOT NULL DEFAULT '{}',
                stem_positive_json TEXT NOT NULL DEFAULT '{}',
                failure_reason_json TEXT NOT NULL DEFAULT '{}',
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (track_a_id, track_b_id)
            );
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                listener TEXT NOT NULL,
                outcome REAL NOT NULL,
                weight REAL NOT NULL,
                track_a_id TEXT NOT NULL,
                track_b_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_events_pair
                ON memory_events(track_a_id, track_b_id, recorded_at_utc);
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(CRATE_MEMORY_SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('runtime_version', ?)",
            (CRATE_MEMORY_VERSION,),
        )

    def observe_track_analysis(self, track: TrackAnalysis) -> None:
        """Persist deterministic record facts once per process.

        These are not preference labels and never add ranking authority.  They
        make stable record knowledge (for example where vocals end) queryable
        beside learned human experience.
        """
        track_id = track_memory_id(track)
        if track_id in _OBSERVED_ANALYSIS_IDS:
            return
        vocal_bars = [int(item.bar_index) for item in track.bar_features if float(item.vocal_probability) >= 0.34]
        vocal_end_bar = max(vocal_bars) if vocal_bars else None
        loop_candidates = [
            int(item.bar_index)
            for item in sorted(
                (
                    item
                    for item in track.exit_candidates
                    if float(item.boundary_strength) >= 0.45
                    and float(item.vocal_probability) <= 0.35
                    and int(item.available_bars) >= 4
                ),
                key=lambda item: (float(item.score), float(item.boundary_strength), -int(item.bar_index)),
                reverse=True,
            )[:8]
        ]
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO track_memory(
                    track_id, label, vocal_end_bar, structural_loop_bars_json, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    label=excluded.label,
                    vocal_end_bar=excluded.vocal_end_bar,
                    structural_loop_bars_json=excluded.structural_loop_bars_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    track_id,
                    track.title,
                    vocal_end_bar,
                    json.dumps(loop_candidates, separators=(",", ":")),
                    now,
                ),
            )
        _OBSERVED_ANALYSIS_IDS.add(track_id)

    def _ensure_track_row(self, connection: sqlite3.Connection, track: str | Path | Mapping[str, Any]) -> str:
        track_id = track_memory_id(track)
        connection.execute(
            """
            INSERT INTO track_memory(track_id, label, updated_at_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET label=excluded.label
            """,
            (track_id, _track_label(track), _utc_now()),
        )
        return track_id

    def record_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        outcome: float,
        weight: float = 1.0,
        reasons: Sequence[str] = (),
        source: str = "human_preference",
        listener: str = "local",
    ) -> None:
        track_a = str(candidate.get("track_a") or "").strip()
        track_b = str(candidate.get("track_b") or "").strip()
        if not track_a or not track_b:
            return
        move = _candidate_move(candidate)
        signed = max(-1.0, min(1.0, float(outcome)))
        magnitude = max(0.05, min(10.0, float(weight)))
        positive = magnitude * max(0.0, signed)
        negative = magnitude * max(0.0, -signed)
        observation = magnitude * max(0.15, abs(signed))
        exit_bar = max(0, _safe_int(move.get("exit_bar", candidate.get("exit_bar", 0))))
        entry_bar = max(0, _safe_int(move.get("entry_bar", candidate.get("entry_bar", 0))))
        transition_bars = max(0, _safe_int(move.get("transition_bars", candidate.get("transition_bars", 0))))
        technique = str(move.get("technique") or candidate.get("technique") or "unknown").strip().lower() or "unknown"
        stem_strategy = str(move.get("stem_strategy") or "master").strip().lower() or "master"
        tempo_ratio = _tempo_ratio(move)
        loop_bar = _loop_start_bar(move)
        now = _utc_now()

        with self._connect() as connection:
            a_id = self._ensure_track_row(connection, track_a)
            b_id = self._ensure_track_row(connection, track_b)
            connection.execute(
                """
                INSERT INTO pair_memory(track_a_id, track_b_id, label_a, label_b, updated_at_utc)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(track_a_id, track_b_id) DO UPDATE SET
                    label_a=excluded.label_a,
                    label_b=excluded.label_b,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (a_id, b_id, _track_label(track_a), _track_label(track_b), now),
            )

            # Record-specific positive experience: an exit that was preferred,
            # a successful outgoing loop and the incoming record's comfortable
            # stretch are useful independently of the exact pair.
            connection.execute(
                """
                UPDATE track_memory SET
                    observations=observations + ?,
                    positive=positive + ?,
                    negative=negative + ?,
                    exit_bar_sum=exit_bar_sum + ?,
                    exit_bar_weight=exit_bar_weight + ?,
                    loop_bar_sum=loop_bar_sum + ?,
                    loop_bar_weight=loop_bar_weight + ?,
                    updated_at_utc=?
                WHERE track_id=?
                """,
                (
                    observation,
                    positive,
                    negative,
                    exit_bar * positive,
                    positive if exit_bar > 0 else 0.0,
                    (loop_bar or 0) * positive,
                    positive if loop_bar is not None else 0.0,
                    now,
                    a_id,
                ),
            )
            connection.execute(
                """
                UPDATE track_memory SET
                    observations=observations + ?,
                    positive=positive + ?,
                    negative=negative + ?,
                    tempo_ratio_sum=tempo_ratio_sum + ?,
                    tempo_ratio_sq_sum=tempo_ratio_sq_sum + ?,
                    tempo_ratio_weight=tempo_ratio_weight + ?,
                    updated_at_utc=?
                WHERE track_id=?
                """,
                (
                    observation,
                    positive,
                    negative,
                    (tempo_ratio or 1.0) * positive,
                    ((tempo_ratio or 1.0) ** 2) * positive,
                    positive if tempo_ratio is not None else 0.0,
                    now,
                    b_id,
                ),
            )

            row = connection.execute(
                "SELECT technique_positive_json, technique_negative_json, stem_positive_json, failure_reason_json FROM pair_memory WHERE track_a_id=? AND track_b_id=?",
                (a_id, b_id),
            ).fetchone()
            technique_positive = str(row["technique_positive_json"] if row is not None else "{}")
            technique_negative = str(row["technique_negative_json"] if row is not None else "{}")
            stem_positive = str(row["stem_positive_json"] if row is not None else "{}")
            failure_reasons = str(row["failure_reason_json"] if row is not None else "{}")
            if positive > 0:
                technique_positive = _inc_counter(technique_positive, technique, positive)
                stem_positive = _inc_counter(stem_positive, stem_strategy, positive)
            if negative > 0:
                technique_negative = _inc_counter(technique_negative, technique, negative)
                for reason in reasons:
                    normalized_reason = str(reason).strip().lower()
                    if normalized_reason and normalized_reason != "great_payoff":
                        failure_reasons = _inc_counter(failure_reasons, normalized_reason, negative)

            connection.execute(
                """
                UPDATE pair_memory SET
                    observations=observations + ?,
                    positive=positive + ?,
                    negative=negative + ?,
                    exit_bar_sum=exit_bar_sum + ?,
                    exit_bar_weight=exit_bar_weight + ?,
                    entry_bar_sum=entry_bar_sum + ?,
                    entry_bar_weight=entry_bar_weight + ?,
                    transition_bars_sum=transition_bars_sum + ?,
                    transition_bars_weight=transition_bars_weight + ?,
                    technique_positive_json=?,
                    technique_negative_json=?,
                    stem_positive_json=?,
                    failure_reason_json=?,
                    updated_at_utc=?
                WHERE track_a_id=? AND track_b_id=?
                """,
                (
                    observation,
                    positive,
                    negative,
                    exit_bar * positive,
                    positive if exit_bar > 0 else 0.0,
                    entry_bar * positive,
                    positive if entry_bar >= 0 else 0.0,
                    transition_bars * positive,
                    positive if transition_bars > 0 else 0.0,
                    technique_positive,
                    technique_negative,
                    stem_positive,
                    failure_reasons,
                    now,
                    a_id,
                    b_id,
                ),
            )
            event_payload = {
                "exit_bar": exit_bar,
                "entry_bar": entry_bar,
                "transition_bars": transition_bars,
                "technique": technique,
                "stem_strategy": stem_strategy,
                "tempo_ratio": tempo_ratio,
                "loop_bar": loop_bar,
                "reasons": [str(item) for item in reasons],
            }
            connection.execute(
                """
                INSERT INTO memory_events(
                    recorded_at_utc, source, listener, outcome, weight,
                    track_a_id, track_b_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    str(source),
                    str(listener or "local"),
                    signed,
                    magnitude,
                    a_id,
                    b_id,
                    json.dumps(event_payload, sort_keys=True, separators=(",", ":")),
                ),
            )

    def record_preference(
        self,
        *,
        candidate_a: Mapping[str, Any],
        candidate_b: Mapping[str, Any],
        choice: str,
        reasons: Sequence[str] = (),
        listener: str = "local",
        weight: float = 1.0,
        source: str = "taste_lab",
    ) -> None:
        normalized = str(choice).strip().lower()
        if normalized == "a":
            outcomes = ((candidate_a, 1.0, reasons), (candidate_b, -0.65, reasons))
        elif normalized == "b":
            outcomes = ((candidate_a, -0.65, reasons), (candidate_b, 1.0, reasons))
        elif normalized == "tie":
            outcomes = ((candidate_a, 0.18, ()), (candidate_b, 0.18, ()))
        elif normalized == "neither":
            outcomes = ((candidate_a, -1.0, reasons), (candidate_b, -1.0, reasons))
        else:
            return
        for candidate, outcome, candidate_reasons in outcomes:
            self.record_candidate(
                candidate,
                outcome=outcome,
                weight=weight,
                reasons=candidate_reasons,
                source=source,
                listener=listener,
            )

    def hint(
        self,
        current_track: TrackAnalysis | str | Path,
        next_track: TrackAnalysis | str | Path,
        move: DJMove | Mapping[str, Any],
        *,
        max_adjustment: float | None = None,
    ) -> CrateMemoryHint:
        current_id = track_memory_id(current_track)
        next_id = track_memory_id(next_track)
        if not self.path.exists():
            return CrateMemoryHint(False, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, current_id, next_id)
        payload = _move_payload(move)
        with self._connect() as connection:
            track_row = connection.execute("SELECT * FROM track_memory WHERE track_id=?", (current_id,)).fetchone()
            incoming_row = connection.execute("SELECT * FROM track_memory WHERE track_id=?", (next_id,)).fetchone()
            pair_row = connection.execute(
                "SELECT * FROM pair_memory WHERE track_a_id=? AND track_b_id=?",
                (current_id, next_id),
            ).fetchone()
        track_obs = _safe_float(track_row["observations"] if track_row is not None else 0.0)
        pair_obs = _safe_float(pair_row["observations"] if pair_row is not None else 0.0)
        observations = pair_obs + 0.35 * track_obs

        learned_exit = None
        learned_loop = None
        vocal_end = None
        structural_loops: tuple[int, ...] = ()
        if track_row is not None:
            if _safe_float(track_row["exit_bar_weight"]) > 0:
                learned_exit = _safe_float(track_row["exit_bar_sum"]) / _safe_float(track_row["exit_bar_weight"])
            if _safe_float(track_row["loop_bar_weight"]) > 0:
                learned_loop = _safe_float(track_row["loop_bar_sum"]) / _safe_float(track_row["loop_bar_weight"])
            vocal_end = _safe_int(track_row["vocal_end_bar"]) if track_row["vocal_end_bar"] is not None else None
            structural_loops = tuple(_json_list(track_row["structural_loop_bars_json"]))

        learned_entry = None
        learned_transition = None
        preferred_techniques: tuple[str, ...] = ()
        preferred_stems: tuple[str, ...] = ()
        failure_reasons: tuple[str, ...] = ()
        pair_sentiment = 0.0
        technique_signal = 0.0
        stem_signal = 0.0
        failure_signal = 0.0
        if pair_row is not None:
            positive = _safe_float(pair_row["positive"])
            negative = _safe_float(pair_row["negative"])
            pair_sentiment = (positive - negative) / max(1.5, positive + negative + 0.75)
            if _safe_float(pair_row["entry_bar_weight"]) > 0:
                learned_entry = _safe_float(pair_row["entry_bar_sum"]) / _safe_float(pair_row["entry_bar_weight"])
            if _safe_float(pair_row["transition_bars_weight"]) > 0:
                learned_transition = _safe_float(pair_row["transition_bars_sum"]) / _safe_float(pair_row["transition_bars_weight"])
            positive_tech = _json_dict(pair_row["technique_positive_json"])
            negative_tech = _json_dict(pair_row["technique_negative_json"])
            stem_counts = _json_dict(pair_row["stem_positive_json"])
            failure_counts = _json_dict(pair_row["failure_reason_json"])
            preferred_techniques = tuple(key for key, _ in sorted(positive_tech.items(), key=lambda item: (-item[1], item[0]))[:4])
            preferred_stems = tuple(key for key, _ in sorted(stem_counts.items(), key=lambda item: (-item[1], item[0]))[:4])
            failure_reasons = tuple(key for key, _ in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:5])
            technique = str(payload.get("technique") or "unknown").lower()
            pos_count = positive_tech.get(technique, 0.0)
            neg_count = negative_tech.get(technique, 0.0)
            if pos_count + neg_count > 0:
                technique_signal = (pos_count - neg_count) / max(1.0, pos_count + neg_count)
            stem = str(payload.get("stem_strategy") or "master").lower()
            total_stem = sum(stem_counts.values())
            if total_stem > 0 and stem_counts.get(stem, 0.0) > 0:
                stem_signal = _clamp(stem_counts.get(stem, 0.0) / total_stem)

            failure_total = sum(failure_counts.values())
            if failure_total > 0:
                conditional = 0.0
                technique_text = technique + " " + str(payload.get("intent") or "") + " " + stem
                if "vocal_clash" in failure_counts and "vocal" in technique_text:
                    conditional += failure_counts["vocal_clash"]
                ratio = _tempo_ratio(payload)
                if "tempo_felt_forced" in failure_counts and ratio is not None and abs(ratio - 1.0) >= 0.025:
                    conditional += failure_counts["tempo_felt_forced"]
                if "wrong_next_track" in failure_counts:
                    conditional += 0.55 * failure_counts["wrong_next_track"]
                if "energy_wrong" in failure_counts:
                    conditional += 0.35 * failure_counts["energy_wrong"]
                failure_signal = -_clamp(conditional / max(1.0, failure_total))

        learned_tempo_pct = None
        tempo_signal = 0.0
        if incoming_row is not None and _safe_float(incoming_row["tempo_ratio_weight"]) > 0:
            weight = _safe_float(incoming_row["tempo_ratio_weight"])
            mean_ratio = _safe_float(incoming_row["tempo_ratio_sum"]) / weight
            learned_tempo_pct = (mean_ratio - 1.0) * 100.0
            ratio = _tempo_ratio(payload)
            if ratio is not None:
                closeness = math.exp(-abs(ratio - mean_ratio) / 0.018)
                tempo_signal = 2.0 * closeness - 1.0

        exit_signal = 0.0
        if learned_exit is not None:
            exit_signal = 2.0 * math.exp(-abs(_safe_int(payload.get("exit_bar")) - learned_exit) / 8.0) - 1.0
        pair_exit_signal = 0.0
        if pair_row is not None and _safe_float(pair_row["exit_bar_weight"]) > 0:
            pair_exit = _safe_float(pair_row["exit_bar_sum"]) / _safe_float(pair_row["exit_bar_weight"])
            pair_exit_signal = 2.0 * math.exp(-abs(_safe_int(payload.get("exit_bar")) - pair_exit) / 7.0) - 1.0
        entry_signal = 0.0
        if learned_entry is not None:
            entry_signal = 2.0 * math.exp(-abs(_safe_int(payload.get("entry_bar")) - learned_entry) / 7.0) - 1.0
        length_signal = 0.0
        if learned_transition is not None:
            length_signal = 2.0 * math.exp(-abs(_safe_int(payload.get("transition_bars")) - learned_transition) / 7.0) - 1.0
        loop_signal = 0.0
        loop_bar = _loop_start_bar(payload)
        if loop_bar is not None and learned_loop is not None:
            loop_signal = 2.0 * math.exp(-abs(loop_bar - learned_loop) / 6.0) - 1.0

        raw = (
            0.32 * pair_sentiment
            + 0.15 * pair_exit_signal
            + 0.10 * entry_signal
            + 0.10 * length_signal
            + 0.09 * technique_signal
            + 0.06 * stem_signal
            + 0.07 * exit_signal
            + 0.06 * tempo_signal
            + 0.03 * loop_signal
            + 0.12 * failure_signal
        )
        raw = max(-1.0, min(1.0, raw))
        authority = _clamp(observations / 6.0)
        max_delta = _maximum_adjustment() if max_adjustment is None else max(0.0, min(0.12, float(max_adjustment)))
        adjustment = max(-max_delta, min(max_delta, raw * authority * max_delta))
        score = _clamp(0.5 + 0.5 * raw)
        return CrateMemoryHint(
            available=observations > 0.0,
            authority=authority,
            score=score,
            adjustment=adjustment,
            observations=observations,
            track_observations=track_obs,
            pair_observations=pair_obs,
            track_id=current_id,
            next_track_id=next_id,
            learned_exit_bar=learned_exit,
            learned_entry_bar=learned_entry,
            learned_transition_bars=learned_transition,
            learned_tempo_percent=learned_tempo_pct,
            learned_loop_bar=learned_loop,
            preferred_techniques=preferred_techniques,
            preferred_stem_strategies=preferred_stems,
            known_failure_reasons=failure_reasons,
            vocal_end_bar=vocal_end,
            structural_loop_bars=structural_loops,
        )

    def purge_listener(self, listener: str) -> int:
        """Remove one listener's evidence and rebuild learned aggregates.

        Account deletion must remove not only raw preference events but also the
        aggregate influence derived from them. Structural analysis facts remain
        because they are properties of local audio, not listener data.
        """
        normalized = str(listener or "").strip()
        if not normalized or not self.path.exists():
            return 0
        with self._connect() as connection:
            deleted = int(
                connection.execute(
                    "SELECT COUNT(*) FROM memory_events WHERE listener=?",
                    (normalized,),
                ).fetchone()[0]
            )
            if deleted <= 0:
                return 0
            connection.execute("DELETE FROM memory_events WHERE listener=?", (normalized,))
            connection.execute(
                """
                UPDATE track_memory SET
                    observations=0, positive=0, negative=0,
                    exit_bar_sum=0, exit_bar_weight=0,
                    loop_bar_sum=0, loop_bar_weight=0,
                    tempo_ratio_sum=0, tempo_ratio_sq_sum=0, tempo_ratio_weight=0
                """
            )
            connection.execute(
                """
                UPDATE pair_memory SET
                    observations=0, positive=0, negative=0,
                    exit_bar_sum=0, exit_bar_weight=0,
                    entry_bar_sum=0, entry_bar_weight=0,
                    transition_bars_sum=0, transition_bars_weight=0,
                    technique_positive_json='{}', technique_negative_json='{}',
                    stem_positive_json='{}', failure_reason_json='{}'
                """
            )
            events = connection.execute("SELECT * FROM memory_events ORDER BY id").fetchall()
            now = _utc_now()
            for event in events:
                try:
                    payload = json.loads(str(event["payload_json"]))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                signed = max(-1.0, min(1.0, _safe_float(event["outcome"])))
                magnitude = max(0.05, min(10.0, _safe_float(event["weight"], 1.0)))
                positive = magnitude * max(0.0, signed)
                negative = magnitude * max(0.0, -signed)
                observation = magnitude * max(0.15, abs(signed))
                a_id = str(event["track_a_id"])
                b_id = str(event["track_b_id"])
                exit_bar = max(0, _safe_int(payload.get("exit_bar")))
                entry_bar = max(0, _safe_int(payload.get("entry_bar")))
                transition_bars = max(0, _safe_int(payload.get("transition_bars")))
                technique = str(payload.get("technique") or "unknown").lower()
                stem_strategy = str(payload.get("stem_strategy") or "master").lower()
                tempo_ratio = _safe_float(payload.get("tempo_ratio"), 0.0) or None
                loop_bar = payload.get("loop_bar")
                loop_value = _safe_int(loop_bar) if loop_bar is not None else None
                reasons = payload.get("reasons", []) if isinstance(payload.get("reasons"), list) else []
                connection.execute(
                    """
                    UPDATE track_memory SET
                        observations=observations + ?, positive=positive + ?, negative=negative + ?,
                        exit_bar_sum=exit_bar_sum + ?, exit_bar_weight=exit_bar_weight + ?,
                        loop_bar_sum=loop_bar_sum + ?, loop_bar_weight=loop_bar_weight + ?, updated_at_utc=?
                    WHERE track_id=?
                    """,
                    (
                        observation, positive, negative,
                        exit_bar * positive, positive if exit_bar > 0 else 0.0,
                        (loop_value or 0) * positive, positive if loop_value is not None else 0.0,
                        now, a_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE track_memory SET
                        observations=observations + ?, positive=positive + ?, negative=negative + ?,
                        tempo_ratio_sum=tempo_ratio_sum + ?, tempo_ratio_sq_sum=tempo_ratio_sq_sum + ?,
                        tempo_ratio_weight=tempo_ratio_weight + ?, updated_at_utc=?
                    WHERE track_id=?
                    """,
                    (
                        observation, positive, negative,
                        (tempo_ratio or 1.0) * positive,
                        ((tempo_ratio or 1.0) ** 2) * positive,
                        positive if tempo_ratio is not None else 0.0,
                        now, b_id,
                    ),
                )
                row = connection.execute(
                    "SELECT technique_positive_json, technique_negative_json, stem_positive_json, failure_reason_json FROM pair_memory WHERE track_a_id=? AND track_b_id=?",
                    (a_id, b_id),
                ).fetchone()
                if row is None:
                    continue
                technique_positive = str(row["technique_positive_json"])
                technique_negative = str(row["technique_negative_json"])
                stem_positive = str(row["stem_positive_json"])
                failure_reasons = str(row["failure_reason_json"])
                if positive > 0:
                    technique_positive = _inc_counter(technique_positive, technique, positive)
                    stem_positive = _inc_counter(stem_positive, stem_strategy, positive)
                if negative > 0:
                    technique_negative = _inc_counter(technique_negative, technique, negative)
                    for reason in reasons:
                        normalized_reason = str(reason).strip().lower()
                        if normalized_reason and normalized_reason != "great_payoff":
                            failure_reasons = _inc_counter(failure_reasons, normalized_reason, negative)
                connection.execute(
                    """
                    UPDATE pair_memory SET
                        observations=observations + ?, positive=positive + ?, negative=negative + ?,
                        exit_bar_sum=exit_bar_sum + ?, exit_bar_weight=exit_bar_weight + ?,
                        entry_bar_sum=entry_bar_sum + ?, entry_bar_weight=entry_bar_weight + ?,
                        transition_bars_sum=transition_bars_sum + ?, transition_bars_weight=transition_bars_weight + ?,
                        technique_positive_json=?, technique_negative_json=?, stem_positive_json=?,
                        failure_reason_json=?, updated_at_utc=?
                    WHERE track_a_id=? AND track_b_id=?
                    """,
                    (
                        observation, positive, negative,
                        exit_bar * positive, positive if exit_bar > 0 else 0.0,
                        entry_bar * positive, positive,
                        transition_bars * positive, positive if transition_bars > 0 else 0.0,
                        technique_positive, technique_negative, stem_positive, failure_reasons, now,
                        a_id, b_id,
                    ),
                )
            return deleted

    def status(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": CRATE_MEMORY_VERSION,
                "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
                "path": str(self.path),
                "available": False,
                "track_records": 0,
                "pair_records": 0,
                "preference_events": 0,
                "enabled": _memory_enabled(),
                "maximum_live_adjustment": _maximum_adjustment(),
                "hard_audio_safety_authoritative": True,
            }
        with self._connect() as connection:
            tracks = int(connection.execute("SELECT COUNT(*) FROM track_memory").fetchone()[0])
            pairs = int(connection.execute("SELECT COUNT(*) FROM pair_memory WHERE observations > 0").fetchone()[0])
            events = int(connection.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0])
            positive_pairs = int(connection.execute("SELECT COUNT(*) FROM pair_memory WHERE positive > negative AND observations > 0").fetchone()[0])
            failure_pairs = int(connection.execute("SELECT COUNT(*) FROM pair_memory WHERE negative > 0").fetchone()[0])
        return {
            "version": CRATE_MEMORY_VERSION,
            "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
            "path": str(self.path),
            "available": True,
            "track_records": tracks,
            "pair_records": pairs,
            "preference_events": events,
            "positive_pair_memories": positive_pairs,
            "failure_pair_memories": failure_pairs,
            "enabled": _memory_enabled(),
            "maximum_live_adjustment": _maximum_adjustment(),
            "directional_pairs": True,
            "structural_track_facts": True,
            "human_preference_authority": True,
            "self_training_from_planner_choices": False,
            "hard_audio_safety_authoritative": True,
        }


def record_preference_to_crate_memory(
    *,
    candidate_a: Mapping[str, Any],
    candidate_b: Mapping[str, Any],
    choice: str,
    listener: str,
    context: Mapping[str, Any] | None,
    weight: float,
    preferences_path: str | Path,
) -> Path | None:
    target = _preference_db_for_path(preferences_path)
    if target is None:
        return None
    reasons_raw = context.get("reasons", []) if isinstance(context, Mapping) else []
    reasons = [str(item) for item in reasons_raw] if isinstance(reasons_raw, (list, tuple)) else []
    store = CrateMemoryStore(target)
    store.record_preference(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        choice=choice,
        reasons=reasons,
        listener=listener,
        weight=weight,
        source="taste_lab_preference",
    )
    return target


def crate_memory_hint(
    current_track: TrackAnalysis,
    next_track: TrackAnalysis,
    move: DJMove | Mapping[str, Any],
    *,
    workspace: str | Path | None = None,
) -> dict[str, Any]:
    store = CrateMemoryStore(workspace=workspace)
    try:
        if not _memory_enabled():
            return {
                "version": CRATE_MEMORY_VERSION,
                "available": False,
                "enabled": False,
                "authority": 0.0,
                "score": 0.5,
                "adjustment": 0.0,
                "hard_safety_authoritative": True,
            }
        if not store.path.exists():
            return CrateMemoryHint(
                False, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
                track_memory_id(current_track), track_memory_id(next_track),
            ).to_dict()
        store.observe_track_analysis(current_track)
        store.observe_track_analysis(next_track)
        return store.hint(current_track, next_track, move).to_dict()
    except (OSError, sqlite3.Error, ValueError) as exc:
        return {
            "version": CRATE_MEMORY_VERSION,
            "available": False,
            "authority": 0.0,
            "score": 0.5,
            "adjustment": 0.0,
            "reason": f"crate memory unavailable: {type(exc).__name__}: {exc}",
            "hard_safety_authoritative": True,
        }


def crate_memory_status(*, workspace: str | Path | None = None) -> dict[str, Any]:
    try:
        return CrateMemoryStore(workspace=workspace).status()
    except (OSError, sqlite3.Error, ValueError) as exc:
        return {
            "version": CRATE_MEMORY_VERSION,
            "schema_version": CRATE_MEMORY_SCHEMA_VERSION,
            "available": False,
            "reason": f"crate memory unavailable: {type(exc).__name__}: {exc}",
            "hard_audio_safety_authoritative": True,
        }
