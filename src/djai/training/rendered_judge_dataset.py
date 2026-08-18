from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from ..intelligence.neural_policy import rendered_judge_payload_features
from ..judge.musical import measure_musical_execution
from ..measurement import measure_transition_audio
from ..serialization import transition_plan_from_dict
from .synthetic_failures import corrupt_audio

SYNTHETIC_RENDERED_SCHEMA_VERSION = "1.1"
DEFAULT_SYNTHETIC_KINDS = (
    "clipping",
    "silence_gap",
    "gain_jump",
    "phase_inversion",
    "transient_double",
    "bass_overload",
    "timing_shift",
)
DEFAULT_SYNTHETIC_SEVERITIES = (0.35, 0.72)


def _enabled() -> bool:
    return os.environ.get("DJAI_TASTE_SYNTHETIC_FAILURES", "1").strip().lower() not in {
        "0", "false", "no", "off"
    }


def _hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]




def _content_track_id(workspace: Path, value: str, cache: dict[str, str]) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    key = str(path)
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    if path.exists() and path.is_file():
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        value_hash = digest.hexdigest()
    else:
        value_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    track_id = f"track-{value_hash[:24]}"
    cache[key] = track_id
    return track_id


def _pair_group(workspace: Path, track_a: str, track_b: str, cache: dict[str, str]) -> str:
    left = _content_track_id(workspace, track_a, cache)
    right = _content_track_id(workspace, track_b, cache)
    return "pair-" + _hash(f"{left}|{right}", 18)

def _rights_cleared() -> bool:
    return os.environ.get("SAUND_TRAINING_SOURCE_RIGHTS_CONFIRMED", "0").strip() == "1"


def _iter_manifests(workspace: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    root = workspace / "output" / "listening-lab"
    if not root.exists():
        return []
    output: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            output.append((path, payload))
    return output


def _as_channel_first(values: np.ndarray) -> np.ndarray:
    audio = np.asarray(values, dtype=np.float32)
    if audio.ndim == 1:
        return audio[None, :]
    if audio.ndim == 2:
        return audio.T
    return audio.reshape(1, -1)


def _measurement_payload(plan_payload: dict[str, Any], measurement: Any, musical: Any) -> dict[str, Any]:
    measurement.rhythm_coherence = musical.rhythm_coherence
    measurement.transient_clarity = musical.transient_clarity
    measurement.groove_continuity = musical.groove_continuity
    measurement.harmonic_continuity = musical.harmonic_continuity
    measurement.energy_intent_fit = musical.energy_intent_fit
    measurement.drop_payoff = musical.drop_payoff
    measurement.musical_overall = musical.overall
    measurement.warnings.extend(musical.warnings)
    payload = json.loads(json.dumps(plan_payload, ensure_ascii=False))
    payload["measurement"] = measurement.to_dict()
    payload["measured_score"] = float(measurement.overall)
    # A corruption changes the waveform, so reusing the clean candidate's
    # rendered embedding would leak the answer. Synthetic rows deliberately
    # train from actual corruption measurements unless a fresh embedding is
    # explicitly computed in a future dataset revision.
    payload["rendered_transition_embedding"] = []
    return payload


def build_synthetic_rendered_judge_rows(
    workspace: str | Path,
    *,
    kinds: tuple[str, ...] = DEFAULT_SYNTHETIC_KINDS,
    severities: tuple[float, ...] = DEFAULT_SYNTHETIC_SEVERITIES,
    maximum_candidates: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create known-bad rendered-judge examples from real blind-session audio.

    Corruptions are measured in memory. No damaged WAVs are persisted by this
    dataset builder, which keeps the loop local, bounded and storage-efficient.
    """

    if not _enabled():
        return [], {"enabled": False, "examples": 0, "candidates": 0, "skipped": 0}
    if maximum_candidates is None:
        try:
            maximum_candidates = max(1, min(4096, int(os.environ.get("DJAI_TASTE_SYNTHETIC_MAX_CANDIDATES", "128"))))
        except ValueError:
            maximum_candidates = 128
    root = Path(workspace).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    candidates_seen = 0
    skipped = 0
    rights_cleared = _rights_cleared()
    track_id_cache: dict[str, str] = {}

    for manifest_path, manifest in _iter_manifests(root):
        preview = manifest.get("preview") if isinstance(manifest.get("preview"), dict) else {}
        default_pre = float(preview.get("pre_roll_seconds", 6.0) or 6.0)
        session_id = str(manifest.get("session_id") or manifest_path.parent.name)
        for item in manifest.get("candidates", []) if isinstance(manifest.get("candidates"), list) else []:
            if maximum_candidates is not None and candidates_seen >= maximum_candidates:
                break
            if not isinstance(item, dict):
                continue
            plan_payload = item.get("plan") if isinstance(item.get("plan"), dict) else None
            full_payload = item.get("plan_full") if isinstance(item.get("plan_full"), dict) else None
            audio_value = item.get("audio_path")
            if plan_payload is None or full_payload is None or not audio_value:
                skipped += 1
                continue
            try:
                plan = transition_plan_from_dict(full_payload)
                audio_path = Path(str(audio_value)).expanduser()
                if not audio_path.is_absolute():
                    audio_path = root / audio_path
                audio, sample_rate = sf.read(audio_path.resolve(), always_2d=True, dtype="float32")
            except (OSError, RuntimeError, TypeError, ValueError):
                skipped += 1
                continue

            pre_roll = float(item.get("pre_roll_seconds", default_pre) or default_pre)
            pre_frames = max(1, int(round(sample_rate * pre_roll)))
            transition_frames = max(1, int(round(sample_rate * float(plan.transition_seconds))))
            if audio.shape[0] < pre_frames + transition_frames + 16:
                skipped += 1
                continue
            pre = _as_channel_first(audio[:pre_frames])
            transition_clean = np.asarray(audio[pre_frames : pre_frames + transition_frames], dtype=np.float32)
            post = _as_channel_first(audio[pre_frames + transition_frames :])
            if post.shape[-1] < 16:
                skipped += 1
                continue

            candidates_seen += 1
            group = _pair_group(
                root,
                str(plan_payload.get("track_a") or plan.track_a),
                str(plan_payload.get("track_b") or plan.track_b),
                track_id_cache,
            )
            candidate_id = str(plan_payload.get("execution_signature") or _hash(json.dumps(plan_payload, sort_keys=True)))
            for kind in kinds:
                for severity in severities:
                    try:
                        damaged_frames = corrupt_audio(transition_clean, sample_rate, kind, severity)
                        damaged = _as_channel_first(damaged_frames)
                        measurement = measure_transition_audio(
                            pre=pre,
                            transition=damaged,
                            post=post,
                            sample_rate=sample_rate,
                            plan=plan,
                        )
                        musical = measure_musical_execution(
                            pre=pre,
                            transition=damaged,
                            post=post,
                            sample_rate=sample_rate,
                            plan=plan,
                        )
                    except (RuntimeError, TypeError, ValueError):
                        skipped += 1
                        continue
                    synthetic_payload = _measurement_payload(plan_payload, measurement, musical)
                    features = rendered_judge_payload_features(synthetic_payload)
                    hard_failed = bool(measurement.hard_failures)
                    severity_value = float(np.clip(severity, 0.0, 1.0))
                    target = 0.0 if hard_failed else max(0.02, 0.30 - 0.22 * severity_value)
                    weight = 0.65 + 0.85 * severity_value + (0.30 if hard_failed else 0.0)
                    row_id = _hash(f"{session_id}|{candidate_id}|{kind}|{severity_value:.4f}", 28)
                    rows.append(
                        {
                            "schema_version": SYNTHETIC_RENDERED_SCHEMA_VERSION,
                            "id": f"synthetic-{row_id}",
                            "group": group,
                            "context_id": session_id,
                            "target": round(target, 6),
                            "weight": round(weight, 6),
                            "rights_cleared": rights_cleared,
                            "label_source": "synthetic_technical_failure",
                            "synthetic_failure": {
                                "kind": kind,
                                "severity": round(severity_value, 4),
                                "hard_failure": hard_failed,
                                "measurement_overall": round(float(measurement.overall), 6),
                            },
                            "render_embedding_present": False,
                            "features": features,
                        }
                    )
        if maximum_candidates is not None and candidates_seen >= maximum_candidates:
            break

    return rows, {
        "enabled": True,
        "schema_version": SYNTHETIC_RENDERED_SCHEMA_VERSION,
        "candidates": candidates_seen,
        "examples": len(rows),
        "skipped": skipped,
        "kinds": list(kinds),
        "severities": [float(value) for value in severities],
        "raw_audio_persisted": False,
        "rights_cleared": rights_cleared,
    }
