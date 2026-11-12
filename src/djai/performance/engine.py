from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import butter, sosfilt

from .program import TransitionProgram, transition_program_from_dict


@dataclass(frozen=True, slots=True)
class ProgramExecutionResult:
    audio: np.ndarray
    source_mode: str
    buses_used: tuple[str, ...]


def _curve_array(points: object, length: int) -> np.ndarray:
    values = list(points or [])
    if not values:
        raise RuntimeError("Executable performance curve is empty.")
    positions = np.asarray([float(item.position) for item in values], dtype=np.float64)
    gains = np.asarray([float(item.value) for item in values], dtype=np.float64)
    x = np.linspace(0.0, 1.0, max(1, length), dtype=np.float64)
    return np.interp(x, positions, gains).astype(np.float32)


def _split_low_high(y: np.ndarray, sample_rate: int, cutoff_hz: float = 180.0) -> tuple[np.ndarray, np.ndarray]:
    cutoff = min(float(cutoff_hz), sample_rate * 0.45)
    sos = butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    low = sosfilt(sos, y, axis=-1).astype(np.float32)
    high = (y - low).astype(np.float32)
    return low, high


def _soft_limit(y: np.ndarray, ceiling: float = 0.985) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32)
    if values.size == 0:
        return values
    peak = float(np.max(np.abs(values)))
    if peak <= ceiling:
        return values
    knee = ceiling * 0.90
    headroom = max(ceiling - knee, 1e-6)
    magnitude = np.abs(values)
    compressed = np.array(magnitude, copy=True)
    mask = magnitude > knee
    compressed[mask] = knee + headroom * np.tanh((magnitude[mask] - knee) / headroom)
    return np.clip(np.copysign(compressed, values), -ceiling, ceiling).astype(np.float32)


def _apply_processor(y: np.ndarray, kind: str, parameters: Mapping[str, float], *, sample_rate: int, bpm: float) -> np.ndarray:
    if kind == "reverse":
        return np.flip(y, axis=-1).copy()
    if kind == "echo":
        result = np.array(y, copy=True, dtype=np.float32)
        beat_seconds = 60.0 / max(float(bpm), 1.0)
        for beats, gain in ((0.5, 0.34), (1.0, 0.22), (1.5, 0.13)):
            delay = int(round(beats * beat_seconds * sample_rate))
            if 0 < delay < y.shape[-1]:
                result[:, delay:] += gain * y[:, :-delay]
        return result
    if kind == "loop_tail":
        beats = max(0.125, float(parameters.get("beats", 1.0)))
        beat_samples = max(64, int(round(beats * sample_rate * 60.0 / max(float(bpm), 1.0))))
        source_start = max(0, y.shape[-1] - beat_samples)
        loop = y[:, source_start:] if y[:, source_start:].shape[-1] else y
        repeats = int(np.ceil(y.shape[-1] / max(loop.shape[-1], 1)))
        return np.tile(loop, (1, repeats))[:, : y.shape[-1]].astype(np.float32)
    raise RuntimeError(f"Unknown executable processor: {kind}")




def _loop_prefix_window(
    y: np.ndarray,
    *,
    sample_rate: int,
    bpm: float,
    beats: float,
    hold_fraction: float,
    source_fraction: float = 1.0,
) -> np.ndarray:
    if y.size == 0:
        return np.asarray(y, dtype=np.float32)
    length = y.shape[-1]
    beat_samples = max(64, int(round(max(0.125, beats) * sample_rate * 60.0 / max(float(bpm), 1.0))))
    beat_samples = min(length, beat_samples)
    source_fraction = max(0.02, min(1.0, float(source_fraction)))
    source_length = max(beat_samples, min(length, int(round(length * source_fraction))))
    source = np.asarray(y[:, :source_length], dtype=np.float32)
    loop = source[:, : min(beat_samples, source_length)]

    # When the committed incoming source is shorter than the audible transition,
    # manufacture runway musically: repeat the instrumental prefix first, then
    # release into the complete planned source so the transition still lands at
    # the exact committed B cursor. Zero padding is never used as audible runway.
    if source_length < length:
        prefix_length = length - source_length
        repeats = int(np.ceil(prefix_length / max(loop.shape[-1], 1)))
        result = np.zeros_like(y, dtype=np.float32)
        result[:, :prefix_length] = np.tile(loop, (1, repeats))[:, :prefix_length]
        result[:, prefix_length:] = source[:, -source_length:]
        fade = min(loop.shape[-1], max(64, int(0.06 * sample_rate)), max(1, source_length // 4))
        if fade > 1 and prefix_length >= fade:
            x = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            left = result[:, prefix_length - fade : prefix_length]
            right = source[:, :fade]
            result[:, prefix_length - fade : prefix_length] = left * (1.0 - x) + right * x
        return result.astype(np.float32)

    hold_end = max(beat_samples, min(length, int(round(length * max(0.0, min(0.92, hold_fraction))))))
    repeats = int(np.ceil(hold_end / max(loop.shape[-1], 1)))
    result = np.array(y, copy=True, dtype=np.float32)
    result[:, :hold_end] = np.tile(loop, (1, repeats))[:, :hold_end]
    fade = min(beat_samples, max(64, length - hold_end), max(64, int(0.08 * sample_rate)))
    if hold_end < length and fade > 0:
        start = max(0, hold_end - fade)
        end = min(length, hold_end + fade)
        width = end - start
        if width > 1:
            x = np.linspace(0.0, 1.0, width, dtype=np.float32)
            result[:, start:end] = result[:, start:end] * (1.0 - x) + y[:, start:end] * x
    return result.astype(np.float32)


def _apply_transition_deck_operations(
    program: TransitionProgram,
    prepared: dict[str, np.ndarray],
    *,
    sample_rate: int,
    bpm: float,
) -> dict[str, np.ndarray]:
    resolved = dict(prepared)
    for operation in program.deck_operations:
        if str(operation.phase) != "transition":
            continue
        kind = str(operation.kind)
        if kind not in {"LOOP_INCOMING", "EXTEND_INTRO"}:
            continue
        if operation.deck != "b":
            raise RuntimeError(f"{kind} is only valid on the incoming deck during a transition.")
        params = operation.parameters
        beats = max(0.125, float(params.get("loop_beats", 4.0)))
        hold_fraction = max(0.0, min(0.92, float(params.get("hold_fraction", 0.55))))
        source_fraction = max(0.02, min(1.0, float(params.get("source_fraction", 1.0))))
        buses = ["deck_b"] if "deck_b" in resolved else sorted(key for key in resolved if key.startswith("b:"))
        if not buses:
            raise RuntimeError(f"{kind} requires an incoming source bus.")
        for bus in buses:
            resolved[bus] = _loop_prefix_window(
                resolved[bus],
                sample_rate=sample_rate,
                bpm=bpm,
                beats=beats,
                hold_fraction=hold_fraction,
                source_fraction=source_fraction,
            )
    return resolved

def _prepare_buses(program: TransitionProgram, sources: Mapping[str, np.ndarray], *, sample_rate: int, bpm: float) -> dict[str, np.ndarray]:
    prepared = {key: np.asarray(value, dtype=np.float32) for key, value in sources.items()}
    lengths = {value.shape[-1] for value in prepared.values()}
    channels = {value.shape[0] for value in prepared.values() if value.ndim == 2}
    if len(lengths) != 1 or len(channels) != 1:
        raise RuntimeError("All executable source buses must have identical channel and frame dimensions.")
    for processor in program.processors:
        if processor.bus not in prepared:
            raise RuntimeError(f"Performance processor references unavailable source bus {processor.bus!r}.")
        prepared[processor.bus] = _apply_processor(
            prepared[processor.bus],
            processor.kind,
            processor.parameters,
            sample_rate=sample_rate,
            bpm=bpm,
        )
    return prepared


def _execute_master(program: TransitionProgram, sources: Mapping[str, np.ndarray], *, sample_rate: int) -> tuple[np.ndarray, tuple[str, ...]]:
    required = ("deck_a", "deck_b")
    for bus in required:
        if bus not in sources:
            raise RuntimeError(f"Master performance program requires source bus {bus!r}.")
    length = sources["deck_a"].shape[-1]
    mixed = np.zeros_like(sources["deck_a"])
    used: list[str] = []
    for deck in ("deck_a", "deck_b"):
        audio = sources[deck]
        gain = program.curves.get(f"{deck}_gain")
        low_gain = program.curves.get(f"{deck}_low_gain")
        high_gain = program.curves.get(f"{deck}_high_gain")
        if low_gain is not None or high_gain is not None:
            low, high = _split_low_high(audio, sample_rate)
            low_curve = _curve_array(low_gain, length) if low_gain is not None else np.ones(length, dtype=np.float32)
            high_curve = _curve_array(high_gain, length) if high_gain is not None else np.ones(length, dtype=np.float32)
            contribution = low * low_curve + high * high_curve
            if gain is not None:
                contribution *= _curve_array(gain, length)
        else:
            if gain is None:
                raise RuntimeError(f"Performance program has no gain automation for {deck!r}.")
            contribution = audio * _curve_array(gain, length)
        mixed += contribution
        used.append(deck)
    return mixed, tuple(used)


def _execute_stems(program: TransitionProgram, sources: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    roles = ("drums", "bass", "vocals", "other")
    first = sources.get("a:drums")
    if first is None:
        raise RuntimeError("Stem performance program requires complete A/B stem buses.")
    length = first.shape[-1]
    mixed = np.zeros_like(first)
    used: list[str] = []
    for deck, curves in (("a", program.stem_curves_a), ("b", program.stem_curves_b)):
        for role in roles:
            bus = f"{deck}:{role}"
            if bus not in sources:
                raise RuntimeError(f"Stem performance program requires source bus {bus!r}.")
            points = curves.get(role)
            if not points:
                raise RuntimeError(f"Stem performance program is missing automation for {bus!r}.")
            mixed += sources[bus] * _curve_array(points, length)
            used.append(bus)
    return mixed, tuple(used)


def _execute_layers(program: TransitionProgram, sources: Mapping[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    first = sources.get("a:master")
    if first is None:
        first = sources.get("b:master")
    if first is None:
        raise RuntimeError("Layer performance program requires at least one full-master source bus.")
    length = first.shape[-1]
    mixed = np.zeros_like(first)
    used: list[str] = []
    for deck, curves in (("a", program.layer_curves_a), ("b", program.layer_curves_b)):
        for role, points in curves.items():
            bus = f"{deck}:{role}"
            if bus not in sources:
                raise RuntimeError(f"Layer performance program requires source bus {bus!r}.")
            mixed += sources[bus] * _curve_array(points, length)
            used.append(bus)
    if not used:
        raise RuntimeError("Layer performance program contains no executable source automation.")
    return mixed, tuple(used)


def execute_transition_program(
    program: TransitionProgram | dict[str, object],
    sources: Mapping[str, np.ndarray],
    *,
    sample_rate: int,
    bpm: float,
) -> ProgramExecutionResult:
    """Execute an immutable TransitionProgram without inspecting technique."""

    resolved = transition_program_from_dict(dict(program)) if isinstance(program, dict) else program
    prepared = _prepare_buses(resolved, sources, sample_rate=sample_rate, bpm=bpm)
    prepared = _apply_transition_deck_operations(resolved, prepared, sample_rate=sample_rate, bpm=bpm)
    if resolved.source_mode == "master":
        audio, used = _execute_master(resolved, prepared, sample_rate=sample_rate)
    elif resolved.source_mode == "stems":
        audio, used = _execute_stems(resolved, prepared)
    elif resolved.source_mode == "layers":
        audio, used = _execute_layers(resolved, prepared)
    else:  # guarded by validation
        raise RuntimeError(f"Unsupported executable source mode {resolved.source_mode!r}.")
    ceiling = max(
        0.50,
        min(
            0.985,
            float(dict(resolved.annotations or {}).get("output_ceiling", 0.985)),
        ),
    )
    return ProgramExecutionResult(
        audio=_soft_limit(audio, ceiling=ceiling),
        source_mode=resolved.source_mode,
        buses_used=used,
    )
