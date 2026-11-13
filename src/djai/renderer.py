from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from .audio import load_audio, pad_to_length, rms, write_audio
from .decision import validate_execution_plan
from .execution import TransitionRenderContext
from .models import TransitionPlan
from .performance.engine import execute_transition_program
from .performance.program import compile_transition_program, transition_program_from_dict

DEFAULT_RENDER_SAMPLE_RATE = 44_100


@dataclass(slots=True)
class RenderedTransitionAudio:
    pre: np.ndarray
    transition: np.ndarray
    post: np.ndarray
    preview: np.ndarray
    gain_b: float
    sample_rate: int


def _extract_with_padding(
    y: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    output_length = max(0, int(round(duration_seconds * sample_rate)))
    output = np.zeros((y.shape[0], output_length), dtype=np.float32)
    requested_start = int(round(start_seconds * sample_rate))
    source_start = max(0, requested_start)
    destination_start = max(0, -requested_start)
    available = max(0, min(y.shape[-1] - source_start, output_length - destination_start))
    if available > 0:
        output[:, destination_start : destination_start + available] = y[
            :, source_start : source_start + available
        ]
    return output


def _smoothstep(length: int, start: float, end: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, max(length, 1), dtype=np.float64)
    denominator = max(end - start, 1e-9)
    t = np.clip((x - start) / denominator, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _equal_power(length: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi / 2.0, max(length, 1), dtype=np.float64)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def _split_low_high(
    y: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    cutoff = min(cutoff_hz, sample_rate * 0.45)
    sos = butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    low = sosfilt(sos, y, axis=-1).astype(np.float32)
    high = (y - low).astype(np.float32)
    return low, high


def _time_stretch(y: np.ndarray, rate: float, target_length: int) -> np.ndarray:
    if abs(rate - 1.0) < 0.002:
        return pad_to_length(y, target_length).astype(np.float32)
    stretched = librosa.effects.time_stretch(y, rate=rate)
    return pad_to_length(np.asarray(stretched, dtype=np.float32), target_length)


def _smooth_rate(start_rate: float, end_rate: float, position: np.ndarray | float) -> np.ndarray:
    x = np.asarray(position, dtype=np.float64)
    t = np.clip(x, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return start_rate + (end_rate - start_rate) * smooth


def _rate_curve_integral(start_rate: float, end_rate: float, fraction: float) -> float:
    # Integral of start + (end-start) * (3x^2 - 2x^3) from 0..fraction.
    x = float(np.clip(fraction, 0.0, 1.0))
    return start_rate * x + (end_rate - start_rate) * (x**3 - 0.5 * x**4)


def _time_stretch_rate_curve(
    y: np.ndarray,
    *,
    start_rate: float,
    end_rate: float,
    target_length: int,
) -> np.ndarray:
    """Pitch-preserving continuous deck tempo ride using a variable phase vocoder.

    The output clock is fixed. Each output STFT frame advances through the
    source according to a smooth rate curve, which is the offline equivalent
    of riding a deck tempo fader after the incoming deck takes ownership.
    """
    if target_length <= 0:
        return np.zeros((y.shape[0], 0), dtype=np.float32)
    if abs(start_rate - end_rate) < 0.001:
        return _time_stretch(y, 0.5 * (start_rate + end_rate), target_length)
    n_fft = 2048
    hop_length = 512
    source = np.asarray(y, dtype=np.float32)
    if source.shape[-1] < n_fft + hop_length * 2:
        average = 0.5 * (start_rate + end_rate)
        return _time_stretch(source, average, target_length)
    spectrum = librosa.stft(source, n_fft=n_fft, hop_length=hop_length)
    output_frames = max(2, 1 + int(np.ceil(target_length / hop_length)))
    positions = np.linspace(0.0, 1.0, output_frames, dtype=np.float64)
    rates = _smooth_rate(start_rate, end_rate, positions)
    time_steps = np.zeros(output_frames, dtype=np.float64)
    if output_frames > 1:
        increments = 0.5 * (rates[:-1] + rates[1:])
        time_steps[1:] = np.cumsum(increments)
    maximum_step = max(0.0, spectrum.shape[-1] - 2.001)
    if time_steps[-1] > maximum_step and time_steps[-1] > 0.0:
        time_steps *= maximum_step / time_steps[-1]

    shape = list(spectrum.shape)
    shape[-1] = output_frames
    stretched = np.zeros(shape, dtype=spectrum.dtype)
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(spectrum[..., 0])
    padding = [(0, 0) for _ in spectrum.shape]
    padding[-1] = (0, 2)
    padded = np.pad(spectrum, padding, mode="constant")
    for index, step in enumerate(time_steps):
        base = int(step)
        columns = padded[..., base : base + 2]
        alpha = float(np.mod(step, 1.0))
        magnitude = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])
        stretched[..., index] = magnitude * np.exp(1j * phase_acc)
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance
        dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase
    output = librosa.istft(stretched, hop_length=hop_length, length=target_length)
    return np.asarray(output, dtype=np.float32)


def _time_stretch_phase_warp(
    y: np.ndarray,
    *,
    sample_rate: int,
    base_rate: float,
    target_length: int,
    correction_points: list[dict[str, float]],
) -> np.ndarray:
    """Pitch-preserving time warp around a base rate using bounded source offsets."""
    if target_length <= 0:
        return np.zeros((y.shape[0], 0), dtype=np.float32)
    if not correction_points:
        return _time_stretch(y, base_rate, target_length)
    n_fft = 2048
    hop_length = 512
    source = np.asarray(y, dtype=np.float32)
    if source.shape[-1] < n_fft + hop_length * 2:
        return _time_stretch(source, base_rate, target_length)

    spectrum = librosa.stft(source, n_fft=n_fft, hop_length=hop_length)
    output_frames = max(2, 1 + int(np.ceil(target_length / hop_length)))
    fractions = np.linspace(0.0, 1.0, output_frames, dtype=np.float64)
    point_x = np.asarray([float(p.get("fraction", 0.0)) for p in correction_points], dtype=np.float64)
    point_y = np.asarray([float(p.get("source_offset_seconds", 0.0)) for p in correction_points], dtype=np.float64)
    offsets_seconds = np.interp(fractions, point_x, point_y)
    base_steps = np.arange(output_frames, dtype=np.float64) * max(float(base_rate), 0.01)
    correction_steps = offsets_seconds * float(sample_rate) / hop_length
    time_steps = np.maximum.accumulate(base_steps + correction_steps)
    maximum_step = max(0.0, spectrum.shape[-1] - 2.001)
    time_steps = np.clip(time_steps, 0.0, maximum_step)

    shape = list(spectrum.shape)
    shape[-1] = output_frames
    stretched = np.zeros(shape, dtype=spectrum.dtype)
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(spectrum[..., 0])
    padding = [(0, 0) for _ in spectrum.shape]
    padding[-1] = (0, 2)
    padded = np.pad(spectrum, padding, mode="constant")
    for index, step in enumerate(time_steps):
        base = int(step)
        columns = padded[..., base : base + 2]
        alpha = float(np.mod(step, 1.0))
        magnitude = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])
        stretched[..., index] = magnitude * np.exp(1j * phase_acc)
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance
        dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase
    output = librosa.istft(stretched, hop_length=hop_length, length=target_length)
    return np.asarray(output, dtype=np.float32)


def _render_tempo_release(
    audio: np.ndarray,
    sample_rate: int,
    plan: TransitionPlan,
    *,
    gain: float,
    output_seconds: float | None = None,
) -> tuple[np.ndarray, float]:
    """Render the post-handoff deck-B tempo glide and return source consumed."""
    total = max(0.0, float(plan.tempo_release_seconds_b))
    if total <= 0.0:
        return np.zeros((audio.shape[0], 0), dtype=np.float32), 0.0
    requested = total if output_seconds is None else max(0.0, min(total, float(output_seconds)))
    if requested <= 0.0:
        return np.zeros((audio.shape[0], 0), dtype=np.float32), 0.0
    fraction = requested / total
    start_rate = max(0.01, float(plan.tempo_release_start_rate_b))
    final_rate = max(0.01, float(plan.tempo_release_end_rate_b))
    end_rate = float(_smooth_rate(start_rate, final_rate, fraction))
    integrated = _rate_curve_integral(start_rate, final_rate, fraction)
    source_seconds = total * integrated
    source = _extract_with_padding(
        audio,
        sample_rate,
        plan.source_end_b,
        source_seconds,
    )
    target_length = max(1, int(round(requested * sample_rate)))
    rendered = _time_stretch_rate_curve(
        source,
        start_rate=start_rate,
        end_rate=end_rate,
        target_length=target_length,
    )
    return np.asarray(rendered * gain, dtype=np.float32), float(source_seconds)


def _match_gain(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_rms = rms(reference)
    candidate_rms = rms(candidate)
    if reference_rms <= 1e-8 or candidate_rms <= 1e-8:
        return 1.0
    raw_gain = reference_rms / candidate_rms
    minimum = 10.0 ** (-6.0 / 20.0)
    maximum = 10.0 ** (6.0 / 20.0)
    return float(np.clip(raw_gain, minimum, maximum))


def _add_echo(y: np.ndarray, sample_rate: int, bpm: float) -> np.ndarray:
    result = np.array(y, copy=True, dtype=np.float32)
    beat_seconds = 60.0 / max(bpm, 1.0)
    for beats, gain in ((0.5, 0.34), (1.0, 0.22), (1.5, 0.13)):
        delay = int(round(beats * beat_seconds * sample_rate))
        if 0 < delay < y.shape[-1]:
            result[:, delay:] += gain * y[:, :-delay]
    return result


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


def _edge_fades(y: np.ndarray, sample_rate: int, milliseconds: float = 20.0) -> np.ndarray:
    length = min(y.shape[-1] // 2, int(sample_rate * milliseconds / 1000.0))
    if length <= 1:
        return y
    result = np.array(y, copy=True)
    fade = np.linspace(0.0, 1.0, length, dtype=np.float32)
    result[:, :length] *= fade
    result[:, -length:] *= fade[::-1]
    return result


def _fade_in(y: np.ndarray, sample_rate: int, milliseconds: float = 18.0) -> np.ndarray:
    length = min(y.shape[-1], int(sample_rate * milliseconds / 1000.0))
    if length <= 1:
        return y
    result = np.array(y, copy=True)
    result[:, :length] *= np.linspace(0.0, 1.0, length, dtype=np.float32)
    return result


def _fade_out(y: np.ndarray, sample_rate: int, milliseconds: float = 25.0) -> np.ndarray:
    length = min(y.shape[-1], int(sample_rate * milliseconds / 1000.0))
    if length <= 1:
        return y
    result = np.array(y, copy=True)
    result[:, -length:] *= np.linspace(1.0, 0.0, length, dtype=np.float32)
    return result


def _render_segment(
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    *,
    rate: float,
    gain: float,
    target_length: int | None = None,
) -> np.ndarray:
    start = max(0.0, float(start_seconds))
    end = max(start, float(end_seconds))
    segment = _extract_with_padding(audio, sample_rate, start, end - start)
    if target_length is None:
        target_length = max(0, int(round(segment.shape[-1] / max(rate, 1e-9))))
    if target_length <= 0:
        return np.zeros((audio.shape[0], 0), dtype=np.float32)
    if abs(rate - 1.0) >= 0.002:
        segment = _time_stretch(segment, rate, target_length)
    else:
        segment = pad_to_length(segment, target_length).astype(np.float32)
    return np.asarray(segment * gain, dtype=np.float32)


def _mix_transition(
    a: np.ndarray,
    b: np.ndarray,
    *,
    sample_rate: int,
    bpm: float,
    technique: str,
) -> np.ndarray:
    length = min(a.shape[-1], b.shape[-1])
    a = a[:, :length]
    b = b[:, :length]
    a_equal, b_equal = _equal_power(length)

    if technique == "equal_power_blend":
        return a * a_equal + b * b_equal

    low_a, high_a = _split_low_high(a, sample_rate)
    low_b, high_b = _split_low_high(b, sample_rate)

    if technique == "eq_bass_swap":
        low_out = 1.0 - _smoothstep(length, 0.34, 0.56)
        low_in = _smoothstep(length, 0.48, 0.70)
        return high_a * a_equal + high_b * b_equal + low_a * low_out + low_b * low_in

    if technique == "filter_build":
        a_out = 1.0 - _smoothstep(length, 0.25, 0.92)
        b_high_in = _smoothstep(length, 0.04, 0.68)
        b_low_in = _smoothstep(length, 0.56, 0.94)
        return a * a_out + high_b * b_high_in + low_b * b_low_in

    if technique == "echo_out_drop_in":
        echoed_a = _add_echo(a, sample_rate, bpm)
        outgoing = echoed_a * (1.0 - _smoothstep(length, 0.04, 0.90))
        activity = np.flatnonzero(np.max(np.abs(b), axis=0) > 1e-7)
        start_fraction = (
            float(activity[0]) / max(length - 1, 1)
            if activity.size
            else 0.80
        )
        incoming = b * _smoothstep(
            length,
            max(0.0, start_fraction - 0.02),
            min(1.0, start_fraction + 0.16),
        )
        return outgoing + incoming

    if technique == "quick_cut":
        cut = max(1, int(length * 0.18))
        result = np.zeros_like(a)
        a_fade, b_fade = _equal_power(cut)
        pivot = max(0, length // 2 - cut // 2)
        result[:, :pivot] = a[:, :pivot]
        result[:, pivot : pivot + cut] = a[:, pivot : pivot + cut] * a_fade + b[:, pivot : pivot + cut] * b_fade
        result[:, pivot + cut :] = b[:, pivot + cut :]
        return result

    if technique == "backspin":
        spun = np.flip(a, axis=-1).copy()
        spun *= 1.0 - _smoothstep(length, 0.10, 0.72)
        b_in = _smoothstep(length, 0.48, 0.76)
        return spun + b * b_in

    if technique == "loop_roll":
        beat_samples = max(64, int(round(sample_rate * 60.0 / max(bpm, 1.0))))
        source_start = max(0, length - beat_samples)
        loop = a[:, source_start:] if a[:, source_start:].shape[-1] else a
        repeats = int(np.ceil(length / max(loop.shape[-1], 1)))
        rolled = np.tile(loop, (1, repeats))[:, :length]
        a_out = 1.0 - _smoothstep(length, 0.30, 0.92)
        b_in = _smoothstep(length, 0.16, 0.78)
        return rolled * a_out + b * b_in

    if technique == "double_drop":
        pivot = length // 2
        a_gain = 1.0 - _smoothstep(length, 0.42, 0.60)
        b_gain = _smoothstep(length, 0.05, 0.48)
        mixed = a * a_gain + b * b_gain
        if pivot < length:
            mixed[:, pivot:] = b[:, pivot:]
        return mixed

    if technique == "stem_vocal_overlay":
        low_out = 1.0 - _smoothstep(length, 0.30, 0.55)
        low_in = _smoothstep(length, 0.55, 0.80)
        return high_a * a_equal + high_b * b_equal + low_a * low_out + low_b * low_in

    raise ValueError(f"Unknown transition technique: {technique}")



STEM_AWARE_TECHNIQUES = {
    "eq_bass_swap",
    "filter_build",
    "double_drop",
    "stem_vocal_overlay",
    "stem_choreography",
}

LAYER_AWARE_TECHNIQUES = {
    "layer_vocal_bridge",
    "layer_incoming_vocal_handoff",
    "layer_instrumental_handoff",
}


def _render_plan_stems(
    stem_paths: dict[str, str],
    *,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    rate: float,
    gain: float,
    target_length: int,
) -> dict[str, np.ndarray]:
    required = ("vocals", "drums", "bass", "other")
    if not all(stem_paths.get(name) for name in required):
        raise RuntimeError("The compiled stem-aware plan is missing one or more stems.")
    rendered: dict[str, np.ndarray] = {}
    for name in required:
        source, loaded_sr = load_audio(
            stem_paths[name], sample_rate=sample_rate, mono=False
        )
        if loaded_sr != sample_rate:
            raise RuntimeError("Unexpected stem sample-rate mismatch.")
        rendered[name] = _render_segment(
            source,
            sample_rate,
            start_seconds,
            end_seconds,
            rate=rate,
            gain=gain,
            target_length=target_length,
        )
    return rendered


def _performance_curve_array(payload: dict[str, object], group: str, name: str, length: int) -> np.ndarray:
    curves = payload.get(group)
    if not isinstance(curves, dict):
        raise RuntimeError(f"Performance program is missing {group} curves.")
    points = curves.get(name)
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"Performance program is missing curve {group}.{name}.")
    positions: list[float] = []
    values: list[float] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        positions.append(float(point.get("position", 0.0)))
        values.append(float(point.get("value", 0.0)))
    if not positions:
        raise RuntimeError(f"Performance program curve {group}.{name} is empty.")
    x = np.linspace(0.0, 1.0, max(1, length), dtype=np.float64)
    return np.interp(x, positions, values).astype(np.float32)


def _render_stem_aware_transition_window(
    plan: TransitionPlan,
    *,
    sample_rate: int,
    gain_a: float,
    gain_b: float,
    target_length: int,
) -> np.ndarray:
    """Compatibility wrapper around the 6.0 generic program engine.

    Production rendering enters through `_render_transition_window`, which
    executes the signed program directly. This helper intentionally compiles
    from the supplied plan so low-level tests/tools that attach stems after
    constructing a synthetic plan still exercise the same DSP engine.
    """
    program = compile_transition_program(plan)
    if program.source_mode != "stems":
        raise RuntimeError("The supplied plan does not compile to a stem-source program.")
    stems_a = _render_plan_stems(
        plan.stem_paths_a,
        sample_rate=sample_rate,
        start_seconds=plan.exit_time_a,
        end_seconds=plan.source_end_a,
        rate=plan.render_rate_a,
        gain=gain_a,
        target_length=target_length,
    )
    stems_b = _render_plan_stems(
        plan.stem_paths_b,
        sample_rate=sample_rate,
        start_seconds=plan.source_start_b,
        end_seconds=plan.source_end_b,
        rate=plan.render_rate_b,
        gain=gain_b,
        target_length=target_length,
    )
    sources = {
        **{f"a:{name}": audio for name, audio in stems_a.items()},
        **{f"b:{name}": audio for name, audio in stems_b.items()},
    }
    return execute_transition_program(
        program,
        sources,
        sample_rate=sample_rate,
        bpm=plan.target_bpm,
    ).audio



def _layer_path(payload: dict[str, object], role: str) -> tuple[str, float] | None:
    value = payload.get(role)
    if not isinstance(value, dict):
        return None
    path = value.get("path")
    if not path:
        return None
    return str(path), float(value.get("offset_seconds", 0.0))


def _render_supplied_layer(
    payload: dict[str, object],
    role: str,
    *,
    sample_rate: int,
    start_seconds: float,
    end_seconds: float,
    rate: float,
    target_length: int,
) -> np.ndarray | None:
    source_info = _layer_path(payload, role)
    if source_info is None:
        return None
    path, offset = source_info
    source, loaded_sr = load_audio(path, sample_rate=sample_rate, mono=False)
    if loaded_sr != sample_rate:
        raise RuntimeError("Unexpected supplied-layer sample-rate mismatch.")
    return _render_segment(
        source,
        sample_rate,
        max(0.0, start_seconds + offset),
        max(0.0, end_seconds + offset),
        rate=rate,
        gain=1.0,
        target_length=target_length,
    )


def _match_layer_level(
    layer: np.ndarray | None,
    reference: np.ndarray,
    *,
    target_ratio: float = 1.0,
) -> np.ndarray | None:
    if layer is None:
        return None
    layer_rms = rms(layer)
    reference_rms = rms(reference)
    if layer_rms <= 1e-8 or reference_rms <= 1e-8:
        return layer
    gain = float(
        np.clip(
            target_ratio * reference_rms / layer_rms,
            10 ** (-7.0 / 20.0),
            10 ** (7.0 / 20.0),
        )
    )
    return np.asarray(layer * gain, dtype=np.float32)


def _render_layer_program_sources(
    part_a: np.ndarray,
    part_b: np.ndarray,
    plan: TransitionPlan,
    *,
    program: object,
    sample_rate: int,
    target_length: int,
) -> dict[str, np.ndarray]:
    resolved = program if hasattr(program, "layer_curves_a") else transition_program_from_dict(dict(program))
    layers_a = dict(plan.performance_layers_a or {})
    layers_b = dict(plan.performance_layers_b or {})
    ratios = dict(getattr(resolved, "annotations", {}).get("layer_level_ratios", {}) or {})
    sources: dict[str, np.ndarray] = {"a:master": part_a, "b:master": part_b}

    for deck, bundle, start, end, rate, reference in (
        ("a", layers_a, plan.exit_time_a, plan.source_end_a, plan.render_rate_a, part_a),
        ("b", layers_b, plan.source_start_b, plan.source_end_b, plan.render_rate_b, part_b),
    ):
        curves = resolved.layer_curves_a if deck == "a" else resolved.layer_curves_b
        for role in curves:
            if role == "master":
                continue
            rendered = _render_supplied_layer(
                bundle,
                role,
                sample_rate=sample_rate,
                start_seconds=start,
                end_seconds=end,
                rate=rate,
                target_length=target_length,
            )
            if rendered is None:
                continue
            ratio = float(ratios.get(f"{deck}:{role}", 1.0))
            matched = _match_layer_level(rendered, reference, target_ratio=ratio)
            if matched is not None:
                sources[f"{deck}:{role}"] = matched
    return sources


def _render_layer_aware_transition_window(
    part_a: np.ndarray,
    part_b: np.ndarray,
    plan: TransitionPlan,
    *,
    sample_rate: int,
    gain_a: float,
    gain_b: float,
    target_length: int,
) -> np.ndarray:
    """Compatibility wrapper around the 6.0 generic program engine."""
    del gain_a, gain_b
    program = compile_transition_program(plan)
    if program.source_mode != "layers":
        raise RuntimeError("The supplied plan does not compile to a layer-source program.")
    sources = _render_layer_program_sources(
        part_a,
        part_b,
        plan,
        program=program,
        sample_rate=sample_rate,
        target_length=target_length,
    )
    return execute_transition_program(
        program,
        sources,
        sample_rate=sample_rate,
        bpm=plan.target_bpm,
    ).audio


def _render_transition_window(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    plan: TransitionPlan,
    *,
    sample_rate: int,
    gain_a: float = 1.0,
    gain_b: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Execute exactly the immutable 6.0 TransitionProgram.

    Source extraction remains an execution concern; musical strategy does not.
    The signed program decides source mode, bus automation, processors and
    incoming alignment. No production branch dispatches on `plan.technique`.
    """
    validate_execution_plan(plan)
    program = transition_program_from_dict(dict(plan.performance_program or {}))
    target_length = max(1, int(round(plan.transition_seconds * sample_rate)))
    part_a = _render_segment(
        audio_a,
        sample_rate,
        plan.exit_time_a,
        plan.source_end_a,
        rate=plan.render_rate_a,
        gain=gain_a,
        target_length=target_length,
    )

    if plan.source_duration_b > 0.0:
        rendered_b_seconds = plan.source_duration_b / max(plan.render_rate_b, 1e-9)
        rendered_b_length = min(
            target_length,
            max(1, int(round(rendered_b_seconds * sample_rate))),
        )
        alignment = str(program.annotations.get("incoming_alignment", "full"))
        incoming_length = rendered_b_length if alignment == "tail" else target_length
        phase_integrity = dict(program.annotations.get("deck_performance", {}) or {}).get("phase_integrity", {})
        phase_points = (
            list(phase_integrity.get("points", []))
            if isinstance(phase_integrity, dict)
            and bool(phase_integrity.get("enabled", False))
            and bool(phase_integrity.get("safe", False))
            and alignment == "full"
            else []
        )
        if phase_points:
            source = _extract_with_padding(
                audio_b,
                sample_rate,
                plan.source_start_b,
                plan.source_duration_b,
            )
            incoming = _time_stretch_phase_warp(
                source,
                sample_rate=sample_rate,
                base_rate=plan.render_rate_b,
                target_length=incoming_length,
                correction_points=phase_points,
            )
        else:
            incoming = _render_segment(
                audio_b,
                sample_rate,
                plan.source_start_b,
                plan.source_end_b,
                rate=plan.render_rate_b,
                gain=1.0,
                target_length=incoming_length,
            )
        if alignment == "tail" and incoming_length < target_length:
            part_b = np.zeros_like(part_a)
            part_b[:, target_length - incoming_length :] = incoming[:, -incoming_length:]
        else:
            part_b = incoming
    else:
        part_b = np.zeros_like(part_a)

    if gain_b is None:
        # Pair-mode backwards compatibility: infer a bounded local level when
        # no exact set render context was supplied. Set mode always supplies
        # the fixed normalization gain rehearsed for final execution.
        probe = min(target_length, int(sample_rate * 8.0))
        gain_b = _match_gain(part_a[:, :probe], part_b[:, :probe]) if plan.source_duration_b > 0 else 1.0
    part_b *= gain_b

    if program.source_mode == "master":
        sources: dict[str, np.ndarray] = {"deck_a": part_a, "deck_b": part_b}
    elif program.source_mode == "stems":
        stems_a = _render_plan_stems(
            plan.stem_paths_a,
            sample_rate=sample_rate,
            start_seconds=plan.exit_time_a,
            end_seconds=plan.source_end_a,
            rate=plan.render_rate_a,
            gain=gain_a,
            target_length=target_length,
        )
        stems_b = _render_plan_stems(
            plan.stem_paths_b,
            sample_rate=sample_rate,
            start_seconds=plan.source_start_b,
            end_seconds=plan.source_end_b,
            rate=plan.render_rate_b,
            gain=float(gain_b),
            target_length=target_length,
        )
        sources = {
            **{f"a:{name}": audio for name, audio in stems_a.items()},
            **{f"b:{name}": audio for name, audio in stems_b.items()},
        }
    elif program.source_mode == "layers":
        sources = _render_layer_program_sources(
            part_a,
            part_b,
            plan,
            program=program,
            sample_rate=sample_rate,
            target_length=target_length,
        )
    else:  # pragma: no cover - validated by transition_program_from_dict
        raise RuntimeError(f"Unsupported performance source mode: {program.source_mode}")

    mixed = execute_transition_program(
        program,
        sources,
        sample_rate=sample_rate,
        bpm=plan.target_bpm,
    ).audio
    if (
        os.environ.get("DJAI_PHASE6_RENDER_TRACE", "0").strip().lower()
        in {"1", "true", "yes", "on"}
        and plan.technique == "echo_out_drop_in"
        and program.source_mode == "master"
    ):
        # Optional diagnostics only: rerun the same immutable program with
        # one deck muted at a time, never modifying the authoritative mix.
        def _level_db(segment: np.ndarray) -> float:
            energy = float(np.mean(np.square(np.asarray(segment, dtype=np.float64))))
            return round(10.0 * np.log10(max(energy, 1e-12)), 3)

        frames = min(mixed.shape[-1], max(1, int(round(sample_rate * 0.6))))
        deck_a_only = execute_transition_program(
            program,
            {"deck_a": sources["deck_a"], "deck_b": np.zeros_like(sources["deck_b"])},
            sample_rate=sample_rate,
            bpm=plan.target_bpm,
        ).audio
        deck_b_only = execute_transition_program(
            program,
            {"deck_a": np.zeros_like(sources["deck_a"]), "deck_b": sources["deck_b"]},
            sample_rate=sample_rate,
            bpm=plan.target_bpm,
        ).audio
        def _levels(audio: np.ndarray) -> dict[str, float]:
            return {
                "head_rms_dbfs": _level_db(audio[..., :frames]),
                "tail_rms_dbfs": _level_db(audio[..., -frames:]),
            }

        print(
            "\n=== PHASE 6 RENDER DECK LEVELS ===\n"
            + json.dumps({
                "track_a": Path(plan.track_a).name,
                "track_b": Path(plan.track_b).name,
                "exit_time_a_seconds": plan.exit_time_a,
                "entry_time_b_seconds": plan.entry_time_b,
                "transition_seconds": plan.transition_seconds,
                "gain_a": gain_a,
                "gain_b": float(gain_b),
                "source_a": _levels(sources["deck_a"]),
                "source_b": _levels(sources["deck_b"]),
                "rendered_deck_a_only": _levels(deck_a_only),
                "rendered_deck_b_only": _levels(deck_b_only),
                "rendered_mix": _levels(mixed),
            }, indent=2),
            flush=True,
        )
    return mixed, plan.source_duration_a, plan.source_duration_b, float(gain_b)


def render_transition_audio_from_loaded(
    track_a: np.ndarray,
    track_b: np.ndarray,
    plan: TransitionPlan,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    pre_roll_seconds: float = 8.0,
    post_roll_seconds: float = 8.0,
    render_context: TransitionRenderContext | None = None,
) -> RenderedTransitionAudio:
    """Render preview components from already-loaded source audio.

    Candidate measurement calls this function so several plans for the same
    track pair share audio decoding while using the production DSP path.
    """
    if pre_roll_seconds < 0.0 or post_roll_seconds < 0.0:
        raise ValueError("Pre-roll and post-roll must be non-negative.")
    validate_execution_plan(plan)
    resolved_context = render_context.normalized() if render_context is not None else None
    if resolved_context is not None and resolved_context.sample_rate != sample_rate:
        raise ValueError(
            "Transition render context sample rate does not match the loaded rehearsal audio."
        )
    gain_a = resolved_context.gain_a if resolved_context is not None else 1.0
    fixed_gain_b = resolved_context.gain_b if resolved_context is not None else None
    pre_source = pre_roll_seconds * plan.render_rate_a
    pre = _soft_limit(
        _render_segment(
            track_a,
            sample_rate,
            max(0.0, plan.exit_time_a - pre_source),
            plan.exit_time_a,
            rate=plan.render_rate_a,
            gain=gain_a,
            target_length=int(round(pre_roll_seconds * sample_rate)),
        )
    )
    transition, _, _, gain_b = _render_transition_window(
        track_a,
        track_b,
        plan,
        sample_rate=sample_rate,
        gain_a=gain_a,
        gain_b=fixed_gain_b,
    )
    release_preview_seconds = min(post_roll_seconds, max(0.0, plan.tempo_release_seconds_b))
    release, _ = _render_tempo_release(
        track_b,
        sample_rate,
        plan,
        gain=float(gain_b),
        output_seconds=release_preview_seconds,
    )
    remaining_post = max(0.0, post_roll_seconds - release_preview_seconds)
    if remaining_post > 0.0:
        native_post_source = remaining_post * plan.post_transition_rate_b
        native_post = _render_segment(
            track_b,
            sample_rate,
            plan.post_transition_source_end_b,
            plan.post_transition_source_end_b + native_post_source,
            rate=plan.post_transition_rate_b,
            gain=gain_b,
            target_length=int(round(remaining_post * sample_rate)),
        )
        post = _soft_limit(np.concatenate([release, native_post], axis=-1))
    else:
        post = _soft_limit(release)
    post = pad_to_length(post, int(round(post_roll_seconds * sample_rate))).astype(np.float32)
    preview = _edge_fades(np.concatenate([pre, transition, post], axis=-1), sample_rate)
    return RenderedTransitionAudio(
        pre=pre,
        transition=transition,
        post=post,
        preview=preview,
        gain_b=float(gain_b),
        sample_rate=sample_rate,
    )


def render_transition_audio(
    plan: TransitionPlan,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    pre_roll_seconds: float = 8.0,
    post_roll_seconds: float = 8.0,
) -> RenderedTransitionAudio:
    validate_execution_plan(plan)
    track_a, sr_a = load_audio(plan.track_a, sample_rate=sample_rate, mono=False)
    track_b, sr_b = load_audio(plan.track_b, sample_rate=sample_rate, mono=False)
    if sr_a != sample_rate or sr_b != sample_rate:
        raise RuntimeError("Unexpected sample-rate mismatch after loading audio.")
    return render_transition_audio_from_loaded(
        track_a,
        track_b,
        plan,
        sample_rate=sample_rate,
        pre_roll_seconds=pre_roll_seconds,
        post_roll_seconds=post_roll_seconds,
    )


def render_transition(
    plan: TransitionPlan,
    output_path: str | Path,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
    pre_roll_seconds: float = 8.0,
    post_roll_seconds: float = 8.0,
) -> Path:
    """Render a stereo preview that uses the exact compiled execution plan."""
    rendered = render_transition_audio(
        plan,
        sample_rate=sample_rate,
        pre_roll_seconds=pre_roll_seconds,
        post_roll_seconds=post_roll_seconds,
    )
    return write_audio(output_path, rendered.preview, sample_rate)


def render_full_mix(
    plan: TransitionPlan,
    output_path: str | Path,
    *,
    sample_rate: int = DEFAULT_RENDER_SAMPLE_RATE,
) -> Path:
    """Render full Track A -> exact transition -> remainder of Track B."""
    validate_execution_plan(plan)
    track_a, sr_a = load_audio(plan.track_a, sample_rate=sample_rate, mono=False)
    track_b, sr_b = load_audio(plan.track_b, sample_rate=sample_rate, mono=False)
    if sr_a != sample_rate or sr_b != sample_rate:
        raise RuntimeError("Unexpected sample-rate mismatch after loading audio.")

    body_a = _fade_in(
        _soft_limit(
            _render_segment(
                track_a,
                sample_rate,
                0.0,
                plan.exit_time_a,
                rate=plan.render_rate_a,
                gain=1.0,
            )
        ),
        sample_rate,
    )
    transition, _, _, gain_b = _render_transition_window(
        track_a,
        track_b,
        plan,
        sample_rate=sample_rate,
        gain_a=1.0,
        gain_b=None,
    )
    release, _ = _render_tempo_release(
        track_b,
        sample_rate,
        plan,
        gain=float(gain_b),
    )
    body_b = _fade_out(
        _soft_limit(
            _render_segment(
                track_b,
                sample_rate,
                plan.post_transition_source_end_b,
                track_b.shape[-1] / sample_rate,
                rate=plan.post_transition_rate_b,
                gain=gain_b,
            )
        ),
        sample_rate,
    )
    full = _edge_fades(np.concatenate([body_a, transition, release, body_b], axis=-1), sample_rate)
    return write_audio(output_path, full, sample_rate)
