from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from ..audio import load_audio
from ..models import TrackAnalysis, TransitionCandidate
from .edm_cue import EDM_CUE_MODEL_ID

# CUE-DETR official inference constants.
# Source: ETH-DISCO/cue-detr cue_points.py
CUE_DETR_SAMPLE_RATE = 22050
CUE_DETR_N_MELS = 128
CUE_DETR_N_FFT = 2048
CUE_DETR_HOP_LENGTH = 512
CUE_DETR_WINDOW_FRAMES = 355
CUE_DETR_WINDOW_OVERLAP = 0.75
CUE_DETR_PADDING = 266
CUE_DETR_CONFIDENCE = 0.90
CUE_DETR_PROCESSOR_ID = "facebook/detr-resnet-50"


@dataclass(frozen=True, slots=True)
class CuePrediction:
    time_seconds: float
    confidence: float


def _model_device(requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # CUE-DETR/DETR can hit unsupported MPS operators depending on Torch.
    # Prefer MPS when available, but inference has a CPU retry below.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_hf_snapshot(
    repo_id: str,
    filenames: tuple[str, ...],
    *,
    allow_download: bool,
) -> Path:
    """Resolve required Hugging Face files to one concrete local snapshot.

    Loading Transformers components directly from a remote model ID has proven
    brittle across recent Transformers / huggingface_hub combinations.  DJAI
    therefore resolves the exact files with huggingface_hub first and then asks
    Transformers to load only from that local snapshot directory.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'CUE-DETR requires huggingface_hub. Install DJAI with ".[cue]".'
        ) from exc

    local_only = not allow_download
    resolved: list[Path] = []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=local_only,
            )
        )
        if not path.is_file():
            raise RuntimeError(f"Could not resolve {filename!r} for {repo_id!r}.")
        resolved.append(path)

    snapshot = resolved[0].parent
    if any(path.parent != snapshot for path in resolved[1:]):
        raise RuntimeError(f"Hugging Face files for {repo_id!r} resolved to different snapshots.")
    return snapshot


def _resolve_model_snapshot(model_id: str, *, allow_download: bool) -> Path:
    """Resolve the published CUE-DETR checkpoint to a concrete local directory."""
    return _resolve_hf_snapshot(
        model_id,
        ("config.json", "model.safetensors"),
        allow_download=allow_download,
    )


def _resolve_processor_snapshot(*, allow_download: bool) -> Path:
    """Resolve the Facebook DETR image-processor config locally first."""
    return _resolve_hf_snapshot(
        CUE_DETR_PROCESSOR_ID,
        ("preprocessor_config.json",),
        allow_download=allow_download,
    )


def _load_model(
    model_id: str,
    *,
    allow_download: bool,
    device: str,
) -> tuple[Any, Any]:
    """Load CUE-DETR using the exact processor family used by the authors.

    The official repository uses DetrImageProcessor from
    ``facebook/detr-resnet-50`` and DetrForObjectDetection for the CUE-DETR
    checkpoint.  It does not use an image processor stored in the CUE-DETR
    repository itself.
    """
    try:
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'CUE-DETR requires Transformers. Install with: pip install -e ".[cue]"'
        ) from exc

    # The processor belongs to the base Facebook DETR model, exactly as in the
    # official ETH-DISCO inference script. Resolve its config explicitly first
    # and then load from the concrete local snapshot. This avoids remote model-
    # ID resolution failures in recent Transformers / huggingface_hub versions.
    processor_snapshot = _resolve_processor_snapshot(allow_download=allow_download)
    processor = DetrImageProcessor.from_pretrained(
        str(processor_snapshot),
        local_files_only=True,
    )

    # Resolve the fine-tuned checkpoint first.  Loading from a concrete local
    # snapshot avoids remote weight-resolution bugs seen in newer Transformers.
    snapshot = _resolve_model_snapshot(model_id, allow_download=allow_download)

    # low_cpu_mem_usage=False avoids meta-device initialization paths that can
    # produce "copying ... to a meta parameter is a no-op" warnings with some
    # recent Torch/Transformers combinations.
    model = DetrForObjectDetection.from_pretrained(
        str(snapshot),
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.to(device)
    return model, processor


def _mel_to_viridis_rgb(audio: np.ndarray, sr: int) -> np.ndarray:
    """Reproduce the official CUE-DETR spectrogram-image conversion."""
    try:
        from matplotlib import cm
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'CUE-DETR requires matplotlib and Pillow. Install DJAI with ".[cue]".'
        ) from exc

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=CUE_DETR_N_FFT,
        hop_length=CUE_DETR_HOP_LENGTH,
        n_mels=CUE_DETR_N_MELS,
    )
    if mel.size == 0 or mel.shape[1] == 0:
        return np.zeros((CUE_DETR_N_MELS, 0, 3), dtype=np.uint8)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Official code flips the spectrogram vertically, maps through viridis,
    # materializes an RGBA PIL image, then discards alpha.
    arr = mel_db[::-1]
    sm = cm.ScalarMappable(cmap="viridis")
    sm.set_clim(float(arr.min()), float(arr.max()))
    rgba = sm.to_rgba(arr, bytes=True)
    rgb_shape = (rgba.shape[1], rgba.shape[0])
    rgba = np.require(rgba, requirements="C")
    im = Image.frombuffer("RGBA", rgb_shape, rgba, "raw", "RGBA", 0, 1)
    image = np.asarray(im)[:, :, :3]
    return np.ascontiguousarray(image)


def _sliding_windows(image: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """Create the official padded 355-frame, 75%-overlap image windows."""
    if image.shape[1] == 0:
        return [], []

    image_w = image.shape[1] + CUE_DETR_PADDING
    step = CUE_DETR_WINDOW_FRAMES * (1.0 - CUE_DETR_WINDOW_OVERLAP)
    n_windows = int(np.floor(image_w / step))

    images: list[np.ndarray] = []
    borders: list[int] = []

    for i in range(n_windows):
        left = int(np.floor(i * step)) - CUE_DETR_PADDING
        right = left + CUE_DETR_WINDOW_FRAMES
        borders.append(left)

        if left < 0:
            segment = image[:, :right]
            pad = -left
            segment = np.pad(
                segment,
                ((0, 0), (pad, 0), (0, 0)),
                mode="linear_ramp",
            )
        elif right > image.shape[1]:
            segment = image[:, left:]
            pad = right - left - segment.shape[1]
            segment = np.pad(
                segment,
                ((0, 0), (0, pad), (0, 0)),
                mode="linear_ramp",
            )
        else:
            segment = image[:, left:right]

        # Defensive geometry check; the official model expects 128 x 355.
        if segment.shape[1] != CUE_DETR_WINDOW_FRAMES:
            delta = CUE_DETR_WINDOW_FRAMES - segment.shape[1]
            if delta > 0:
                segment = np.pad(segment, ((0, 0), (0, delta), (0, 0)), mode="linear_ramp")
            else:
                segment = segment[:, :CUE_DETR_WINDOW_FRAMES]
        images.append(np.ascontiguousarray(segment))

    return images, borders


def _run_model(model: Any, pixel_values: Any, *, device: str) -> tuple[Any, str]:
    """Run DETR and retry on CPU if Apple MPS rejects an operation."""
    try:
        return model(pixel_values), device
    except (RuntimeError, NotImplementedError) as exc:
        if device != "mps":
            raise
        message = str(exc).lower()
        if not any(token in message for token in ("mps", "not implemented", "unsupported")):
            raise
        model.to("cpu")
        return model(pixel_values.to("cpu")), "cpu"


def _scale_scores(scores: list[float]) -> np.ndarray:
    """Match the authors' global min/max normalization, safely."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def predict_cue_points(
    track: str | Path,
    *,
    model_id: str = EDM_CUE_MODEL_ID,
    confidence: float = CUE_DETR_CONFIDENCE,
    minimum_spacing_bars: int = 16,
    bpm: float | None = None,
    allow_download: bool | None = None,
    device: str | None = None,
) -> list[CuePrediction]:
    """Predict cue points using the published CUE-DETR inference recipe.

    This ports ETH-DISCO's ``cue_points.py`` pipeline into DJAI while preserving
    DJAI's local-first model policy and ``CuePrediction`` return type.

    The model sees a viridis Mel-spectrogram as overlapping 128x355 RGB image
    windows.  DETR bounding-box x-centres are mapped back to spectrogram frames,
    scores are normalized over the whole track, and scipy peak detection yields
    final cue positions.
    """
    resolved = Path(track).expanduser().resolve()
    audio, sr = load_audio(resolved, sample_rate=CUE_DETR_SAMPLE_RATE, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []

    # Retained for API compatibility / future DJAI post-filtering.  The official
    # CUE-DETR peak detector itself uses an index-distance radius, not BPM.
    _ = bpm

    if allow_download is None:
        allow_download = os.getenv("DJAI_CUE_MODEL_ALLOW_DOWNLOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
    selected_device = _model_device(device or os.getenv("DJAI_CUE_MODEL_DEVICE", "auto"))

    model, processor = _load_model(
        model_id,
        allow_download=bool(allow_download),
        device=selected_device,
    )

    try:
        import torch
        from scipy.signal import find_peaks
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            'CUE-DETR requires PyTorch and SciPy. Install DJAI with ".[cue]".'
        ) from exc

    image = _mel_to_viridis_rgb(audio, sr)
    images, borders = _sliding_windows(image)
    if not images:
        return []

    # Official preprocessing: no resize.  All windows stay 128 x 355 so DETR's
    # predicted x coordinates map directly back to time.
    encoding = processor.preprocess(images, do_resize=False, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(selected_device)

    with torch.no_grad():
        outputs, actual_device = _run_model(model, pixel_values, device=selected_device)

    # Official post-processing uses threshold=0 and target image size 128x355.
    target_sizes = [(CUE_DETR_N_MELS, CUE_DETR_WINDOW_FRAMES)] * pixel_values.shape[0]
    predictions = processor.post_process_object_detection(
        outputs,
        threshold=0.0,
        target_sizes=target_sizes,
    )

    raw_scores: list[float] = []
    raw_positions: list[int] = []
    for prediction, left in zip(predictions, borders, strict=True):
        raw_scores.extend(float(x) for x in prediction["scores"].detach().cpu().tolist())
        boxes = prediction["boxes"].detach().cpu()
        centers = ((boxes[:, 0] + boxes[:, 2]) // 2).long() + int(left)
        raw_positions.extend(int(x) for x in centers.tolist())

    if not raw_scores:
        return []

    scaled_scores = _scale_scores(raw_scores)
    ordered = sorted(zip(raw_positions, scaled_scores.tolist()), key=lambda item: item[0])
    if not ordered:
        return []

    positions = np.asarray([item[0] for item in ordered], dtype=np.int64)
    scores = np.asarray([item[1] for item in ordered], dtype=np.float64)

    # This intentionally mirrors the official script: sensitivity is a height
    # threshold after whole-track min/max scaling, and radius is passed as the
    # minimum sample distance in the sorted detection sequence.
    peak_idx, properties = find_peaks(
        scores,
        height=float(confidence),
        distance=max(1, int(minimum_spacing_bars)),
    )

    result: list[CuePrediction] = []
    heights = properties.get("peak_heights", np.asarray([], dtype=np.float64))
    max_frame = image.shape[1] - 1
    for j, idx in enumerate(peak_idx.tolist()):
        frame = int(positions[idx])
        # Padding can generate detections outside the actual spectrogram.
        if frame < 0 or frame > max_frame:
            continue
        seconds = float(librosa.frames_to_time(frame, sr=sr, hop_length=CUE_DETR_HOP_LENGTH))
        score = float(heights[j]) if j < len(heights) else float(scores[idx])
        result.append(CuePrediction(round(seconds, 6), round(score, 6)))

    result.sort(key=lambda item: item.time_seconds)
    return result


def cue_model_fingerprint(model_id: str = EDM_CUE_MODEL_ID) -> str:
    return hashlib.sha256(f"cue-detr-official-v2|{model_id}".encode("utf-8")).hexdigest()[:20]


def _candidate_boost(
    candidate: TransitionCandidate,
    predictions: list[CuePrediction],
    bpm: float,
) -> float:
    if not predictions:
        return 0.0
    beat_seconds = 60.0 / max(bpm, 1.0)
    nearest = min(predictions, key=lambda item: abs(item.time_seconds - candidate.time_seconds))
    distance_beats = abs(nearest.time_seconds - candidate.time_seconds) / beat_seconds
    proximity = float(np.exp(-0.5 * (distance_beats / 1.0) ** 2))
    return proximity * nearest.confidence


def rerank_analysis_candidates_with_cue_model(
    analysis: TrackAnalysis,
    predictions: list[CuePrediction],
    *,
    learned_weight: float = 0.45,
) -> TrackAnalysis:
    """Blend CUE-DETR evidence into DJAI's beat/downbeat-safe cue candidates.

    CUE-DETR contributes preference evidence only.  DJAI still owns the actual
    executable cue locations, so neural predictions cannot introduce off-grid
    transition timing.
    """
    weight = float(np.clip(learned_weight, 0.0, 0.75))
    for candidates in (analysis.entry_candidates, analysis.exit_candidates):
        for candidate in candidates:
            learned = _candidate_boost(candidate, predictions, analysis.bpm)
            candidate.score = round((1.0 - weight) * candidate.score + weight * learned, 6)
            candidate.phrase_strength = round(
                max(candidate.phrase_strength, min(1.0, learned)), 6
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
    return analysis
