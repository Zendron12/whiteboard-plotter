from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy


@dataclass(frozen=True)
class DetectionResult:
    input_type: str
    confidence: float
    reason: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_bgr(image_bytes: bytes) -> numpy.ndarray:
    encoded = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('uploaded raster image could not be decoded')
    return image


def _forced_result(input_type: str, *, reason: str) -> DetectionResult:
    return DetectionResult(
        input_type=input_type,
        confidence=1.0,
        reason=reason,
        metrics={'forced': True},
    )


def detect_raster_input_type(
    image_bytes: bytes,
    *,
    filename: str = '',
    content_type: str = '',
    requested_input_type: str = 'auto',
) -> DetectionResult:
    requested = str(requested_input_type or 'auto').strip().lower()
    if requested == 'sketch_image':
        return _forced_result('sketch_image', reason='forced sketch/line-art input')
    if requested in {'colored_image', 'image'}:
        return _forced_result('colored_image', reason='forced colored image input')
    if requested != 'auto':
        raise ValueError('requested_input_type must be one of: auto, sketch_image, colored_image, image')

    image = _decode_bgr(image_bytes)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(numpy.float32)
    value = hsv[:, :, 2].astype(numpy.float32)
    color_mask = numpy.logical_and(saturation > 35.0, value > 35.0)
    color_pixel_ratio = float(numpy.count_nonzero(color_mask)) / float(max(1, height * width))
    mean_saturation = float(numpy.mean(saturation))
    p90_saturation = float(numpy.percentile(saturation, 90.0))
    channels = image.astype(numpy.int16)
    channel_spread = numpy.max(channels, axis=2) - numpy.min(channels, axis=2)
    mean_channel_spread = float(numpy.mean(channel_spread))
    filename_suffix = Path(filename or '').suffix.lower()
    normalized_content_type = str(content_type or '').split(';', 1)[0].strip().lower()

    colored_score = 0.0
    if color_pixel_ratio >= 0.08:
        colored_score += min(0.45, color_pixel_ratio * 1.8)
    if mean_saturation >= 22.0:
        colored_score += min(0.25, mean_saturation / 180.0)
    if p90_saturation >= 55.0:
        colored_score += 0.2
    if mean_channel_spread >= 8.0:
        colored_score += min(0.1, mean_channel_spread / 255.0)
    colored_score = max(0.0, min(1.0, colored_score))

    metrics = {
        'width_px': int(width),
        'height_px': int(height),
        'color_pixel_ratio': color_pixel_ratio,
        'mean_saturation': mean_saturation,
        'p90_saturation': p90_saturation,
        'mean_channel_spread': mean_channel_spread,
        'filename_suffix': filename_suffix,
        'content_type': normalized_content_type,
        'forced': False,
    }
    if colored_score >= 0.42:
        return DetectionResult(
            input_type='colored_image',
            confidence=colored_score,
            reason='high saturation/color-channel variation detected',
            metrics=metrics,
        )
    return DetectionResult(
        input_type='sketch_image',
        confidence=1.0 - colored_score,
        reason='low saturation raster treated as sketch/line-art',
        metrics=metrics,
    )
